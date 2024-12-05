import random
import numpy as np
import asyncio
from collections import deque
from queue import Queue
from itertools import product
from typing import List, Tuple, Optional, Union
from itertools import chain

import invertedai as iai
from invertedai import (
    async_drive, 
    async_initialize,
    drive, 
    initialize,
    light, 
    location_info
)
from invertedai.common import (
    AgentProperties,
    AgentState, 
    Image,
    InfractionIndicators, 
    Point,
    RecurrentState,
    TrafficLightStatesDict
)


class BasicCosimulation:
    """
    Stateful wrapper around the Inverted AI API to simplify co-simulation.
    All arguments to :func:`initialize` can be passed to the constructor here
    and a sufficient combination of them must be passed as required by :func:`initialize`.
    This wrapper caches static agent attributes and propagates the recurrent state,
    so that only states of ego agents and NPCs need to be exchanged with it to
    perform co-simulation. Typically, each time step requires a single call to
    :func:`self.npc_states` and a single call to :func:`self.step`.

    This wrapper only supports a minimal co-simulation functionality.
    For more advanced use cases, call :func:`initialize` and :func:`drive` directly.

    :param location: Location name as expected by :func:`initialize`.
    :param ego_agent_mask: List indicating which agent is ego, meaning that it is
        controlled by you externally. The order in this list should be the same as that
        used in arguments to :func:`initialize`.
    :param monitor_infraction: Whether to monitor driving infractions, at a small increase
        in latency and payload size.
    :param get_birdview: Whether to render the bird's eye view of the simulation state
        at each time step. It drastically increases the payload received from Inverted AI
        servers and therefore slows down the simulation - use only for debugging.
    :param random_seed: Controls the stochastic aspects of simulation for reproducibility.
    """

    def __init__(
        self,
        location: str,
        ego_agent_mask: Optional[List[bool]] = None,
        monitor_infractions: bool = False,
        get_birdview: bool = False,
        random_seed: Optional[int] = None,
        traffic_lights: bool = False,
        # sufficient arguments to initialize must also be included
        **kwargs,
    ):
        self._location = location
        self.rng = None if random_seed is None else random.Random(random_seed)
        self.light_flag = False
        self.light_recurrent_state = None
        self._light_state = None
        if traffic_lights:
            location_info_response = location_info(location=location)
            static_actors = location_info_response.static_actors
            if any(actor.agent_type == "traffic_light" for actor in static_actors):
                self.light_flag = True
        response = initialize(
            location=location,
            get_birdview=get_birdview,
            get_infractions=monitor_infractions,
            random_seed=None if self.rng is None else self.rng.randint(0, int(9e6)),
            traffic_light_state_history=None,
            **kwargs,
        )
        if self.light_flag:
            self._light_state = response.traffic_lights_states
            self.light_recurrent_state = response.light_recurrent_states
        if monitor_infractions and (response.infractions is not None):
            self._infractions = response.infractions
        else:
            self._infractions = None
        self._agent_count = len(
            response.agent_properties
        )  # initialize may produce different agent count
        self._agent_properties = response.agent_properties
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        self._monitor_infractions = monitor_infractions
        self._birdview = response.birdview if get_birdview else None
        self._get_birdview = get_birdview
        if ego_agent_mask is None:
            self._ego_agent_mask = [False] * self._agent_count
        else:
            self._ego_agent_mask = ego_agent_mask[:self._agent_count]
        # initialize might not return the exact number of agents requested,
        # in which case we need to adjust the ego agent mask
        if len(self._ego_agent_mask) > self._agent_count:
            self._ego_agent_mask = self._ego_agent_mask[:self._agent_count]
        if len(self._ego_agent_mask) < self._agent_count:
            self._ego_agent_mask += [False] * (self._agent_count - len(self._ego_agent_mask))
        self._time_step = 0

    @property
    def location(self) -> str:
        """
        Location name as recognized by Inverted AI API.
        """
        return self._location

    @property
    def agent_count(self) -> int:
        """
        The total number of agents, both ego and NPCs.
        """
        return self._agent_count

    @property
    def agent_states(self) -> List[AgentState]:
        """
        The predicted states for all agents, including ego.
        """
        return self._agent_states

    @property
    def agent_properties(self) -> List[AgentProperties]:
        """
        The attributes (length, width, rear_axis_offset) for all agents, including ego.
        """
        return self._agent_properties

    @property
    def ego_agent_mask(self) -> List[bool]:
        """
        Lists which agents are ego, which means that you control them externally.
        It can be updated during the simulation, but see caveats in user guide
        regarding the quality of resulting predictions.
        """
        return self._ego_agent_mask

    @ego_agent_mask.setter
    def ego_agent_mask(self, value):
        self.ego_agent_mask = value

    @property
    def ego_states(self):
        """
        Returns the predicted states of ego agents in order.
        The NPC agents are excluded.
        """
        return [d for d, s in zip(self._agent_states, self._ego_agent_mask) if s]

    @property
    def ego_attributes(self):
        """
        Returns the attributes of ego agents in order.
        The NPC agents are excluded.
        """
        return [attr for attr, s in zip(self._agent_properties, self._ego_agent_mask) if s]

    @property
    def infractions(self) -> Optional[List[InfractionIndicators]]:
        """
        If `monitor_infractions` was set in the constructor,
        lists infractions currently committed by each agent, including ego agents.
        """
        return self._infractions

    @property
    def birdview(self) -> Image:
        """
        If `get_birdview` was set in the constructor,
        this is the image showing the current state of the simulation.
        """
        return self._birdview

    @property
    def npc_states(self) -> List[AgentState]:
        """
        Returns the predicted states of NPCs (non-ego agents) in order.
        The predictions for ego agents are excluded.
        """
        npc_states = []
        for (i, s) in enumerate(self._agent_states):
            if not self._ego_agent_mask[i]:
                npc_states.append(s)
        return npc_states

    @property
    def light_states(self) -> Optional[TrafficLightStatesDict]:
        """
        Returns the traffic light states if any exists on the map.
        """
        return self._light_state

    def step(self, current_ego_agent_states: List[AgentState]) -> None:
        """
        Calls :func:`drive` to advance the simulation by one time step.
        Current states of ego agents need to be provided to synchronize with
        your local simulator.

        :param current_ego_agent_states:  States of ego agents before the step.
        :return: None - call :func:`self.npc_states` to retrieve predictions.
        """
        self._update_ego_states(current_ego_agent_states)
        
        response = drive(
            location=self.location,
            agent_properties=self._agent_properties,
            agent_states=self.agent_states,
            recurrent_states=self._recurrent_states,
            get_infractions=self._monitor_infractions,
            get_birdview=self._get_birdview,
            random_seed=None if self.rng is None else self.rng.randint(0, int(9e6)),
            light_recurrent_states=self.light_recurrent_state,
        )
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        if self._monitor_infractions and (response.infractions is not None):
            self._infractions = response.infractions
        if self._get_birdview:
            self._birdview = response.birdview
        self._time_step += 1
        if self.light_flag:
            self._light_state = response.traffic_lights_states
            self.light_recurrent_state = response.light_recurrent_states

    def _update_ego_states(self, ego_agent_states):
        new_states = []
        ego_idx = 0
        for (i, s) in enumerate(self.agent_states):
            if self.ego_agent_mask[i]:
                new_states.append(ego_agent_states[ego_idx])
                ego_idx += 1
            else:
                new_states.append(self.agent_states[i])
        self._agent_states = new_states
