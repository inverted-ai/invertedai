from typing import List, Tuple, Optional, Union
import random
from collections import deque
from queue import Queue
from pydantic import BaseModel, validate_arguments
from itertools import product
import numpy as np
import asyncio
from itertools import chain

import invertedai as iai
from invertedai import drive, initialize, location_info, light, async_drive, async_initialize
from invertedai.common import (AgentState, InfractionIndicators, Image,
                               TrafficLightStatesDict, AgentAttributes, RecurrentState, Point)

MAX_HISTORY_LEN = 10
RE_INITIALIZATION_PERIOD = 200  # : In frames


class Car:
    @validate_arguments
    def __init__(self,
                 agent_attributes: Optional[AgentAttributes],
                 agent_states: Optional[AgentState],
                 recurrent_states: Optional[RecurrentState],):

        self._agent_attributes = agent_attributes
        self._recurrent_states = recurrent_states

        # TODO: Must check if the order is correct
        self._states_history = deque([agent_states], maxlen=MAX_HISTORY_LEN)

    def update(self,
               agent_states: Optional[AgentState],
               recurrent_states: Optional[RecurrentState],):
        self._states_history.append(agent_states)
        self._recurrent_states = recurrent_states

    @property
    def recurrent_states(self):
        return self._recurrent_states

    @property
    def agent_attributes(self):
        return self._agent_attributes

    @ property
    def center(self):
        return self.agent_states.center

    @ property
    def agent_states(self):
        return self._states_history[-1]

    @ agent_states.setter
    def agent_states(self, agent_states: AgentState):
        # Setting the sate of the ego agent
        self._states_history[-1] = agent_states


class Region:
    def __init__(self, location, center: Point, region_fov: float, npcs: Optional[List[Car]] = [],
                 re_initialization: Optional[int] = RE_INITIALIZATION_PERIOD) -> None:
        self.center = center
        self.npcs = npcs
        self.region_fov = region_fov
        self.re_initialization_period = re_initialization
        self.time_to_reinitialize = self.re_initialization_period
        self.location = location
        self.incoming = []  # List of incomming NPCs

    def incoming(self, incoming_npcs):
        self.incoming = incoming_npcs

    async def drive(self):
        """_summary_
        updates the state of all NPCs inside the region and returns a list of NPCs that are not longer int the region
        """
        if self.incoming:
            # TODO: something with the incomming or just apppend them to
            pass
        if self.empty:
            return []
        if self.time_to_reinitialize == 0:
            # TODO: Initialization
            self.time_to_reinitialize = self.re_initialization_period
            pass
        else:
            self.time_to_reinitialize -= 1
        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        drive_response = await async_drive(location=self.location,
                                           agent_attributes=agent_attributes,
                                           agent_states=agent_states,
                                           recurrent_states=recurrent_states)
        outgoing_npcs = []
        remaining_npcs = []
        for npc, state, rs in zip(self.npcs, drive_response.agent_states, drive_response.recurrent_states):
            npc.update(state, rs)
            if self.inside_fov(self.center, self.region_fov, npc):
                remaining_npcs.append(npc)
            else:
                outgoing_npcs.append(npc)

        self.npcs = remaining_npcs
        return outgoing_npcs

    @staticmethod
    def inside_fov(center: Point, region_fov: float, npc: Car) -> bool:
        return ((center.x - (region_fov / 2) < npc.center.x < center.x + (region_fov / 2)) and
                (center.y - (region_fov / 2) < npc.center.y < center.y + (region_fov / 2)))

    @property
    def empty(self):
        return not bool(self.npcs)


class Simulation:
    """
    Stateful simulation for large maps with async calls to Inverted AI API for simultaneously driving npcs
    in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(self,
                 location: str,
                 center: Tuple[float],
                 width: float,
                 height: float,
                 agent_per_region: float,
                 random_seed: Optional[float] = None,
                 region_fov: Optional[float] = 120,
                 ):
        self.location = location
        self.center = Point(x=center[0], y=center[1])
        self.width = width
        self.height = height
        self.agent_per_region = agent_per_region
        self.random_seed = random_seed
        self.region_fov = region_fov
        self.location_info = iai.location_info(location=self.location)
        self.re_initialization = RE_INITIALIZATION_PERIOD  # :
        self.regions, self.npcs, self.initialize = self._initialize_regions()

    def _initialize_regions(self):
        try:
            light_response = iai.light(location=self.location)
            traffic_lights_states = [light_response.traffic_lights_states]
        except:
            traffic_lights_states = None
        initialize_response = iai.utils.area_initialization(self.location, self.agent_per_region,
                                                            traffic_lights_states=traffic_lights_states,
                                                            random_seed=self.random_seed,
                                                            map_center=(self.center.x, self.center.y),
                                                            width=self.width, height=self.height, stride=self.region_fov/2)

        npcs = [Car(agent_attributes=attr, agent_states=state, recurrent_states=rs) for attr, state,
                rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states)]

        h_start, h_end = self.center.x - (self.height / 2) + (self.region_fov /
                                                              2), self.center.x + (self.height / 2) - (self.region_fov / 2) + 1
        w_start, w_end = self.center.y - (self.width / 2) + (self.region_fov /
                                                             2), self.center.y + (self.width / 2) - (self.region_fov / 2) + 1
        region_centers = product(np.arange(h_start, h_end, self.region_fov / 2),
                                 np.arange(w_start, w_end, self.region_fov / 2))
        regions = []
        for region_center in map(Point.fromlist, region_centers):
            region_npcs = list(filter(lambda x: Region.inside_fov(region_center, self.region_fov, x), npcs))
            regions.append(Region(location=self.location, center=region_center,
                           region_fov=self.region_fov, npcs=region_npcs))

        return regions, npcs, initialize_response

    async def drive_regions(self):
        outgoings = await asyncio.gather(*[region.drive() for region in self.regions])
        return outgoings

    def drive(self):
        outgoing_npcs = asyncio.run(self.drive_regions())
        outgoing_npcs = list(chain.from_iterable(outgoing_npcs))
        for region in self.regions:
            region_npcs = list(filter(lambda x: Region.inside_fov(region.center, region.region_fov, x), outgoing_npcs))
            region.incomming(region_npcs)

        return npcs


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
        self._light_state = None
        if traffic_lights:
            location_info_response = location_info(location=location)
            static_actors = location_info_response.static_actors
            if any(actor.agent_type == "traffic-light" for actor in static_actors):
                self.light_flag = True
                light_response = light(location=location)
                self._light_state = light_response.traffic_lights_states
                self.light_recurrent_state = light_response.recurrent_states
        response = initialize(
            location=location,
            get_birdview=get_birdview,
            get_infractions=monitor_infractions,
            random_seed=None if self.rng is None else self.rng.randint(0, int(9e6)),
            traffic_light_state_history=[self._light_state] if self._light_state else None,
            **kwargs,
        )
        if monitor_infractions and (response.infractions is not None):
            self._infractions = response.infractions
        else:
            self._infractions = None
        self._agent_count = len(
            response.agent_attributes
        )  # initialize may produce different agent count
        self._agent_attributes = response.agent_attributes
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
        if self.light_flag:
            light_response = light(location=self.location, recurrent_states=self.light_recurrent_state)
            self.light_recurrent_state = light_response.recurrent_states
            self._light_state = light_response.traffic_lights_states
        else:
            light_state = None

        response = drive(
            location=self.location,
            agent_attributes=self._agent_attributes,
            agent_states=self.agent_states,
            recurrent_states=self._recurrent_states,
            get_infractions=self._monitor_infractions,
            get_birdview=self._get_birdview,
            random_seed=None if self.rng is None else self.rng.randint(0, int(9e6)),
            traffic_lights_states=self._light_state,
        )
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        if self._monitor_infractions and (response.infractions is not None):
            self._infractions = response.infractions
        if self._get_birdview:
            self._birdview = response.birdview
        self._time_step += 1

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
