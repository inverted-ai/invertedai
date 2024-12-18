from typing import List, Optional, Union
from copy import deepcopy

import invertedai as iai
from invertedai.common import (
    AgentProperties,
    AgentState, 
    RecurrentState,
    TrafficLightStatesDict
)
from invertedai.large.drive import large_drive
from invertedai.large.initialize import large_initialize, get_regions_default
from invertedai.api.drive import DriveResponse
from invertedai.api.initialize import InitializeResponse

class BasicCosimulation:
    """
    Stateful wrapper around the Inverted AI API to simplify co-simulation.
    All keyword arguments to :func:`large_initialize` can be passed to the constructor 
    here and a sufficient combination of them must be passed as required by :func:`large_initialize`.
    This wrapper caches static agent properties and propagates the recurrent state,
    so that only states of ego agents and NPCs need to be exchanged with it to
    perform co-simulation. Typically, each time step requires a single call to :func:`self.step`.

    This wrapper only supports a minimal co-simulation functionality.
    For more advanced use cases, call :func:`large_initialize` and :func:`large_drive` directly.

    :param location: 
        Location name as expected by :func:`large_initialize` and :func:`large_drive`.
    :param ego_agent_properties: 
        Agent properties for all ego agents NOT controlled by the Inverted AI API but nonetheless
        must be given to the API so the NPC agents are aware of them. Please refer to the documentation
        for :func:`large_initialize` for more information on how to format this parameter (treating the
        ego agents as "predefined agents".)
    :param ego_agent_agent_states: 
        Agent states for all ego agents NOT controlled by the Inverted AI API but nonetheless
        must be given to the API so the NPC agents are aware of their states. Please refer to the documentation
        for :func:`large_initialize` for more information on how to format this parameter (treating the
        ego agents as "predefined agents".)
    """

    def __init__(
        self,
        location: str,
        ego_agent_properties: Optional[List[AgentProperties]] = None,
        ego_agent_agent_states: Optional[List[AgentState]] = None,
        **kwargs # sufficient arguments to initialize must also be included
    ):
        self._ego_agent_properties = ego_agent_properties
        self._ego_agent_agent_states = ego_agent_agent_states

        self._location = location
        self._response = large_initialize(
            location=self._location,
            agent_properties=self._ego_agent_properties,
            agent_states=self._ego_agent_agent_states,
            **kwargs,
        )
        self.init_response = deepcopy(self._response)
        self._light_state = self.init_response.traffic_lights_states
        self._light_recurrent_state = self.init_response.light_recurrent_states

        self._total_agent_count = len(
            self.init_response.agent_properties
        )  # initialize may produce different agent count
        self._ego_agent_count = len(self._ego_agent_properties)
        self._npc_agent_count = self._total_agent_count - self._ego_agent_count
        
        self._agent_properties = self.init_response.agent_properties
        self._agent_states = self.init_response.agent_states
        self._recurrent_states = self.init_response.recurrent_states
        
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
        return self._total_agent_count

    @property
    def agent_states(self) -> List[AgentState]:
        """
        The predicted states for all agents, including ego.
        """
        return self._agent_states

    @property
    def agent_properties(self) -> List[AgentProperties]:
        """
        The properties (length, width, rear_axis_offset, max_speed) for all agents, including ego.
        """
        return self._agent_properties

    @property
    def ego_states(self) -> List[AgentState]:
        """
        Returns the predicted states of ego agents in order.
        The NPC agents are excluded.
        """
        return self._agent_states[:self._ego_agent_count]

    @property
    def ego_properties(self) -> List[AgentProperties]:
        """
        Returns the properties of ego agents in order.
        The NPC agents are excluded.
        """
        return self._agent_properties[:self._ego_agent_count]

    @property
    def npc_states(self) -> List[AgentState]:
        """
        Returns the predicted states of NPCs (non-ego agents) in order.
        The predictions for ego agents are excluded.
        """
        return self._agent_states[self._ego_agent_count:]

    @property
    def npc_properties(self) -> List[AgentProperties]:
        """
        Returns the properties of NPCs (non-ego agents) in order.
        The ego agents are excluded.
        """
        return self._agent_properties[self._ego_agent_count:]

    @property
    def npc_recurrent_states(self) -> List[RecurrentState]:
        """
        Returns the recurrent states of NPCs (non-ego agents) in order.
        The ego agents are excluded.
        """
        return self._recurrent_states[self._ego_agent_count:]

    @property
    def light_states(self) -> Optional[TrafficLightStatesDict]:
        """
        Returns the traffic light states if any exists on the map.
        """
        return self._light_state

    @property
    def response(self) -> Union[DriveResponse,InitializeResponse]:
        """
        Get the current response data object containing all information received from the API.
        """
        return self._response

    def step(
        self, 
        current_ego_agent_states: List[AgentState],
        **kwargs
    ) -> None:
        """
        Calls :func:`large_drive` to advance the simulation by one time step.
        Current states of ego agents need to be provided to synchronize with
        your local simulator.
        All remaining keyword arguments to :func:`large_drive` can be passed 
        to this function here to receive the desired information from the API.

        :param current_ego_agent_states:  States of ego agents before the step.
        :return: None - call :func:`self.npc_states` to retrieve predictions.
        """
        self._update_ego_states(current_ego_agent_states)
        
        try:
            self._response = large_drive(
                location=self.location,
                agent_properties=self._agent_properties,
                agent_states=self._agent_states,
                recurrent_states=self._recurrent_states,
                light_recurrent_states=self._light_recurrent_state,
                **kwargs
            )
        except Exception as e:
            print(e)
            breakpoint()
        self._agent_states = self._response.agent_states
        self._recurrent_states = self._response.recurrent_states
        self._light_state = self._response.traffic_lights_states
        self._light_recurrent_state = self._response.light_recurrent_states

    def _update_ego_states(self, ego_agent_states):
        self._agent_states[:self._ego_agent_count] = ego_agent_states
