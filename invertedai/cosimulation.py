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
    so that only states of conditional agents need to be exchanged with it to
    perform co-simulation. Typically, each time step requires a single call to :func:`self.step`.

    This wrapper only supports a minimal co-simulation functionality.
    For more advanced use cases, call :func:`large_initialize` and :func:`large_drive` directly.

    :param location: 
        Location name as expected by :func:`large_initialize` and :func:`large_drive`.
    :param conditional_agent_properties: 
        Agent properties for all conditional agents that must be given to the API so the NPC agents are aware of them. 
        Please refer to the documentation for :func:`large_initialize` for more information on how to format this parameter (treating the
        conditional agents as "predefined agents"). Furthermore, any predefined agents for which the user
        wishes the IAI API to control must be defined at the end of this list.
    :param conditional_agent_agent_states: 
        Agent states for all conditional agents that must be given to the API so the NPC agents are aware of their states. 
        Please refer to the documentation for :func:`large_initialize` for more information on how to format this parameter (treating the
        ego agents as "predefined agents"). Furthermore, any predefined agents for which the user
        wishes the IAI API to control must be defined at the end of this list.
    :param num_non_ego_conditional_agents: 
        The ego agents are the subset of the conditional agents that are NOT controlled by the Inverted AI API. This
        parameter allows some of the conditional agents with predefined states and properties to nonetheless be 
        controlled by the Inverted AI API. The non-ego conditional agents must be placed at the end of the conditional
        agents list and the ego agents must be placed at the beginning of the conditional agents list.
    """

    def __init__(
        self,
        location: str,
        conditional_agent_properties: Optional[List[AgentProperties]] = None,
        conditional_agent_agent_states: Optional[List[AgentState]] = None,
        num_non_ego_conditional_agents: Optional[int] = 0,
        **kwargs # sufficient arguments to initialize must also be included
    ):
        self._conditional_agent_properties = conditional_agent_properties
        self._conditional_agent_agent_states = conditional_agent_agent_states

        self._location = location
        self._response = large_initialize(
            location=self._location,
            agent_properties=self._conditional_agent_properties,
            agent_states=self._conditional_agent_agent_states,
            **kwargs,
        )
        self.init_response = deepcopy(self._response)
        self._light_state = self.init_response.traffic_lights_states
        self._light_recurrent_state = self.init_response.light_recurrent_states

        self._total_agent_count = len(
            self.init_response.agent_properties
        )  # initialize may produce different agent count
        self._conditional_agent_count = len(self._conditional_agent_agent_states) - num_non_ego_conditional_agents
        assert self._conditional_agent_count > 0, "Invalid number of ego and conditional agents."
        self._npc_agent_count = self._total_agent_count - self._conditional_agent_count
        
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
        return self._agent_states[:self._conditional_agent_count]

    @property
    def ego_properties(self) -> List[AgentProperties]:
        """
        Returns the properties of ego agents in order.
        The NPC agents are excluded.
        """
        return self._agent_properties[:self._conditional_agent_count]

    @property
    def npc_states(self) -> List[AgentState]:
        """
        Returns the predicted states of NPCs (non-ego agents) in order.
        The predictions for ego agents are excluded.
        """
        return self._agent_states[self._conditional_agent_count:]

    @property
    def npc_properties(self) -> List[AgentProperties]:
        """
        Returns the properties of NPCs (non-ego agents) in order.
        The ego agents are excluded.
        """
        return self._agent_properties[self._conditional_agent_count:]

    @property
    def npc_recurrent_states(self) -> List[RecurrentState]:
        """
        Returns the recurrent states of NPCs (non-ego agents) in order.
        The ego agents are excluded.
        """
        return self._recurrent_states[self._conditional_agent_count:]

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
        current_conditional_agent_states: List[AgentState],
        **kwargs
    ) -> None:
        """
        Calls :func:`large_drive` to advance the simulation by one time step.
        Current states of ego agents need to be provided to synchronize with
        your local simulator.
        This function assumes the conditional agents are placed at the beginning
        of given agent states list.
        All remaining keyword arguments to :func:`large_drive` can be passed 
        to this function here to receive the desired information from the API.


        :param current_conditional_agent_states:  States of ego agents before the step
            which must match the number of given ego, conditional agents during initialization.
        :return: None - call :func:`self.npc_states` to retrieve predictions.
        """
        self._update_conditional_states(current_conditional_agent_states)
        
        self._response = large_drive(
            location=self.location,
            agent_properties=self._agent_properties,
            agent_states=self._agent_states,
            recurrent_states=self._recurrent_states,
            light_recurrent_states=self._light_recurrent_state,
            **kwargs
        )
        self._agent_states = self._response.agent_states
        self._recurrent_states = self._response.recurrent_states
        self._light_state = self._response.traffic_lights_states
        self._light_recurrent_state = self._response.light_recurrent_states

    def _update_conditional_states(self, conditional_agent_states):
        assert len(conditional_agent_states) == self._conditional_agent_count, "Given number of agents in this step must match the number of ego agents in the co-simulation."

        self._agent_states[:self._conditional_agent_count] = conditional_agent_states
