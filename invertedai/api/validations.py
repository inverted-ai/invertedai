from typing import List, Optional, Dict
from invertedai.error import InvalidInputType, InvalidInput
from invertedai.common import (
    RecurrentState,
    AgentState,
    AgentAttributes,
    TrafficLightId,
    TrafficLightState,
)


def drive_validation(drive):
    def wrapped_drive(
        location: str,
        agent_states: List[AgentState],
        agent_attributes: List[AgentAttributes],
        recurrent_states: List[RecurrentState],
        traffic_lights_states: Optional[Dict[TrafficLightId, TrafficLightState]] = None,
        get_birdview: bool = False,
        get_infractions: bool = False,
        random_seed: Optional[int] = None,
    ):
        try:
            assert isinstance(location, str)
            assert all(map(lambda s: isinstance(s, AgentState), agent_states))
            assert all(map(lambda s: isinstance(s, AgentAttributes), agent_attributes))
            assert all(map(lambda s: isinstance(s, RecurrentState), recurrent_states))
            assert isinstance(traffic_lights_states, (list, type(None)))
            assert isinstance(get_birdview, bool)
            assert isinstance(get_infractions, bool)
            assert isinstance(random_seed, (int, type(None)))
        except AssertionError:
            raise InvalidInputType("iai.drive: invalid input type.")

        try:
            assert len(agent_states) == len(agent_attributes)
            assert len(agent_states) == len(recurrent_states)
        except AssertionError or TypeError:
            raise InvalidInput("iai.drive: invalid input size.")

        return drive(location,
                     agent_states,
                     agent_attributes,
                     recurrent_states,
                     traffic_lights_states,
                     get_birdview,
                     get_infractions,
                     random_seed,)
    return wrapped_drive


def initialize_validation(initialize):
    def wrapped_initialize(
        location: str,
        agent_attributes: Optional[List[AgentAttributes]] = None,
        states_history: Optional[List[List[AgentState]]] = None,
        traffic_light_state_history: Optional[
            List[Dict[TrafficLightId, TrafficLightState]]
        ] = None,
        get_birdview: bool = False,
        get_infractions: bool = False,
        agent_count: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        try:
            assert isinstance(location, str)
            assert isinstance(agent_attributes, (list, type(None)))
            assert isinstance(states_history, (list, type(None)))
            assert isinstance(traffic_light_state_history, (list, type(None)))
            assert isinstance(get_birdview, bool)
            assert isinstance(get_infractions, bool)
            assert isinstance(agent_count, (int, type(None)))
            assert isinstance(random_seed, (int, type(None)))
        except AssertionError:
            raise InvalidInputType("iai.initialize: invalid input type.")

        if agent_attributes or states_history:
            try:
                assert all(map(lambda s: isinstance(s, AgentAttributes), agent_attributes))
                num_agents = len(agent_attributes)
                for states in states_history:
                    assert len(states) == num_agents
                    assert all(map(lambda s: isinstance(s, AgentState), states))
            except AssertionError or TypeError:
                raise InvalidInput("iai.initialize: invalid input size.")

        return initialize(
            location,
            agent_attributes,
            states_history,
            traffic_light_state_history,
            get_birdview,
            get_infractions,
            agent_count,
            random_seed,
        )
    return wrapped_initialize
