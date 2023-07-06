import pytest
from typing import List, Tuple
from invertedai.api.blame import BlameResponse, blame
from invertedai.common import AgentState, RecurrentState, AgentAttributes


@pytest.fixture
def location():
    return "carla:Town03"


@pytest.fixture
def candidate_agents():
    return (0, 1)


@pytest.fixture
def agent_state_history() -> List[List[AgentState]]:
    return [[
        AgentState.fromlist((0.0, 0.0, 0.0, 0.0)),
        AgentState.fromlist((10.0, 10.0, 0.0, 0.0))
    ], [
        AgentState.fromlist((0.0, 0.0, 0.0, 0.0)),
        AgentState.fromlist((10.0, 10.0, 0.0, 0.0))
    ]]


@pytest.fixture
def agent_attributes() -> List[AgentAttributes]:
    return [
        AgentAttributes.fromlist((2.0, 1.0, 0.5)),
        AgentAttributes.fromlist((2.0, 1.0, 0.5))
    ]


@pytest.fixture
def get_reason():
    return True


@pytest.fixture
def get_confidence_score():
    return True

#    location: str,
#    candidate_agents: Tuple[int, int],
#    agent_state_history: List[List[AgentState]],
#    agent_attributes: List[AgentAttributes],
#    traffic_light_state_history: List[TrafficLightStatesDict],
#    get_birdview: bool = False,
#    detect_collisions: bool = False


def test_blame(location, candidate_agents, agent_state_history, agent_attributes, get_reason, get_confidence_score):
    response = blame(location=location,
                     candidate_agents=candidate_agents,
                     agent_state_history=agent_state_history,
                     agent_attributes=agent_attributes,
                     get_reason=get_reason,
                     get_confidence_score=get_confidence_score)
    assert response is not None
    assert isinstance(response, BlameResponse)
    assert isinstance(response.blamed_result, Tuple)
    assert isinstance(response.reason, str)
    assert isinstance(response.confidence_score, float)
    assert len(response.birdviews) == 0


def test_invalid_agent_states(location, candidate_agents, agent_attributes):
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=[],
              agent_attributes=agent_attributes)
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=[[AgentState()]],
              agent_attributes=agent_attributes)
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=[[AgentState(x=[1.0], y=[2.0], orientation=[3.0], speed=[4.0]),
                                    AgentState(x=[1.0], y=[2.0], orientation=[3.0])]],
              agent_attributes=agent_attributes)


def test_invalid_agent_attributes(location, candidate_agents, agent_state_history):
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=agent_state_history,
              agent_attributes=[])
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=agent_state_history,
              agent_attributes=[AgentAttributes()])
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=candidate_agents,
              agent_state_history=agent_state_history,
              agent_attributes=[AgentAttributes(length=[1.0], width=[2.0], rear_axis_offset=[
              3.0]), AgentAttributes(length=[1.0], width=[2.0])])


def test_invalid_candidate_agents(location, agent_state_history, agent_attributes):
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=(),
              agent_state_history=agent_state_history,
              agent_attributes=agent_attributes)
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=(0, ),
              agent_state_history=agent_state_history,
              agent_attributes=agent_attributes)
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=(0, 1, 2),
              agent_state_history=agent_state_history,
              agent_attributes=agent_attributes)
    with pytest.raises(Exception):
        blame(location=location,
              candidate_agents=(10000),
              agent_state_history=agent_state_history,
              agent_attributes=agent_attributes)
