import pytest
from typing import List
import invertedai.error as error
from invertedai.api import initialize, InitializeResponse
from invertedai.common import AgentAttributes, AgentState, Image, InfractionIndicators, RecurrentState, TrafficLightStatesDict

# Test case for a successful initialization


@pytest.fixture
def location():
    return "carla:Town03"


@pytest.fixture
def agent_states() -> List[AgentState]:
    return [
        AgentState.fromlist((0.0, 0.0, 0.0, 0.0)),
        AgentState.fromlist((10.0, 10.0, 0.0, 0.0))
    ]


@pytest.fixture
def agent_attributes() -> List[AgentAttributes]:
    return [
        AgentAttributes.fromlist((2.0, 1.0, 0.5)),
        AgentAttributes.fromlist((2.0, 1.0, 0.5))
    ]


def test_initialize_zero(location):
    response = initialize(
        location=location,
        get_birdview=False,
        location_of_interest=None,
        get_infractions=False,
        agent_count=0,
        random_seed=None
    )
    assert isinstance(response, InitializeResponse)
    assert len(response.agent_states) == 0
    assert len(response.recurrent_states) == 0
    assert len(response.birdview.encoded_image) == 0
    assert len(response.infractions) == 0

# Test case for initialization with an agent count but no agent attributes


def test_initialize(location):
    response = initialize(
        location=location,
        agent_attributes=None,
        states_history=None,
        traffic_light_state_history=None,
        get_birdview=False,
        location_of_interest=None,
        get_infractions=False,
        agent_count=10,
        random_seed=None
    )
    assert isinstance(response, InitializeResponse)
    assert len(response.agent_states) == 10
    assert len(response.recurrent_states) == 10
    assert len(response.birdview.encoded_image) == 0
    assert len(response.infractions) == 0


def test_initialize_with_conditional_agents(location, agent_attributes, agent_states):
    response = initialize(
        location=location,
        agent_attributes=agent_attributes,
        states_history=[agent_states],
        get_birdview=False,
        location_of_interest=None,
        get_infractions=False,
        agent_count=len(agent_attributes),
        random_seed=None
    )
    assert isinstance(response, InitializeResponse)
    assert len(response.agent_states) == len(agent_attributes)
    assert len(response.recurrent_states) == len(agent_attributes)
    assert len(response.birdview.encoded_image) == 0
    assert len(response.infractions) == 0


# Test case for initialization without agent count or attributes
def test_initialize_without_agent_count_and_attributes(location):
    with pytest.raises(error.InvalidRequestError):
        initialize(
            location=location,
            get_birdview=False,
            location_of_interest=None,
            get_infractions=False,
            agent_count=None,
            random_seed=None
        )

# Test case for initialization with agent count less than the number of agents in states_history


def test_initialize_with_agent_count_less_than_history(location, agent_attributes, agent_states):
    with pytest.raises(error.InvalidRequestError):
        initialize(
            location=location,
            agent_attributes=agent_attributes,
            states_history=[agent_states],
            traffic_light_state_history=None,
            get_birdview=False,
            location_of_interest=None,
            get_infractions=False,
            agent_count=1,
            random_seed=None
        )

# Test case for initialization with incompatible number of agents in states_history and agent_attributes


def test_initialize_with_incompatible_agents(location):
    with pytest.raises(ValueError):
        initialize(
            location=location,
            agent_attributes=[AgentAttributes(speed=0.5, height=1.8)],
            states_history=[[AgentState(position=(0, 0, 0))]],
            traffic_light_state_history=None,
            get_birdview=False,
            location_of_interest=None,
            get_infractions=False,
            agent_count=None,
            random_seed=None
        )

# Test case for using mock api


def test_initialize_with_mock_api(monkeypatch, location):
    monkeypatch.setattr("invertedai.api.config.should_use_mock_api", lambda: True)
    response = initialize(
        location=location,
        agent_attributes=None,
        states_history=None,
        traffic_light_state_history=None,
        get_birdview=False,
        location_of_interest=None,
        get_infractions=False,
        agent_count=10,
        random_seed=None
    )
    assert isinstance(response, InitializeResponse)
