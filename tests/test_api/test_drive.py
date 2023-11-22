import pytest
from typing import List
from invertedai.api.drive import DriveResponse, drive
from invertedai.common import AgentState, RecurrentState, AgentAttributes


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


@pytest.fixture
def recurrent_states() -> List[RecurrentState]:
    return [
        RecurrentState(),
        RecurrentState()
    ]


def test_drive(location, agent_states, agent_attributes, recurrent_states):
    response = drive(location, agent_states, agent_attributes, recurrent_states)
    assert response is not None
    assert isinstance(response, DriveResponse)
    assert len(response.agent_states) == len(agent_states)
    assert len(response.recurrent_states) == len(recurrent_states)
    assert len(response.birdview.encoded_image) == 0
    assert len(response.infractions) == 0


def test_invalid_agent_states(location, agent_attributes, recurrent_states):
    with pytest.raises(Exception):
        drive(location, [], agent_attributes, recurrent_states)
    with pytest.raises(Exception):
        drive(location, [AgentState()], agent_attributes, recurrent_states)
    with pytest.raises(Exception):
        drive(location, [AgentState(x=[1.0], y=[2.0], orientation=[3.0], speed=[4.0]), AgentState(
            x=[1.0], y=[2.0], orientation=[3.0])], agent_attributes, recurrent_states)


def test_invalid_agent_attributes(location, agent_states, recurrent_states):
    with pytest.raises(Exception):
        drive(location, agent_states, [], recurrent_states)
    with pytest.raises(Exception):
        drive(location, agent_states, [AgentAttributes()], recurrent_states)
    with pytest.raises(Exception):
        drive(location, agent_states, [AgentAttributes(length=[1.0], width=[2.0], rear_axis_offset=[
              3.0]), AgentAttributes(length=[1.0], width=[2.0])], recurrent_states)


def test_invalid_recurrent_states(location, agent_states, agent_attributes):
    with pytest.raises(Exception):
        drive(location, agent_states, agent_attributes, [])
    with pytest.raises(Exception):
        drive(location, agent_states, agent_attributes, [RecurrentState()])
    with pytest.raises(Exception):
        drive(location, agent_states, agent_attributes, [
              RecurrentState(packed=[0.0 for _ in range(128)]), RecurrentState()])
