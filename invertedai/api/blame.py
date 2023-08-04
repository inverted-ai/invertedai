import time
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, validate_arguments

import sys
sys.path.append("..")

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.api.mock import (
    get_mock_birdview,
    get_mock_agents_at_fault,
    get_mock_blamed_reasons,
    get_mock_confidence_score,
)
from invertedai.error import APIConnectionError, InvalidInput
from invertedai.common import (
    AgentState,
    Image,
    AgentAttributes,
    TrafficLightStatesDict,
)

class BlameResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.blame`.
    """
    agents_at_fault: Optional[Tuple[int, ...]] #: A tuple containing all agents predicted to be at fault. If empty, the model has predicated no agents are at fault.
    reasons: Optional[Dict[int, List[str]]] #: A dictionary with agent IDs as keys and a list of fault class strings for why the keyed agent is to blame.
    confidence_score: Optional[float] #: Float value between [0,1] indicating the models confidence in the response.
    birdviews: Optional[List[Image]]  #: If `get_birdviews` was set, this contains the resulting image.

@validate_arguments
def blame(
    location: str,
    colliding_agents: Tuple[int, int],
    agent_state_history: List[List[AgentState]],
    agent_attributes: List[AgentAttributes],
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_reasons: bool = False,
    get_confidence_score: bool = False,
    get_birdviews: bool = False
) -> BlameResponse:
    """
    Parameters
    ----------
    location:
        Location name in IAI format.

    colliding_agents:
        Two agents involved in the collision. These integers should correspond to the 
        indices of the relevant agents in the lists within agent_state_history.

    agent_state_history:
        Lists containing AgentState objects for every agent within the scene (up to 100
        agents) for each time step within the relevant continuous sequence immediately 
        preceding the collision.
        The list of AgentState objects should include the first time step of the collision
        and no time steps afterwards. The lists of AgentState objects preceding the 
        collision should capture enough of the scenario context before the collision for 
        the Blame model to analyze and assign fault. For best results it is recommended to 
        input 30-100 time steps of 0.1s each preceding the collision.
        Each AgentState state must include x: [float], y: [float] coordinates in meters, 
        orientation: [float] in radians with 0 pointing along the positive x axis and 
        pi/2 pointing along the positive y axis, and speed: [float] in m/s.

    agent_attributes:
        List of static AgentAttribute objects for all agents. Each agent requires, length: 
        [float], width: [float], and rear_axis_offset: [float] all in meters.

    traffic_light_state_history:
        List of TrafficLightStatesDict objects containing the state of all traffic lights
        for every time step. The dictionary keys are the traffic-light IDs and value is 
        the state, i.e., 'green', 'yellow', 'red', or None.

    get_reasons:
        Whether to return the reasons regarding why each agent was blamed.

    get_confidence_score:
        Whether to return how confident the Blame model is in its response.

    get_birdviews:
        Whether to return images visualizing the collision case.This is very slow and 
        should only be used for debugging.

    See Also
    --------
    :func:`drive`
    :func:`initialize`
    :func:`location_info`
    :func:`light`
    """
    if len(agent_state_history[0]) != len(agent_attributes):
        raise InvalidInput("Incompatible Number of Agents in either 'agent_state_history' or 'agent_attributes'.")

    if should_use_mock_api():
        agents_at_fault = get_mock_agents_at_fault()
        birdviews = [get_mock_birdview()]
        reasons = get_mock_blamed_reasons()
        confidence_score = get_mock_confidence_score()
        response = BlameResponse(
            agents_at_fault=agents_at_fault,
            birdviews=birdviews,
            reasons=reasons,
            confidence_score=confidence_score
        )
        return response

    model_inputs = dict(
        location=location,
        colliding_agents=colliding_agents,
        agent_state_history=[[state.tolist() for state in agent_states] for agent_states in agent_state_history],
        agent_attributes=[attr.tolist() for attr in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_reasons=get_reasons ,
        get_confidence_score=get_confidence_score,
        get_birdviews=get_birdviews
    )
    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            response = iai.session.request(model="blame", data=model_inputs)

            response = BlameResponse(
                agents_at_fault=response["agents_at_fault"],
                reasons=response["reasons"],
                confidence_score=response["confidence_score"],
                birdviews=[Image.fromval(birdview) for birdview in response["birdviews"]]
            )

            return response
        except APIConnectionError as e:
            iai.logger.warning("Retrying")
            if (
                timeout is not None and time.time() > start + timeout
            ) or not e.should_retry:
                raise e


@validate_arguments
async def async_blame(
    location: str,
    colliding_agents: Tuple[int, int],
    agent_state_history: List[List[AgentState]],
    agent_attributes: List[AgentAttributes],
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_reasons: bool = False,
    get_confidence_score: bool = False,
    get_birdviews: bool = False
) -> BlameResponse:
    """
    A light async version of :func:`blame`
    """
    model_inputs = dict(
        location=location,
        colliding_agents=colliding_agents,
        agent_state_history=[[state.tolist() for state in agent_states] for agent_states in agent_state_history],
        agent_attributes=[attr.tolist() for attr in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_reasons=get_reasons,
        get_confidence_score=get_confidence_score,
        get_birdviews=get_birdviews
    )

    response = await iai.session.async_request(model="blame", data=model_inputs)

    response = BlameResponse(
        agents_at_fault=response["agents_at_fault"],
        reasons=response["reasons"],
        confidence_score=response["confidence_score"],
        birdviews=[Image.fromval(birdview) for birdview in response["birdviews"]]
    )

    return response
