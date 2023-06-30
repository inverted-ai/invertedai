import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, validate_arguments

import sys
sys.path.append("..")
sys.path.insert(0, "..")

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.api.mock import (
    get_mock_birdview,
    get_mock_blamed_result,
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
    blamed_collisions: Optional[List[Tuple[Tuple[int, int], Tuple[int, int], int]]]
    blamed_result: Optional[Tuple[bool, bool]]
    birdviews: Optional[List[Image]]  #: If `get_birdview` was set, this contains the resulting image.


@validate_arguments
def blame(
    location: str,
    candidate_agents: Tuple[int, int],
    agent_state_history: List[List[AgentState]],
    agent_attributes: List[AgentAttributes],
    traffic_light_state_history: List[TrafficLightStatesDict],
    get_birdview: bool = False,
    detect_collisions: bool = False
) -> BlameResponse:
    """
    Parameters
    ----------

    See Also
    --------
    :func:`drive`
    :func:`initialize`
    :func:`location_info`
    :func:`light`
    """
    if len(agent_state_history[0]) != len(agent_attributes):
        raise InvalidInput("Incompatible Number of Agents in either 'agent_states' or 'agent_attributes'.")

    if should_use_mock_api():
        blamed_result = get_mock_blamed_result()
        birdviews = [get_mock_birdview()]
        response = BlameResponse(
            blamed_result=blamed_result,
            birdviews=birdviews
        )
        return response

    model_inputs = dict(
        location=location,
        candidate_agents=candidate_agents,
        agent_state_history=[[state.tolist() for state in agent_states] for agent_states in agent_state_history],
        agent_attributes=[attr.tolist() for attr in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        detect_collisions=detect_collisions
    )
    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            response = iai.session.request(model="blame", data=model_inputs)

            if detect_collisions:
                response = BlameResponse(
                    blamed_collisions=response["blamed_collisions"],
                    birdviews=[Image.fromval(birdview) for birdview in response["birdviews"]]
                )
            else:
                response = BlameResponse(
                    blamed_result=response["blamed_result"],
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
    candidate_agents: Tuple[int, int],
    agent_state_history: List[List[AgentState]],
    agent_attributes: List[AgentAttributes],
    traffic_light_state_history: List[TrafficLightStatesDict],
    get_birdview: bool = False,
    detect_collisions: bool = False
) -> BlameResponse:
    """
    A light async version of :func:`blame`
    """
    model_inputs = dict(
        location=location,
        candidate_agents=candidate_agents,
        agent_state_history=[[state.tolist() for state in agent_states] for agent_states in agent_state_history],
        agent_attributes=[attr.tolist() for attr in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        detect_collisions=detect_collisions
    )

    response = await iai.session.async_request(model="blame", data=model_inputs)

    if detect_collisions:
        response = BlameResponse(
            blamed_collisions=response["blamed_collisions"],
            birdviews=[Image.fromval(birdview) for birdview in response["birdviews"]]
        )
    else:
        response = BlameResponse(
            blamed_result=response["blamed_result"],
            birdviews=[Image.fromval(birdview) for birdview in response["birdviews"]]
        )

    return response
