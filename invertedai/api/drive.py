import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, validate_arguments
import asyncio

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.api.mock import (
    mock_update_agent_state,
    get_mock_birdview,
    get_mock_infractions,
)
from invertedai.error import APIConnectionError, InvalidInput
from invertedai.common import (
    AgentState,
    RecurrentState,
    Image,
    InfractionIndicators,
    AgentAttributes,
    TrafficLightStatesDict,
)


class DriveResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.drive`.
    """

    agent_states: List[
        AgentState
    ]  #: Predicted states for all agents at the next time step.
    recurrent_states: List[
        RecurrentState
    ]  #: To pass to :func:`iai.drive` at the subsequent time step.
    birdview: Optional[
        Image
    ]  #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[
        List[InfractionIndicators]
    ]  #: If `get_infractions` was set, they are returned here.
    is_inside_supported_area: List[
        bool
    ]  #: For each agent, indicates whether the predicted state is inside supported area.


@validate_arguments
def drive(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: List[AgentAttributes],
    recurrent_states: List[RecurrentState],
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
) -> DriveResponse:
    """
    Parameters
    ----------
    location:
        Location name in IAI format.

    agent_states:
        Current states of all agents.
        The state must include x: [float], y: [float] coordinate in meters
        orientation: [float] in radians with 0 pointing along x and pi/2 pointing along y and
        speed: [float] in m/s.

    agent_attributes:
        Static attributes of all agents.
        List of agent attributes. Each agent requires, length: [float]
        width: [float] and rear_axis_offset: [float] all in meters. agent_type: [str],
        currently only supports 'car', but support for 'pedestrian' will be added in the future

    recurrent_states:
        Recurrent states for all agents, obtained from the previous call to
        :func:`drive` or :func:`initialize`.

    get_birdview:
        Whether to return an image visualizing the simulation state.
        This is very slow and should only be used for debugging.

    rendering_center:
        Optional center coordinates for the rendered birdview.

    rendering_fov:
        Optional fov for the rendered birdview.

    get_infractions:
        Whether to check predicted agent states for infractions.
        This introduces some overhead, but it should be relatively small.

    traffic_lights_states:
       If the location contains traffic lights within the supported area,
       their current state should be provided here. Any traffic light for which no
       state is provided will be ignored by the agents.

    random_seed:
        Controls the stochastic aspects of agent behavior for reproducibility.

    See Also
    --------
    :func:`initialize`
    :func:`location_info`
    :func:`light`
    :func:`blame`
    """
    validate_drive_flows(agent_states, agent_attributes, recurrent_states)

    if should_use_mock_api():
        agent_states = [mock_update_agent_state(s) for s in agent_states]
        present_mask = [True for _ in agent_states]
        birdview = get_mock_birdview()
        infractions = get_mock_infractions(len(agent_states))
        response = DriveResponse(
            agent_states=agent_states,
            is_inside_supported_area=present_mask,
            recurrent_states=recurrent_states,
            birdview=birdview,
            infractions=infractions,
        )
        return response

    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    recurrent_states = (
        _tolist(recurrent_states) if recurrent_states is not None else None
    )  # AxTx2x64
    model_inputs = dict(
        location=location,
        agent_states=[state.tolist() for state in agent_states],
        agent_attributes=[state.tolist() for state in agent_attributes],
        recurrent_states=[r.packed for r in recurrent_states],
        traffic_lights_states=traffic_lights_states,
        get_birdview=get_birdview,
        get_infractions=get_infractions,
        random_seed=random_seed,
        rendering_center=rendering_center,
        rendering_fov=rendering_fov
    )
    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            response = iai.session.request(model="drive", data=model_inputs)

            response = DriveResponse(
                agent_states=[
                    AgentState.fromlist(state) for state in response["agent_states"]
                ],
                recurrent_states=[
                    RecurrentState.fromval(r) for r in response["recurrent_states"]
                ],
                birdview=Image.fromval(response["birdview"])
                if response["birdview"] is not None
                else None,
                infractions=[
                    InfractionIndicators.fromlist(infractions)
                    for infractions in response["infraction_indicators"]
                ]
                if response["infraction_indicators"]
                else [],
                is_inside_supported_area=response["is_inside_supported_area"],
            )

            return response
        except APIConnectionError as e:
            iai.logger.warning("Retrying")
            if (
                timeout is not None and time.time() > start + timeout
            ) or not e.should_retry:
                raise e


@validate_arguments
async def async_drive(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: List[AgentAttributes],
    recurrent_states: List[RecurrentState],
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
) -> DriveResponse:
    """
    A light async version of :func:`drive`
    """
    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    validate_drive_flows(agent_states, agent_attributes, recurrent_states)
    recurrent_states = (
        _tolist(recurrent_states) if recurrent_states is not None else None
    )  # AxTx2x64
    model_inputs = dict(
        location=location,
        agent_states=[state.tolist() for state in agent_states],
        agent_attributes=[state.tolist() for state in agent_attributes],
        recurrent_states=[r.packed for r in recurrent_states],
        traffic_lights_states=traffic_lights_states,
        get_birdview=get_birdview,
        get_infractions=get_infractions,
        random_seed=random_seed,
        rendering_center=rendering_center,
        rendering_fov=rendering_fov
    )
    response = await iai.session.async_request(model="drive", data=model_inputs)

    response = DriveResponse(
        agent_states=[
            AgentState.fromlist(state) for state in response["agent_states"]
        ],
        recurrent_states=[
            RecurrentState.fromval(r) for r in response["recurrent_states"]
        ],
        birdview=Image.fromval(response["birdview"])
        if response["birdview"] is not None
        else None,
        infractions=[
            InfractionIndicators.fromlist(infractions)
            for infractions in response["infraction_indicators"]
        ]
        if response["infraction_indicators"]
        else [],
        is_inside_supported_area=response["is_inside_supported_area"],
    )

    return response
