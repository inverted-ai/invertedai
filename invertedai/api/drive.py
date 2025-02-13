import time
import asyncio
import warnings
from typing import List, Optional, Tuple
from pydantic import BaseModel, validate_call

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import APIConnectionError, InvalidInput
from invertedai.api.mock import (
    mock_update_agent_state,
    get_mock_birdview,
    get_mock_infractions,
    get_mock_light_recurrent_states
)
from invertedai.common import (
    AgentState,
    RecurrentState,
    Image,
    InfractionIndicators,
    AgentAttributes,
    AgentProperties,
    TrafficLightStatesDict,
    LightRecurrentStates,
    LightRecurrentState,
)


class DriveResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.drive`.
    """

    agent_states: List[AgentState] #: Predicted states for all agents at the next time step.
    recurrent_states: List[RecurrentState] #: To pass to :func:`iai.drive` at the subsequent time step.
    birdview: Optional[Image] #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[List[InfractionIndicators]]  #: If `get_infractions` was set, they are returned here.
    is_inside_supported_area: List[bool] #: For each agent, indicates whether the predicted state is inside supported area.
    traffic_lights_states: Optional[TrafficLightStatesDict] #: Traffic light states for the full map, as seen by the agents before they performed their actions resulting in the returned state. Each key-value pair corresponds to one particular traffic light.
    light_recurrent_states: Optional[LightRecurrentStates] #: Light recurrent states for the full map, each element corresponds to one light group. Pass this to the next call of :func:`iai.drive` for the server to realistically update the traffic light states.
    api_model_version: str # Model version used for this API call

    def serialize_drive_response_parameters(self):
        output_dict = dict(self)
        output_dict["agent_states"] = [state.tolist() for state in output_dict["agent_states"]]
        output_dict["recurrent_states"] = [r.packed for r in output_dict["recurrent_states"]] if output_dict["recurrent_states"] is not None else None
        output_dict["light_recurrent_states"] = [light_recurrent_state.tolist() for light_recurrent_state in output_dict["light_recurrent_states"]] if output_dict["light_recurrent_states"] is not None else None
        output_dict["infractions"] = [infrac.tolist() for infrac in output_dict["infractions"]] if output_dict["infractions"] is not None else None

        return output_dict

@validate_call
def serialize_drive_request_parameters(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    recurrent_states: Optional[List[RecurrentState]] = None,
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    light_recurrent_states: Optional[LightRecurrentStates] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
):
    return dict(
        location=location,
        agent_states=[state.tolist() for state in agent_states],
        agent_attributes=[attr.tolist() for attr in agent_attributes] if agent_attributes is not None else None,
        agent_properties=[ap.serialize() for ap in agent_properties] if agent_properties is not None else None,
        recurrent_states=[r.packed for r in recurrent_states] if recurrent_states is not None else None,
        traffic_lights_states=traffic_lights_states,
        light_recurrent_states=[light_recurrent_state.tolist() for light_recurrent_state in light_recurrent_states] if light_recurrent_states is not None else None,
        get_birdview=get_birdview,
        get_infractions=get_infractions,
        random_seed=random_seed,
        rendering_center=rendering_center,
        rendering_fov=rendering_fov,
        model_version=api_model_version
    )


@validate_call
def drive(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    recurrent_states: Optional[List[RecurrentState]] = None,
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    light_recurrent_states: Optional[LightRecurrentStates] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
) -> DriveResponse:
    """
    Update the state of all given agents forward one time step. Agents are identified by their list index.

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
        Deprecated. Static attributes of all agents.
        List of agent attributes. Each agent requires, length: [float]
        width: [float] and rear_axis_offset: [float] all in meters. agent_type: [str],
        currently supports 'car' and 'pedestrian'.
        waypoint: optional [Point], the target waypoint of the agent.

    agent_properties:
        Agent properties for all agents, replacing soon to be deprecated `agent_attributes`.
        List of agent attributes. Each agent requires, length: [float]
        width: [float] and rear_axis_offset: [float] all in meters. agent_type: [str],
        currently supports 'car' and 'pedestrian'.
        waypoint: optional [Point], the target waypoint of the agent.
        max_speed: optional [float], the desired maximum speed of the agent in m/s.

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
       their current state can be provided here. It is legal to not provide this field, and use
       'light_recurrent_states' to step the traffic lights. If provided, light states from 'traffic_light_states' will override
       the original light states given by 'light_recurrent_states'. The server does not currently support continuing user-provided light state sequences, 
       so once the states are provided at any step, they should also be provided on all subsequent steps to guarantee coherent light sequences.
       If neither 'traffic_lights_states' nor 'light_recurrent_states' are provided, the server will arbitrarily initialize the traffic light states,
       and return the associated 'light_recurrent_states' in the response.

    light_recurrent_states:
       Light recurrent states for all agents, obtained from the previous call to
        :func:`drive` or :func:`initialize`.
       Specifies the state and time remaining for each light group in the map.
       To let the server manage all light states in the scene, 
       pass 'light_recurrent_states' from the previous response of :func:`drive` here and leave `traffic_light_states=None`.

    random_seed:
        Controls the stochastic aspects of agent behavior for reproducibility.

    api_model_version:
        Optionally specify the version of the model. If None is passed which is by default, the best model will be used.
    See Also
    --------
    :func:`initialize`
    :func:`location_info`
    :func:`light`
    :func:`blame`
    """

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
            traffic_lights_states=traffic_lights_states if traffic_lights_states is not None else None,
            light_recurrent_states=get_mock_light_recurrent_states(len(traffic_lights_states)) if traffic_lights_states is not None else None,
            api_model_version=api_model_version if api_model_version is not None else "best"
        )
        return response

    if agent_attributes is not None:
        warnings.warn('agent_attributes is deprecated. Please use agent_properties.',category=DeprecationWarning) 

    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    recurrent_states = _tolist(recurrent_states) if recurrent_states is not None else None
    model_inputs = serialize_drive_request_parameters(
        location=location,
        agent_states=agent_states,
        agent_attributes=agent_attributes,
        agent_properties=agent_properties,
        recurrent_states=recurrent_states,
        traffic_lights_states=traffic_lights_states,
        light_recurrent_states=light_recurrent_states,
        get_birdview=get_birdview,
        rendering_center=rendering_center,
        rendering_fov=rendering_fov,
        get_infractions=get_infractions,
        random_seed=random_seed,
        api_model_version=api_model_version
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
                api_model_version=response["model_version"],
                traffic_lights_states=response["traffic_lights_states"]
                if response["traffic_lights_states"] is not None 
                else None,
                light_recurrent_states=[
                    LightRecurrentState(state=state_arr[0], time_remaining=state_arr[1]) 
                    for state_arr in response["light_recurrent_states"]
                ] 
                if response["light_recurrent_states"] is not None 
                else None
            )

            return response

        except APIConnectionError as e:
            iai.logger.warning("Retrying")
            if (timeout is not None and time.time() > start + timeout) or not e.should_retry:
                raise e


@validate_call
async def async_drive(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: Optional[List[AgentAttributes]]=None,
    agent_properties: Optional[List[AgentProperties]]=None,
    recurrent_states: Optional[List[RecurrentState]] = None,
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    light_recurrent_states: Optional[LightRecurrentStates] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
) -> DriveResponse:
    """
    A light async version of :func:`drive`
    """

    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    recurrent_states = _tolist(recurrent_states) if recurrent_states is not None else None
    model_inputs = dict(
        location=location,
        agent_states=[state.tolist() for state in agent_states],
        agent_attributes=[state.tolist() for state in agent_attributes] if agent_attributes is not None else None,
        agent_properties=[ap.serialize() for ap in agent_properties] if agent_properties is not None else None,
        recurrent_states=[r.packed for r in recurrent_states] if recurrent_states is not None else None,
        traffic_lights_states=traffic_lights_states,
        light_recurrent_states=[light_recurrent_state.tolist() for light_recurrent_state in light_recurrent_states] 
        if light_recurrent_states is not None else None,
        get_birdview=get_birdview,
        get_infractions=get_infractions,
        random_seed=random_seed,
        rendering_center=rendering_center,
        rendering_fov=rendering_fov,
        model_version=api_model_version
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
        api_model_version=response["model_version"],
        traffic_lights_states=response["traffic_lights_states"] 
        if response["traffic_lights_states"] is not None 
        else None,
        light_recurrent_states=[
            LightRecurrentState(state=state_arr[0], time_remaining=state_arr[1]) 
            for state_arr in response["light_recurrent_states"]
        ] 
        if response["light_recurrent_states"] is not None 
        else None
    )

    return response
