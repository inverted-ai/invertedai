import time
import asyncio
import warnings
from pydantic import BaseModel, validate_call
from typing import List, Optional, Dict, Tuple

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import TryAgain, InvalidInputType, InvalidInput
from invertedai.api.mock import (
    get_mock_agent_attributes,
    get_mock_agent_properties,
    get_mock_agent_state,
    get_mock_recurrent_state,
    get_mock_birdview,
    get_mock_infractions,
    get_mock_traffic_light_states,
    get_mock_light_recurrent_states
)
from invertedai.common import (
    AgentAttributes,
    AgentProperties,
    AgentState,
    Image,
    InfractionIndicators,
    LightRecurrentState,
    LightRecurrentStates,
    RecurrentState,
    TrafficLightStatesDict
)


class InitializeResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.initialize`.
    """
    agent_states: List[AgentState] #: Initial states of all initialized agents.
    recurrent_states: List[Optional[RecurrentState]] #: To pass to :func:`iai.drive` at the first time step.
    agent_attributes: List[Optional[AgentAttributes]] #: Static attributes of all initialized agents.
    agent_properties: List[AgentProperties]  #: Static agent properties of all initialized agents.
    birdview: Optional[Image] #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[List[InfractionIndicators]] #: If `get_infractions` was set, they are returned here.
    traffic_lights_states: Optional[TrafficLightStatesDict] #: Traffic light states for the full map, each key-value pair corresponds to one particular traffic light.
    light_recurrent_states: Optional[LightRecurrentStates] #: Light recurrent states for the full map. Pass this to :func:`iai.drive` at the first time step to let the server generate a realistic continuation of the traffic light state sequence. This does not work correctly if any specific light states were specified as input to `initialize`.
    api_model_version: str #: Model version used for this API call

    def serialize_initialize_response_parameters(self):
        output_dict = dict(self)
        output_dict["agent_states"] = [state.tolist() for state in output_dict["agent_states"]]
        output_dict["recurrent_states"] = [r.packed for r in output_dict["recurrent_states"]] if output_dict["recurrent_states"] is not None else None
        output_dict["agent_attributes"] = [attr.tolist() for attr in output_dict["agent_attributes"]] if output_dict["agent_attributes"] is not None else None
        output_dict["agent_properties"] = [ap.serialize() for ap in output_dict["agent_properties"]] if output_dict["agent_properties"] is not None else None
        output_dict["birdview"] = None if output_dict["birdview"] is None else output_dict["birdview"].encoded_image
        output_dict["light_recurrent_states"] = [light_recurrent_state.tolist() for light_recurrent_state in output_dict["light_recurrent_states"]] if output_dict["light_recurrent_states"] is not None else None
        output_dict["infractions"] = [infrac.tolist() for infrac in output_dict["infractions"]] if output_dict["infractions"] is not None else None

        return output_dict

@validate_call
def serialize_initialize_request_parameters(
    location: str,
    agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_birdview: bool = False,
    location_of_interest: Optional[Tuple[float, float]] = None,
    get_infractions: bool = False,
    agent_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
):
    return dict(
        location=location,
        num_agents_to_spawn=agent_count,
        states_history=states_history if states_history is None else [[st.tolist() for st in states] for states in states_history],
        agent_attributes=agent_attributes if agent_attributes is None else [state.tolist() for state in agent_attributes],
        agent_properties=agent_properties if agent_properties is None else [ap.serialize() if ap else None for ap in agent_properties] ,
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        location_of_interest=location_of_interest,
        get_infractions=get_infractions,
        random_seed=random_seed,
        model_version=api_model_version if api_model_version is not None else "best"
    )

@validate_call
def initialize(
    location: str,
    agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_birdview: bool = False,
    location_of_interest: Optional[Tuple[float, float]] = None,
    get_infractions: bool = False,
    agent_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None  # Model version used for this API call
) -> InitializeResponse:
    """
    Initializes a simulation in a given location, using a combination of **user-defined** and **sampled** agents.
    The `agent_properties` parameter is used to determine the agents that are placed into the given `location`. 
    **User-defined** agents are placed into a scene first, after which a number of agents are sampled conditionally.
    Any **user-defined** agent must have a corresponding fully specified static `AgentProperties` object in `agent_properties`.
    Furthermore for all **user-defined** agents, `states_history` must contain a list of `AgentState's` of all **user-defined** agents per historical time step. 
    Per desired **sampled** agent, an `AgentProperties` object must be provided at the end of `agent_properties` with only its `agent_type` specified.
    Agents are identified by their list index, so ensure the indices of each agent match in `states_history` and`agent_properties` when applicable. 
    The `agent_attributes` and `agent_count` parameters are deprecated.
    If traffic lights are present in the scene, their states history can be specified with a list of `TrafficLightStatesDict`, each 
    represent light states for one timestep, with the last element representing the current time step. It is legal to omit the traffic 
    light state specification, and the scene will be initialized with a light state configuration consistent with agent states.
    Every simulation must start with a call to this function in order to obtain correct recurrent states for :func:`drive`.

    Parameters
    ----------
    location:
        Location name in IAI format.

    agent_attributes:
        Deprecated. Static attributes for all agents.
        The pre-defined agents should be specified first, followed by the sampled agents.
        The optional waypoint passed will be ignored for Initialize.
    agent_properties:
        Agent properties for all agents, replacing the deprecated `agent_attributes`.
        The pre-defined agents should be specified first, followed by the sampled agents.
        The optional waypoint passed will be ignored for Initialize.
        max_speed: optional [float], the desired maximum speed of the agent in m/s.

    states_history:
        History of pre-defined agent states - the outer list is over time and the inner over agents,
        in chronological order, i.e., index 0 is the oldest state and index -1 is the current state.
        The order of agents should be the same as in `agent_attributes`.
        For best results, provide at least 10 historical states for each agent.

    traffic_light_state_history:
       History of traffic light states - the list is over time, in chronological order, i.e.
       the last element is the current state. If there are traffic lights in the map, 
       not specifying traffic light state is equivalent to using server generated light states.

    location_of_interest:
        Optional coordinates for spawning agents with the given location as center instead of the default map center

    get_birdview:
        If True, a birdview image will be returned representing the current world. Note this will significantly
        impact on the latency.

    get_infractions:
        If True, infraction metrics will be returned for each agent.

    agent_count:
        Deprecated. Equivalent to padding the `agent_attributes` list to this length with default `AgentAttributes`.

    random_seed:
        Controls the stochastic aspects of initialization for reproducibility.

    api_model_version:
        Optionally specify the version of the model. If None is passed which is by default, the best model will be used.

    See Also
    --------
    :func:`drive`
    :func:`location_info`
    :func:`light`
    :func:`blame`
    """

    if should_use_mock_api():
        assert agent_properties is not None or agent_attributes is not None or agent_count is not None
        if agent_properties is not None:
            agent_count = len(agent_properties)
        elif agent_attributes is not None:
            agent_count = len(agent_attributes)

        agent_properties = [get_mock_agent_properties() for _ in range(agent_count)]
        agent_attributes = [get_mock_agent_attributes() for _ in range(agent_count)]
        if agent_attributes is None:
            agent_states = [get_mock_agent_state() for _ in range(agent_count)]
        else:
            agent_states = states_history[-1] if states_history is not None else []
        recurrent_states = [get_mock_recurrent_state() for _ in range(agent_count)]
        birdview = get_mock_birdview()
        infractions = get_mock_infractions(len(agent_states))
        response = InitializeResponse(
            agent_states=agent_states,
            agent_attributes=agent_attributes,
            agent_properties=agent_properties,
            recurrent_states=recurrent_states,
            birdview=birdview,
            infractions=infractions,
            api_model_version=api_model_version if api_model_version is not None else "best",
            traffic_lights_states=traffic_light_state_history[-1] if traffic_light_state_history is not None else None,
            light_recurrent_states=get_mock_light_recurrent_states(len(traffic_light_state_history[0])) if traffic_light_state_history is not None else None
        )
        return response

    if agent_attributes is not None:
        warnings.warn('agent_attributes is deprecated. Please use agent_properties.',category=DeprecationWarning)

    model_inputs = serialize_initialize_request_parameters(
        location=location,
        agent_attributes=agent_attributes,
        agent_properties=agent_properties,
        states_history=states_history,
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        location_of_interest=location_of_interest,
        get_infractions=get_infractions,
        agent_count=agent_count,
        random_seed=random_seed,
        api_model_version=api_model_version
    )
    start = time.time()
    timeout = TIMEOUT
    while True:
        try:
            response = iai.session.request(model="initialize", data=model_inputs)
            response = InitializeResponse(
                agent_states=[
                    AgentState.fromlist(state) for state in response["agent_states"]
                ],
                agent_attributes=[
                    AgentAttributes.fromlist(attr) for attr in response["agent_attributes"]
                ] if response["agent_attributes"] is not None else [],
                agent_properties=[
                    AgentProperties.deserialize(ap) for ap in response["agent_properties"]
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
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


@validate_call
async def async_initialize(
    location: str,
    agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_birdview: bool = False,
    location_of_interest: Optional[Tuple[float, float]] = None,
    get_infractions: bool = False,
    agent_count: Optional[int] = None,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
) -> InitializeResponse:
    """
    The async version of :func:`initialize`
    """

    model_inputs = dict(
        location=location,
        num_agents_to_spawn=agent_count,
        states_history=states_history
        if states_history is None
        else [[st.tolist() for st in states] for states in states_history],
        agent_attributes=agent_attributes
        if agent_attributes is None
        else [state.tolist() for state in agent_attributes],
        agent_properties=agent_properties if agent_properties is None 
        else [ap.serialize() if ap else None for ap in agent_properties] ,
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        location_of_interest=location_of_interest,
        get_infractions=get_infractions,
        random_seed=random_seed,
        model_version=api_model_version
    )

    response = await iai.session.async_request(model="initialize", data=model_inputs)
    agents_spawned = len(response["agent_states"])
    if agents_spawned != agent_count:
        iai.logger.warning(
            f"Unable to spawn a scenario for {agent_count} agents,  {agents_spawned} spawned instead."
        )
    response = InitializeResponse(
        agent_states=[
            AgentState.fromlist(state) for state in response["agent_states"]
        ],
        agent_attributes=[
            AgentAttributes.fromlist(attr) for attr in response["agent_attributes"]
        ],
        agent_properties=[
                    AgentProperties.deserialize(ap) for ap in response["agent_properties"]
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
