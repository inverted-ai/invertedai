import time
from pydantic import BaseModel, validate_arguments
from typing import List, Optional, Dict, Tuple
import asyncio

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import TryAgain, InvalidInputType, InvalidInput
from invertedai.api.mock import (
    get_mock_agent_attributes,
    get_mock_agent_state,
    get_mock_recurrent_state,
    get_mock_birdview,
    get_mock_infractions,
)
from invertedai.common import (
    RecurrentState,
    AgentState,
    AgentAttributes,
    TrafficLightStatesDict,
    Image,
    InfractionIndicators,
    LightRecurrentStates,
    LightRecurrentState,
)


class InitializeResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.initialize`.
    """

    recurrent_states: List[
        Optional[RecurrentState]
    ]  #: To pass to :func:`iai.drive` at the first time step.
    agent_states: List[Optional[AgentState]]  #: Initial states of all initialized agents.
    agent_attributes: List[
        Optional[AgentAttributes]
    ]  #: Static attributes of all initialized agents.
    birdview: Optional[
        Image
    ]  #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[
        List[InfractionIndicators]
    ]  #: If `get_infractions` was set, they are returned here.
    traffic_lights_states: Optional[TrafficLightStatesDict]  #: Traffic light states for the full map, each key-value pair corresponds to one particular traffic light.
    light_recurrent_states: Optional[LightRecurrentStates] #: Light recurrent states for the full map, each element corresponds to one light group.
    model_version: str # Model version used for this API call


@validate_arguments
def initialize(
        location: str,
        agent_attributes: Optional[List[AgentAttributes]] = None,
        states_history: Optional[List[List[AgentState]]] = None,
        traffic_light_state_history: Optional[
            List[TrafficLightStatesDict]
        ] = None,
        get_birdview: bool = False,
        location_of_interest: Optional[Tuple[float, float]] = None,
        get_infractions: bool = False,
        agent_count: Optional[int] = None,
        random_seed: Optional[int] = None,
        model_version: Optional[str] = None  # Model version used for this API call
) -> InitializeResponse:
    """
    Initializes a simulation in a given location, using a combination of **user-defined** and **sampled** agents.
    **User-defined** agents are placed in a scene first, after which a number of agents are sampled conditionally 
    inferred from the `agent_count` argument.
    If **user-defined** agents are desired, `states_history` must contain a list of `AgentState's` of all **user-defined** 
    agents per historical time step.
    Any **user-defined** agent must have a corresponding fully specified static `AgentAttribute` in `agent_attributes`. 
    Any **sampled** agents not specified in `agent_attributes` will be generated with default static attribute values however **sampled** 
    agents may be defined by specifying `agent_type` only. 
    Agents are identified by their list index, so ensure the indices of each agent match in `states_history` and
    `agent_attributes` when applicable. 
    If traffic lights are present in the scene, for best results their state should be specified for the current time in a 
    `TrafficLightStatesDict`, and all historical time steps for which `states_history` is provided. It is legal to omit
    the traffic light state specification, but the scene will be initialized as if the traffic lights were disabled.
    Every simulation must start with a call to this function in order to obtain correct recurrent states for :func:`drive`.

    Parameters
    ----------
    location:
        Location name in IAI format.

    agent_attributes:
        Static attributes for all agents.
        The pre-defined agents should be specified first, followed by the sampled agents.
        The optional waypoint passed will be ignored for Initialize.

    states_history:
        History of pre-defined agent states - the outer list is over time and the inner over agents,
        in chronological order, i.e., index 0 is the oldest state and index -1 is the current state.
        The order of agents should be the same as in `agent_attributes`.
        For best results, provide at least 10 historical states for each agent.

    traffic_light_state_history:
       History of traffic light states - the list is over time, in chronological order, i.e.
       the last element is the current state. If there are traffic lights in the map, 
       not specifying traffic light state is equivalent to using iai generated light states.

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

    model_version:
        Optionally specify the version of the model. If None is passed which is by default, the best model will be used.

    See Also
    --------
    :func:`drive`
    :func:`location_info`
    :func:`light`
    :func:`blame`
    """

    if should_use_mock_api():
        if agent_attributes is None:
            assert agent_count is not None
            agent_attributes = [get_mock_agent_attributes() for _ in range(agent_count)]
            agent_states = [get_mock_agent_state() for _ in range(agent_count)]
        else:
            agent_states = states_history[-1]
        recurrent_states = [get_mock_recurrent_state() for _ in range(agent_count)]
        birdview = get_mock_birdview()
        infractions = get_mock_infractions(len(agent_states))
        response = InitializeResponse(
            agent_states=agent_states,
            agent_attributes=agent_attributes,
            recurrent_states=recurrent_states,
            birdview=birdview,
            infractions=infractions,
        )
        return response

    model_inputs = dict(
        location=location,
        num_agents_to_spawn=agent_count,
        states_history=states_history
        if states_history is None
        else [[st.tolist() for st in states] for states in states_history],
        agent_attributes=agent_attributes
        if agent_attributes is None
        else [state.tolist() for state in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        location_of_interest=location_of_interest,
        get_infractions=get_infractions,
        random_seed=random_seed,
        model_version=model_version
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
                model_version=response["model_version"],
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


@validate_arguments
async def async_initialize(
        location: str,
        agent_attributes: Optional[List[AgentAttributes]] = None,
        states_history: Optional[List[List[AgentState]]] = None,
        traffic_light_state_history: Optional[
            List[TrafficLightStatesDict]
        ] = None,
        get_birdview: bool = False,
        location_of_interest: Optional[Tuple[float, float]] = None,
        get_infractions: bool = False,
        agent_count: Optional[int] = None,
        random_seed: Optional[int] = None,
        model_version: Optional[str] = None
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
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        location_of_interest=location_of_interest,
        get_infractions=get_infractions,
        random_seed=random_seed,
        model_version=model_version
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
        model_version=response["model_version"],
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
