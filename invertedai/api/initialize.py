import time
from pydantic import BaseModel, validate_arguments, root_validator
from typing import List, Optional, Dict

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
)


class InitializeResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.initialize`.
    """

    recurrent_states: List[
        RecurrentState
    ]  #: To pass to :func:`iai.drive` at the first time step.
    agent_states: List[AgentState]  #: Initial states of all initialized agents.
    agent_attributes: List[
        AgentAttributes
    ]  #: Static attributes of all initialized agents.
    birdview: Optional[
        Image
    ]  #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[
        List[InfractionIndicators]
    ]  #: If `get_infractions` was set, they are returned here.


@validate_arguments
def initialize(
    location: str,
    conditional_agent_states:  Optional[List[AgentState]] = None,
    conditional_agent_attributes: Optional[List[AgentAttributes]] = None,
    agent_attributes: Optional[List[AgentAttributes]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[
        List[TrafficLightStatesDict]
    ] = None,
    get_birdview: bool = False,
    get_infractions: bool = False,
    agent_count: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> InitializeResponse:
    """
    Initializes a simulation in a given location.
    Either `agent_count` or both `agent_attributes` and `states_history` need to be provided.
    In the latter case, the simulation is initialized with the specific history,
    and if traffic lights are present then `traffic_light_state_history` should also be provided.
    If only `agent_count` is specified, a new initial state is generated with the requested
    total number of agents, which can be optionally conditioned on `conditional_agent_states` and
    `conditional_agent_attributes`. Every simulation needs to start with a call to this function
    in order to obtain correct recurrent states for :func:`drive`.

    Parameters
    ----------
    location:
        Location name in IAI format.

    agent_attributes:
        Static attributes for all agents.

    states_history:
        History of agent states - the outer list is over agents and the inner over time,
        in chronological order.
        For best results, provide at least 10 historical states for each agent.

    traffic_light_state_history:
       History of traffic light states - the list is over time, in chronological order.
       Traffic light states should be provided for all time steps where agent states are specified.

    get_birdview:
        If True, a birdview image will be returned representing the current world. Note this will significantly
        impact on the latency.

    get_infractions:
        If True, infraction metrics will be returned for each agent.

    agent_count:
        If `states_history` is not specified, this needs to be provided and dictates how many
        agents will be spawned.

    conditional_agent_states:
        Optional conditional agent states when `agent_count` is passed. When passed, `agent_count` includes the number of
        conditional agents passed.

    conditional_agent_attributes:
        Optional agent attributes when `conditional_agent_states` is passed

    random_seed:
        Controls the stochastic aspects of initialization for reproducibility.

    See Also
    --------
    :func:`drive`
    :func:`location_info`
    :func:`light`
    """

    if (states_history is not None) or (agent_attributes is not None):
        if (agent_attributes is None) or (states_history is None):
            raise InvalidInput("'agent_attributes' or 'states_history' are not provided.")
        for agent_states in states_history:
            if len(agent_states) != len(agent_attributes):
                raise InvalidInput("Incompatible Number of Agents in either 'agent_states' or 'agent_attributes'.")

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
        conditional_agent_states=conditional_agent_states if conditional_agent_states is None
        else [states.tolist() for states in conditional_agent_states],
        conditional_agent_attributes=conditional_agent_attributes if conditional_agent_attributes is None
        else [state.tolist() for state in conditional_agent_attributes],
        states_history=states_history
        if states_history is None
        else [[st.tolist() for st in states] for states in states_history],
        agent_attributes=agent_attributes
        if agent_attributes is None
        else [state.tolist() for state in agent_attributes],
        traffic_light_state_history=traffic_light_state_history,
        get_birdview=get_birdview,
        get_infractions=get_infractions,
        random_seed=random_seed,
    )
    start = time.time()
    timeout = TIMEOUT
    while True:
        try:
            response = iai.session.request(model="initialize", data=model_inputs)
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
            )
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))
