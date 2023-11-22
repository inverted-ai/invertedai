from invertedai.error import InvalidInputType, InvalidInput
from typing import Optional, List
from invertedai.common import AgentAttributes, AgentState, RecurrentState


def all_float_gt_zero(agent: AgentAttributes):
    return all([type(attr) is float and attr > 0 for attr in agent.tolist()[:3]])


def validate_attributes_without_history(agent_attributes: List[AgentAttributes]):
    if agent_attributes is not None:
        if not all([len(agent.tolist()) == 1 and type(agent.agent_type) is str for agent in agent_attributes]):
            raise InvalidInputType(
                "agent_attributes: cannot specify this combination of attributes without full state history.")


def validate_attributes_with_history(states_history: List[List[AgentState]], agent_attributes: List[AgentAttributes],
                                     agent_count: int):
    if agent_attributes is None:
        raise InvalidInputType(
            "agent attributes: need to specify length, width, rear_axis_offsets given full state history.")
    else:
        if len(states_history[0]) != len(agent_attributes):
            raise InvalidInput(
                f"The number of agents in agents_attributes({len(agent_attributes)}) does not "
                f"match the number of agents in states_history({len(states_history[0])})."
            )
        if len(states_history[0]) > agent_count:
            raise InvalidInput(
                f"num_agents_to_spawn ({agent_count}) must be larger than the number of conditional agents"
                f" passed ({len(states_history[0])})."
            )
        if not all([len(agent.tolist()) >= 3 and all_float_gt_zero(agent) for agent in agent_attributes]):
            raise InvalidInput(
                "agent attributes: need to specify length, width, rear_axis_offsets in valid float range given full state history."
            )


def validate_initialize_flows(agent_attributes: Optional[List[AgentAttributes]] = None,
                              states_history: Optional[List[List[AgentState]]] = None,
                              agent_count: Optional[int] = None):
    if states_history is None:
        validate_attributes_without_history(agent_attributes)
    else:
        validate_attributes_with_history(states_history, agent_attributes, agent_count)


def validate_drive_flows(agent_states: List[AgentState],
                         agent_attributes: List[AgentAttributes],
                         recurrent_states: List[RecurrentState]):
    if len(agent_states) != len(agent_attributes):
        raise InvalidInput("Incompatible Number of Agents in either 'agent_states' or 'agent_attributes'.")
    if len(agent_states) != len(recurrent_states):
        raise InvalidInput("Incompatible Number of Agents in either 'agent_states' or 'recurrent_states'.")
    if not all([agent.agent_type == "car" for agent in agent_attributes]):
        raise InvalidInput("unsupported agent type in agents_attributes.")
