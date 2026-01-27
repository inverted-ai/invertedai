from typing import Dict, List, Optional, Tuple
from invertedai.common import AgentState, AgentProperties, RecurrentState
from pydantic import BaseModel
import invertedai as iai
from invertedai.api.drive import DriveResponse
from invertedai.api.initialize import InitializeResponse

class AgentData(BaseModel):
    """
    Container for all agent data
    """
    state: Optional[AgentState] = None
    properties: Optional[AgentProperties] = None
    recurrent: Optional[RecurrentState] = None

class KeyedAgents(BaseModel):
    @classmethod
    def unpack_agents(
        cls,
        agents: Dict[str, AgentData],
    ) -> Tuple[
        List[str],
        List[AgentState],
        List[AgentProperties],
        List[RecurrentState],
    ]:
        """
        Convert keyed AgentData dict into List[AgentID] + 3 lists for API
        """
        agent_ids: List[str] = []
        states: List[AgentState] = []
        properties: List[AgentProperties] = []
        recurrent_states: List[RecurrentState] = []
        for aid, data in agents.items():
            agent_ids.append(aid)
            if data.state is not None:
                states.append(data.state)
            if data.properties is not None:
                properties.append(data.properties)
            if data.recurrent is not None:
                recurrent_states.append(data.recurrent)
        return agent_ids, states, properties, recurrent_states
    @classmethod
    def pack_agents(
        cls,
        agent_ids: List[str],
        states: List[AgentState],
        properties: Optional[List[AgentProperties]],
        recurrent_states: Optional[List[RecurrentState]],
    ) -> Dict[str, AgentData]:
        """"
        Reconstruct keyed AgentData dictionary from API responses
        """
        out: Dict[str, AgentData] = {}
        for i, aid in enumerate(agent_ids):
            out[aid] = AgentData(
                state=states[i] if states else None,
                properties=properties[i] if properties else None,
                recurrent=recurrent_states[i] if recurrent_states else None,
            )
        return out
    @classmethod
    def keyed_initialize(
        cls,
        agents: Dict[str, AgentData],
        *,
        large: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, AgentData], InitializeResponse]:
        """
        Wrapper around initialize or large_initialize
        takes in keyed AgentData dict
        """
        agent_ids, _, properties, _ = KeyedAgents.unpack_agents(agents)
        if large:
            response = iai.large_initialize(
                agent_properties=properties if properties else None,
                **kwargs,
            )
        else:
            response = iai.initialize(
                agent_properties=properties if properties else None,
                **kwargs,
            )
        agent_ids = [f"agent_{i}" for i in range(len(response.agent_states))]
        updated_agents = KeyedAgents.pack_agents(
            agent_ids=agent_ids,
            states=response.agent_states,
            properties=response.agent_properties,
            recurrent_states=response.recurrent_states,
        )
        return updated_agents, response
    @classmethod
    def keyed_drive(
        cls,
        agents: Dict[str, AgentData],
        *,
        large: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, AgentData], DriveResponse]:
        """
        Wrapper around drive or large_drive
        """
        agent_ids, states, properties, recurrent = KeyedAgents.unpack_agents(agents)
        if large:
            response = iai.large_drive(
                agent_states=states,
                agent_properties=properties if properties else None,
                recurrent_states=recurrent if recurrent else None,
                **kwargs,
            )
        else:
            response = iai.drive(
                agent_states=states,
                agent_properties=properties if properties else None,
                recurrent_states=recurrent if recurrent else None,
                **kwargs,
            )
        updated_agents = KeyedAgents.pack_agents(
            agent_ids=agent_ids,
            states=response.agent_states,
            properties=properties,
            recurrent_states=response.recurrent_states,
        )
        return updated_agents, response
