from typing import Dict, List, Optional, Tuple
from invertedai.common import AgentState, AgentProperties, RecurrentState, AgentType
from pydantic import BaseModel, Field
import invertedai as iai
from invertedai.utils import get_default_agent_properties

AgentID = str
class AgentData(BaseModel):
    """
    Container for all agent data
    """
    state: Optional[AgentState] = None
    properties: Optional[AgentProperties] = None
    recurrent: Optional[RecurrentState] = None

AgentDict = Dict[AgentID, AgentData]
class KeyedAgents(BaseModel):
    """
    Class for managing keyed agents with an internal dictionary structure to manage AgentData by AgentID
    """
    agents_dict: AgentDict
    def __init__(self, num_agents: int = 0, **data):
        agents_dict = {}
        for i in range(num_agents):
            aid = f"agent_{i}"
            agents_dict[aid] = AgentData(
                properties=get_default_agent_properties({AgentType.car: 1})[0]
            )
        super().__init__(agents_dict=agents_dict, **data)
    def add_agent(
        self,
        agent_id: str,
        agent: AgentData,
        overwrite: bool = False,
    ):
        """
        Add one agent to the dictionary
        Checks for overwriting 
        """
        if agent_id in self.agents_dict and not overwrite:
            raise ValueError(f"Agent '{agent_id}' already exists")
        self.agents_dict[agent_id] = agent

    def remove_agent(self, agent_id: str) -> AgentData:
        """
        Remove an agent from the container
        Returns the removed AgentData so the caller may:
            discard it, store it or reinsert it later with preserved states
        """
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict.pop(agent_id)

    def unpack(
        self
    ) -> Tuple[
        List[str],
        List[AgentState],
        List[AgentProperties],
        List[RecurrentState],
    ]:
        """
        Convert AgentDict into List[AgentID] + 3 lists for API
        """
        agent_ids: List[str] = []
        states: List[AgentState] = []
        properties: List[AgentProperties] = []
        recurrent_states: List[RecurrentState] = []
        for aid, data in self.agents_dict.items():
            agent_ids.append(aid)
            if data.state is not None:
                states.append(data.state)
            if data.properties is not None:
                properties.append(data.properties)
            if data.recurrent is not None:
                recurrent_states.append(data.recurrent)
        return agent_ids, states, properties, recurrent_states
    
    def pack(
        self,
        agent_ids: List[str],
        states: List[AgentState],
        properties: Optional[List[AgentProperties]],
        recurrent_states: Optional[List[RecurrentState]],
    ):
        """"
        Reconstruct AgentDict from API response
        """
        self.agents_dict = {
            aid: AgentData(
                state=states[i],
                properties=properties[i] if properties else None,
                recurrent=recurrent_states[i] if recurrent_states else None,
            )
            for i, aid in enumerate(agent_ids)
        }

    #Getters
    def get_states(self) -> List[AgentState]:
        return [data.state for data in self.agents_dict.values()]
    def get_agent_ids(self) -> List[str]:
        return list(self.agents_dict.keys())
    def get_properties(self) -> List[AgentProperties]:
        return [data.properties for data in self.agents_dict.values()]
    def get_recurrent_states(self) -> List[RecurrentState]:
        return [data.recurrent for data in self.agents_dict.values()]