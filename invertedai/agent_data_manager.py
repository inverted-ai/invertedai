from typing import Dict, List, Optional, Tuple
from invertedai.common import AgentState, AgentProperties, RecurrentState, AgentType, Point
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.api.location import LocationResponse
from invertedai.helpers.waypoints import WaypointManagerConfig, WaypointManager
from pydantic import BaseModel
from invertedai.utils import get_default_agent_properties
import invertedai as iai
import uuid

AgentID = str                
class AgentData(BaseModel):
    """
    Container for all agent data
    """
    state: Optional[AgentState] = None
    properties: Optional[AgentProperties] = None
    recurrent: Optional[RecurrentState] = None
    # waypoint: Optional[Point] = None
    # waypoints: Optional[List[Point]] = None for later improvements

AgentDict = Dict[AgentID, AgentData]
class AgentDataManager: 
    """
    Class for managing keyed agents with an internal dictionary structure to manage AgentData by AgentID
    """
    def __init__(
            self,
            location_info_response: Optional[LocationResponse] = None,
            agents_dict: Optional[AgentDict] = None,
            num_agents: int = 0,   
            waypoint_cfg : Optional[WaypointManagerConfig] = None
        ):
            self.location_info_response = location_info_response
            if agents_dict is None:
                agents_dict = {
                    str(uuid.uuid4()): AgentData(
                        properties=get_default_agent_properties({AgentType.car: 1})[0]
                    )
                    for _ in range(num_agents)
            }
            self.agents_dict = agents_dict
            self.waypoint_manager: Optional[WaypointManager] = None
            if location_info_response:
                self.waypoint_manager = WaypointManager(location_info_response, waypoint_cfg)

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
        Remove an agent from the dictionary
        Returns the removed AgentData
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
    
    def initialize(self, location: str, **kwargs)->InitializeResponse:
        """
        Wrapper around iai.large_initialize
        """
        agent_ids, states, properties, recurrent_states = self.unpack()

        regions = iai.get_regions_default(
            location = location,
            agent_count_dict = {AgentType.car: len(self.agents_dict)},
            # area_shape = (int(args.width/2),int(args.height/2)),
            # map_center = map_center, 
        )
        response = iai.large_initialize( 
            location=location,
            regions=regions,
            agent_properties=properties,
            **kwargs
        )
        properties = response.agent_properties
        if self.waypoint_manager:
            properties = self.waypoint_manager.update( # consider isolating the waypoints later
                response = response,
                agent_properties = response.agent_properties,
            )
        self.pack(
            agent_ids=agent_ids,
            states=response.agent_states,
            properties=properties, ## # consider isolating later
            recurrent_states=response.recurrent_states,
        )
        return response
    
    def drive(self, location: str, **kwargs)-> DriveResponse:
        """
        Wrapper around iai.large_drive
        """
        agent_ids, states, properties, recurrent_states = self.unpack()
        response = iai.large_drive(
            location=location,
            agent_states=states,
            agent_properties=properties,
            recurrent_states=recurrent_states,
            **kwargs
        )
        if self.waypoint_manager:
            properties = self.waypoint_manager.update(
                response = response,
                agent_properties = properties, # consdier isolating later
            )
        self.pack(
            agent_ids=agent_ids,
            states=response.agent_states,
            properties=properties,# consdier isolating later
            recurrent_states=response.recurrent_states,
        )

        return response
    #Getters
    def get_states(self) -> List[AgentState]:
        return [data.state for data in self.agents_dict.values()]
    def get_agent_ids(self) -> List[str]:
        return list(self.agents_dict.keys())
    def get_properties(self) -> List[AgentProperties]:
        return [data.properties for data in self.agents_dict.values()]
    def get_recurrent_states(self) -> List[RecurrentState]:
        return [data.recurrent for data in self.agents_dict.values()]
    def get_agent_data(self, agent_id:str) -> AgentData:
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict[agent_id]
    def get_agent_dict(self)-> AgentDict:
        return self.agents_dict
    # Setters for individual agents
    def set_state(self, agent_id: str, state: AgentState):
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        self.agents_dict[agent_id].state = state
    def set_property(self, agent_id: str, properties: AgentProperties):
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        self.agents_dict[agent_id].properties = properties
    def set_recurrent_state(self, agent_id: str, recurrent: RecurrentState):
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        self.agents_dict[agent_id].recurrent = recurrent
    #Setters for all agents
    def set_states(self, states: List[AgentState]):
        if len(states) != len(self.agents_dict):
            raise ValueError(f"Expected {len(self.agents_dict)} states, got {len(states)}")
        for i, agent_id in enumerate(self.agents_dict.keys()):
            self.agents_dict[agent_id].state = states[i]
    def set_properties(self, properties: List[AgentProperties]):
        if len(properties) != len(self.agents_dict):
            raise ValueError(f"Expected {len(self.agents_dict)} properties, got {len(properties)}")
        for i, agent_id in enumerate(self.agents_dict.keys()):
            self.agents_dict[agent_id].properties = properties[i]
    def set_recurrent_states(self, recurrent_states: List[RecurrentState]):
        if len(recurrent_states) != len(self.agents_dict):
            raise ValueError(f"Expected {len(self.agents_dict)} recurrent states, got {len(recurrent_states)}")
        for i, agent_id in enumerate(self.agents_dict.keys()):
            self.agents_dict[agent_id].recurrent = recurrent_states[i]