from typing import DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
from invertedai.common import AgentState, AgentProperties, RecurrentState, AgentType, Point
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.api.location import LocationResponse
from invertedai.helpers.waypoints import WaypointManagerConfig, WaypointManager
from pydantic import BaseModel
from invertedai.utils import get_default_agent_properties, ScenePlotterConfig, ScenePlotter, WaypointsDict
from dataclasses import dataclass
from invertedai.logs.logger import LogWriterConfig, ScenarioLog, LogWriter
from invertedai.large.common import Region
from matplotlib.animation import FuncAnimation
import invertedai as iai
import uuid

AgentID = str   
@dataclass             
class AgentData:
    """
    Container for all agent data
    """
    state: Optional[AgentState] = None
    properties: Optional[AgentProperties] = None
    recurrent: Optional[RecurrentState] = None

SimulationAgentDict = DefaultDict[AgentID, AgentData]

class SimulationManager: 
    """
    Stateful class for managing keyed agents with an internal dictionary structure to manage AgentData by AgentID 
        and provides wrappers around the IAI large_initialize and large_drive APIs

    Parameters:
    scene_plotter_cfg : 
        Configuration object used to initialize a ScenePlotter instance
        Enables birdview visualization and animation of the simulation

    waypoint_cfg : 
        Configuration for initializing a WaypointManager
        If provided waypoints will be dynamically updated during simulation

    log_writer_cfg :
        Configuration for enabling structured logging of the simulation
        If provided all initialize and drive steps will be recorded to a JSON log
    """
    def __init__(
            self,
            scene_plotter_cfg: Optional[ScenePlotterConfig] = None, # can optionally initialize a scene plotter for visualization
            waypoint_cfg : Optional[WaypointManagerConfig] = None, # can optionally initialize a waypointManager to manage waypoints
            log_writer_cfg: Optional[LogWriterConfig] = None # can optionally initialize a log_writer_cfg to write a json file log of the simulation
        ):
            self.scene_plotter = None
            if scene_plotter_cfg:
                self.scene_plotter = ScenePlotter(
                    scene_plotter_cfg.location_info_response.birdview_image.decode(),
                    scene_plotter_cfg.location_info_response.map_fov,
                    (scene_plotter_cfg.location_info_response.map_center.x, scene_plotter_cfg.location_info_response.map_center.y),
                    scene_plotter_cfg.location_info_response.static_actors,
                    left_hand_coordinates = scene_plotter_cfg.location.split(":")[0] == "carla"
                )
            self.agents_dict: SimulationAgentDict = defaultdict(AgentData)
            self.waypoint_manager: Optional[WaypointManager] = None
            if waypoint_cfg:
                self.waypoint_manager = WaypointManager(cfg=waypoint_cfg)
            self.log_writer = None
            self.log_writer_cfg = log_writer_cfg
            if log_writer_cfg:
                self.log_writer = LogWriter()   
    
    def insert_agents(
        self,
        agent_data_list: List[AgentData],
        ids: Optional[List[str]],
        overwrite: bool = False,
    ):
        """
        Insert multiple agents into the existing agents_dict using their AgentData
        """
        if ids is None:
            new_ids = [str(uuid.uuid4()) for _ in agent_data_list]
        else:
            new_ids = ids
        if len(new_ids) != len(agent_data_list):
            raise ValueError("Length of ids provided and agent_data_list is not equal")
        for i, agent_id in enumerate(new_ids):
            if agent_id in self.agents_dict and not overwrite:
                raise ValueError(f"Agent '{agent_id}' already exists. Cannot be inserted again with overwrite=False.")
            self.agents_dict[agent_id] = agent_data_list[i]
        return new_ids
        
    def remove_agents(
        self,
        agent_ids: List[str],
    ):
        """
        Removes multiple agents from the SimulationManager given their AgentIDs

        Parameters:
        agent_ids : List[str]
            List of AgentIDs to remove

        Returns:
        Dict[str, AgentData]
            Dictionary mapping removed AgentIDs to their AgentData

        Raises:
        KeyError
            If any AgentID does not exist
        """
        missing = [aid for aid in agent_ids if aid not in self.agents_dict]
        if missing:
            raise KeyError(f"Agents do not exist: {missing}. Cannot be removed.")
        for aid in agent_ids:
            self.agents_dict.pop(aid)
    
    def _unpack(
        self
    ) -> Tuple[
        List[str],
        List[AgentState],
        List[AgentProperties],
        List[RecurrentState],
    ]:
        """
        Unpack agent data, ensuring agents with populated states/properties are processed first
        This maintains proper index alignment with API responses
        """
        # Separate agents into two groups: with states/without states
        agents_with_states = []
        agents_without_states = []
        
        for aid, data in self.agents_dict.items():
            if data.properties is not None:
                if data.state is not None:
                    agents_with_states.append((aid, data)) # place as tuple along with aid
                else:
                    agents_without_states.append((aid, data))
        
        # agents_with_states in front of agents_without_states
        ordered_agents = agents_with_states + agents_without_states
        
        agent_ids: List[str] = []
        states: List[AgentState] = []
        properties: List[AgentProperties] = []
        recurrent_states: List[RecurrentState] = []
        
        for aid, data in ordered_agents:
            agent_ids.append(aid)
            states.append(data.state)  # Can be None for agents_without_states
            properties.append(data.properties)
            recurrent_states.append(data.recurrent)  # Can be None
        
        return agent_ids, states, properties, recurrent_states

    def _pack(
        self,
        agent_ids: List[str],
        states: List[AgentState],
        properties: List[AgentProperties],
        recurrent_states: List[RecurrentState],
    ):
        self.agents_dict = {
            aid: AgentData(
                state=states[i],
                properties=properties[i] if properties else None,
                recurrent=recurrent_states[i] if recurrent_states else None,
            )
            for i, aid in enumerate(agent_ids)
        }
    
    def initialize(
        self, 
        regions: List[Region],
        external_agent_data: Optional[SimulationAgentDict] = None, # optional param for passing in external agent data in the form of a SimulationAgentDict. Overwrite = True 
        **kwargs
    ) -> InitializeResponse:
        """
        Wrapper around iai.large_initialize

        Parameters:
        regions : List[Region]
            Regions with presampled agents. use iai.get_default_regions() to obtain list of Regions

        external_agent_data : Optional[SimulationAgentDict]
            Optional dictionary of externally created agents to merge into global self.agents_dict before initialization
        
        Please see :func:`large_initialize` for documentation on kwargs
        Note:
        - agent_states, agent_properties, and recurrent_states should not be
          provided in kwargs. These values are automatically derived from the
          internal agent dictionary and managed by this wrapper.
        - For all other supported parameters, please refer to the documentation for :func:`large_initialize`        """
        # must first merge external agents into global agents dictionary
        if external_agent_data:
            self.insert_agents(ids=external_agent_data.keys(), agent_data_list=external_agent_data.values(), overwrite=True)

        agent_ids, states, properties, recurrent_states = self._unpack()
        original_agent_count=len(agent_ids)

        response = iai.large_initialize( 
            regions=regions,
            agent_properties=properties,
            agent_states=states,
            return_exact_agents=True,
            **kwargs
        )

        num_new_agents = len(response.agent_states) - original_agent_count
        new_ids = [str(uuid.uuid4()) for _ in range(num_new_agents)]
        all_agent_ids = agent_ids + new_ids
        properties = response.agent_properties
        
        if self.waypoint_manager:
            properties = self.waypoint_manager.update(
                response = response,
                agent_properties = response.agent_properties,
            )
        self._pack(
            agent_ids=all_agent_ids,
            states=response.agent_states,
            properties=properties,
            recurrent_states=response.recurrent_states,
        )
        if self.scene_plotter:
            self.scene_plotter.initialize_recording(
                agent_states=response.agent_states,
                agent_properties=response.agent_properties,
            )
        if self.log_writer is not None:
            if self.waypoint_manager is not None:
                waypoints = {
                    aid: data.properties.waypoints
                    for aid, data in self.agents_dict.items()
                    if data.properties is not None and data.properties.waypoints is not None
                }
            self.log_writer.initialize(
                location=self.log_writer_cfg.location,
                location_info_response=self.log_writer_cfg.location_info_response,
                init_response=response,
                waypoints=waypoints 
            )

        return response
    
    def drive(
        self, 
        location: str, 
        **kwargs
    )-> DriveResponse:
        """
        Advance the simulation by one timestep using the current agents in self.agent_dict

        This method:
        - updated the self.agent_dict with results from DRIVE
        - Applies waypoint-based modifications if configured
        - Records visualization and logging outputs if configured

        Returns:
            DriveResponse

        Please see :func:`large_drive` for information on kwargs
        Note:
        - agent_states, agent_properties, and recurrent_states should not be
          provided in kwargs. These values are automatically derived from the
          internal agent dictionary and managed by this wrapper.
        - For all other supported parameters, please refer to the documentation for :func:`large_drive`
          
        """
        agent_ids, states, properties, recurrent_states = self._unpack()
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
                agent_properties = properties,
            )
        self._pack(
            agent_ids=agent_ids,
            states=response.agent_states,
            properties=properties,
            recurrent_states=response.recurrent_states,
        )
        if self.scene_plotter:
            self.scene_plotter.record_step(
                response.agent_states,
                traffic_light_states=response.traffic_lights_states,
                agent_properties=properties,
            )
        if self.log_writer is not None:
            waypoints: Optional[WaypointsDict] = None
            if self.waypoint_manager is not None:
                waypoints = {
                    aid: data.properties.waypoints
                    for aid, data in self.agents_dict.items()
                    if data.properties is not None and data.properties.waypoints is not None
                }
            self.log_writer.drive(
                drive_response=response,
                waypoints=waypoints 
            )
        return response
    
    def visualize_data(self, **kwargs) -> FuncAnimation:
        """
        Produce an animation of sequentially recorded steps. A matplotlib animation object can be returned and/or a gif saved of the scene.

        For kwargs, please see documentation for ScenePlotter.animate_scene
        """
        if self.scene_plotter is None:
            raise ValueError("ScenePlotter not initialized, failed to animate scene")
        self.scene_plotter.animate_scene(**kwargs)
    
    def export_log(self, path: Optional[str] = None):
        if self.log_writer is None:
            raise ValueError("Logging not enabled.")
        log_path = path or self.log_writer_cfg.log_path
        if log_path is None:
            raise ValueError("No export path specified.")
        self.log_writer.export_to_file(log_path=log_path)

    #Getters
    def get_scene_plotter(self) -> Optional[ScenePlotter]:
        return self.scene_plotter
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
    def get_agent_dict(self)-> SimulationAgentDict:
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
