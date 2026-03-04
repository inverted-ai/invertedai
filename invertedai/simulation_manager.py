from typing import Dict, List, Optional, Tuple
from invertedai.common import AgentState, AgentProperties, RecurrentState, AgentType, Point
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.api.location import LocationResponse
from invertedai.helpers.waypoints import WaypointManagerConfig, WaypointManager
from pydantic import BaseModel
from invertedai.utils import get_default_agent_properties, ScenePlotterConfig, ScenePlotter, WaypointsDict
from invertedai.logs.logger import LogWriterConfig, ScenarioLog, LogWriter
from invertedai.large.common import Region
from matplotlib.animation import FuncAnimation
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

AgentDict = Dict[AgentID, AgentData]

class SimulationManager: 
    """
    Class for managing keyed agents with an internal dictionary structure to manage AgentData by AgentID 
        and provides wrappers around the IAI large_initialize and large_drive APIs

    Parameters:
    scene_plotter_cfg : 
        Configuration object used to initialize a ScenePlotter instance
        Enables birdview visualization and animation of the simulation

    agents_dict : 
        A pre-existing dictionary mapping AgentID (str) to AgentData
        Use this when restoring from logs or inserting externally managed agents

    num_agents : 
        Number of agents to initialize in the agent_dict
        Each agent will be assigned:
            - a UUID-based AgentID
            - default car AgentProperties
        Ignored if agents_dict is provided

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
            agents_dict: Optional[AgentDict] = None, # can pass in a pre-existing agent dict
            num_agents: Optional[int] = 0, # can pass in number of agents to initialize if not passing in pre-existing agent_dict
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
            if agents_dict is None:
                agents_dict = {
                    str(uuid.uuid4()): AgentData(
                        properties=get_default_agent_properties({AgentType.car: 1})[0]
                    )
                    for _ in range(num_agents)
                }
            self.agents_dict = agents_dict
            self.waypoint_manager: Optional[WaypointManager] = None
            if waypoint_cfg:
                self.waypoint_manager = WaypointManager(cfg=waypoint_cfg)
            self.log_writer = None
            self.log_writer_cfg = log_writer_cfg
            if log_writer_cfg:
                self.log_writer = LogWriter()   
    
    def insert_agents(
        self,
        states: List[AgentState],
        properties: List[AgentProperties],
        recurrent_states: Optional[List[Optional[RecurrentState]]] = None,
        overwrite: bool = False,
    ) -> List[str]:
        """
        Insert multiple agents into the existing agents_dict using their list of agent_states, agent_properties and recurrent_states
        """
        if len(states) != len(properties):
            raise ValueError("Length of states and properties must match.")
        if recurrent_states is not None and len(recurrent_states) != len(states):
            raise ValueError("Length of recurrent_states must match states.")
        
        new_ids = []
        for i, agent_id in enumerate(new_ids):
            agent=AgentData(
                state=states[i],
                properties=properties[i],
                recurrent=recurrent_states[i] if recurrent_states else None,
            )
            if agent_id in self.agents_dict and not overwrite:
                raise ValueError(f"Agent '{agent_id}' already exists")
            self.agents_dict[agent_id] = agent
        return new_ids
        
    def remove_agents(
        self,
        agent_ids: List[str],
    ) -> Dict[str, AgentData]:
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
            raise KeyError(f"Agents do not exist: {missing}")

        removed = {}
        for aid in agent_ids:
            if aid not in self.agents_dict:
                raise KeyError(f"Agent '{aid}' does not exist")
            return self.agents_dict.pop(aid)

        return removed
    
    def _unpack(
        self
    ) -> Tuple[
        List[str],
        List[AgentState],
        List[AgentProperties],
        List[RecurrentState],
    ]:
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
    
    def _pack(
        self,
        agent_ids: List[str],
        states: List[AgentState],
        properties: Optional[List[AgentProperties]],
        recurrent_states: Optional[List[RecurrentState]],
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
            location: str, 
            regions: Optional[List[Region]] =None, 
            num_new_agents: Optional[int] = None,
            **kwargs
        )->InitializeResponse:
        """
        Wrapper around iai.large_initialize
        
        Parameters:

        """
        agent_ids, states, properties, recurrent_states = self._unpack()
        if num_new_agents is not None and regions is None:
            regions = iai.get_regions_default(
                location = location,
                agent_count_dict = {AgentType.car: num_new_agents},
            )
        if regions is None:
            regions = iai.get_regions_default(
                location = location,
                agent_count_dict = {AgentType.car: len(self.agents_dict)},
            )
        num_existing = len(self.agents_dict)
        response = iai.large_initialize( 
            location=location,
            regions=regions,
            agent_properties=properties,
            agent_states=states,
            return_exact_agents=True,
            **kwargs
        )
        
        num_returned = len(response.agent_states)
        num_new = num_returned - num_existing

        # generate new ids for the new agents
        if num_new > 0:
            new_ids = [str(uuid.uuid4()) for _ in range(num_new)]
            agent_ids = agent_ids + new_ids
        properties = response.agent_properties
        if self.waypoint_manager:
            properties = self.waypoint_manager.update(
                response = response,
                agent_properties = response.agent_properties,
            )
        self._pack(
            agent_ids=agent_ids,
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
        # agents_mask: Optional[List[bool]] = None, # make it its separate thing later
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
        """
        kwargs.pop("recurrent_states", None)
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
                agent_properties = properties, # consdier isolating later
                # agents_mask=agents_mask 
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

def build_agent_dict_from_lists(
    states: List[AgentState],
    properties: List[AgentProperties],
    recurrent_states: Optional[List[Optional[RecurrentState]]] = None,
    existing_agents_dict: Optional[AgentDict] = None,
    overwrite: bool = False,
) -> AgentDict:
    """
    Construct or extend an AgentDict from parallel lists of states, properties,
    and recurrent states.

    Parameters
    states : List[AgentState]
        List of agent states

    properties : List[AgentProperties]
        List of agent properties. Must match length of states

    recurrent_states : Optional[List[Optional[RecurrentState]]]
        List of recurrent states. If provided, must match length of states
        If None, recurrent field will be set to None

    existing_agents_dict : Optional[AgentDict]
        If provided, new agents will be merged into this dictionary
        If None, a new dictionary will be created

    overwrite : bool, default=False
        If False and an AgentID already exists in existing_agents_dict,
        raises ValueError.
        If True, existing entries will be replaced.

    Returns
    AgentDict
        Newly constructed or merged dictionary of agents.

    """

    if len(states) != len(properties):
        raise ValueError("Length of states and properties must match.")

    if recurrent_states is not None and len(recurrent_states) != len(states):
        raise ValueError("Length of recurrent_states must match states.")

    agent_dict = existing_agents_dict.copy() if existing_agents_dict else {}
    new_ids = [str(uuid.uuid4()) for _ in states]
    if not overwrite:
        conflicts = [aid for aid in new_ids if aid in agent_dict]
        if conflicts:
            raise ValueError(
                f"Generated AgentIDs already exist and overwrite=False: {conflicts}"
            )
    for i, agent_id in enumerate(new_ids):
        agent_dict[agent_id] = AgentData(
            state=states[i],
            properties=properties[i],
            recurrent=recurrent_states[i] if recurrent_states else None,
        )

    return agent_dict