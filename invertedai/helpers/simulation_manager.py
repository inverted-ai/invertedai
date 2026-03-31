from typing import List, Optional, Tuple
from collections import defaultdict
from copy import deepcopy
from invertedai.common import RECURRENT_SIZE, AgentState, AgentProperties, RecurrentState, SimulationAgentDict, AgentData
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.helpers.waypoints import WaypointManagerConfig, WaypointManager
from pydantic import BaseModel
from invertedai.utils import ScenePlotterConfig, ScenePlotter, WaypointsDict
from invertedai.large.initialize import large_initialize, get_regions_default, RegionsConfig
from invertedai.large.drive import large_drive
from invertedai.logs.logger import LogWriterConfig, LogWriter
from invertedai.large.common import Region
from invertedai.api.location import location_info
from matplotlib.animation import FuncAnimation
import uuid

class SimulationManager: 
    """
    Stateful class for managing keyed agents with an internal dictionary SimulationAgentDict to manage AgentData by AgentID 
        and provides wrappers around the IAI large_initialize and large_drive APIs

    Parameters:
    scene_plotter_cfg : Optional[ScenePlotterConfig]
        Configuration object used to initialize a ScenePlotter instance
        Enables birdview visualization and animation of the simulation

    waypoint_cfg : Optional[WaypointManagerConfig]
        Configuration for initializing a WaypointManager
        If provided waypoints will be dynamically updated during simulation

    log_writer_cfg : Optional[LogWriterConfig]
        Configuration for enabling structured logging of the simulation
        If provided all initialize and drive steps will be recorded to a JSON log
    """
    def __init__(
            self,
            scene_plotter_cfg: Optional[ScenePlotterConfig] = None, # can optionally initialize a scene plotter for visualization
            waypoint_cfg : Optional[WaypointManagerConfig] = None, # can optionally initialize a waypointManager to manage waypoints
            log_writer_cfg: Optional[LogWriterConfig] = None, # can optionally initialize a log_writer_cfg to write a json file log of the simulation
        ):
            self.scene_plotter = None
            self.scene_plotter_cfg = scene_plotter_cfg
            if scene_plotter_cfg:
                map_center = scene_plotter_cfg.map_center if scene_plotter_cfg.map_center is not None else \
                    (scene_plotter_cfg.location_info_response.map_center.x, scene_plotter_cfg.location_info_response.map_center.y)
                fov = scene_plotter_cfg.fov if scene_plotter_cfg.fov is not None else \
                    scene_plotter_cfg.location_info_response.map_fov
                birdview_image = scene_plotter_cfg.location_info_response.birdview_image.decode()
                if scene_plotter_cfg.fov is not None or scene_plotter_cfg.map_center is not None:
                    correct_li = location_info(
                        location=scene_plotter_cfg.location,
                        rendering_fov=fov,
                        rendering_center=map_center,
                    )
                    birdview_image = correct_li.birdview_image.decode()
                self.scene_plotter = ScenePlotter(
                    birdview_image,
                    fov,
                    map_center,
                    scene_plotter_cfg.location_info_response.static_actors,
                    left_hand_coordinates = scene_plotter_cfg.location.split(":")[0] == "carla",
                    resolution=(2048,2048)
                )
            self.agents_dict: SimulationAgentDict = defaultdict(AgentData)
            self.waypoint_manager: Optional[WaypointManager] = None
            if waypoint_cfg:
                self.waypoint_manager = WaypointManager(cfg=waypoint_cfg)
            self.log_writer = None
            self.log_writer_cfg = log_writer_cfg
            if log_writer_cfg:
                self.log_writer = LogWriter()   
    
    def form_regions(
        self,
        regions_config: RegionsConfig,
    ) -> List[Region]:
        """
        Uses :func:`get_regions_default` to generate regions based on the configuration provided from dataclass :class:`RegionsConfig`

        The returned list of regions can be passed directly to :func:`initialize`

        Parameters:
        regions_config : RegionsConfig
            Configuration specifying location, agent counts, area shape, etc.
        """
        return get_regions_default(
            location=regions_config.location,
            agent_count_dict=regions_config.agent_count_dict,
            area_shape=regions_config.area_shape,
            map_center=regions_config.map_center,
            random_seed=regions_config.random_seed,
            display_progress_bar=regions_config.display_progress_bar,
        )

    def insert_agents(
        self,
        agent_data_list: List[AgentData],
        ids: Optional[List[str]],
        overwrite: bool = False,
    ):
        """
        Insert multiple agents into the existing agents_dict using their AgentData

        Parameters:
        agent_data_list : List[AgentData]
            List of AgentData for each agent to be inserted
        ids : Optional[List[str]]
            Optional list of AgentIDs to use for the new agents. 
            If None, random UUIDs will be generated for each new agent.
        overwrite: bool
            If True, allows new agents to overwrite existing agents with the same ID.
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

        Raises:
        KeyError
            If any AgentID does not exist in self.agents_dict
        """
        missing = [aid for aid in agent_ids if aid not in self.agents_dict]
        if missing:
            raise KeyError(f"Agents do not exist: {missing}. Cannot be removed.")
        for aid in agent_ids:
            self.agents_dict.pop(aid)
    
    def _unpack(
        self,
        agent_dict: Optional[SimulationAgentDict] = None
    ) -> Tuple[
        List[str],
        List[AgentState],
        List[AgentProperties],
        List[RecurrentState],
    ]:
        # agents with both properties and states will be placed at the front of the list
        # Separate agents into two groups: with states & without states
        agents_with_states = []
        agents_without_states = []
        if agent_dict is None:
            agent_dict = self.agents_dict
        for aid, data in agent_dict.items():
            if data.properties is not None:
                if data.state is not None:
                    agents_with_states.append((aid, data))
                else:
                    agents_without_states.append((aid, data))
        
        # agents_with_states in front of agents_without_states for API alignment
        ordered_agents = agents_with_states + agents_without_states
        
        agent_ids: List[str] = []
        states: List[AgentState] = []
        properties: List[AgentProperties] = []
        recurrent_states: List[RecurrentState] = []
        
        # Determine actual recurrent size from existing agents
        recurrent_size = RECURRENT_SIZE
        for _, data in ordered_agents:
            if data.recurrent is not None:
                recurrent_size = len(data.recurrent.packed)
                break

        for aid, data in ordered_agents:
            agent_ids.append(aid)
            states.append(data.state)  
            properties.append(data.properties)
            recurrent_states.append(data.recurrent if data.recurrent is not None else RecurrentState(packed=[0.0] * recurrent_size))
        if states == [None] * len(states):
            states = None
        
        return agent_ids, states, properties, recurrent_states

    def _pack(
        self,
        agent_ids: List[str],
        states: List[AgentState],
        properties: List[AgentProperties],
        recurrent_states: List[RecurrentState],
    ) -> SimulationAgentDict:
        agents_dict = defaultdict(AgentData)
        for i, aid in enumerate(agent_ids):
            agents_dict[aid] = AgentData(
                state=states[i],
                properties=properties[i] if properties else None,
                recurrent=recurrent_states[i] if recurrent_states else None,
            )
        return agents_dict

    def initialize(
        self,
        regions: List[Region],
        external_agent_data: Optional[SimulationAgentDict] = None,
        return_external_dict: bool = False,
        **kwargs
    ) -> InitializeResponse:
        """
        Initialize simulation using :func:`large_initialize` with agents in self.agents_dict along with any optionally provided external_agent_data

        Parameters:
        regions : List[Region]
            Regions with presampled agents. use iai.get_regions_default() to obtain list of Regions

        external_agent_data : Optional[SimulationAgentDict]
            Optional stateless dictionary of externally created agents to initialize alongside internal self.agents_dict
            Requires valid properties and optional state to be provided for each agent in the dictionary
            You can use the Scenario Builder tool available on the Inverted AI website to validate an agent's state and properties
        
        Please see :func:`large_initialize` for documentation on **kwargs

        Note:
        - agent_states, agent_properties, and recurrent_states should not be
          provided in **kwargs. These values are automatically derived from the
          internal agent dictionary and managed by this wrapper
        - For all other supported parameters, please refer to the documentation for :func:`large_initialize`        
        """
        if external_agent_data:
            overlap = set(self.agents_dict.keys()) & set(external_agent_data.keys())
            if overlap:
                raise ValueError(f"External agent IDs conflict with internal agents: {overlap}")
            
        agent_ids, states, properties, recurrent_states = self._unpack(
            {**self.agents_dict, **external_agent_data} if external_agent_data else self.agents_dict
        )
        external_ids = set(external_agent_data.keys()) if external_agent_data else set()

        if any(p is None for p in properties):
            raise ValueError("All agents must have non-None properties before initialization.")

        original_agent_count = len(agent_ids)

        response = large_initialize( 
            regions=regions,
            agent_properties=properties,
            agent_states=states,
            return_exact_agents=True,
            **kwargs
        )

        num_new_agents = len(response.agent_states) - original_agent_count
        new_ids = [str(uuid.uuid4()) for _ in range(num_new_agents)]
        all_agent_ids = agent_ids + new_ids

        new_properties = response.agent_properties
        if self.waypoint_manager:
            new_properties = self.waypoint_manager.update(
                response = response,
                agent_properties = response.agent_properties,
            )
        internal_indices = []
        for i, aid in enumerate(all_agent_ids):
            if aid not in external_ids:
                internal_indices.append(i)

        self.agents_dict = self._pack(
            agent_ids=[all_agent_ids[i] for i in internal_indices],
            states=[response.agent_states[i] for i in internal_indices],
            properties=[new_properties[i] for i in internal_indices],
            recurrent_states=[response.recurrent_states[i] for i in internal_indices],
        )
        if self.scene_plotter:
            self.scene_plotter.initialize_recording(
                agent_states=response.agent_states,
                agent_properties=response.agent_properties,
            )
        if self.log_writer is not None or return_external_dict:
            all_agents_dict = self._pack( # both internal+external agents
                agent_ids=all_agent_ids,
                states=response.agent_states,
                properties=new_properties,
                recurrent_states=response.recurrent_states,
            )
            if self.log_writer is not None:
                self.log_writer.initialize(
                    location=self.log_writer_cfg.location,
                    location_info_response=self.log_writer_cfg.location_info_response,
                    agents_dict=all_agents_dict,
                    init_response=response,
                )
        if return_external_dict:
            external_dict = {aid: all_agents_dict[aid] for aid in external_ids}
            return response, external_dict
        return response
    
    def drive(
        self, 
        external_agent_data: Optional[SimulationAgentDict] = None,
        return_external_dict: bool = False,
        **kwargs
    )-> DriveResponse:
        """
        Advance the simulation by one timestep with :func:`large_drive` using the data from agents 
            in self.agents_dict and optionally provided external_agent_data

        This method:
        - updates self.agents_dict with results from large_drive
        - uses iai.WaypointManager to update waypoints if configured
        - Records visualization and logging outputs if configured

        Parameters:
        external_agent_data : Optional[SimulationAgentDict]
            Optional stateless dictionary of externally created agents to 
                drive alongside internal self.agents_dict at this timestep
            Requires both valid state and properties to be provided for each agent in the dictionary
            You can use the Scenario Builder tool available on the Inverted AI website to validate an agent's state and properties

        Returns:
            DriveResponse

        Please see :func:`large_drive` for information on kwargs

        Note:
        - agent_states, agent_properties, and recurrent_states should not be
          provided in kwargs. These values are automatically derived from the
          internal agent dictionary and managed by this wrapper
        - For all other supported parameters, please refer to the documentation for :func:`large_drive`
        """
        if external_agent_data:
            overlap = set(self.agents_dict.keys()) & set(external_agent_data.keys())
            if overlap:
                raise ValueError(f"External agent IDs conflict with internal agents: {overlap}")
            
        agent_ids, states, properties, recurrent_states = self._unpack(self.agents_dict)
        if len(recurrent_states) > 0: 
             internal_recur_size = len(recurrent_states[0].packed)
        else: 
            internal_recur_size = RECURRENT_SIZE
        if external_agent_data:
            # external data validation
            missing_states = [aid for aid, data in external_agent_data.items() if data.state is None]
            if missing_states:
                raise ValueError(f"External agents must have a state for drive: {missing_states}")
            missing_props = [aid for aid, data in external_agent_data.items() if data.properties is None]
            if missing_props:
                raise ValueError(f"External agents must have properties for drive: {missing_props}")
            ext_ids, ext_states, ext_props, _ = self._unpack(external_agent_data)
            ext_recurrent_states = [RecurrentState(packed=[0.0] * internal_recur_size) for _ in ext_ids]

            agent_ids = agent_ids + ext_ids
            states = states + ext_states
            properties = properties + ext_props
            recurrent_states = recurrent_states + ext_recurrent_states

        external_ids = set(ext_ids) if external_agent_data else set()

        response = large_drive(
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
        internal_indices = []
        for i, aid in enumerate(agent_ids):
            if aid not in external_ids:
                internal_indices.append(i)
        self.agents_dict = self._pack(
            agent_ids=[agent_ids[i] for i in internal_indices],
            states=[response.agent_states[i] for i in internal_indices],
            properties=[properties[i] for i in internal_indices],
            recurrent_states=[response.recurrent_states[i] for i in internal_indices],
        )
        if self.scene_plotter:
            self.scene_plotter.record_step(
                response.agent_states,
                traffic_light_states=response.traffic_lights_states,
                agent_properties=properties,
            )
        if self.log_writer is not None or return_external_dict is not None:
            all_agents_dict = self._pack(
                agent_ids=agent_ids,
                states=response.agent_states,
                properties=properties,
                recurrent_states=response.recurrent_states,
            )
            if self.log_writer is not None:
                self.log_writer.drive(
                    drive_response=response,
                    agents_dict=all_agents_dict,
                )
        if return_external_dict:
            external_dict = {aid: all_agents_dict[aid] for aid in external_ids}
            return response, external_dict
        return response
    
    def visualize_data(self, **kwargs) -> FuncAnimation:
        """
        Produce an animation of sequentially recorded steps. If a ScenePlotter was configured during initialization,
            recorded steps from each drive will be visualized using the birdview map and static actors.

        A matplotlib animation object can be returned and/or a gif saved of the scene.

        If fov or xy_offset are provided, a new birdview image will be fetched from location_info
        to match the updated view.

        For kwargs, please see documentation from :func:`animate_scene` in the ScenePlotter class
        """
        if self.scene_plotter is None:
            raise ValueError("ScenePlotter not initialized, failed to animate scene")
        self.scene_plotter.animate_scene(**kwargs)
    
    def export_log(self, path: Optional[str] = None):
        """
        Export the log of the simulation to a JSON file if logging was enabled with a 
            LogWriterConfig during initialization of the SimulationManager
        
        Parameters:
        path : Optional[str]
            Optional path to specify where the log should be saved. If None, will default to the path provided in LogWriterConfig during initialization. 
        
        If no path is provided in either place, will raise an error.
        """
        if self.log_writer is None:
            raise ValueError("Logging not enabled.")
        log_path = path or self.log_writer_cfg.log_path
        if log_path is None:
            raise ValueError("No export path specified.")
        self.log_writer.export_to_file(log_path=log_path)

    # Getters
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
    
    def get_state(self, agent_id: str) -> AgentState:
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict[agent_id].state

    def get_property(self, agent_id: str) -> AgentProperties:
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict[agent_id].properties

    def get_recurrent_state(self, agent_id: str) -> RecurrentState:
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict[agent_id].recurrent

    def get_agent_data(self, agent_id:str) -> AgentData:
        if agent_id not in self.agents_dict:
            raise KeyError(f"Agent '{agent_id}' does not exist")
        return self.agents_dict[agent_id]
    
    def get_agent_dict(self)-> SimulationAgentDict:
        return self.agents_dict
    
    # Setters for individual agents in self.agents_dict
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

    #Setters for all agents in self.agents_dict
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
