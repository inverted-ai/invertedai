from collections import defaultdict

from pydantic import BaseModel, validate_arguments, model_validator
from typing import List, Optional, Dict, Tuple, Any, Union
from copy import deepcopy

import matplotlib.pyplot as plt
import json

from invertedai.utils import ScenePlotter, agents_from_lists, WaypointsDict, convert_attributes_to_properties
from invertedai import location_info
from invertedai.api.location import LocationResponse
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.common import ( 
    AgentAttributes,
    AgentData, 
    AgentProperties,
    AgentState, 
    LightRecurrentState,
    LightRecurrentStates,
    Point,
    RecurrentState,
    SimulationAgentDict,
    TrafficLightStatesDict 
)

class ScenarioLog(BaseModel):
    """
    A log containing simulation information for storage, replay, or an initial state from which a simulation 
    can be continued. Some data fields contain data for all historic time steps while others contain information
    for the most recent time step to be used to continue a simulation.
    """
    agent_data: List[SimulationAgentDict] #: Historic data for all SimulationAgentDict up until the most recent time step.
    traffic_lights_states: Optional[List[TrafficLightStatesDict]] = None #: Historic data for all TrafficLightStatesDict up until the most recent time step.

    location: str #: Location name in IAI format.
    rendering_center: Optional[Tuple[float, float]] = None #: Please refer to the documentation of :func:`location_info` for information on this parameter.
    rendering_fov: Optional[int] = None #: Please refer to the documentation of :func:`location_info` for information on this parameter.

    lights_random_seed: Optional[int] = None #: Controls the stochastic aspects of the the traffic lights states.
    initialize_random_seed: Optional[int] = None #: Please refer to the documentation of :func:`initialize` for information on the random_seed parameter.
    drive_random_seed: Optional[int] = None #: Please refer to the documentation of :func:`drive` for information on the random_seed parameter.

    initialize_model_version: Optional[str] = "best" #: Please refer to the documentation of :func:`initialize` for information on the api_model_version parameter.
    drive_model_version: Optional[str] = "best" #: Please refer to the documentation of :func:`drive` for information on the api_model_version parameter.
    
    light_recurrent_states: Optional[LightRecurrentStates] = None #: As of the most recent time step. Please refer to the documentation of :func:`drive` for further information on this parameter.
    recurrent_states: Optional[List[RecurrentState]] = None #: As of the most recent time step. Please refer to the documentation of :func:`drive` for further information on this parameter.

    def get_agent_states(self, timestep: Optional[int] = None) -> List[AgentState]:
        """Agent states at a specific timestep. If None, returns the latest."""
        if timestep is None:
            timestep = -1
        return [data.state for data in self.agent_data[timestep].values()]

    def get_agent_properties(self, timestep: Optional[int] = None) -> List[AgentProperties]:
        """Agent properties at a specific timestep. If None, returns the latest."""
        if timestep is None:
            timestep = -1
        return [data.properties for data in self.agent_data[timestep].values()]

    def get_agent_ids(self, timestep: Optional[int] = None) -> List[str]:
        """Agent IDs at a specific timestep. If None, returns the latest."""
        if timestep is None:
            timestep = -1
        return list(self.agent_data[timestep].keys())

    def get_agents(self, timestep: Optional[int] = None) -> SimulationAgentDict:
        """Agent dict at a specific timestep. If None, returns the latest."""
        if timestep is None:
            timestep = -1
        return self.agent_data[timestep]

    def get_agent_ids_all(self) -> List[str]:
        """All unique agent IDs across all timesteps, in order of first appearance."""
        keys = []
        seen = set()
        for snap in self.agent_data:
            for k in snap.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        return keys

    def get_agent_properties_all(self) -> Dict[str, AgentProperties]:
        """Map of agent ID -> latest properties for all agents that ever appeared."""
        props = {}
        for snap in self.agent_data:
            for key, data in snap.items():
                props[key] = data.properties
        return props

    @model_validator(mode='after')
    def validate_agent_data(self):
        for t, agent_dict in enumerate(self.agent_data):
            self._validate_agent_dict(agent_dict, timestep=t)
        return self

    @staticmethod
    def _validate_agent_dict(agent_dict: SimulationAgentDict, timestep: Optional[int] = None):
        """Validate that every agent in the dict has a non-None state."""
        ts_label = f" at timestep {timestep}" if timestep is not None else ""
        for aid, data in agent_dict.items():
            assert data.state is not None, f"Agent '{aid}'{ts_label} has no state."

    def add_time_step_data(self, agent_dict: SimulationAgentDict):
        """Append a deep-copied agent_dict entry"""
        self._validate_agent_dict(agent_dict)
        self.agent_data.append(deepcopy(agent_dict))
        
class ScenarioLogLegacy(BaseModel):
    """
    Deprecated, do not use. Please use ScenarioLog instead.

    A log containing simulation information for storage, replay, or an initial state from which a simulation 
    can be continued. Some data fields contain data for all historic time steps while others contain information
    for the most recent time step to be used to continue a simulation.
    """

    agent_states: List[List[AgentState]] #: Historic data for all agents states up until the most recent time step.
    agent_properties: List[AgentProperties] #: Agent properties data for all agents in this scenario/log.
    traffic_lights_states: Optional[List[TrafficLightStatesDict]] = None #: Historic data for all TrafficLightStatesDict up until the most recent time step.

    location: str #: Location name in IAI format.
    rendering_center: Optional[Tuple[float, float]] = None #: Please refer to the documentation of :func:`location_info` for information on this parameter.
    rendering_fov: Optional[int] = None #: Please refer to the documentation of :func:`location_info` for information on this parameter.

    lights_random_seed: Optional[int] = None #: Controls the stochastic aspects of the the traffic lights states.
    initialize_random_seed: Optional[int] = None #: Please refer to the documentation of :func:`initialize` for information on the random_seed parameter.
    drive_random_seed: Optional[int] = None #: Please refer to the documentation of :func:`drive` for information on the random_seed parameter.

    initialize_model_version: Optional[str] = "best" #: Please refer to the documentation of :func:`initialize` for information on the api_model_version parameter.
    drive_model_version: Optional[str] = "best" #: Please refer to the documentation of :func:`drive` for information on the api_model_version parameter.
    
    light_recurrent_states: Optional[LightRecurrentStates] = None #: As of the most recent time step. Please refer to the documentation of :func:`drive` for further information on this parameter.
    recurrent_states: Optional[List[RecurrentState]] = None #: As of the most recent time step. Please refer to the documentation of :func:`drive` for further information on this parameter.

    waypoints_per_frame: Optional[List[WaypointsDict]] = None # As of the most recent time step. A list of waypoints keyed to agent ID's not including waypoints already passed. These waypoints are not automatically populated into the agent properties.
    present_indexes: List[List[int]] = None #: List of indexes corresponding to agent_properties for which agents are present at each time step. If None, all agents are present at every time step.

    @model_validator(mode='after')
    def validate_states_and_present_indexes_init(self):
        if self.present_indexes is not None:
            assert len(self.agent_states) == len(self.present_indexes), "Given different number of time steps for agent states and present indexes."

            for states, pres_ids in zip(self.agent_states,self.present_indexes):
                self.validate_states_and_present_indexes_time_step(
                    current_agent_states=states,
                    current_present_indexes=pres_ids
                )
        return self

    def validate_states_and_present_indexes_time_step(
        self,
        current_agent_states: List[AgentState],
        current_present_indexes: List[int]
    ):
        assert min(current_present_indexes) >= 0, "Invalid agent ID's in given list of present indexes."
        assert len(current_present_indexes) == len(current_agent_states), "Given number of agent states does not match number of present agents."

    def add_time_step_data(
        self,
        current_agent_states: List[AgentState],
        current_present_indexes: List[int]
    ):
        self.validate_states_and_present_indexes_time_step(
            current_agent_states=current_agent_states,
            current_present_indexes=current_present_indexes
        )
        self.present_indexes.append(current_present_indexes)
        self.agent_states.append(current_agent_states)

    def to_scenario_log(self):
        """Convert this deprecated ScenarioLogLegacy into a ScenarioLog"""
        agent_data = []
        for t, (states, present) in enumerate(zip(self.agent_states, self.present_indexes)):
            agent_dict: SimulationAgentDict = {}
            for pos, idx in enumerate(present):
                props = self.agent_properties[idx]
                if self.waypoints_per_frame is not None and t < len(self.waypoints_per_frame):
                    frame_wp = self.waypoints_per_frame[t]
                    if frame_wp and str(idx) in frame_wp:
                        props = deepcopy(props)
                        props.waypoints = frame_wp[str(idx)]
                agent_dict[str(idx)] = AgentData(
                    state=states[pos],
                    properties=props,
                    recurrent=None,
                )
            agent_data.append(agent_dict)

        return ScenarioLog(
            agent_data=agent_data,
            traffic_lights_states=self.traffic_lights_states,
            location=self.location,
            rendering_center=self.rendering_center,
            rendering_fov=self.rendering_fov,
            lights_random_seed=self.lights_random_seed,
            initialize_random_seed=self.initialize_random_seed,
            drive_random_seed=self.drive_random_seed,
            initialize_model_version=self.initialize_model_version,
            drive_model_version=self.drive_model_version,
            light_recurrent_states=self.light_recurrent_states,
            recurrent_states=self.recurrent_states,
        )

    @classmethod
    def from_scenario_log(
        cls, 
        scenario_log: ScenarioLog
    ):
        """Convert a ScenarioLog into the deprecated ScenarioLogLegacy format"""
        all_keys = scenario_log.get_agent_ids_all()
        id_to_idx = {aid: i for i, aid in enumerate(all_keys)}
        props_map = scenario_log.get_agent_properties_all()

        agent_properties_list = [props_map[k] for k in all_keys]
        agent_states_list = []
        present_indexes_list = []
        waypoints_per_frame_list = []

        for agent_dict in scenario_log.agent_data:
            states = []
            present = []
            wp_dict = {}
            for aid, data in agent_dict.items():
                idx = id_to_idx[aid]
                present.append(idx)
                states.append(data.state)
                if data.properties and data.properties.waypoints:
                    wp_dict[str(idx)] = data.properties.waypoints
            present_indexes_list.append(present)
            agent_states_list.append(states)
            waypoints_per_frame_list.append(wp_dict)

        return cls(
            agent_states=agent_states_list,
            agent_properties=agent_properties_list,
            traffic_lights_states=scenario_log.traffic_lights_states,
            location=scenario_log.location,
            rendering_center=scenario_log.rendering_center,
            rendering_fov=scenario_log.rendering_fov,
            lights_random_seed=scenario_log.lights_random_seed,
            initialize_random_seed=scenario_log.initialize_random_seed,
            drive_random_seed=scenario_log.drive_random_seed,
            initialize_model_version=scenario_log.initialize_model_version,
            drive_model_version=scenario_log.drive_model_version,
            light_recurrent_states=scenario_log.light_recurrent_states,
            recurrent_states=scenario_log.recurrent_states,
            waypoints_per_frame=waypoints_per_frame_list if any(wp_dict for wp_dict in waypoints_per_frame_list) else None,
            present_indexes=present_indexes_list,
        )

class LogBase():
    """
    A class for containing features relevant to both log reading and writing such as visualization.
    """

    def __init__(self):
        self._scenario_log = None
        self.simulation_length = None

    @classmethod
    def scenario_log_from_debug_log(
        cls,
        debug_log_path
    ):
        with open(debug_log_path) as f:
            DEBUG_LOG_DATA = json.load(f)

        last_init_res = json.loads(DEBUG_LOG_DATA["initialize_responses"][-1])
        last_init_req = json.loads(DEBUG_LOG_DATA["initialize_requests"][-1])
        last_drive_req = json.loads(DEBUG_LOG_DATA["drive_requests"][-1])
        all_drive_responses = []
        all_agent_states = []
        all_agent_properties = []
        all_traffic_lights_states = []
        for res_, req_ in zip(DEBUG_LOG_DATA["drive_responses"], DEBUG_LOG_DATA["drive_requests"]):
            res = json.loads(res_)
            req = json.loads(req_)
            all_drive_responses.append(res)
            all_agent_states.append([AgentState.fromlist(state) for state in res["agent_states"]])
            all_agent_properties.append([AgentProperties.deserialize(prop) for prop in req["agent_properties"]])
            if res["traffic_lights_states"] is not None:
                all_traffic_lights_states.append(res["traffic_lights_states"])
            else:
                all_traffic_lights_states = None

        log_location = last_init_req["location"]

        if len(DEBUG_LOG_DATA["location_responses"]) > 0:
            loc_res = json.loads(DEBUG_LOG_DATA["location_responses"][-1])
            rendering_center = loc_res["map_center"]
            rendering_fov=loc_res["map_fov"]
        else:
            location_info_response = location_info(
                location=log_location,
            )
            rendering_center = [
                location_info_response.map_center.x,
                location_info_response.map_center.y
            ]
            rendering_fov=location_info_response.map_fov

        agent_data = []
        for t, (states_list, props_list) in enumerate(zip(all_agent_states, all_agent_properties)):
            agent_dict: SimulationAgentDict = defaultdict(AgentData)
            for i, (state, props) in enumerate(zip(states_list, props_list)):
                agent_dict[str(i)] = AgentData(
                    state=state,
                    properties=props,
                    recurrent=None,
                )
            agent_data.append(agent_dict)

        scenario_log = ScenarioLog(
            agent_data=agent_data,
            traffic_lights_states=all_traffic_lights_states, 
            location=log_location,
            rendering_center=rendering_center,
            rendering_fov=rendering_fov,
            lights_random_seed=last_drive_req["random_seed"],
            initialize_random_seed=last_init_req["random_seed"],
            drive_random_seed=last_drive_req["random_seed"],
            initialize_model_version=last_init_res["model_version"],
            drive_model_version=all_drive_responses[-1]["model_version"],
            light_recurrent_states=all_drive_responses[-1]["light_recurrent_states"],
            recurrent_states=[RecurrentState.fromval(rec_state) for rec_state in all_drive_responses[-1]["recurrent_states"]],
        )

        return scenario_log
    
    @validate_arguments
    def visualize_range(
        self,
        timestep_range: Tuple[int,int],
        gif_path: str,
        fov: int = 200,
        resolution: Tuple[int,int] = (2048,2048),
        dpi: int = 300,
        map_center: Optional[Tuple[float,float]] = None,
        direction_vec: bool = False,
        velocity_vec: bool = False,
        plot_frame_number: bool = True,
        left_hand_coordinates: Optional[bool] = None,
        agent_ids: Optional[List[int]] = None
    ):
        """
        Use the available internal tools to visualize the a specific range of time steps within the log and save it to a given location. If
        an invalid time step range is given, the function will fail. Please refer to ScenePlotter for details on the visualization tool.
        """

        def format_agent_properties(ts, agent_id):
            agent_properties = deepcopy(self._scenario_log.agent_properties[agent_id])
            if self._scenario_log.waypoints_per_frame is not None:
                if self._scenario_log.waypoints_per_frame[ts] is not None:
                    agent_id_str = str(agent_id)
                    if agent_id_str in self._scenario_log.waypoints_per_frame[ts]:
                        agent_properties.waypoints = self._scenario_log.waypoints_per_frame[ts][agent_id_str]

            return agent_properties

        if left_hand_coordinates is None:
            left_hand_coordinates = self._scenario_log.location.split(":")[0] == "carla"

        for timestep in timestep_range:
            assert timestep >= 0 or timestep <= (self.simulation_length - 1), "Visualization time range valid."
        assert timestep_range[1] >= timestep_range[0], "Visualization time range valid."

        location_info_response = location_info(
            location=self._scenario_log.location,
            rendering_fov=fov,
            rendering_center=map_center
        )
        rendered_static_map = location_info_response.birdview_image.decode()
        map_center = tuple([location_info_response.map_center.x, location_info_response.map_center.y]) if map_center is None else map_center
        n_frames = len(self._scenario_log.agent_states)
        traffic_lights_states = [None] * n_frames if self._scenario_log.traffic_lights_states is None else self._scenario_log.traffic_lights_states

        def _build_agents_dict(ts):
            """Build a keyed agent dict from the ScenarioLog's parallel lists at a given timestep."""
            states = self._scenario_log.agent_states[ts]
            present = self._scenario_log.present_indexes[ts] if self._scenario_log.present_indexes is not None else list(range(len(states)))
            props = [format_agent_properties(ts, i) for i in present]
            return agents_from_lists(states, props, agent_ids=[str(i) for i in present])

        scene_plotter = ScenePlotter(
            map_image=rendered_static_map,
            fov=fov,
            xy_offset=map_center,
            static_actors=location_info_response.static_actors,
            resolution=resolution,
            dpi=dpi,
            left_hand_coordinates=left_hand_coordinates
        )
        scene_plotter.initialize_recording(
            agents=_build_agents_dict(0),
            traffic_light_states=traffic_lights_states[timestep_range[0]],
        )

        for ts in range(n_frames):
            lights = traffic_lights_states[ts] if ts < len(traffic_lights_states) else None
            scene_plotter.record_step(
                agents=_build_agents_dict(ts),
                traffic_light_states=lights,
            )

        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        scene_plotter.animate_scene(
            output_name=gif_path,
            start_idx=timestep_range[0],
            end_idx=timestep_range[1],
            ax=ax,
            direction_vec=direction_vec,
            velocity_vec=velocity_vec,
            plot_frame_number=plot_frame_number,
            agent_ids=agent_ids
        )

        plt.close(fig)

    @validate_arguments
    def visualize(
        self,
        gif_path: str,
        fov: int = 200,
        resolution: Tuple[int,int] = (2048,2048),
        dpi: int = 300,
        map_center: Optional[Tuple[float,float]] = None,
        direction_vec: bool = False,
        velocity_vec: bool = False,
        plot_frame_number: bool = True,
        left_hand_coordinates: Optional[bool] = None,
        agent_ids: Optional[List[int]] = None
    ):
        """
        Use the available internal tools to visualize the entire log and save it to a given location. Please refer to ScenePlotter for details on 
        the visualization tool.
        """

        self.visualize_range(
            timestep_range = tuple([0,self.simulation_length-1]),
            gif_path = gif_path,
            fov = fov,
            resolution = resolution,
            dpi = dpi,
            map_center = map_center,
            direction_vec = direction_vec,
            velocity_vec = velocity_vec,
            plot_frame_number = plot_frame_number,
            left_hand_coordinates = left_hand_coordinates,
            agent_ids = agent_ids
        )

    def initialize(self):
        pass

    def drive(self):
        pass

class LogWriterConfig(BaseModel):
    """
    Configuration for logging + exporting simulation runs.
    """
    log_path: str
    location: str
    location_info_response: Optional[LocationResponse] = None


class LogWriter(LogBase):
    """
    A class for conveniently writing a log to a JSON log format. 
    """

    def __init__(self):
        super().__init__()

    @validate_arguments
    def export_to_file(
        self,
        log_path: str,
        scenario_log: Optional[ScenarioLog] = None
    ):  
        """
        Convert the data currently contained within the log into a JSON format and export it to a given file. This function can furthermore be 
        used to export a given scenario log instead of the log contained within the object.
        """

        def _format_waypoints_json(
            wps: List[Point]
        ):
            return {
                "suggestion_strength": 0.8, #Default value
                "states": [{
                    "center": {
                        "x": wp.x,
                        "y": wp.y
                    }
                } for wp in wps]
            }

        if scenario_log is None:
            scenario_log = self._scenario_log

        all_keys = scenario_log.get_agent_ids_all()
        agent_properties = scenario_log.get_agent_properties_all()

        # Build waypoints from agent properties map
        individual_suggestions_dict = {}
        for key, prop in agent_properties.items():
            if prop.waypoints:
                individual_suggestions_dict[key] = _format_waypoints_json(prop.waypoints)
            elif prop.waypoint:
                individual_suggestions_dict[key] = _format_waypoints_json([prop.waypoint])

        num_cars, num_pedestrians = 0, 0
        for prop in agent_properties.values():
            if prop.agent_type == "car":
                num_cars += 1
            elif prop.agent_type == "pedestrian":
                num_pedestrians += 1

        num_controls_light, num_controls_yield, num_controls_stop, num_controls_other = 0, 0, 0, 0
        static_actors_list = location_info(location=scenario_log.location).static_actors
        for actor in static_actors_list:
            if actor.agent_type == "traffic_light":
                num_controls_light += 1
            elif actor.agent_type == "yield_sign":
                num_controls_yield += 1
            elif actor.agent_type == "stop_sign":
                num_controls_stop += 1
            else:
                num_controls_other += 1

        # Build predetermined_agents_dict using agent keys from agent_data
        predetermined_agents_dict = {}
        for key in all_keys:
            prop = agent_properties[key]
            states_dict = {}
            for t, agent_dict in enumerate(scenario_log.agent_data):
                if key in agent_dict:
                    state = agent_dict[key].state
                    states_dict[str(t)] = {
                        "center": {"x": state.center.x, "y": state.center.y},
                        "orientation": state.orientation,
                        "speed": state.speed,
                    }
            predetermined_agents_dict[key] = {
                "entity_type": prop.agent_type,
                "static_attributes": {
                    "length": prop.length,
                    "width": prop.width,
                    "rear_axis_offset": prop.rear_axis_offset,
                },
                "states":states_dict
            }

        predetermined_controls_dict = {}
        if scenario_log.traffic_lights_states is not None:
            for actor in [actor for actor in static_actors_list if actor.agent_type == "traffic_light"]:
                actor_id = actor.actor_id
                states_dict = {}

                for t, tls in enumerate(scenario_log.traffic_lights_states):
                    states_dict[str(t)] = {
                        "center": {"x": actor.center.x, "y": actor.center.y},
                        "orientation": actor.orientation,
                        "speed": 0,
                        "control_state": tls[actor_id]
                    }

                predetermined_controls_dict[actor_id] = {
                    "entity_type": "traffic_light",
                    "static_attributes": {
                        "length": actor.length,
                        "width": actor.width,
                        "rear_axis_offset": 0,
                    },
                    "states":states_dict
                }

        self.output_dict = {
            "location": {
                "identifier": scenario_log.location
            },
            "scenario_length": len(scenario_log.agent_data),
            "num_agents": {
                "car": num_cars,
                "pedestrian": num_pedestrians
            },
            "predetermined_agents": predetermined_agents_dict,
            "num_controls": {
                "traffic_light": num_controls_light,
                "yield_sign": num_controls_yield,
                "stop_sign": num_controls_stop,
                "other": num_controls_other,
            },
            "predetermined_controls": predetermined_controls_dict,
            "individual_suggestions": individual_suggestions_dict,
            "initialize_random_seed": scenario_log.initialize_random_seed,
            "lights_random_seed": scenario_log.lights_random_seed,
            "drive_random_seed": scenario_log.drive_random_seed,
            "drive_model_version": scenario_log.drive_model_version,
            "initialize_model_version": scenario_log.initialize_model_version,
            "birdview_options": {
                "rendering_center": [
                    scenario_log.rendering_center[0],
                    scenario_log.rendering_center[1]
                ],
                "renderingFOV": scenario_log.rendering_fov
            },
            "light_recurrent_states": [] if scenario_log.light_recurrent_states is None else [lrs.tolist() for lrs in scenario_log.light_recurrent_states],
            "rendering_centers": [
                scenario_log.rendering_center[0],
                scenario_log.rendering_center[1]
            ]
        }

        with open(log_path, "w") as outfile:
            json.dump(
                self.output_dict, 
                outfile,
                indent=4
            )
    
    @classmethod
    def export_log_to_file(
        cls, 
        log_path: str,
        scenario_log: ScenarioLog
    ):
        """
        Class function to convert a given log data type into a JSON format and export it to a given file.
        """

        cls.export_to_file(cls,log_path,scenario_log)

    
    @validate_arguments
    def initialize(
        self,
        location: Optional[str] = None,
        location_info_response: Optional[LocationResponse] = None,
        init_response: Optional[InitializeResponse] = None,
        lights_random_seed: Optional[int] = None,
        initialize_random_seed: Optional[int] = None,
        drive_random_seed: Optional[int] = None,
        drive_model_version: Optional[str] = None,
        scenario_log: Optional[Union[ScenarioLogLegacy, ScenarioLog]] = None,
        waypoints: Optional[WaypointsDict] = None,
        agents_dict: Optional[SimulationAgentDict] = None,
        agent_ids: Optional[List[str]] = None,
    ):
        """
        Consume and store all initial information within a ScenarioLog data object. If random seed information is desired to be stored, it
        must be given separately but is not mandatory.

        Preferred: pass agents_dict (SimulationAgentDict) so UUID keys are preserved.
        Fallback:  pass init_response for backwards compatibility — integer string keys
                   ("0", "1", ...) will be assigned automatically.
        Legacy:    pass scenario_log (ScenarioLog) to initialize from an existing log.

        Deprecated parameters (kept for backwards compatibility, use agents_dict instead):
            waypoints: Waypoints are now stored directly in AgentProperties within the agents_dict.
            agent_ids: Agent IDs are now the keys of agents_dict.
        """

        if scenario_log is not None:
            if isinstance(scenario_log, ScenarioLog):
                self._scenario_log = scenario_log
            else:
                self._scenario_log = scenario_log.to_scenario_log()
            self.simulation_length = len(self._scenario_log.agent_data)
            return

        assert location is not None, "No scenario log given, must provide a location argument."
        assert location_info_response is not None, "No scenario log given, must provide a location_info_response argument."
        assert init_response is not None, "No scenario log given, must provide a init_response argument."

        if agents_dict is None:
            # Build from init_response (backwards compatibility)
            agent_properties = init_response.agent_properties
            if type(agent_properties[0]) == AgentAttributes:
                agent_properties = [convert_attributes_to_properties(attr) for attr in agent_properties]

            keys = agent_ids if agent_ids is not None else [str(i) for i in range(len(init_response.agent_states))]
            assert len(keys) == len(init_response.agent_states), (
                f"agent_ids length ({len(keys)}) must match number of agents in init_response ({len(init_response.agent_states)})."
            )
            agent_dict: SimulationAgentDict = {}
            for key, state, prop, rec in zip(keys, init_response.agent_states, agent_properties, init_response.recurrent_states):
                if waypoints is not None and key in waypoints:
                    prop = deepcopy(prop)
                    prop.waypoints = waypoints[key]
                agent_dict[key] = AgentData(state=state, properties=prop, recurrent=rec)
        else:
            agent_dict = agents_dict

        self._scenario_log = ScenarioLog(
            agent_data=[deepcopy(agent_dict)],
            traffic_lights_states=[init_response.traffic_lights_states] if init_response.traffic_lights_states is not None else None,
            location=location,
            rendering_center=[
                location_info_response.map_center.x,
                location_info_response.map_center.y
            ],
            rendering_fov=location_info_response.map_fov,
            lights_random_seed=lights_random_seed,
            initialize_random_seed=initialize_random_seed,
            drive_random_seed=drive_random_seed,
            initialize_model_version=init_response.api_model_version,
            drive_model_version=drive_model_version,
            light_recurrent_states=init_response.light_recurrent_states,
            recurrent_states=init_response.recurrent_states,
        )
        self.simulation_length = 1

    @validate_arguments
    def drive(
        self,
        drive_response: DriveResponse,
        current_present_indexes: Optional[List[int]] = None,
        new_agent_properties: Optional[List[AgentProperties]] = None,
        waypoints: Optional[WaypointsDict] = None, #unused, only for backwards compatibility with older logs that may have waypoints stored separately from agent properties
        agent_properties: Optional[List[AgentProperties]] = None,
        agents_dict: Optional[SimulationAgentDict] = None,
        agent_ids: Optional[List[str]] = None,
    ):
        """
        Consume and store driving response information from a single timestep and append it to the end of the log.

        Preferred: pass agents_dict (SimulationAgentDict) with all agents for this timestep.
        Fallback:  pass drive_response with optional agent_ids and agent_properties to build the dict.
        Legacy:    pass current_present_indexes and new_agent_properties for index-based tracking.

        Deprecated parameters (kept for backwards compatibility, use agents_dict instead):
            waypoints: Waypoints are now stored directly in AgentProperties within the agents_dict.
            agent_ids: Agent IDs are now the keys of agents_dict.
        """

        if agents_dict is not None:
            self._scenario_log.add_time_step_data(agents_dict)
        elif current_present_indexes is not None:
            # Backwards compatibility: index-based agent tracking
            all_props_map = dict(self._scenario_log.get_agent_properties_all())

            if new_agent_properties is not None:
                next_idx = len(all_props_map)
                for j, prop in enumerate(new_agent_properties):
                    all_props_map[str(next_idx + j)] = prop

            recurrent_states = drive_response.recurrent_states or [None] * len(current_present_indexes)
            agent_dict: SimulationAgentDict = {}
            for state_idx, prop_idx in enumerate(current_present_indexes):
                key = str(prop_idx)
                agent_dict[key] = AgentData(
                    state=drive_response.agent_states[state_idx],
                    properties=all_props_map[key],
                    recurrent=recurrent_states[state_idx],
                )
            self._scenario_log.add_time_step_data(agent_dict)
        else:
            # No agents_dict and no current_present_indexes provided:
            # assume agent keys have not changed from the previous timestep.

            # if agent_ids is provided use them, otherwise assume ids are the same from previous timestep
            keys = agent_ids if agent_ids is not None else list(self._scenario_log.get_agents().keys())
            assert len(keys) == len(drive_response.agent_states), (
                f"Number of agent_ids ({len(keys)}) does not match number of agents "
                f"in drive_response ({len(drive_response.agent_states)}). "
                f"If agents were added or removed, provide agents_dict or current_present_indexes."
            )
            properties = agent_properties or [self._scenario_log.get_agents()[k].properties for k in keys]
            recurrent_states = drive_response.recurrent_states or [None] * len(keys)

            agent_dict: SimulationAgentDict = {}
            for key, state, prop, rec in zip(keys, drive_response.agent_states, properties, recurrent_states):
                agent_dict[key] = AgentData(state=state, properties=prop, recurrent=rec)
            self._scenario_log.add_time_step_data(agent_dict)

        if drive_response.traffic_lights_states is not None:
            self._scenario_log.traffic_lights_states.append(drive_response.traffic_lights_states)
        
        waypoints = self._format_waypoints(
            current_present_indexes = current_present_indexes,
            agent_properties = agent_properties,
            waypoints = waypoints
        )
        if self._scenario_log.waypoints_per_frame is not None:
            self._scenario_log.waypoints_per_frame.append(waypoints)
        elif waypoints:
            self._scenario_log.waypoints_per_frame = [waypoints]

        self._scenario_log.drive_model_version = drive_response.api_model_version
        self._scenario_log.light_recurrent_states = drive_response.light_recurrent_states
        self._scenario_log.recurrent_states = drive_response.recurrent_states

        self.simulation_length += 1

    @property
    def current_present_indexes(self):
        """
        Returns the indexes of the agents that are currently present within the simulation.
        """
        all_ids = self._scenario_log.get_agent_ids_all()
        id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
        current_snap = self._scenario_log.agent_data[self.simulation_length - 1]
        return [id_to_idx[k] for k in current_snap.keys() if k in id_to_idx]

    @property
    def all_agent_properties(self):
        """
        Returns all agent properties that have been present in the simulation this log is capturing.
        """

        return list(self._scenario_log.get_agent_properties_all().values())


class LogReader(LogBase):
    """
    A class for conveniently reading in a log file then rendering it and/or plugging it into a simulation. Once the log is read, it is 
    intended to be used in place of calling the API.
    """

    def __init__(
        self,
        log_path: str
    ):
        """
        The initialization of this object must be given the path to a JSON file in the IAI format. Assume that the 0th time step is taken
        from the output of :func:`initialize` and set the time step to the 1st time step whic correlates to the first time step produced 
        by :func:`drive`.
        """

        super().__init__()

        with open(log_path) as f:
            LOG_DATA = json.load(f)

        location = LOG_DATA["location"]["identifier"]

        agent_waypoints = None
        if "individual_suggestions" in LOG_DATA:
            agent_waypoints = {}
            for agent_id, waypoints in LOG_DATA["individual_suggestions"].items():
                agent_waypoints[agent_id] = []
                for pt in waypoints["states"]:
                    data = pt["center"]
                    agent_waypoints[agent_id].append(Point.fromlist([data["x"],data["y"]]))

        # Build agent_data from JSON data, using agent keys
        agent_data: List[SimulationAgentDict] = []
        all_agent_properties: Dict[str, AgentProperties] = {}

        for i in range(LOG_DATA["scenario_length"]):
            agent_dict: SimulationAgentDict = defaultdict(AgentData)

            for agent_id, agent in LOG_DATA["predetermined_agents"].items():
                if not agent_id in all_agent_properties:
                    agent_attributes_json = agent["static_attributes"]
                    agent_properties = AgentProperties()
                    agent_properties.length = agent_attributes_json["length"]
                    agent_properties.width = agent_attributes_json["width"]
                    agent_properties.rear_axis_offset = agent_attributes_json["rear_axis_offset"]
                    agent_properties.agent_type = agent["entity_type"]
                    if agent_waypoints is not None:
                        if agent_id in agent_waypoints:
                            agent_properties.waypoints = agent_waypoints[agent_id]
                    all_agent_properties[agent_id] = agent_properties

                ts_key = str(i)
                if ts_key in agent["states"]:
                    agent_state_data = agent["states"][ts_key]
                    state = AgentState.fromlist([
                        agent_state_data["center"]["x"],
                        agent_state_data["center"]["y"],
                        agent_state_data["orientation"],
                        agent_state_data["speed"],
                    ])
                    agent_dict[agent_id] = AgentData(
                        state=state,
                        properties=all_agent_properties[agent_id],
                        recurrent=None,
                    )

            agent_data.append(agent_dict)

        all_traffic_light_states = []
        for i in range(LOG_DATA["scenario_length"]):
            traffic_light_states_ts = {}
            for actor_id, actor in LOG_DATA["predetermined_controls"].items():
                if actor["entity_type"] == "traffic_light":
                    actor_info_ts = actor["states"][str(i)]
                    traffic_light_states_ts[int(actor_id)] = actor_info_ts["control_state"]
            if traffic_light_states_ts:
                all_traffic_light_states.append(traffic_light_states_ts)

        if not all_traffic_light_states:
            all_traffic_light_states = None

        rendering_center = None
        rendering_fov = None
        if "birdview_options" in LOG_DATA:
            rendering_center = tuple([LOG_DATA["birdview_options"]["rendering_center"][0],LOG_DATA["birdview_options"]["rendering_center"][1]])
            rendering_fov = LOG_DATA["birdview_options"]["renderingFOV"]

        light_recurrent_states = None
        if "light_recurrent_states" in LOG_DATA:
            light_recurrent_states = None if (LOG_DATA["light_recurrent_states"] == [] or LOG_DATA["light_recurrent_states"] is None) else [LightRecurrentState(state=state[0],time_remaining=state[1]) for state in LOG_DATA["light_recurrent_states"]]

        self._scenario_log = ScenarioLog(
            agent_data=agent_data,
            traffic_lights_states=all_traffic_light_states, 
            location=location, 
            rendering_center=rendering_center,
            rendering_fov=rendering_fov,
            lights_random_seed=None if not "lights_random_seed" in LOG_DATA else LOG_DATA["lights_random_seed"],
            initialize_random_seed=None if not "initialize_random_seed" in LOG_DATA else LOG_DATA["initialize_random_seed"],
            drive_random_seed=None if not "drive_random_seed" in LOG_DATA else LOG_DATA["drive_random_seed"],
            initialize_model_version=None if not "initialize_model_version" in LOG_DATA else LOG_DATA["initialize_model_version"],
            drive_model_version=None if not "drive_model_version" in LOG_DATA else LOG_DATA["drive_model_version"],
            light_recurrent_states=light_recurrent_states,
            recurrent_states=None,
        )
        self._scenario_log_original = self._scenario_log

        self.reset_log()

        self.simulation_length = len(agent_data)
        self.initialize_model_version = self._scenario_log.initialize_model_version
        self.drive_model_version = self._scenario_log.drive_model_version
        self.all_waypoints = agent_waypoints
        
        self.location_info_response = location_info(
            location=self._scenario_log.location,
            rendering_fov=self._scenario_log.rendering_fov,
            rendering_center=self._scenario_log.rendering_center,
        )
    @validate_arguments
    def return_scenario_logV2(
        self,
        timestep_range: Optional[Tuple[int,int]] = None
    ) -> ScenarioLog:
        """
        Return the original scenario log. Optionally choose a time range within the log of interest.
        """

        if timestep_range is None:
            return self._scenario_log_original
        else:
            for timestep in timestep_range:
                assert timestep >= 0 or timestep <= (self.simulation_length - 1), "Visualization time range valid."
            assert timestep_range[1] >= timestep_range[0], "Visualization time range valid."

            i, j = timestep_range[0], timestep_range[1]
            returned_log = deepcopy(self._scenario_log_original)
            returned_log.agent_data = returned_log.agent_data[i:j]
            if returned_log.traffic_lights_states is not None:
                returned_log.traffic_lights_states = returned_log.traffic_lights_states[i:j]

            return returned_log
        
    @validate_arguments
    def return_scenario_log(
        self,
        timestep_range: Optional[Tuple[int,int]] = None
    ) -> ScenarioLogLegacy:
        """
        Return the original scenario log as a legacy ScenarioLogLegacy. Optionally choose a time range within the log of interest.
        """

        log_to_convert = self._scenario_log_original
        if timestep_range is not None:
            for timestep in timestep_range:
                assert timestep >= 0 or timestep <= (self.simulation_length - 1), "Visualization time range valid."
            assert timestep_range[1] >= timestep_range[0], "Visualization time range valid."

            i, j = timestep_range[0], timestep_range[1]
            log_to_convert = deepcopy(self._scenario_log_original)
            log_to_convert.agent_data = log_to_convert.agent_data[i:j]
            if log_to_convert.traffic_lights_states is not None:
                log_to_convert.traffic_lights_states = log_to_convert.traffic_lights_states[i:j]

        return ScenarioLogLegacy.from_scenario_log(log_to_convert)

    @validate_arguments
    def _return_state_at_timestep(
        self,
        timestep: int
    ):
        """
        Populate all state data from the given time step into the relevant member variables.
        """

        if timestep >= self.simulation_length:
            return False

        agent_dict = self._scenario_log.agent_data[timestep]
        self.agent_states = [data.state for data in agent_dict.values()]
        self.agent_properties = [data.properties for data in agent_dict.values()]
        self.recurrent_states = None
        self.traffic_lights_states = None if self._scenario_log.traffic_lights_states is None else self._scenario_log.traffic_lights_states[timestep]
        self.light_recurrent_states = self._scenario_log.light_recurrent_states if timestep == (self.simulation_length - 1) else None
        self.present_agent_ids = [str(i) for i in self._scenario_log.present_indexes[timestep]]
        self.agent_properties = [self._scenario_log.agent_properties[i] for i in self._scenario_log.present_indexes[timestep]]

        return True

    @validate_arguments
    def return_last_state(self):
        """
        Read and make available state data from the final time step contained within the log which is useful as a launching point for another simulation.
        """

        return self._return_state_at_timestep(timestep=self.simulation_length-1)

    @validate_arguments
    def initialize(self):
        """
        Read and make available state data from the 0th time step into the relevant state member variables e.g. agent_states.
        """

        is_init_response = self._return_state_at_timestep(timestep=0)
        self.current_timestep = 1

        return is_init_response

    @validate_arguments
    def drive(self):
        """
        Read and make available state data from the current time step into the relevant member variables then increment the current time step so that this 
        function may be called again. If the end of the log has been reached, return False otherwise return True.
        """

        if self.current_timestep >= self.simulation_length:
            return False

        is_drive_response = self._return_state_at_timestep(timestep=self.current_timestep)
        self.current_timestep += 1

        return is_drive_response

    @validate_arguments
    def reset_log(self):
        """
        In the case the log was modified, revert the log to its initial state after being read and clear all state data. Furthermore, change the current 
        time step such that the first :func:`drive` time step can be read.
        """
        
        self._scenario_log = self._scenario_log_original

        self.agent_states = None
        self.agent_properties = None
        self.recurrent_states = None
        self.traffic_lights_states = None
        self.light_recurrent_states = None
        self.present_agent_ids = None
        self.current_timestep = 1

    @property
    def all_agent_properties(self):
        """
        Return a list of agent properties of all agents present in the simulation.
        """

        return self._scenario_log.get_agent_properties()

    
    @property
    def location(self):
        """
        Return the location from the log.
        """

        return self._scenario_log.location
    
    @property
    def log_length(self):
        """
        Return the length of the simulation in time steps captured in this log.
        """

        return len(self._scenario_log.agent_data)

    @property
    def agents(self) -> SimulationAgentDict:
        """
        Return the agent dict for the most recently read timestep.
        """
        ts = max(0, self.current_timestep - 1)
        return self._scenario_log.agent_data[ts]
