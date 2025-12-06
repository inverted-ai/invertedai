from pydantic import BaseModel, validate_arguments, model_validator
from typing import List, Optional, Dict, Tuple, Any
from copy import deepcopy

import matplotlib.pyplot as plt
import json

from invertedai import location_info
from invertedai.utils import ScenePlotter
from invertedai.api.location import LocationResponse
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.common import ( 
    AgentAttributes, 
    AgentProperties,
    AgentState, 
    LightRecurrentState,
    LightRecurrentStates,
    Point,
    RecurrentState,
    TrafficLightStatesDict 
)


class ScenarioLog(BaseModel):
    """
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

    waypoints: Optional[Dict[str,List[Point]]] = None #: As of the most recent time step. A list of waypoints keyed to agent ID's not including waypoints already passed. These waypoints are not automatically populated into the agent properties.
    waypoints_per_frame: Optional[List[Dict[int,Point]]] = None # for visualization
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
    

class LogBase():
    """
    A class for containing features relevant to both log reading and writing such as visualization.
    """

    def __init__(self):
        self._scenario_log = None
        self.simulation_length = None

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
        left_hand_coordinates: bool = False,
        agent_ids: Optional[List[int]] = None
    ):
        """
        Use the available internal tools to visualize the a specific range of time steps within the log and save it to a given location. If
        an invalid time step range is given, the function will fail. Please refer to ScenePlotter for details on the visualization tool.
        """

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
        traffic_lights_states = [None]*len(self._scenario_log.agent_states) if self._scenario_log.traffic_lights_states is None else self._scenario_log.traffic_lights_states
        
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
            agent_states=self._scenario_log.agent_states[0],
            agent_properties=[self._scenario_log.agent_properties[i] for i in self._scenario_log.present_indexes[0]],
            traffic_light_states=traffic_lights_states[timestep_range[0]],
            waypoints_per_frame = self._scenario_log.waypoints_per_frame
        )

        for states, lights, present in zip(self._scenario_log.agent_states[0:],traffic_lights_states[0:],self._scenario_log.present_indexes[0:]):
            scene_plotter.record_step(
                agent_states=states, 
                traffic_light_states=lights,
                agent_properties=[self._scenario_log.agent_properties[i] for i in present]
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
            numbers=agent_ids
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
        left_hand_coordinates: bool = False,
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

        individual_suggestions_dict = {}
        if scenario_log is None:
            scenario_log = self._scenario_log
            for i, prop in enumerate(scenario_log.agent_properties):
                wp = prop.waypoint
                if wp is not None:
                    individual_suggestions_dict[str(i)] = {
                        "suggestion_strength": 0.8, #Default value
                        "states":[{
                            "center": {
                                "x": wp.x,
                                "y": wp.y
                            }
                        }]
                    }
        else:
            if scenario_log.waypoints is not None:
                for agent_id, wps in scenario_log.waypoints.items():
                        individual_suggestions_dict[agent_id] = {
                            "suggestion_strength": 0.8, #Default value
                            "states": [{
                                "center": {
                                    "x": wp.x,
                                    "y": wp.y
                                }
                            } for wp in wps]
                        }

        num_cars, num_pedestrians = 0, 0
        for prop in scenario_log.agent_properties:
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

        predetermined_agents_dict = {}
        for i, prop in enumerate(scenario_log.agent_properties):
            states_dict = {}
            for t, states in enumerate(scenario_log.agent_states):
                if i in scenario_log.present_indexes[t]:
                    ind = scenario_log.present_indexes[t].index(i)

                    states_dict[str(t)] = {
                        "center": {"x": states[ind].center.x, "y": states[ind].center.y},
                        "orientation": states[ind].orientation,
                        "speed": states[ind].speed
                    }

            predetermined_agents_dict[str(i)] = {
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
            "scenario_length": len(scenario_log.agent_states),
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
        scenario_log: Optional[ScenarioLog] = None
    ): 
        """
        Consume and store all initial information within a ScenarioLog data object. If random seed information is desired to be stored, it 
        must be given separately but is not mandatory.
        """

        if scenario_log is None:
            assert location is not None, "No scenario log given, must provide a location argument."
            assert location_info_response is not None, "No scenario log given, must provide a location_info_response argument."
            assert init_response is not None, "No scenario log given, must provide a init_response argument."

            agent_properties = init_response.agent_properties
            if type(agent_properties[0]) == AgentAttributes:
                agent_properties = [convert_attributes_to_properties(attr) for attr in agent_properties]

            self._scenario_log = ScenarioLog(
                agent_states=[init_response.agent_states], 
                agent_properties=agent_properties, 
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
                waypoints=None,
                present_indexes=[list(range(len(agent_properties)))]
            )
            self.simulation_length = 1

        else:
            self._scenario_log = scenario_log
            if self._scenario_log.present_indexes is None:
                self._scenario_log.present_indexes = [list(range(len(self._scenario_log.agent_properties)))]

            self.simulation_length = len(self._scenario_log.agent_states)

    @validate_arguments
    def drive(
        self,
        drive_response: DriveResponse,
        current_present_indexes: Optional[List[int]] = None,
        new_agent_properties: Optional[List[AgentProperties]] = None,
        waypoints: Optional[Dict[int, Optional[Point]]] = None
    ): 
        """
        Consume and store driving response information from a single timestep and append it to the end of the log. If the number of agents
        changes during this time step, a new list of present agent ID's must be given indicating which agents are now present. If agents have been 
        added, their AgentProperties must be given as well and will be added in the given order. If no present indexes list is given, it is assumed
        which agents are present has not changed since the previous time step.
        """

        if new_agent_properties is not None:
            self._scenario_log.agent_properties.extend(new_agent_properties)

        if current_present_indexes is None:
            current_present_indexes = deepcopy(self._scenario_log.present_indexes[self.simulation_length-1])
        self._scenario_log.add_time_step_data(
            current_agent_states=drive_response.agent_states,
            current_present_indexes=current_present_indexes
        )

        if drive_response.traffic_lights_states is not None:
            self._scenario_log.traffic_lights_states.append(drive_response.traffic_lights_states)
        if waypoints is not None:
            if self._scenario_log.waypoints_per_frame is None:
                self._scenario_log.waypoints_per_frame = []
            cleaned_waypoints = {aid: wp for aid, wp in waypoints.items() if wp is not None}
            if self._scenario_log.waypoints_per_frame is None:
                self._scenario_log.waypoints_per_frame = []
            self._scenario_log.waypoints_per_frame.append(cleaned_waypoints)
        self._scenario_log.drive_model_version = drive_response.api_model_version
        self._scenario_log.light_recurrent_states = drive_response.light_recurrent_states
        self._scenario_log.recurrent_states = drive_response.recurrent_states

        self.simulation_length += 1

    @property
    def current_present_indexes(self): 
        """
        Returns the indexes of the agents that are currently present within the simulation.
        """

        return self._scenario_log.present_indexes[self.simulation_length-1]

    @property
    def all_agent_properties(self):
        """
        Returns all agent properties that have been present in the simulation this log is capturing.
        """

        return self._scenario_log.agent_properties


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

        all_agent_states_unsorted = []
        all_agent_properties_unsorted = {}
        present_indexes_unsorted = []
        agent_id_list = {}
        agent_id_sequence_num = 0
        for i in range(LOG_DATA["scenario_length"]):
            agent_states_ts = {}
            present_indexes_ts = []
            
            for agent_id, agent in LOG_DATA["predetermined_agents"].items():
                if not agent_id in agent_id_list:
                    agent_attributes_json = agent["static_attributes"]
                    agent_properties = AgentProperties()
                    agent_properties.length = agent_attributes_json["length"]
                    agent_properties.width = agent_attributes_json["width"]
                    agent_properties.rear_axis_offset = agent_attributes_json["rear_axis_offset"]
                    agent_properties.agent_type = agent["entity_type"]
                    all_agent_properties_unsorted[agent_id] = agent_properties
                    agent_id_list[agent_id] = agent_id_sequence_num
                    agent_id_sequence_num += 1

                ts_key = str(i)
                if ts_key in agent["states"]:
                    present_indexes_ts.append(agent_id_list[agent_id])
                    agent_state = agent["states"][ts_key]
                    agent_states_ts[agent_id] = AgentState.fromlist([
                        agent_state["center"]["x"],
                        agent_state["center"]["y"],
                        agent_state["orientation"],
                        agent_state["speed"],
                    ])

            all_agent_states_unsorted.append(agent_states_ts)
            present_indexes_unsorted.append(present_indexes_ts)

        #Sort agents by index if not in the correct order from the JSON dict
        all_agent_properties = self._sort_unsorted_dict(
            unsorted_dict=all_agent_properties_unsorted,
            index_key=agent_id_list
        )
        all_agent_states = []
        for agent_states_ts in all_agent_states_unsorted:
            all_agent_states.append(self._sort_unsorted_dict(
                unsorted_dict=agent_states_ts,
                index_key=agent_id_list
            ))
        log_present_indexes = []
        for present_indexes_ts in present_indexes_unsorted:
            log_present_indexes.append(sorted(present_indexes_ts))

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

        agent_waypoints = {}
        for agent_id, waypoints in LOG_DATA["individual_suggestions"].items():
            agent_waypoints[agent_id] = []
            for pt in waypoints["states"]:
                data = pt["center"]
                agent_waypoints[agent_id].append(Point.fromlist([data["x"],data["y"]]))
        if not agent_waypoints:
            agent_waypoints = None

        self._scenario_log = ScenarioLog(
            agent_states=all_agent_states, 
            agent_properties=all_agent_properties, 
            traffic_lights_states=all_traffic_light_states, 
            location=location, 
            rendering_center=tuple([LOG_DATA["birdview_options"]["rendering_center"][0],LOG_DATA["birdview_options"]["rendering_center"][1]]),
            rendering_fov=LOG_DATA["birdview_options"]["renderingFOV"],
            lights_random_seed=None if not "lights_random_seed" in LOG_DATA else LOG_DATA["lights_random_seed"],
            initialize_random_seed=None if not "initialize_random_seed" in LOG_DATA else LOG_DATA["initialize_random_seed"],
            drive_random_seed=LOG_DATA["drive_random_seed"],
            initialize_model_version=None if not "initialize_model_version" in LOG_DATA else LOG_DATA["initialize_model_version"],
            drive_model_version=LOG_DATA["drive_model_version"],
            light_recurrent_states=None if (LOG_DATA["light_recurrent_states"] is [] or LOG_DATA["light_recurrent_states"] is None) else [LightRecurrentState(state=state[0],time_remaining=state[1]) for state in LOG_DATA["light_recurrent_states"]],
            recurrent_states=None,
            waypoints=agent_waypoints,
            present_indexes=log_present_indexes
        )
        self._scenario_log_original = self._scenario_log

        self.reset_log()

        self.simulation_length = len(all_agent_states)
        self.initialize_model_version = self._scenario_log.initialize_model_version
        self.drive_model_version = self._scenario_log.drive_model_version
        self.all_waypoints = agent_waypoints
        
        self.location_info_response = location_info(
            location=self._scenario_log.location,
            rendering_fov=self._scenario_log.rendering_fov,
            rendering_center=self._scenario_log.rendering_center,
        )

    def _sort_unsorted_dict(
        self,
        unsorted_dict: Dict[str,Any],
        index_key: Dict[str,int]
    ):
        sorted_list = []
        present_indexes = {index_key[k]: k for k in list(unsorted_dict.keys())}
        for agent_id in sorted(present_indexes.keys()):
            sorted_list.append(unsorted_dict[present_indexes[agent_id]])

        return sorted_list

    @validate_arguments
    def return_scenario_log(
        self,
        timestep_range: Optional[Tuple[int,int]] = None
    ):
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
            returned_log.agent_states = returned_log.agent_states[i:j]
            returned_log.present_indexes = returned_log.present_indexes[i:j]
            if returned_log.traffic_lights_states is not None:
                returned_log.traffic_lights_states = returned_log.traffic_lights_states[i:j]

            return returned_log

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

        self.agent_states = self._scenario_log.agent_states[timestep]
        self.recurrent_states = None
        self.traffic_lights_states = None if self._scenario_log.traffic_lights_states is None else self._scenario_log.traffic_lights_states[timestep]
        self.light_recurrent_states = self._scenario_log.light_recurrent_states if timestep == (self.simulation_length - 1) else None
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
        self.current_timestep = 1

    @property
    def all_agent_properties(self):
        """
        Return a list of agent properties of all agents present in the simulation.
        """

        return self._scenario_log.agent_properties

    @property
    def waypoint_dictionary(self):
        """
        Return all waypoints in the simulation keyed to the index of agents corresponding to the full agent properties list.
        """

        return self._scenario_log.waypoints
    
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

        return len(self._scenario_log.agent_states)
