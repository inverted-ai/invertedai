import logging
import json
import os

import invertedai as iai
from invertedai.common import AgentState, AgentProperties, TrafficLightState, RecurrentState, LightRecurrentState, Image, StaticMapActor, Point
from invertedai.api.location import LocationResponse

from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DebugLogger:
    def __new__(cls, *args, **kwargs):
        if not cls.check_instance_exists(cls):
            cls.instance = super(DebugLogger, cls).__new__(cls)
        return cls.instance

    def check_instance_exists(cls):
        return hasattr(cls, 'instance')

    def __init__(
        self,
        debug_dir_path: Optional[str] = None
    ):
        if debug_dir_path is not None:
            self.debug_dir_path = debug_dir_path
            self._create_directory()

            self.data = defaultdict(list)

            self.init_time = datetime.timestamp(datetime.now())
            file_name = "iai_log_" + self._get_current_time_human_readable_UTC() + "_UTC.json"
            self.debug_log_path = os.path.join(self.debug_dir_path,file_name)

    def reinitialize_logger(self):
        self.__init__(debug_dir_path = self.debug_dir_path)

    def _get_current_time_human_readable_UTC(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S:%f")

    def _create_directory(self):
        if not os.path.isdir(self.debug_dir_path):
            logger.info(f"Debug log directory does not exist: Created new directory at {self.debug_dir_path}")
            os.makedirs(self.debug_dir_path)

    def append_request(
        self,
        model: str,
        data_dict: Optional[dict] = None
    ):
        ts = self._get_current_time_human_readable_UTC()
        data_str = json.dumps(data_dict)

        if model == "location_info":
            self.data["location_info_requests"].append(data_str)
            self.data["location_info_request_timestamps"].append(ts)

        elif model == "initialize":
            self.data["initialize_requests"].append(data_str)
            self.data["initialize_request_timestamps"].append(ts)

        elif model == "drive":
            self.data["drive_requests"].append(data_str)
            self.data["drive_request_timestamps"].append(ts)

        elif model == "large_initialize":
            self.data["large_initialize_requests"].append(data_str)
            self.data["large_initialize_request_timestamps"].append(ts)

        elif model == "large_drive":
            self.data["large_drive_requests"].append(data_str)
            self.data["large_drive_request_timestamps"].append(ts)

        self.write_data_to_log()

    def append_response(
        self,
        model: str,
        data_dict: Optional[dict] = None
    ):
        ts = self._get_current_time_human_readable_UTC()
        data_str = json.dumps(data_dict)

        if model == "location_info":
            self.data["location_info_responses"].append(data_str)
            self.data["location_info_response_timestamps"].append(ts)

        elif model == "initialize":
            self.data["initialize_responses"].append(data_str)
            self.data["initialize_response_timestamps"].append(ts)

        elif model == "drive":
            self.data["drive_responses"].append(data_str)
            self.data["drive_response_timestamps"].append(ts)

        elif model == "large_initialize":
            self.data["large_initialize_responses"].append(data_str)
            self.data["large_initialize_response_timestamps"].append(ts)

        elif model == "large_drive":
            self.data["large_drive_responses"].append(data_str)
            self.data["large_drive_response_timestamps"].append(ts)

        self.write_data_to_log()

    def write_data_to_log(self):
        with open(self.debug_log_path, "w") as outfile:
            json.dump(self.data, outfile)

    def _get_scene_plotter(
        self,
        log_data: Dict,
        fov: int = 100,
        map_center: Tuple[float,float] = None
    ):
        location_info_response = None
        location = None
        if "location_info_responses" in log_data:
            if len(log_data["location_info_responses"]) > 0:
                location = json.loads(log_data["location_info_requests"][-1])["location"]
                lir = json.loads(log_data["location_info_responses"][-1])
                location_info_response = LocationResponse(
                    version=lir["version"],
                    max_agent_number=lir["max_agent_number"],
                    map_fov=lir["map_fov"],
                    bounding_polygon=[Point.fromlist(p) for p in lir["bounding_polygon"]],
                    map_center=Point.fromlist(lir["map_center"]),
                    static_actors=[StaticMapActor.fromdict(sa) for sa in lir["static_actors"]],
                    birdview_image=Image(encoded_image=lir["birdview_image"]),
                    osm_map=None
                )

        if location_info_response is None:
            if "initialize_requests" in log_data:
                if len(log_data["initialize_requests"]) > 0:
                    location = json.loads(log_data["initialize_requests"][-1])["location"]
                    location_info_response = iai.location_info(
                        location = location,
                        rendering_fov = fov,
                        rendering_center = map_center
                    )
                    map_center = tuple([location_info_response.map_center.x, location_info_response.map_center.y]) if map_center is None else map_center

        if location_info_response is None:
            raise Exception("No location data in the log to be able to visualize the data.")
        if len(log_data["initialize_responses"]) <= 0:
            raise Exception("No initialize responses to visualize.")
        rendered_static_map = location_info_response.birdview_image.decode()

        all_properties = [AgentProperties(
            length=s["length"],
            width=s["width"],
            rear_axis_offset=s["rear_axis_offset"],
            agent_type=s["agent_type"],
            waypoint=s["waypoint"],
            max_speed=s["max_speed"]) for s in json.loads(log_data["initialize_responses"][-1])["agent_properties"]
        ]
        agent_states = [AgentState.fromlist(s) for s in json.loads(log_data["initialize_responses"][-1])["agent_states"]]
        recurrent_states = [RecurrentState.fromval(s) for s in json.loads(log_data["initialize_responses"][-1])["recurrent_states"]]
        traffic_light_states = json.loads(log_data["initialize_responses"][-1])["traffic_lights_states"]
        lrs = json.loads(log_data["initialize_responses"][-1])["light_recurrent_states"]
        light_recurrent_states = [LightRecurrentState(
            state=s[0], 
            time_remaining=s[1]) for s in lrs
        ] if lrs is not None else lrs
        response_data = {
            "location": location,
            "agent_properties": all_properties,
            "agent_states": agent_states,
            "recurrent_states": recurrent_states,
            "traffic_light_states": traffic_light_states,
            "light_recurrent_states": light_recurrent_states
        }

        scene_plotter = iai.utils.ScenePlotter(
            map_image=rendered_static_map,
            fov=fov,
            xy_offset=map_center,
            static_actors=location_info_response.static_actors,
            resolution=(2048,2048),
            dpi=300
        )
        scene_plotter.initialize_recording(
            agent_states=agent_states,
            agent_properties=all_properties,
            traffic_light_states=traffic_light_states
        )

        return scene_plotter, response_data

    @classmethod
    def visualize_log(
        cls,
        log_data: Dict,
        gif_name: str = "./debug_log_visualization.gif",
        fov: int = 100,
        map_center: Tuple[float,float] = None
    ):
        scene_plotter, _ = cls._get_scene_plotter(
            cls,
            log_data=log_data,
            fov=fov,
            map_center=map_center
        )

        for response_json in log_data["drive_responses"]:
            response = json.loads(response_json)
            scene_plotter.record_step([AgentState.fromlist(s) for s in response["agent_states"]],response["traffic_lights_states"])

        # save the visualization to disk
        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        scene_plotter.animate_scene(
            output_name=gif_name,
            ax=ax,
            direction_vec=False,
            velocity_vec=False,
            plot_frame_number=True,
        )
        plt.close(fig)

    @classmethod
    def reproduce_log(
        cls,
        log_data: Dict,
        gif_name: str = "./debug_log_reproduction.gif",
        fov: int = 100,
        map_center: Tuple[float,float] = None,
        use_log_seed: bool = True
    ):
        scene_plotter, response_data = cls._get_scene_plotter(
            cls,
            log_data=log_data,
            fov=fov,
            map_center=map_center
        )
        agent_states = response_data["agent_states"]
        agent_properties = response_data["agent_properties"]
        recurrent_states = response_data["recurrent_states"]
        traffic_lights_states = response_data["traffic_light_states"]
        light_recurrent_states = response_data["light_recurrent_states"]

        for request_json in log_data["drive_requests"]:
            request = json.loads(request_json)
            response = iai.large_drive(
                location = response_data["location"],
                agent_states = agent_states,
                agent_properties = agent_properties,
                recurrent_states = recurrent_states,
                light_recurrent_states = light_recurrent_states,
                random_seed = request["random_seed"] if use_log_seed else None,
                api_model_version = request["model_version"]
            )

            agent_states = response.agent_states
            recurrent_states = response.recurrent_states
            traffic_lights_states = response.traffic_lights_states
            light_recurrent_states = response.light_recurrent_states
            
            scene_plotter.record_step(agent_states,traffic_lights_states)

        # save the visualization to disk
        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        scene_plotter.animate_scene(
            output_name=gif_name,
            ax=ax,
            direction_vec=False,
            velocity_vec=False,
            plot_frame_number=True,
        )
        plt.close(fig)

    @classmethod
    def read_log_from_path(
        cls,
        debug_log_path: str,
        is_visualize_log: bool = False,
        is_reproduce_log: bool = False,
        **kwargs
    ):
        with open(debug_log_path) as json_file:
            log_data = json.load(json_file)

            if is_visualize_log:
                cls.visualize_log(
                    log_data=log_data,
                    **kwargs
                )
            if is_reproduce_log:
                cls.reproduce_log(
                    log_data=log_data,
                    **kwargs
                )

            return log_data


    