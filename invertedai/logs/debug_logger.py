import logging
import json
import os

import invertedai as iai
from invertedai.common import AgentState, AgentProperties, TrafficLightState

from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DebugLogger:
    def __init__(
        self,
        debug_log_path: str
    ):
        self.debug_log_path = debug_log_path
        self._create_directory()

        self.data = defaultdict(list)

        file_name = "iai_log_" + self._get_current_time_human_readable_UTC() + "_UTC.json"
        self.log_path = self.debug_log_path + file_name

    def _get_current_time_human_readable_UTC(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S:%f")

    def _create_directory(self):
        if not os.path.isdir(self.debug_log_path):
            logger.info(f"Debug log directory does not exist: Created new directory at {self.debug_log_path}")
            os.makedirs(self.debug_log_path)

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

        self.write_data_to_log()

    def write_data_to_log(self):
        with open(self.log_path, "w") as outfile:
            json.dump(self.data, outfile)

    @classmethod
    def visualize_log(
        cls,
        log_data: Dict,
        gif_name: str = "./debug_log_visualization.gif",
        fov: int = 100,
        map_center: Tuple[float] = None
    ):
        location_info_response = None
        if "location_info_responses" in log_data:
            if len(log_data["location_info_responses"]) > 0:
                location_info_response = json.loads(log_data["location_info_responses"][-1])

        if location_info_response is None:
            if "initialize_requests" in log_data:
                if len(log_data["initialize_requests"]) > 0:
                    location_info_response = iai.location_info(
                        location = json.loads(log_data["initialize_requests"][-1])["location"],
                        rendering_fov = fov,
                        rendering_center = map_center
                    )
                    map_center = tuple([location_info_response.map_center.x, location_info_response.map_center.y]) if map_center is None else map_center

        if location_info_response is None:
            raise Exception("No location data in the log to be able to visualize the data.")
        rendered_static_map = location_info_response.birdview_image.decode()

        scene_plotter = iai.utils.ScenePlotter(
            map_image=rendered_static_map,
            fov=fov,
            xy_offset=map_center,
            static_actors=location_info_response.static_actors,
            resolution=(2048,2048),
            dpi=300
        )
        scene_plotter.initialize_recording(
            agent_states=[AgentState.fromlist(s) for s in json.loads(log_data["initialize_responses"][-1])["agent_states"]],
            agent_properties=[AgentProperties(length=s["length"],width=s["width"],rear_axis_offset=s["rear_axis_offset"],agent_type=s["agent_type"],waypoint=s["waypoint"],max_speed=s["max_speed"]) for s in json.loads(log_data["initialize_responses"][-1])["agent_properties"]],
            traffic_light_states=json.loads(log_data["initialize_responses"][-1])["traffic_lights_states"]
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
    def read_log_from_path(
        cls,
        debug_log_path: str,
        is_visualize_log: bool = False,
        **kwargs
    ):
        with open(debug_log_path) as json_file:
            log_data = json.load(json_file)

            if is_visualize_log:
                cls.visualize_log(
                    log_data=log_data,
                    **kwargs
                )

            return log_data


    