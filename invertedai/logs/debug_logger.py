import logging
import json
import os

from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone

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
        data_dict: dict
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
        data_dict: dict
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