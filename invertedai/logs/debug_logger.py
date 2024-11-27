from pydantic import BaseModel, validate_arguments
from typing import List, Optional, Dict, Tuple
from datetime import datetime

import json

class DebugLogger:

    def __init__(self,debug_log_path):
        self.debug_log_path = debug_log_path

        self.data = {
            "location_info_requests":[],
            "location_info_responses":[],
            "location_info_request_timestamps":[],
            "location_info_response_timestamps":[],
            "initialize_requests":[],
            "initialize_responses":[],
            "initialize_request_timestamps":[],
            "initialize_response_timestamps":[],
            "drive_requests":[],
            "drive_responses":[],
            "drive_request_timestamps":[],
            "drive_response_timestamps":[],
        }

        file_name = "iai_log_" + self._get_current_time_human_readable_UTC() + "_UTC.json"
        self.log_path = self.debug_log_path + file_name

    def _create_directory(self):
        if not os.path.isdir(self.debug_log_path):
            os.makedirs(self.debug_log_path)
            print(f"Debug log directory does not exist: Created new directory at {self.debug_log_path}")
        print(f"Directory already exists!")

    def append_request(self,model,data_dict):
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

    def append_response(self,model,data_dict):
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

    def _get_current_time_human_readable_UTC(self):
        return datetime.now().strftime("%d-%m-%Y_%H:%M:%S:%f")

    def write_data_to_log(self):
        with open(self.log_path, "w") as outfile:
            json.dump(self.data, outfile)