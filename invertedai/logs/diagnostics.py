import os
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')


import invertedai as iai
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt

class DiagnosticTool:
    def __init__(
        self,
        debug_log_path: str
    ):
        self.debug_log_path = debug_log_path
        self.log_data = None
        with open(debug_log_path) as json_file:
            self.log_data = json.load(json_file)

    def full_diagnostic_test(self):
        pass

    def check_drive_response_equivalence(self):
        is_equal_agent_states = []
        is_equal_recurrent_states = []

        req_groupings, res_groupings = self._get_all_drive_agents(log_data = self.log_data)

        for req_dict, res_dict in zip(req_groupings["agent_states"][1:],res_groupings["agent_states"][:-1]):
            is_equal_agent_states.append(self._check_states_equal(req_dict,res_dict))
        for req_dict, res_dict in zip(req_groupings["recurrent_states"][1:],res_groupings["recurrent_states"][:-1]):
            is_equal_recurrent_states.append(self._check_states_equal(req_dict,res_dict))

        return all(is_equal_agent_states) and all(is_equal_recurrent_states)

    def _get_all_drive_agents(
        self,
        log_data
    ):
        STATE_SIGDIG = 2
        RECURR_SIGDIG = 6

        req_agent_state_dict = {"agent_states":[],"recurrent_states":[]}
        res_agent_state_dict = {"agent_states":[],"recurrent_states":[]}

        is_large_simulation = len(log_data["large_drive_responses"]) > 0
        if is_large_simulation:
            drive_req_data = log_data["large_drive_requests"]
            drive_res_data = log_data["large_drive_responses"]
        else:
            drive_req_data = log_data["drive_requests"]
            drive_res_data = log_data["drive_responses"]

        for req_data, res_data in zip(drive_req_data,drive_res_data):
            res_data = json.loads(res_data)
            req_data = json.loads(req_data)

            for (state_dict, data) in zip([req_agent_state_dict,res_agent_state_dict],[req_data,res_data]):
                state_dict["agent_states"].append([[round(x,STATE_SIGDIG) for x in st] for st in data["agent_states"]])
                state_dict["recurrent_states"].append([[round(x,RECURR_SIGDIG) for x in st] for st in data["recurrent_states"]])

        return req_agent_state_dict, res_agent_state_dict

    def _check_individual_states_equal(self,sa_0,sa_1):
        for param_0, param_1 in zip(sa_0,sa_1):
            if not param_0 == param_1: return False

        return True

    def _check_states_equal(self,states_0,states_1):
        if not len(states_0) == len(states_1): return False

        for i, sa_0 in enumerate(states_0):
            is_states_equal = False
            for j, sa_1 in enumerate(states_1):
                is_states_equal = self._check_individual_states_equal(sa_0,sa_1)
                if is_states_equal: break
            if not is_states_equal: return False

        return True

    