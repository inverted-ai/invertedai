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
        # for curr_drive_req, prev_drive_res in zip(self.log_data["drive_requests"][1:],self.log_data["drive_responses"][:-2]):
        #     curr_req = json.loads(curr_drive_req)
        #     prev_res = json.loads(prev_drive_res)
        #     is_equal_agent_states.append(self._check_agent_states_equal(
        #         curr_req["agent_states"],
        #         prev_res["agent_states"]
        #     ))
        #     is_equal_recurrent_states.append(self._check_recurrent_states_equal(
        #         curr_req["recurrent_states"],
        #         prev_res["recurrent_states"]
        #     ))

        num_req_res_pairs = len(self.log_data["drive_requests"])
        curr_index = 0

        req_groupings = []
        res_groupings = []

        while curr_index < num_req_res_pairs - 1:
            req_agent_dict, res_agent_dict, curr_index = self._get_all_agents_from_large_drive(
                num_requests = num_req_res_pairs,
                start_index = curr_index,
                drive_req_list = self.log_data["drive_requests"],
                drive_res_list = self.log_data["drive_responses"]
            )
            req_groupings.append(req_agent_dict)
            res_groupings.append(res_agent_dict)

        for req_dict, res_dict in zip(req_groupings[1:],res_groupings[:-1]):
            is_equal_agent_states.append(self._check_states_equal(
                [v["agent_state"] for k, v in req_dict.items()],
                [v["agent_state"] for k, v in res_dict.items()]
            ))
            is_equal_recurrent_states.append(self._check_states_equal(
                [v["recurrent_state"] for k, v in req_dict.items()],
                [v["recurrent_state"] for k, v in res_dict.items()]
            ))

        return all(is_equal_agent_states) and all(is_equal_recurrent_states)

    def _get_all_agents_from_large_drive(
        self,
        num_requests,
        start_index,
        drive_req_list,
        drive_res_list
    ):
        STATE_SIGDIG = 2
        RECURR_SIGDIG = 6
        is_remaining_unique_agents = True

        curr_index = start_index
        req_agent_prop_dict = {}
        res_agent_state_dict = {}

        while True:
            drive_req_obj = json.loads(drive_req_list[curr_index])
            is_remaining_unique_agents = False

            for i, agent_prop in enumerate(drive_req_obj["agent_properties"]):
                agent_prop_str = json.dumps(agent_prop)

                if not agent_prop_str in req_agent_prop_dict:
                    req_agent_prop_dict[agent_prop_str] = {
                        "agent_state": [round(x,STATE_SIGDIG) for x in drive_req_obj["agent_states"][i]],
                        "recurrent_state": [round(x,RECURR_SIGDIG) for x in drive_req_obj["recurrent_states"][i]]
                    }
                    is_remaining_unique_agents = True

            if not is_remaining_unique_agents:
                break

            curr_index += 1

            if curr_index >= num_requests:
                break

        for index in range(start_index,curr_index):
            drive_res_obj = json.loads(drive_res_list[index])

            for i, agent_state in enumerate(drive_res_obj["agent_states"]):
                agent_state_str = json.dumps([round(x,STATE_SIGDIG) for x in agent_state])

                if not agent_state_str in res_agent_state_dict:
                    res_agent_state_dict[agent_state_str] = {
                        "agent_state": [round(x,STATE_SIGDIG) for x in drive_res_obj["agent_states"][i]],
                        "recurrent_state": [round(x,RECURR_SIGDIG) for x in drive_res_obj["recurrent_states"][i]]
                    }
                else:
                    res_agent_state_dict[agent_state_str]

        return req_agent_prop_dict, res_agent_state_dict, curr_index

    #This function is useful for checking equality between 2 numerical variables ignoring rounding errors
    def _check_equality_with_error(self,a,b,err):
        return abs(a-b) <= err

    # def _check_agent_states_equal(self,states_0,states_1):
    #     if not len(states_0) == len(states_1): return False

    #     for sa_0, sa_1 in zip(states_0,states_1):
    #         for param_0, param_1 in zip(sa_0,sa_1):
    #             if not param_0 == param_1: return False

    #     return True

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

    # def _check_recurrent_states_equal(self,rs_0,rs_1):
    #     for param_0, param_1 in zip(rs_0,rs_1):
    #         if not param_0 == param_1: return False

    #     return True


    