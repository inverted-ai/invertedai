import json
import argparse
import invertedai as iai
import matplotlib.pyplot as plt

from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional, Union

DIAGNOSTIC_ISSUE_LIBRARY = {
    10:"Agent state index change",
    11:"Agent state modified or removed before next request",
    20:"Recurrent state index change",
    21:"Recurrent state modified or removed before next request",
    50:"Static agent properties/attributes deviation from expected range",
    51:"Static agent properties/attributes modified or removed between time steps",
    52:"Static agent properties/attributes index change",
    100:"Agents removed before next request",
    101:"Agents added before next request"
}

class DiagnosticMessage(BaseModel):
    timestep: int
    issue_type: int
    agent_list: Optional[List[int]] = None

class DiagnosticTool:
    """
    A user-side tool that examines a debug log checking for common implementation mistakes 
    and provides feedback.

    Parameters
    ----------
    debug_log_path:
        The full path to the debug log file to be loaded and analyzed.
    ego_indexes:
        A list of index IDs for ego vehicles that will be analyzed differently (e.g. it is expected
        that the state of an ego vehicle may be modified external to the API).
    """

    def __init__(
        self,
        debug_log_path: str,
        ego_indexes: List[int] = None
    ):
        self.debug_log_path = debug_log_path
        self.log_data = None
        with open(debug_log_path) as json_file:
            self.log_data = json.load(json_file)
        (
            self.req_groupings, 
            self.res_groupings,
            self.req_agent_details,
            self.init_agent_details
        ) = self._parse_log_data(log_data = self.log_data)
        self.num_timesteps = len(self.res_groupings["agent_states"])

        self.DIAGNOSTIC_ISSUE_LIBRARY = DIAGNOSTIC_ISSUE_LIBRARY

        self.ego_indexes = [] if ego_indexes is None else ego_indexes

    def _format_message(
        self,
        timestep: int,
        issue_type: str,
        agent_list: Optional[List[int]] = None
    ):
        return DiagnosticMessage(
            timestep=timestep,
            issue_type=issue_type,
            agent_list=agent_list
        )

    def full_diagnostic_test(self):
        """
        Main access point for this class. This function runs all available diagnostic tests and prints 
        human-readable feedback for the benefit of a user to fix issues in their integration.
        """

        diagnostic_message_codes = []

        diagnostic_message_codes.extend(self._check_drive_response_equivalence())
        diagnostic_message_codes.extend(self._check_number_of_agents())
        diagnostic_message_codes.extend(self._check_agent_details_realistic())
        diagnostic_message_codes.extend(self._check_agent_details_static())

        diagnostic_message_dict = {}
        for msg in diagnostic_message_codes:
            if msg.timestep in diagnostic_message_dict:
                diagnostic_message_dict[msg.timestep].append(msg)
            else:
                diagnostic_message_dict[msg.timestep] = [msg]

        print(f"===========================================================================================")
        print(f"Printing potential implementation issues that might cause performance degradation:")
        print(f"")

        for ts in range(self.num_timesteps):
            if ts not in diagnostic_message_dict:
                print(f"At timestep {ts}: No issues detected.")
            else:
                for msg in diagnostic_message_dict[ts]:
                    if msg.agent_list is not None:
                        print(f"At timestep {ts}: Issue code {msg.issue_type} applicable to agent IDs {msg.agent_list}.")
                    else:
                        print(f"At timestep {ts}: Issue code {msg.issue_type}.")

        print(f"")
        print(f"===========================================================================================")
        print(f"Diagnostic Issue Legend:")
        print(f"")
        for code, message in self.DIAGNOSTIC_ISSUE_LIBRARY.items():
            print(f"{code}:{message}")
        print(f"")
        print(f"===========================================================================================")
        print(f"")

        print(f"Finished diagnostic analysis.")

    def _check_agent_details_realistic(self):
        diagnostic_message_codes = []

        flagged_agents = []
        for ind, detes in enumerate(self.init_agent_details):
            if detes[3] == "car":
                if not ((3.0 < detes[0] < 7.0) and (1.0 < detes[1] < 3.0) and (detes[0]*0.05 < detes[2] < detes[0]*0.95)):
                    flagged_agents.append(ind)
            if detes[3] == "pedestrian":
                if not ((0.5 < detes[0] < 2.0) and (0.5 < detes[1] < 2.0)):
                    flagged_agents.append(ind)
        if len(flagged_agents) > 0:
            diagnostic_message_codes.append(
                self._format_message(
                    timestep = 0,
                    issue_type = 50,
                    agent_list = flagged_agents
                )
            )

        return diagnostic_message_codes

    def _check_agent_details_static(self):
        diagnostic_message_codes = []

        is_equal_agent_details = {"details_equal":[],"same_index":[]}
        all_agent_details = [self.init_agent_details] + self.req_agent_details

        for ts in range(len(all_agent_details)-1):
            details_equal, is_index_equal = self._check_states_equal(
                [prop[:-1] for prop in all_agent_details[ts]],
                [prop[:-1] for prop in all_agent_details[ts+1]]
            )
            is_equal_agent_details["details_equal"].append(details_equal)
            is_equal_agent_details["same_index"].append(is_index_equal)

        for ind, (ts_agent_exists, ts_index) in enumerate(zip(is_equal_agent_details["details_equal"],is_equal_agent_details["same_index"])):
            if not all(ts_agent_exists):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind,
                        issue_type = 51,
                        agent_list = list(filter(lambda i: not ts_agent_exists[i], range(len(ts_agent_exists))))
                    )
                )
            elif not all(ts_index):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind,
                        issue_type = 52,
                        agent_list = list(filter(lambda i: ts_agent_exists[i] and not ts_index[i], range(len(ts_index))))
                    )
                )

        return diagnostic_message_codes

    def _check_number_of_agents(self):
        diagnostic_message_codes = []

        for ind, (req_list, res_list) in enumerate(zip(self.req_groupings["agent_states"][1:],self.res_groupings["agent_states"][:-1])):
            if len(res_list) < len(req_list):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = 101
                    )
                )

        return diagnostic_message_codes

    def _check_drive_response_equivalence(self):
        diagnostic_message_codes = []

        is_equal_agent_states = {"agents_equal":[],"same_index":[]}
        is_equal_recurrent_states = {"agents_equal":[],"same_index":[]}

        for req_dict, res_dict in zip(self.req_groupings["agent_states"][1:],self.res_groupings["agent_states"][:-1]):
            states_equal, is_index_equal = self._check_states_equal(res_dict,req_dict)
            is_equal_agent_states["agents_equal"].append(states_equal)
            is_equal_agent_states["same_index"].append(is_index_equal)
        for req_dict, res_dict in zip(self.req_groupings["recurrent_states"][1:],self.res_groupings["recurrent_states"][:-1]):
            states_equal, is_index_equal = self._check_states_equal(res_dict,req_dict)
            is_equal_recurrent_states["agents_equal"].append(states_equal)
            is_equal_recurrent_states["same_index"].append(is_index_equal)

        for ind, (ts_agent_exists, ts_index) in enumerate(zip(is_equal_agent_states["agents_equal"],is_equal_agent_states["same_index"])):
            if not all(ts_agent_exists):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = 11,
                        agent_list = list(filter(lambda i: not ts_agent_exists[i], range(len(ts_agent_exists))))
                    )
                )
            elif not all(ts_index):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = 10,
                        agent_list = list(filter(lambda i: ts_agent_exists[i] and not ts_index[i], range(len(ts_index))))
                    )
                )

        for ind, (ts_agent_exists, ts_index) in enumerate(zip(is_equal_recurrent_states["agents_equal"],is_equal_recurrent_states["same_index"])):
            if not all(ts_agent_exists):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = 21,
                        agent_list = list(filter(lambda i: not ts_agent_exists[i], range(len(ts_agent_exists))))
                    )
                )
            elif not all(ts_index):
                diagnostic_message_codes.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = 20,
                        agent_list = list(filter(lambda i: ts_agent_exists[i] and not ts_index[i], range(len(ts_index))))
                    )
                )

        return diagnostic_message_codes

    def _get_agent_details(
        self,
        agent_dict: Dict
    ):
        ts_agent_details = []

        agent_details = []
        #Covers cases where agent attributes is either None or an empty list
        if not agent_dict["agent_attributes"]:
            for detes in agent_dict["agent_properties"]:
                ts_agent_details.append([
                    detes["length"],
                    detes["width"],
                    detes["rear_axis_offset"],
                    detes["agent_type"],
                    detes["waypoint"]
                ])
        else:
            ts_agent_details = agent_dict["agent_attributes"]

        return ts_agent_details

    def _parse_log_data(
        self,
        log_data: Dict
    ):
        STATE_DECIMAL = 2
        RECURR_DECIMAL = 6

        req_agent_state_dict = {"agent_states":[],"recurrent_states":[]}
        res_agent_state_dict = {"agent_states":[],"recurrent_states":[]}
        req_agent_details = []
        init_agent_details = []

        if "large_drive_responses" in log_data:
            drive_req_data = log_data["large_drive_requests"]
            drive_res_data = log_data["large_drive_responses"]
        else:
            drive_req_data = log_data["drive_requests"]
            drive_res_data = log_data["drive_responses"]

        for req_json, res_json in zip(drive_req_data,drive_res_data):
            res_data = json.loads(res_json)
            req_data = json.loads(req_json)

            for (state_dict, data) in zip([req_agent_state_dict,res_agent_state_dict],[req_data,res_data]):
                state_dict["agent_states"].append([[round(x,STATE_DECIMAL) for x in st] for st in data["agent_states"]])
                recurr_state = None
                if data["recurrent_states"] is not None:
                    recurr_state = [[round(x,RECURR_DECIMAL) for x in st] for st in data["recurrent_states"]]
                state_dict["recurrent_states"].append(recurr_state)

        for req_json in drive_req_data:
            req_data = json.loads(req_json)
            req_agent_details.append(self._get_agent_details(req_data))
        
        if "large_initialize_responses" in log_data:
            init_res_data = log_data["large_initialize_responses"]
        else:
            init_res_data = log_data["initialize_responses"]
        res_data = json.loads(init_res_data[-1])
        init_agent_details = self._get_agent_details(res_data)

        return req_agent_state_dict, res_agent_state_dict, req_agent_details, init_agent_details

    def _check_states_equal(
        self,
        res_states: List[Union[float,str,List[float]]],
        req_states: Optional[List[Union[float,str,List[float]]]] = None
    ):
        num_res_agents = len(res_states)
        states_equal = [False]*num_res_agents
        is_same_index = [False]*num_res_agents

        if res_states is not None and req_states is not None:
            for i, sa_0 in enumerate(res_states):
                is_state_equal = False
                is_index_equal = False
                for j, sa_1 in enumerate(req_states):
                    is_state_equal = sa_0 == sa_1
                    if is_state_equal: 
                        is_index_equal = i == j
                        break
                
                # Some behaviour is expected with ego agents controlled externally to the API
                states_equal[i] = is_state_equal or i in self.ego_indexes
                is_same_index[i] = is_index_equal or i in self.ego_indexes

        return states_equal, is_same_index

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--debug_log_path',
        type=str,
        help=f"Full path to an IAI debug log to be analyzed.",
        default='None'
    )
    argparser.add_argument(
        '--ego_agent_list',
        type=int,
        nargs='+',
        help=f"IDs of ego agents.",
        default=None
    )
    args = argparser.parse_args()

    diagnostic_tool = DiagnosticTool(
        debug_log_path=args.debug_log_path,
        ego_indexes=args.ego_agent_list
    )
    diagnostic_tool.full_diagnostic_test()