import json
import argparse
import invertedai as iai
import matplotlib.pyplot as plt

from typing import Dict, List, Optional

class DiagnosticTool:
    def __init__(
        self,
        debug_log_path: str,
        ego_indexes: Optional[List[int]] = []
    ):
        """
        A user-side tool that examines a debug log checking for common implementation mistakes 
        and provides feedback.

        Parameters
        ----------
        debug_log_path:
            The full path to the debug log file to be loaded and analyzed.
        """

        self.debug_log_path = debug_log_path
        self.log_data = None
        with open(debug_log_path) as json_file:
            self.log_data = json.load(json_file)

        self.DIAGNOSTIC_ISSUE_LIBRARY = {
            10: "Agent state index change",
            11: "Agent state modified or removed before next request",
            20: "Recurrent state index change",
            21: "Recurrent state modified or removed before next request",
            100: "Agents removed before next request",
            101: "Agents added before next request"
        }

        self.ego_indexes = ego_indexes

    def _format_message(
        self,
        timestep: int,
        issue_type: str,
        agent_list: Optional[str] = None
    ):
        if agent_list is None:
            output_str = f"Potential issue that might cause degradation in performance at timestep {timestep}: {issue_type}"
        else:
            output_str = f"Potential issue that might cause degradation in performance at timestep {timestep}: {issue_type} applicable to agents {agent_list}"
        return output_str

    def full_diagnostic_test(self):
        """
        Main access point for this class. This function runs all available diagnostic tests and prints 
        human-readable feedback for the benefit of a user to fix issues in their integration.
        """

        for msg in self._check_drive_response_equivalence():
            print(msg)

        for msg in self._check_number_of_agents():
            print(msg)

        print(f"Finished diagnostic analysis.")

    def _check_number_of_agents(self):
        diagnostic_messages = []

        req_groupings, res_groupings = self._get_all_drive_agents(log_data = self.log_data)
        for ind, (req_list, res_list) in enumerate(zip(req_groupings["agent_states"][1:],res_groupings["agent_states"][:-1])):
            if len(res_list) < len(req_list):
                diagnostic_messages.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = self.DIAGNOSTIC_ISSUE_LIBRARY[101]
                    )
                )

        return diagnostic_messages

    def _check_drive_response_equivalence(self):
        diagnostic_messages = []

        is_equal_agent_states = {"agents_equal":[],"same_index":[]}
        is_equal_recurrent_states = {"agents_equal":[],"same_index":[]}

        req_groupings, res_groupings = self._get_all_drive_agents(log_data = self.log_data)

        for req_dict, res_dict in zip(req_groupings["agent_states"][1:],res_groupings["agent_states"][:-1]):
            states_equal, is_index_equal = self._check_states_equal(res_dict,req_dict)
            is_equal_agent_states["agents_equal"].append(states_equal)
            is_equal_agent_states["same_index"].append(is_index_equal)
        for req_dict, res_dict in zip(req_groupings["recurrent_states"][1:],res_groupings["recurrent_states"][:-1]):
            states_equal, is_index_equal = self._check_states_equal(res_dict,req_dict)
            is_equal_recurrent_states["agents_equal"].append(states_equal)
            is_equal_recurrent_states["same_index"].append(is_index_equal)

        for ind, (ts_agent_exists, ts_index) in enumerate(zip(is_equal_agent_states["agents_equal"],is_equal_agent_states["same_index"])):
            if not all(ts_agent_exists):
                diagnostic_messages.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = self.DIAGNOSTIC_ISSUE_LIBRARY[11],
                        agent_list = str(list(filter(lambda i: not ts_agent_exists[i], range(len(ts_agent_exists)))))
                    )
                )
            elif not all(ts_index):
                diagnostic_messages.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = self.DIAGNOSTIC_ISSUE_LIBRARY[10],
                        agent_list = str(list(filter(lambda i: ts_agent_exists[i] and not ts_index[i], range(len(ts_index)))))
                    )
                )

        for ind, (ts_agent_exists, ts_index) in enumerate(zip(is_equal_recurrent_states["agents_equal"],is_equal_recurrent_states["same_index"])):
            if not all(ts_agent_exists):
                diagnostic_messages.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = self.DIAGNOSTIC_ISSUE_LIBRARY[21],
                        agent_list = str(list(filter(lambda i: not ts_agent_exists[i], range(len(ts_agent_exists)))))
                    )
                )
            elif not all(ts_index):
                diagnostic_messages.append(
                    self._format_message(
                        timestep = ind+1,
                        issue_type = self.DIAGNOSTIC_ISSUE_LIBRARY[20],
                        agent_list = str(list(filter(lambda i: ts_agent_exists[i] and not ts_index[i], range(len(ts_index)))))
                    )
                )

        return diagnostic_messages

    def _get_all_drive_agents(
        self,
        log_data: Dict
    ):
        STATE_DECIMAL = 2
        RECURR_DECIMAL = 6

        req_agent_state_dict = {"agent_states":[],"recurrent_states":[]}
        res_agent_state_dict = {"agent_states":[],"recurrent_states":[]}

        if "large_drive_responses" in log_data:
            drive_req_data = log_data["large_drive_requests"]
            drive_res_data = log_data["large_drive_responses"]
        else:
            drive_req_data = log_data["drive_requests"]
            drive_res_data = log_data["drive_responses"]

        for req_data, res_data in zip(drive_req_data,drive_res_data):
            res_data = json.loads(res_data)
            req_data = json.loads(req_data)

            for (state_dict, data) in zip([req_agent_state_dict,res_agent_state_dict],[req_data,res_data]):
                state_dict["agent_states"].append([[round(x,STATE_DECIMAL) for x in st] for st in data["agent_states"]])
                recurr_state = None
                if data["recurrent_states"] is not None:
                    recurr_state = [[round(x,RECURR_DECIMAL) for x in st] for st in data["recurrent_states"]]
                state_dict["recurrent_states"].append(recurr_state)

        return req_agent_state_dict, res_agent_state_dict

    def _check_states_equal(
        self,
        res_states: Optional[List[float]] = None,
        req_states: Optional[List[float]] = None
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