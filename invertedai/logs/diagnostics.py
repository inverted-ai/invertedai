import json
import argparse
import invertedai as iai
import matplotlib.pyplot as plt

from typing import Dict, List

class DiagnosticTool:
    def __init__(
        self,
        debug_log_path: str
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

    def full_diagnostic_test(self):
        """
        Main access point for this class. This function runs all available diagnostic tests and prints 
        human-readable feedback for the benefit of a user to fix issues in their integration.
        """

        drive_state_equivalence_messages = self._check_drive_response_equivalence()

        for msg in drive_state_equivalence_messages:
            print(f"Diagnostic Information: {msg}")

        print(f"Finished diagnostic analysis.")

    def _check_drive_response_equivalence(self):
        diagnostic_messages = []

        is_equal_agent_states = {"is_equal":[],"agents_equal":[]}
        is_any_agent_states_issue = False
        is_equal_recurrent_states = {"is_equal":[],"agents_equal":[]}
        is_any_recurrent_states_issue

        req_groupings, res_groupings = self._get_all_drive_agents(log_data = self.log_data)

        for req_dict, res_dict in zip(req_groupings["agent_states"][1:],res_groupings["agent_states"][:-1]):
            states_equal, is_state_equal = self._check_states_equal(req_dict,res_dict)
            is_equal_agent_states["agents_equal"].append(states_equal)
            is_equal_agent_states["is_equal"].append(is_state_equal)
        for req_dict, res_dict in zip(req_groupings["recurrent_states"][1:],res_groupings["recurrent_states"][:-1]):
            states_equal, is_state_equal = self._check_states_equal(req_dict,res_dict)
            is_equal_recurrent_states["agents_equal"].append(states_equal)
            is_equal_recurrent_states["is_equal"].append(is_state_equal)

        if not all(is_equal_agent_states["is_equal"]):
            res_states = [f"({j}:{is_equal_agent_states["agents_equal"][j]})" for j, val in enumerate(is_equal_agent_states["is_equal"]) if not val]
            diagnostic_messages.append(f"Potential agent state index error detected for (Time step:[Agent Indexes]): {res_states}")

        if not all(is_equal_recurrent_states["is_equal"]):
            res_states = [f"({j}:{is_equal_recurrent_states["agents_equal"][j]})" for j, val in enumerate(is_equal_recurrent_states["is_equal"]) if not val]
            diagnostic_messages.append(f"Potential recurrent state index error detected for (Time step:[Agent Indexes]): {res_states}")

        return diagnostic_messages

    def _get_all_drive_agents(
        self,
        log_data: Dict
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

    def _check_individual_states_equal(
        self,
        sa_0: float,
        sa_1: float
    ):
        for param_0, param_1 in zip(sa_0,sa_1):
            if not param_0 == param_1: return False

        return True

    def _check_states_equal(
        self,
        states_0: List[float],
        states_1: List[float]
    ):
        is_all_states_equal = True
        states_equal = []
        # if not len(states_0) == len(states_1): return False

        for i, sa_0 in enumerate(states_0):
            is_state_equal = False
            for j, sa_1 in enumerate(states_1):
                is_state_equal = self._check_individual_states_equal(sa_0,sa_1)
                if is_state_equal: break
            
            states_equal.append(is_state_equal)
            is_all_states_equal = is_state_equal and is_all_states_equal

        return states_equal, is_all_states_equal


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--debug_log_path',
        type=str,
        help=f"Full path to an IAI debug log to be analyzed.",
        default='None'
    )
    args = argparser.parse_args()

    diagnostic_tool = DiagnosticTool(debug_log_path=args.debug_log_path)
    diagnostic_tool.full_diagnostic_test()