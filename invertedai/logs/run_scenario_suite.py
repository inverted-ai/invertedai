import argparse
import invertedai as iai
import matplotlib.pyplot as plt

from invertedai.common import AgentState
from typing import List

class ScenarioTool:
    """
    A user-side tool that loads a scenario file and runs it, with option to run a fraction
    of the vehicles externally to the API.

    Parameters
    ----------
    scenario_path:
        The full path to the debug scenario file to be loaded and executed.
    ego_indexes:
        A list of index IDs for ego vehicles that will be controlled externally.
    """

    def __init__(
        self,
        scenario_path: str,
        ego_indexes: Optional[List[int]] = None
    ):
        self.scenario_path = scenario_path
        self.ego_indexes = [] if ego_indexes is None else sorted(tuple(ego_indexes))

        self.log_reader = iai.logs.LogReader(log_path=self.scenario_path)
        self.log_reader.initialize()

        num_agents = len(self.log_reader.agent_properties)

        self.cosimulation = iai.cosimulation.BasicCosimulation(
            location=self.log_reader.location,
            conditional_agent_properties=[self.log_reader.agent_properties[i] for i in range(num_agents) if i in self.ego_indexes] +\
                [self.log_reader.agent_properties[i] for i in range(num_agents) if i not in self.ego_indexes],
            conditional_agent_agent_states=[self.log_reader.agent_states[i] for i in range(num_agents) if i in self.ego_indexes] +\
                [self.log_reader.agent_states[i] for i in range(num_agents) if i not in self.ego_indexes],
            num_non_ego_conditional_agents=num_agents-len(self.ego_indexes),
            regions=iai.get_regions_in_grid(
                width=100,
                height=100,
                map_center=(
                    self.log_reader.location_info_response.map_center.x,
                    self.log_reader.location_info_response.map_center.y
                )
            ),
            traffic_light_state_history=[self.log_reader.traffic_lights_states] if self.log_reader.traffic_lights_states is not None else None,
            random_seed=self._scenario_log.initialize_random_seed,
            api_model_version=self._scenario_log.initialize_model_version,
            return_exact_agents=True
        )

    def step(
        self,
        ego_agent_states: List[AgentState],
        **kwargs
    ):
        self.cosimulation.step(
            current_conditional_agent_states=ego_agent_states,
            **kwargs
        )

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--scenario_path',
        type=str,
        help=f"Full path to an IAI debug scenario to rollout.",
        default='None'
    )
    argparser.add_argument(
        '--ego_indexes',
        type=int,
        nargs='+',
        help=f"IDs of ego agents.",
        default=None
    )
    args = argparser.parse_args()

    diagnostic_tool = ScenarioTool(
        scenario_path=args.scenario_path,
        ego_indexes=args.ego_indexes
    )