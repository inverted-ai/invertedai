import argparse
import os
import time
import random
import logging
import invertedai as iai
import matplotlib.pyplot as plt

from invertedai.common import AgentState
from invertedai.api.initialize import InitializeResponse
from typing import List, Optional

logger = logging.getLogger(__name__)

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

        self.log_reader = iai.LogReader(log_path=self.scenario_path)
        self.log_reader.initialize()

        self.scenario_log = self.log_reader.return_scenario_log()

        num_agents = len(self.log_reader.agent_properties)

        self.cosimulation = iai.cosimulation.BasicCosimulation(
            location=self.log_reader.location,
            conditional_agent_properties=[self.log_reader.agent_properties[i] for i in range(num_agents) if i in self.ego_indexes] +\
                [self.log_reader.agent_properties[i] for i in range(num_agents) if i not in self.ego_indexes],
            conditional_agent_agent_states=[self.log_reader.agent_states[i] for i in range(num_agents) if i in self.ego_indexes] +\
                [self.log_reader.agent_states[i] for i in range(num_agents) if i not in self.ego_indexes],
            num_non_ego_conditional_agents=num_agents-len(self.ego_indexes),
            init_response = InitializeResponse(
                agent_states = self.log_reader.agent_states,
                recurrent_states = self.log_reader.recurrent_states if self.log_reader.recurrent_states is not None else [],
                agent_attributes = [],
                agent_properties = self.log_reader.agent_properties, 
                birdview = None,
                infractions = [],
                traffic_lights_states = self.log_reader.traffic_lights_states,
                light_recurrent_states = self.log_reader.light_recurrent_states,
                api_model_version = self.scenario_log.initialize_model_version
            )
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

def _run_simulation(
    scenario_tool: ScenarioTool,
    args: argparse.Namespace,
    scenario_name: str,
    is_visualize: bool = True,
    ego_indexes: List[int] = None
):
    if ego_indexes is not None:
        num_ego_agents = len(ego_indexes)
    else:
        num_ego_agents = 0

    lir = scenario_tool.log_reader.location_info_response

    if is_visualize:
        rendered_static_map = scenario_tool.log_reader.location_info_response.birdview_image.decode()
        scene_plotter = iai.utils.ScenePlotter(
            map_image=rendered_static_map,
            fov=scenario_tool.scenario_log.rendering_fov,
            xy_offset=scenario_tool.scenario_log.rendering_center,
            static_actors=scenario_tool.log_reader.location_info_response.static_actors,
            resolution=(2048,2048),
            dpi=300
        )
        scene_plotter.initialize_recording(
            agent_states=scenario_tool.cosimulation.agent_states,
            agent_properties=scenario_tool.cosimulation.agent_properties,
            traffic_light_states=scenario_tool.cosimulation.light_states
        )

    random.seed(int(time.time()))
    drive_seed = random.randint(1,10000)
    if args.model_version_drive is None: 
        model_version = None
    else:
        model_version = args.model_version_drive

    logger.info(f"Simulation {scenario_name} begin rolling through time steps.")
    for _ in range(args.sim_length):
        ego_agent_states = []
        if ego_indexes is not None:
            response = iai.large_drive(
                location = scenario_tool.log_reader.location,
                agent_states = scenario_tool.cosimulation.agent_states,
                agent_properties = scenario_tool.cosimulation.agent_properties,
                recurrent_states = None if len(scenario_tool.cosimulation.recurrent_states) == 0 else scenario_tool.cosimulation.recurrent_states,
                light_recurrent_states = scenario_tool.cosimulation.light_recurrent_state,
                random_seed = drive_seed+1,
                get_infractions = args.get_infractions,
            )
            ego_agent_states = response.agent_states[:num_ego_agents]
        
        scenario_tool.cosimulation.step(
            current_conditional_agent_states=ego_agent_states,
            random_seed = drive_seed,
            api_model_version = model_version,
            get_infractions = args.get_infractions
        )

        if is_visualize: scene_plotter.record_step(scenario_tool.cosimulation.agent_states,scenario_tool.cosimulation.light_states)

    if is_visualize:
        logger.info(f"Simulation {scenario_name} finished, saving visualization.")
        # save the visualization to disk
        colour_list = None
        if ego_indexes is not None:
            colour_list = [(0.78, 0.0, 0.0) if i in ego_indexes else None for i in range(len(scenario_tool.cosimulation.agent_properties))]

        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        current_time = int(time.time())
        gif_name = f'scenario_visualization_{current_time}_{scenario_name.split(".")[0]}.gif'
        scene_plotter.animate_scene(
            output_name=gif_name,
            ax=ax,
            direction_vec=True,
            velocity_vec=False,
            plot_frame_number=True,
            agent_face_colors=colour_list,
            agent_edge_colors=colour_list
        )
        plt.close(fig)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--scenario_path',
        type=str,
        help=f"Full path to an IAI debug scenario to rollout.",
        default=None
    )
    argparser.add_argument(
        '--scenarios_dir',
        type=str,
        help=f"Directory containing many scenarios to rollout sequentially.",
        default=None
    )
    argparser.add_argument(
        '--ego_indexes',
        type=int,
        nargs='+',
        help=f"IDs of ego agents.",
        default=None
    )
    argparser.add_argument(
        '--sim_length',
        type=int,
        help="Length of the simulation in timesteps (default: 100)",
        default=100
    )
    argparser.add_argument(
        '--model_version_drive',
        type=str,
        help=f"Version of the DRIVE model to use during the simulation.",
        default=None
    )
    argparser.add_argument(
        '--get_infractions',
        type=bool,
        help=f"Should the simulation capture infractions data.",
        default=False
    )
    args = argparser.parse_args()

    if args.scenarios_dir is not None:
        for root, dirs, files in os.walk(args.scenarios_dir):
            for file in files:
                if file.endswith('.json'):
                    scenario_tool = ScenarioTool(
                        scenario_path=os.path.join(root, file),
                        ego_indexes=args.ego_indexes
                    )

                    _run_simulation(
                        scenario_tool=scenario_tool,
                        args=args,
                        is_visualize=True,
                        scenario_name=file,
                        ego_indexes=args.ego_indexes
                    )
    else:
        scenario_tool = ScenarioTool(
            scenario_path=args.scenario_path,
            ego_indexes=args.ego_indexes
        )

        _run_simulation(
            scenario_tool=scenario_tool,
            args=args,
            is_visualize=True,
            scenario_name=args.scenario_path.split("/")[-1],
            ego_indexes=args.ego_indexes
        )