import invertedai as iai
from invertedai.common import AgentType

import argparse
import random
import time

from tqdm import tqdm

def main(args):
    #Read the scenario log containing all safety-critical agents trajectories
    log_reader = iai.LogReader(
        log_path = args.log_path
    )
    location = log_reader.location
    
    #Setup model parameters
    if args.model_version_drive == "None": 
        model_version = None
    else:
        model_version = args.model_version_drive
    random_seed = int(time.time())

    #Acquire the scenario center over which the visualization will be focused
    scenario_center = args.scenario_center if args.scenario_center is None else tuple(args.scenario_center)
    print(f"Call location info.")
    location_info_response = iai.location_info(
        location = location,
        rendering_fov = args.fov,
        rendering_center = scenario_center,
        include_map_source = True
    )
    scenario_center = tuple([location_info_response.map_center.x, location_info_response.map_center.y]) if scenario_center is None else scenario_center

    #Initialize the simulation including placing the safety critical agents and generate background vehicles around them
    print(f"Begin initialization.") 
    log_reader.initialize()

    if args.num_agents > 0:
        regions = iai.get_regions_default(
            location = location,
            agent_count_dict = {AgentType.car: args.num_agents},
            area_shape = (int(args.width/2),int(args.height/2)),
            map_center = scenario_center
        )
    else:
        regions = [
            iai.large.common.Region.create_square_region(
                center = iai.common.Point(
                    x = scenario_center[0],
                    y = scenario_center[1]
                ),
                size = args.fov
            )
        ]

    response = iai.large_initialize(
        location = location,
        regions = regions,
        random_seed = random_seed,
        get_infractions = args.get_infractions,
        agent_properties = log_reader.agent_properties,
        agent_states = log_reader.agent_states,
        traffic_light_state_history = [log_reader.traffic_lights_states] if log_reader.traffic_lights_states is not None else None
    )
    NUM_LOG_AGENTS = len(log_reader.agent_properties)
    LOG_LENGTH = log_reader.log_length

    wp_manager = iai.WaypointManager(
        location_info_response = location_info_response,
        cfg = iai.WaypointManagerConfig(
            random_seed=random_seed,
            fail_soft=True
        )
    )
    response.agent_properties = wp_manager.update(
        response = response,
        agent_properties = response.agent_properties,
    )
    
    print(f"Set up simulation.")
    if args.save_sim:
        log_writer = iai.LogWriter()
        log_writer.initialize(
            location=location,
            location_info_response=location_info_response,
            init_response=response,
        )

    total_num_agents = len(response.agent_states)
    print(f"Number of agents in simulation: {total_num_agents}")

    print(f"Begin stepping through simulation.")
    agent_properties = response.agent_properties
    for ts in tqdm(range(args.sim_length)):
        # Call DRIVE on all safety critical conditional agents and background NPC agents
        is_log_traffic_light_states = log_reader.traffic_lights_states is not None and ts < LOG_LENGTH
        response = iai.large_drive(
            location = location,
            agent_states = response.agent_states,
            agent_properties = agent_properties,
            recurrent_states = response.recurrent_states,
            traffic_lights_states = log_reader.traffic_lights_states if is_log_traffic_light_states else None,
            light_recurrent_states = response.light_recurrent_states if not is_log_traffic_light_states else None,
            random_seed = random_seed,
            api_model_version = model_version,
            get_infractions = args.get_infractions,
        )
        agent_properties = wp_manager.update(
            response = response,
            agent_properties = agent_properties,
        )

        # Get the safety critical agent states
        if ts < LOG_LENGTH:
            log_reader.drive()
            agent_states = response.agent_states
            agent_states[:NUM_LOG_AGENTS] = log_reader.agent_states
            if ts >= args.takeover_timestep:
                agent_states[args.ego_id] = response.agent_states[args.ego_id]
            response.agent_states = agent_states

        if args.save_sim: 
            log_writer.drive(
                drive_response=response,
                agent_properties=agent_properties
            )

    if args.save_sim:
        print("Simulation finished, save visualization.")
        current_time = int(time.time())
        log_name = args.log_path.split("/")[-1].split(".json")[0]
        gif_name = f'safety_critical_scenario_{log_name}_{current_time}_seed-{random_seed}_modelversion-{model_version}.gif'
        log_writer.visualize(
            gif_path=gif_name,
            fov = args.fov,
            resolution = (2048,2048),
            dpi = 300,
            direction_vec = True,
            velocity_vec = False,
            plot_frame_number = True,
            map_center = scenario_center,
            left_hand_coordinates = location.split(":")[0] == "carla",
            agent_ids = list(range(len(agent_properties)))
        )
        log_writer.export_to_file(log_path=gif_name.split(".gif")[0]+".json")
    print("Done")

    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-P',
        '--log-path',
        type=str,
        default="./assets/carla_Town10HD_example_emergency_scenario.json",
        help='Path to the scenario log file containing trajectories of safety critical agents.'
    )
    argparser.add_argument(
        '-N',
        '--num-agents',
        default=1,
        type=int,
        help='Number of background vehicles to spawn in the simulation (default: 10).'
    )
    argparser.add_argument(
        '--sim-length',
        type=int,
        help="Length of the simulation in timesteps (default: 100).",
        default=100
    )
    argparser.add_argument(
        '--takeover-timestep',
        type=int,
        help="Timestep at which control of the ego is passed to DRIVE instead of the log.",
        default=1
    )
    argparser.add_argument(
        '--ego-id',
        type=int,
        help="Index in the list of safety critical agents of the ego agent.",
        default=0
    )
    argparser.add_argument(
        '--fov',
        type=int,
        help=f"Field of view for visualization.",
        default=100
    )
    argparser.add_argument(
        '--width',
        type=int,
        help=f"Full width of the area to initialize.",
        default=100
    )
    argparser.add_argument(
        '--height',
        type=int,
        help=f"Full height of the area to initialize",
        default=100
    )
    argparser.add_argument(
        '--scenario-center',
        type=int,
        nargs='+',
        help=f"Center of the area to initialize.",
        default=None
    )
    argparser.add_argument(
        '--save-sim',
        type=bool,
        help=f"Should the simulation be saved with visualization tool.",
        default=True
    )
    argparser.add_argument(
        '--get-infractions',
        type=bool,
        help=f"Should the simulation capture infractions data.",
        default=False
    )
    argparser.add_argument(
        '--model-version-drive',
        type=str,
        help=f"Version of the DRIVE model to use during the simulation.",
        default="None"
    )
    args = argparser.parse_args()


    main(args)