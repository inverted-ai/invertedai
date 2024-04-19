import sys
sys.path.append('../')

import invertedai as iai
from area_drive.area_drive import AreaDriver, AreaDriverConfig

import argparse
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random

def main(args):
    if args.model_version_drive == "None": 
        model_version = None
    else:
        model_version = args.model_version_drive
    for i in range(args.num_simulations):
        initialize_seed = random.randint(1,10000)
        drive_seed = random.randint(1,10000)

        map_center = tuple(args.map_center)

        print(f"Call location info.")
        location_info_response = iai.location_info(
            location = args.location,
            rendering_fov = args.fov,
            rendering_center = map_center
        )

        print(f"Begin initialization.") 
        response = iai.region_initialize(
            location = args.location,
            regions = iai.get_regions_density_by_road_area(
                location = args.location,
                regions = iai.define_regions_grid(
                    map_center = map_center,
                    width = int(args.width/2), 
                    height = int(args.height/2) 
                ),
                max_agent_density = args.agent_density,
                scaling_factor = 1.0
            ),
            random_seed = initialize_seed
        )

        print(f"Set up simulation.")    
        if args.save_sim_gif:
            rendered_static_map = location_info_response.birdview_image.decode()
            scene_plotter = iai.utils.ScenePlotter(
                rendered_static_map,
                args.fov,
                map_center,
                location_info_response.static_actors,
                resolution=(2048,2048),
                dpi=300
            )
            scene_plotter.initialize_recording(
                agent_states=response.agent_states,
                agent_attributes=response.agent_attributes,
                traffic_light_states=response.traffic_lights_states
            )

        total_num_agents = len(response.agent_states)
        print(f"Number of agents in simulation: {total_num_agents}")

        print(f"Begin stepping through simulation.")
        agent_attributes = response.agent_attributes
        for _ in tqdm(range(args.sim_length)):
            response = iai.region_drive(
                location = args.location,
                agent_states = response.agent_states,
                agent_attributes = agent_attributes,
                recurrent_states = response.recurrent_states,
                traffic_lights_states = response.traffic_lights_states,
                light_recurrent_states = response.light_recurrent_states,
                random_seed = drive_seed,
                api_model_version = model_version,
                capacity = args.capacity,
                is_async = True
            )

            if args.save_sim_gif: scene_plotter.record_step(response.agent_states,response.traffic_lights_states)

        if args.save_sim_gif:
            print("Simulation finished, save visualization.")
            # save the visualization to disk
            fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
            plt.axis('off')
            current_time = int(time.time())
            gif_name = f'large_map_example_{current_time}_location-{args.location.split(":")[-1]}_density-{args.agent_density}_center-x{map_center[0]}y{map_center[1]}_width-{args.width}_height-{args.height}_initseed-{initialize_seed}_driveseed-{drive_seed}_modelversion-{model_version}.gif'
            scene_plotter.animate_scene(
                output_name=gif_name,
                ax=ax,
                direction_vec=False,
                velocity_vec=False,
                plot_frame_number=True,
            )
        print("Done")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-D',
        '--agent-density',
        metavar='D',
        default=1,
        type=int,
        help='Number of vehicles to spawn per 100x100m grid (default: 10)'
    )
    argparser.add_argument(
        '--sim-length',
        type=int,
        help="Length of the simulation in timesteps (default: 100)",
        default=100
    )
    argparser.add_argument(
        '--location',
        type=str,
        help=f"IAI formatted map on which to create simulate.",
        default='None'
    )
    argparser.add_argument(
        '--capacity',
        type=int,
        help=f"The capacity parameter of a quadtree leaf before splitting.",
        default=100
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
        '--map-center',
        type=int,
        nargs='+',
        help=f"Center of the area to initialize",
        default=[0,0]
    )
    argparser.add_argument(
        '--save-sim-gif',
        type=bool,
        help=f"Should the simulation be saved with visualization tool.",
        default=True
    )
    argparser.add_argument(
        '--num-simulations',
        type=int,
        help=f"Number of simulations to run one after the other.",
        default=1
    )
    argparser.add_argument(
        '--model-version-drive',
        type=str,
        help=f"Version of the DRIVE model to use during the simulation.",
        default="None"
    )
    args = argparser.parse_args()

    main(args)