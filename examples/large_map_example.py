import sys
sys.path.append('../')

import invertedai as iai
from area_drive.area_drive import AreaDriver, AreaDriverConfig

import argparse
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def main(args):

    map_center = tuple(args.map_center)

    print(f"Call location info.")
    location_info_response = iai.location_info(
        location = args.location,
        rendering_fov = args.fov,
        rendering_center = map_center
    )

    print(f"Begin initialization.") 
    initialize_response = iai.utils.area_initialization(
        location = args.location, 
        agent_density = args.agent_density, 
        scaling_factor = 1.0,
        width = int(args.width/2),
        height = int(args.height/2),
        map_center = map_center
    )

    print(f"Set up simulation.")    
    map_extent = max([args.width,args.height])
    cfg = AreaDriverConfig(
        location = args.location,
        area_center = map_center,
        area_fov = map_extent,
        quadtree_capacity = args.capacity,
        render_fov=args.fov,
        pygame_window = args.display_sim,
        show_quadtree = args.display_quadtree,
        rendered_static_map = location_info_response.birdview_image.decode()
    )

    simulation = AreaDriver(
        cfg = cfg,
        location_response = location_info_response,
        initialize_response = initialize_response
    )

    if args.save_sim_gif:
        rendered_static_map = location_info_response.birdview_image.decode()
        scene_plotter = iai.utils.ScenePlotter(
            rendered_static_map,
            args.fov,
            map_center,
            location_info_response.static_actors,
            resolution=(1080,1080),
            dpi=200
        )
        scene_plotter.initialize_recording(
            agent_states=initialize_response.agent_states,
            agent_attributes=initialize_response.agent_attributes,
        )

    total_num_agents = len(simulation.agent_states)
    print(f"Number of agents in simulation: {total_num_agents}")

    print(f"Begin stepping through simulation.")
    for _ in tqdm(range(args.sim_length)):
        simulation.drive()
        
        if args.save_sim_gif: scene_plotter.record_step(simulation.agent_states,simulation.traffic_lights_states)


    if args.save_sim_gif:
        print("Simulation finished, save visualization.")
        # save the visualization to disk
        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        gif_name = f'large_map_example_{int(time.time())}_{total_num_agents}agents.gif'
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
        '--display-sim',
        type=bool,
        help=f"Should the in-simulation visualization be displayed.",
        default=False
    )
    argparser.add_argument(
        '--display-quadtree',
        type=bool,
        help=f"If the in-simulation visualization is active, display the quadtree as well.",
        default=False
    )
    args = argparser.parse_args()

    main(args)