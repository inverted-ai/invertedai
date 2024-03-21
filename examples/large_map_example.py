import sys
sys.path.append('../')

import invertedai as iai
from simulation.simulator import Simulation, SimulationConfig

import argparse
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt


def main(args):

	map_center = tuple(args.map_center)

	print(f"Call location info.")
	rendering_fov = 200
	location_info_response = iai.location_info(
		location=args.location,
		rendering_fov=args.fov,
		rendering_center=map_center
	)

	print(f"Begin initialization.")	
	initialize_response = iai.utils.area_initialization(
		location=args.location, 
		agent_density=args.agent_density, 
		scaling_factor = 1.0,
		width = args.width,
		height = args.height,
		map_center = map_center
	)

	print(f"Set up simulation.")	
	map_width = max([abs(pt.x) for pt in location_info_response.bounding_polygon])
	map_height = max([abs(pt.y) for pt in location_info_response.bounding_polygon]) 
	map_extent = max([map_width,map_height])
	cfg = SimulationConfig(
		location = args.location,
		map_center = map_center,
		map_fov = map_extent,
		rendered_static_map= location_info_response.birdview_image.decode(),
	)

	simulation = Simulation(
		cfg = cfg,
		location_response = location_info_response,
		initialize_response = initialize_response
	)

	if args.display_sim:
		rendered_static_map = location_info_response.birdview_image.decode()
		scene_plotter = iai.utils.ScenePlotter(
		    rendered_static_map,
		    args.fov,
		    map_center,
		    location_info_response.static_actors)
		scene_plotter.initialize_recording(
		    agent_states=initialize_response.agent_states,
		    agent_attributes=initialize_response.agent_attributes,
		)


	print(f"Begin stepping through simulation.")
	for _ in tqdm(range(args.sim_length)):
		simulation.drive()
		
		if args.display_sim: scene_plotter.record_step(simulation.agent_states,simulation.traffic_lights_states)


	if args.display_sim:
		print("Simulation finished, save visualization.")
		# save the visualization to disk
		fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
		gif_name = 'large_map_example.gif'
		scene_plotter.animate_scene(
		    output_name=gif_name,
		    ax=ax,
		    direction_vec=False,
		    velocity_vec=False,
		    plot_frame_number=True
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
		default='carla:Town10HD'
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
		help=f"Width of the area to initialize.",
		default=100
	)
	argparser.add_argument(
		'--height',
		type=int,
		help=f"Height of the area to initialize",
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
		'--display-sim',
		type=bool,
		help=f"Should the simulation be visualized and saved.",
		default=True
	)
	args = argparser.parse_args()

	main(args)