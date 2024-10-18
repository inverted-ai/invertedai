import numpy as np
from tqdm import tqdm
import asyncio
import argparse
import matplotlib.pyplot as plt
import time

import invertedai as iai
from invertedai.utils import get_default_agent_properties, InterpolationManager
from invertedai.common import AgentState, Point


async def main(args):
    print("Begin initialization.")
    map_center = args.map_center if args.map_center is None else tuple(args.map_center)
    location_info_response = iai.location_info(
        location=args.location,
        rendering_fov = args.fov,
        rendering_center = map_center
    )
    map_center = tuple([location_info_response.map_center.x, location_info_response.map_center.y]) if map_center is None else map_center

    response = iai.initialize(
        location=args.location,  
        agent_properties=get_default_agent_properties({"car":args.num_agents}),
    )
    external_response = response
    agent_properties = response.agent_properties
    iai_agent_states = response.agent_states[args.num_ego_agents:]
    external_agent_states = response.agent_states[:args.num_ego_agents]

    interpolation_manager = InterpolationManager(
        ext_agent_states = external_agent_states,
        iai_agent_states = iai_agent_states,
        drive_function = iai.drive,
        drive_function_kwargs = {
            "location":args.location,
            "agent_properties":agent_properties,
            "recurrent_states":response.recurrent_states,
            "light_recurrent_states":response.light_recurrent_states
        },
        num_between_steps = args.num_between_steps
    )

    # interpolation_manager.interpolate_iai_agents(interpolation_manager.iai_agent_states[1],interpolation_manager.iai_agent_states[0])
    # breakpoint()

    print(f"Set up simulation.")
    if args.save_sim_gif:
        rendered_static_map = location_info_response.birdview_image.decode()
        scene_plotter = iai.utils.ScenePlotter(
            map_image=rendered_static_map,
            fov=args.fov,
            xy_offset=map_center,
            static_actors=location_info_response.static_actors,
            resolution=(2048,2048),
            dpi=300,
            interval=int(100/(1+args.num_between_steps))
        )
        scene_plotter.initialize_recording(
            agent_states=interpolation_manager.response.agent_states,
            agent_properties=agent_properties,
            traffic_light_states=interpolation_manager.response.traffic_lights_states
        )

    print("Begin stepping through simulation.")
    for _ in tqdm(range(args.sim_length)): 
        response = interpolation_manager.response
        is_finished_calling_drive = await interpolation_manager.async_step(
            drive_function = iai.drive,
            drive_function_kwargs = {
                "location":args.location,
                "agent_properties":agent_properties,
                "recurrent_states":response.recurrent_states,
                "light_recurrent_states":response.light_recurrent_states
            }
        )

        external_response = iai.drive(
            location=args.location,
            agent_properties=agent_properties,
            agent_states=external_agent_states+response.agent_states[args.num_ego_agents:],
            recurrent_states=external_response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states
        )
        external_agent_states_next = external_response.agent_states[:args.num_ego_agents]

        print(f"interpolated_iai_states: {interpolation_manager.interpolated_iai_states}")
        print(f"Ego states current {external_agent_states} and future {external_agent_states_next}")
        print(f"iai_agent_states {interpolation_manager.iai_agent_states[-1]}")
        
        for tts, agent_states in enumerate(interpolation_manager.interpolated_iai_states):
            num_interp_points = args.num_between_steps + 1
            external_agent_states_tts = []
            for next_state, curr_state in zip(external_agent_states_next,external_agent_states):
                external_agent_states_tts.append(
                    AgentState(
                        center=Point(
                            x=curr_state.center.x + (next_state.center.x - curr_state.center.x)*(tts/num_interp_points), 
                            y=curr_state.center.y + (next_state.center.y - curr_state.center.y)*(tts/num_interp_points), 
                        ), 
                        orientation=curr_state.orientation + (next_state.orientation - curr_state.orientation)*(tts/num_interp_points),
                        speed=curr_state.speed + (next_state.speed - curr_state.speed)*(tts/num_interp_points)
                    )
                )


            if args.save_sim_gif: scene_plotter.record_step(external_agent_states_tts+agent_states,response.traffic_lights_states)

        external_agent_states = external_agent_states_next
        interpolation_manager.add_external_agent_states_at_iai_timestep(external_agent_states)
        if is_finished_calling_drive:
            continue

    if args.save_sim_gif:
        print("Simulation finished, save visualization.")
        # save the visualization to disk
        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        gif_name = f'interpolation_example_{int(time.time())}_location-{args.location.split(":")[-1]}.gif'
        scene_plotter.animate_scene(
            output_name=gif_name,
            ax=ax,
            direction_vec=False,
            velocity_vec=False,
            plot_frame_number=True,
        )
        plt.close(fig)
    print("Done")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-N',
        '--num-agents',
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
        '--num-ego-agents',
        type=int,
        help="Number of non-NPC agents other IAI agents are simulated around.",
        default=1
    )
    argparser.add_argument(
        '--num-between-steps',
        type=int,
        help="Number steps to interpolate between standard IAI time steps of 100ms (e.g. )",
        default=1
    )
    argparser.add_argument(
        '--location',
        type=str,
        help=f"IAI formatted map on which to create simulate.",
        default='None'
    )
    argparser.add_argument(
        '--fov',
        type=int,
        help=f"Field of view for visualization.",
        default=100
    )
    argparser.add_argument(
        '--map-center',
        type=int,
        nargs='+',
        help=f"Center of the area to initialize",
        default=None
    )
    argparser.add_argument(
        '--save-sim-gif',
        type=bool,
        help=f"Should the simulation be saved with visualization tool.",
        default=True
    )
    args = argparser.parse_args()

    asyncio.run(main(args))