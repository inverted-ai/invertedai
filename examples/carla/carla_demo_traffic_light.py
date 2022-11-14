"""
This script illustrates a simple application of Inverted AI NPCs in CARLA.
When ran, it will provide visualization in the server window, with
Inverted AI NPCs indicated by blue dots, the ego vehicle by the red dot,
and the NPCs outside the supported area, controlled by CARLA's traffic manager,
not having any dots on them.
The setup is such that NPCs within a given location's supported area
are controlled by Inverted AI, while those outside of it are controlled
by CARLA's traffic manager. As the NPCs enter and exit the supported area,
they're dynamically handed off between the two controllers. The intention
is that the scenario of interest occurs when the ego vehicle traverses
the supported area, so that's where the NPCs need to be maximally realistic.
"""
from tqdm import tqdm
from PIL import Image as PImage
import imageio
import numpy as np
import pygame
import argparse
from carla_simulator import CarlaEnv, CarlaSimulationConfig
import invertedai as iai

# Configuration options to set from command line.
parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("-n", "--location", type=str, default="carla:Town03:Roundabout",
                    help='See data/static_carla.py for a list of available locations.')
parser.add_argument("-c", "--agent_count", type=int, default=8)
parser.add_argument("-l", "--episode_length", type=int, default=30)
parser.add_argument("-e", "--ego_spawn_point", default="demo")
parser.add_argument("-s", "--spectator_transform", default=None)
parser.add_argument("-i", "--initial_states", default=None)
parser.add_argument("-m", "--non_roi_npc_mode", type=int, default=1)
parser.add_argument("-pi", "--npc_population_interval", type=int, default=6)
parser.add_argument("-ca", "--max_cars_in_map", type=int, default=100)
parser.add_argument("-ep", "--episodes", type=int, default=5)
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("-mc", "--manual_control_ego", action="store_true")


# Parse arguments and set defaults
args = parser.parse_args()
if args.non_roi_npc_mode == 0:
    non_roi_npc_mode = "spawn_at_entrance"
elif args.non_roi_npc_mode == 1:
    non_roi_npc_mode = "carla_handoff"
else:
    non_roi_npc_mode = "no_non_roi_npcs"
if args.api_key is not None:
    iai.add_apikey(args.api_key)

# Get static information about a given location.
response = iai.location_info(location=args.location)
static_actors = response.static_actors

# Initialize CARLA with the same state
carla_cfg = CarlaSimulationConfig(
    location=args.location,
    episode_length=args.episode_length,
    non_roi_npc_mode=non_roi_npc_mode,
    npc_population_interval=args.npc_population_interval,
    max_cars_in_map=args.max_cars_in_map,
    manual_control_ego=args.manual_control_ego,
    pygame_window=False,
    spectator_fov=150
)
sim = CarlaEnv(
    cfg=carla_cfg,
    static_actors=static_actors,
)

# Initialize simulation with an API call
initialize_response = iai.initialize(
    location=args.location,
    agent_count=args.agent_count,
    traffic_light_state_history=[sim.traffic_light_states],
)

frames = []
pygame.init()

try:
    # Run simulation for a given number of episodes
    for _ in tqdm(range(args.episodes), position=0):
        agent_states, recurrent_states, agent_attributes = sim.reset(
            initial_states=initialize_response.agent_states,
            ego_spawn_point=args.ego_spawn_point,
            spectator_transform=args.spectator_transform,
            initial_recurrent_states=initialize_response.recurrent_states,
        )
        clock = pygame.time.Clock()
        for i in tqdm(
            range(carla_cfg.episode_length * carla_cfg.fps), position=0, leave=False
        ):
            # Call the API to obtain the NPC behavior
            tl_states = sim.traffic_light_states
            response = iai.drive(
                agent_attributes=agent_attributes,
                agent_states=agent_states,
                recurrent_states=recurrent_states,
                location=args.location,
                traffic_lights_states=tl_states,
                get_birdview=True,
            )
            image = response.birdview.decode()
            frames.append(image)
            im = PImage.fromarray(image)

            # Advance the simulation.
            # Return values are needed to allow the NPCs to enter and
            # exit the simulation dynamically.
            agent_states, recurrent_states, agent_attributes = sim.step(
                npcs=response
            )

            # To prevent the simulation from running faster than real time
            clock.tick_busy_loop(carla_cfg.fps)
    imageio.mimsave("iai-carla.gif", np.array(frames), format="GIF-PIL")
finally:
    # Release the CARLA server
    sim.destroy()
