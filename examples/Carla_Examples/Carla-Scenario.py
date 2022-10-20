import pygame
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()
if os.environ.get("IAI_DEV", False):
    sys.path.append("../../")
from invertedai.simulators import CarlaEnv, CarlaSimulationConfig
import argparse

import invertedai as iai


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("-n", "--scene_name", type=str, default="CARLA:Town03:Roundabout")
parser.add_argument("-c", "--agent_count", type=int, default=10)
parser.add_argument("-l", "--episode_length", type=int, default=20)
parser.add_argument("-e", "--ego_spawn_point", default="demo")
parser.add_argument("-s", "--spectator_transform", default=None)
parser.add_argument("-i", "--initial_states", default=None)
parser.add_argument("-m", "--non_roi_npc_mode", type=int, default=0)
parser.add_argument("-pi", "--npc_population_interval", type=int, default=6)
parser.add_argument("-ca", "--max_cars_in_map", type=int, default=100)
parser.add_argument("-ep", "--episodes", type=int, default=5)

args = parser.parse_args()
if args.non_roi_npc_mode == 0:
    non_roi_npc_mode = "spawn_at_entrance"
elif args.non_roi_npc_mode == 1:
    non_roi_npc_mode = "carla_handoff"
else:
    non_roi_npc_mode = "None"


carla_cfg = CarlaSimulationConfig(
    scene_name=args.scene_name,
    episode_length=args.episode_length,
    non_roi_npc_mode=non_roi_npc_mode,
    npc_population_interval=args.npc_population_interval,
    max_cars_in_map=args.max_cars_in_map,
)
response = iai.initialize(
    location=args.scene_name,
    agent_count=args.agent_count,
)

initial_states = response["states"][0]
sim = CarlaEnv(
    cfg=carla_cfg,
    initial_states=initial_states,
    ego_spawn_point=args.ego_spawn_point,
    spectator_transform=args.spectator_transform,
)
clock = pygame.time.Clock()

for episode in range(args.episodes):
    states, recurrent_states, dimensions = sim.reset()
    for i in range(carla_cfg.episode_length * carla_cfg.fps):
        response = iai.drive(
            agent_attributes=[dimensions],
            states=[states],
            recurrent_states=[recurrent_states],
            location=args.scene_name,
            steps=1,
            traffic_states_id=response["traffic_states_id"],
            get_infractions=True,
        )
        print(
            f"Collision rate: {100*np.array(response['collision'])[-1, 0, :].mean():.2f}% | "
            + f"Off-road rate: {100*np.array(response['offroad'])[-1, 0, :].mean():.2f}% | "
            + f"Wrong-way rate: {100*np.array(response['wrong_way'])[-1, 0, :].mean():.2f}%"
        )
        states, recurrent_states, dimensions = sim.step(npcs=response, ego="autopilot")

        clock.tick_busy_loop(carla_cfg.fps)

sim.destroy()
