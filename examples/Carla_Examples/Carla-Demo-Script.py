import json
import pygame
import os
import sys
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
if os.environ.get("DEV", False):
    sys.path.append("../../")
from invertedai_drive import Drive, Config
from invertedai_drive.simulators import CarlaEnv, CarlaSimulationConfig
import argparse


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("-n", "--scene_name", type=str, default="CARLA:Town03:Roundabout")
parser.add_argument("-c", "--agent_count", type=int, default=50)
parser.add_argument("-l", "--episode_length", type=int, default=20)
parser.add_argument("-e", "--ego_spawn_point", default=None)
parser.add_argument("-s", "--spectator_transform", default=None)
parser.add_argument("-i", "--initial_states", default=None)
parser.add_argument("-m", "--non_roi_npc_mode", type=int, default=0)
parser.add_argument("-pi", "--npc_population_interval", type=int, default=6)
parser.add_argument("-ca", "--max_cars_in_map", type=int, default=100)

args = parser.parse_args()
if args.non_roi_npc_mode == 0:
    non_roi_npc_mode = "spawn_at_entrance"
elif args.non_roi_npc_mode == 1:
    non_roi_npc_mode = "carla_handoff"
else:
    non_roi_npc_mode = "None"


iai_cfg = Config(
    location=args.scene_name, simulator="CARLA", agent_count=args.agent_count
)
carla_cfg = CarlaSimulationConfig(
    scene_name=args.scene_name,
    episode_length=args.episode_length,
    non_roi_npc_mode=non_roi_npc_mode,
    npc_population_interval=args.npc_population_interval,
    max_cars_in_map=args.max_cars_in_map,
)

drive = Drive(iai_cfg)
response = drive.initialize()
initial_states = response["states"][0]
sim = CarlaEnv(
    cfg=carla_cfg,
    initial_states=initial_states,
    ego_spawn_point=args.ego_spawn_point,
    spectator_transform=args.spectator_transform,
)
states, recurrent_states, dimensions = sim.reset()
clock = pygame.time.Clock()


for i in range(carla_cfg.episode_length * carla_cfg.fps):
    response = drive.run(
        agent_attributes=torch.tensor(dimensions).unsqueeze(0).tolist(),
        states=torch.tensor(states).unsqueeze(0).tolist(),
        recurrent_states=torch.tensor(recurrent_states).unsqueeze(0).tolist(),
    )
    states, recurrent_states, dimensions = sim.step(npcs=response, ego="autopilot")

    clock.tick_busy_loop(carla_cfg.fps)

sim.destroy()
