import pygame
import os
import sys

os.environ["IAI_MOCK_API"] = "0"
os.environ["IAI_DEV"] = "1"
os.environ["IAI_DEV_URL"] = "http://localhost:8888"
if os.environ.get("IAI_DEV", False):
    sys.path.append("../")
import invertedai as iai
from simulators import CarlaEnv, CarlaSimulationConfig
import argparse


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("-n", "--scene_name", type=str, default="carla:Town03:Roundabout")
parser.add_argument("-c", "--agent_count", type=int, default=8)
parser.add_argument("-l", "--episode_length", type=int, default=30)
parser.add_argument("-e", "--ego_spawn_point", default="demo")
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


carla_cfg = CarlaSimulationConfig(
    scene_name=args.scene_name,
    episode_length=args.episode_length,
    non_roi_npc_mode=non_roi_npc_mode,
    npc_population_interval=args.npc_population_interval,
    max_cars_in_map=args.max_cars_in_map,
)

iai.add_apikey("")
response = iai.initialize(
    location=args.scene_name,
    agent_count=args.agent_count,
)
initial_states = response.agent_states
sim = CarlaEnv(
    cfg=carla_cfg,
    initial_states=initial_states,
    ego_spawn_point=args.ego_spawn_point,
    spectator_transform=args.spectator_transform,
)

agent_states, recurrent_states, agent_attributes = sim.reset()
clock = pygame.time.Clock()

for i in range(carla_cfg.episode_length * carla_cfg.fps):

    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=agent_states,
        recurrent_states=recurrent_states,
        location=args.scene_name,
    )
    agent_states, recurrent_states, agent_attributes = sim.step(
        npcs=response, ego="autopilot"
    )

    clock.tick_busy_loop(carla_cfg.fps)

sim.destroy()
