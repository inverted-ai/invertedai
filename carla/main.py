import carla
import math
import time
import random
import pygame
from utils import npcs_in_roi, simulation_config

import os
import sys

sys.path.append("../")
os.environ["DEV"] = "1"
from IPython import get_ipython
from invertedai_drive import Drive, Config
from PIL import Image as PImage
import imageio
import numpy as np
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default="")
parser.add_argument("--location", type=str, default="")

if get_ipython() is None:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

iai_config = Config(
    api_key=args.api_key,
    # location=args.location,
    location="Town03_Roundabout",
    obs_length=1,
    step_times=1,
)


sim_config = simulation_config(
    map_name="Town03",
    fps=30,
    traffic_count=100,
    simulation_time=10,  # In Seconds
    roi_center={"x": 0, "y": 0},  # region of interest center
    proximity_threshold=50,
)

world_settings = carla.WorldSettings(
    synchronous_mode=True,
    fixed_delta_seconds=1 / float(sim_config.fps),
)


client = carla.Client("localhost", 2000)
traffic_manager = client.get_trafficmanager(8001)
world = client.load_world(sim_config.map_name)
debug = world.debug
original_settings = client.get_world().get_settings()
world.apply_settings(world_settings)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_hybrid_physics_mode(True)

# spectator as a free camera
spectator = world.get_spectator()
camera_loc = carla.Location(
    sim_config.roi_center["x"], sim_config.roi_center["y"], z=70
)
camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
transform = carla.Transform(camera_loc, camera_rot)
spectator.set_transform(transform)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)
npcs = []

for spawn_point in spawn_points[: sim_config.traffic_count]:
    vehicle_bp = random.choice(bp_lib.filter("vehicle"))
    npc = world.try_spawn_actor(vehicle_bp, spawn_point)
    npc.set_autopilot(True)
    npcs.append(npc)

clock = pygame.time.Clock()

drive = Drive(iai_config)

for i in range(sim_config.simulation_time * sim_config.fps):
    world.tick()
    roi_npcs = npcs_in_roi(npcs, sim_config)
    for npc in roi_npcs["actors"]:
        # npc.set_autopilot(False)
        loc = npc.get_location()
        loc.z += 3
        debug.draw_point(
            location=loc,
            size=0.1,
            color=carla.Color(0, 255, 0, 0),
            life_time=0.1,
        )

    clock.tick_busy_loop(sim_config.fps)


## Clean up
for npc in npcs:
    try:
        npc.set_autopilot(False)
    except:
        pass
    npc.destroy()
client.get_world().apply_settings(original_settings)
traffic_manager.set_synchronous_mode(False)
