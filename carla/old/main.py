import carla
import math
import time
import random
import pygame
import utils
from utils import (
    npcs_in_roi,
    simulation_config,
    get_available_port,
    to_transform,
    get_roi_spawn_points,
    get_entrance,
)

import os
import sys
import torch

sys.path.append("../..")
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
parser.add_argument("--with_init", type=int, default=0)
parser.add_argument("-el", "--episode_length", type=int, default=10)

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
    agent_count=10,
    batch_size=1,
    min_speed=10,
    max_speed=20,
)


sim_config = simulation_config(
    map_name="Town03",
    fps=10,
    traffic_count=10,
    simulation_time=args.episode_length,  # In Seconds
    roi_center={"x": 0, "y": 0},  # region of interest center
    proximity_threshold=50,
    entrance_interval=3,  # In Seconds
    slack=5,
)

world_settings = carla.WorldSettings(
    synchronous_mode=True,
    fixed_delta_seconds=1 / float(sim_config.fps),
)


drive = Drive(iai_config)

client = carla.Client("localhost", 2000)
traffic_manager = client.get_trafficmanager(get_available_port(subsequent_ports=0))
world = client.load_world(sim_config.map_name)
original_settings = client.get_world().get_settings()
world.apply_settings(world_settings)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_hybrid_physics_mode(True)

# spectator as a free camera
spectator = world.get_spectator()
camera_loc = carla.Location(
    sim_config.roi_center["x"], sim_config.roi_center["y"], z=110
)
camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
transform = carla.Transform(camera_loc, camera_rot)
spectator.set_transform(transform)

bp_lib = world.get_blueprint_library()

if args.with_init:
    response = drive.initialize()
    spawn_points = to_transform(response["states"][0])
else:
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

entrance = get_entrance(spawn_points, sim_config)
roi_spawn_points = get_roi_spawn_points(spawn_points, sim_config)

spawn_points = (
    roi_spawn_points
    if len(roi_spawn_points) < sim_config.traffic_count
    else roi_spawn_points[: sim_config.traffic_count]
)

all_npcs = []
for spawn_point in spawn_points[: sim_config.traffic_count]:
    vehicle_bp = random.choice(bp_lib.filter("vehicle"))
    npc = world.try_spawn_actor(vehicle_bp, spawn_point)
    try:
        npc.set_autopilot(True)
        all_npcs.append(npc)
    except:
        pass

world.tick()
clock = pygame.time.Clock()

npcs = dict(
    actors=all_npcs, recurrent_states=torch.zeros(len(all_npcs), 2, 64).tolist()
)

frames = []
ego = []
# recurrent_states = None
for i in range(sim_config.simulation_time * sim_config.fps):
    npcs = npcs_in_roi(npcs, ego, sim_config, world, flag_npcs=True)
    states = torch.tensor(npcs["states"]).unsqueeze(-2).unsqueeze(0).tolist()
    dimensions = torch.tensor(npcs["dimenstions"]).unsqueeze(0).tolist()
    recurrent_states = torch.tensor(npcs["recurrent_states"]).unsqueeze(0).tolist()
    breakpoint()
    response = drive.run(
        agent_attributes=dimensions,
        states=states,
        recurrent_states=recurrent_states,
        return_birdviews=True,
    )
    # recurrent_states = response["recurrent_states"]
    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
    if i % (sim_config.fps * sim_config.entrance_interval):
        vehicle_bp = random.choice(bp_lib.filter("vehicle"))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(entrance))
        try:
            npc.set_autopilot(True)
            npcs["actors"].append(npc)
            npcs["recurrent_states"].append(torch.zeros(2, 64).tolist())
        except:
            pass

        pass

    # for npc in roi_npcs["actors"]:
    # npc.set_autopilot(False)
    world.tick()
    clock.tick_busy_loop(sim_config.fps)


imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
## Clean up
for npc in npcs["actors"]:
    try:
        npc.set_autopilot(False)
    except:
        pass
    npc.destroy()
client.get_world().apply_settings(original_settings)
traffic_manager.set_synchronous_mode(False)
