from utils import CarlaEnv, CarlaSimulationConfig
import pygame
import os
import sys
import numpy as np
from PIL import Image as PImage
import cv2
import imageio
import torch
import carla

sys.path.append("../..")
os.environ["DEV"] = "1"
from invertedai_drive import Drive, Config

iai_config = Config(
    api_key=" ",
    # location=args.location,
    location="Town03_Roundabout",
    # location="Town04_Merging",
    obs_length=1,
    step_times=1,
    agent_count=100,
    batch_size=1,
    min_speed=10,
    max_speed=20,
    carla_simulator=True,
)


drive = Drive(iai_config)
response = drive.initialize()
spawn_points = response["states"][0]

sim = CarlaEnv.from_preset_data(npc_roi_spawn_points=spawn_points)
sim.set_npc_autopilot()
sim.set_ego_autopilot()

clock = pygame.time.Clock()
frames = []

states, recurrent_states, dimensions = sim.get_obs(obs_len=1)

for i in range(10 * sim.config.fps):
    response = drive.run(
        agent_attributes=torch.tensor(dimensions).unsqueeze(0).tolist(),
        states=torch.tensor(states).unsqueeze(0).tolist(),
        recurrent_states=torch.tensor(recurrent_states).unsqueeze(0).tolist(),
        return_birdviews=True,
    )

    # states, recurrent_states, dimensions = sim.step(npcs=None, ego="autopilot")
    states, recurrent_states, dimensions = sim.step(npcs=response, ego="autopilot")
    clock.tick_busy_loop(sim.config.fps)
    # recurrent_states = response["recurrent_states"]
    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)


imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
sim.destroy()
