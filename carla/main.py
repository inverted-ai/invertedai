from utils import CarlaEnv, CarlaSimulationConfig
import pygame
import os
import sys
import numpy as np
from PIL import Image as PImage
import cv2
import imageio
import torch

sys.path.append("../..")
os.environ["DEV"] = "1"
from invertedai_drive import Drive, Config

iai_config = Config(
    api_key=" ",
    location="Town03_Roundabout",
    obs_length=1,
    step_times=1,
    agent_count=100,
    batch_size=1,
    min_speed=1,
    max_speed=5,
    carla_simulator=True,
)


drive = Drive(iai_config)
response = drive.initialize()
initial_states = response["states"][0]

sim = CarlaEnv.from_preset_data(initial_states=initial_states)
states, recurrent_states, dimensions = sim.reset()
clock = pygame.time.Clock()
frames = []


for i in range(sim.config.episode_lenght * sim.config.fps):
    response = drive.run(
        agent_attributes=torch.tensor(dimensions).unsqueeze(0).tolist(),
        states=torch.tensor(states).unsqueeze(0).tolist(),
        recurrent_states=torch.tensor(recurrent_states).unsqueeze(0).tolist(),
        return_birdviews=True,
    )
    states, recurrent_states, dimensions = sim.step(npcs=response, ego="autopilot")

    clock.tick_busy_loop(sim.config.fps)
    # recurrent_states = response["recurrent_states"]
    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)

imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
sim.destroy()
