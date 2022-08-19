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
    # location=args.location,
    location="Town03_Roundabout",
    obs_length=1,
    step_times=1,
    agent_count=10,
    batch_size=1,
    min_speed=10,
    max_speed=20,
    carla_simulator=True,
)


drive = Drive(iai_config)
sim = CarlaEnv.from_preset_data()
sim.set_npc_autopilot()
sim.set_ego_autopilot()


clock = pygame.time.Clock()
frames = []
for i in range(10 * sim.config.fps):
    states, recurrent_states, dimensions = sim.get_obs(obs_len=1)
    response = drive.run(
        agent_attributes=torch.tensor(dimensions).unsqueeze(0).tolist(),
        # states=torch.tensor(states).unsqueeze(-2).unsqueeze(0).tolist(),
        states=torch.tensor(states).unsqueeze(0).tolist(),
        recurrent_states=torch.tensor(recurrent_states).unsqueeze(0).tolist(),
        return_birdviews=True,
    )

    # breakpoint()
    sim.step(npcs=response, ego="autopilot")
    clock.tick_busy_loop(sim.config.fps)
    # recurrent_states = response["recurrent_states"]
    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)


imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
sim.destroy()
