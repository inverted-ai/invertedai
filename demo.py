#!/usr/bin/env ipython
from dataclasses import dataclass
import os

os.environ["DEV"] = "1"
from invertedai_drive import drive
from PIL import Image as PImage
import imageio
import numpy as np
import cv2
import random

maps = [
    "Town06_Merge_Double",
    "Town06_Merge_Single",
    "Town01_Straight",
    "Town02_Straight",
    "Town04_Merging",
]


config = drive.config(
    api_key="",
    location=random.choice(maps),
    # location="Town04_Merging",
    agent_count=30,
    batch_size=1,
    obs_length=1,
    step_times=1,
    min_speed=10,
    max_speed=20,
)

print(config.location)

response = drive.initialize(config)

states = response["initial_condition"]["agent_states"]
agent_sizes = response["initial_condition"]["agent_sizes"]
recurrent_states = None
frames = []

for t in range(10):
    drive_response = drive.run(
        config=config,
        location=config.location,
        x=states["x"],
        y=states["y"],
        psi=states["psi"],
        speed=states["speed"],
        length=agent_sizes["length"],
        width=agent_sizes["width"],
        lr=agent_sizes["lr"],
        recurrent_states=recurrent_states,
        return_birdviews=True,
    )

    states = drive_response["states"]
    recurrent_states = drive_response["recurrent_states"]
    birdview = np.array(drive_response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
