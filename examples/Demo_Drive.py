#!/usr/bin/env ipython
from dataclasses import dataclass
import os

# os.environ["DEV"] = "1"
from invertedai_drive import drive
from PIL import Image as PImage
import imageio
import numpy as np
import cv2
from tqdm import tqdm

config = drive.config(
    api_key="",
    location="",
    agent_count=10,
    batch_size=1,
    obs_length=1,
    step_times=1,
    min_speed=10,
    max_speed=20,
)

print(config.location)
response = drive.initialize(config)
agent_attributes = response["attributes"]
frames = []

for i in tqdm(range(50)):
    response = drive.run(
        config=config,
        location=config.location,
        agent_attributes=agent_attributes,
        states=response["states"],
        recurrent_states=response["recurrent_states"],
        return_birdviews=True,
    )

    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
