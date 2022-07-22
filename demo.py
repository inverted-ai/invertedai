#!/usr/bin/env ipython
import os

os.environ["DEV"] = "1"
from invertedai_drive import drive
from invertedai_drive.utils import MapLocation
from IPython.display import Image
from PIL import Image as PImage
import imageio
import torch
import numpy as np
import matplotlib
import cv2


response = drive.initialize(
    MapLocation.Town03_GasStation,
    agents_counts=10,
    num_samples=1,
    min_speed=10,
    max_speed=20,
)

initial_condition = response["initial_condition"]
initial_states = initial_condition["agent_states"]
agent_sizes = initial_condition["agent_sizes"]
x = initial_states["x"]
y = initial_states["y"]
psi = initial_states["psi"]
speed = initial_states["speed"]
length = agent_sizes["length"]
width = agent_sizes["width"]
lr = agent_sizes["lr"]
recurrent_states = None
frames = []
##

# drive_response = drive.run(
#     api_key="",
#     location=MapLocation.Town03_5way,
#     x=x,
#     y=y,
#     psi=psi,
#     speed=speed,
#     length=length,
#     width=width,
#     lr=lr,
#     batch_size=1,
#     agent_counts=10,
#     obs_length=1,
#     step_times=1,
#     recurrent_states=recurrent_states,
#     return_birdviews=True,
# )


for t in range(10):
    drive_response = drive.run(
        api_key="",
        location=MapLocation.Town03_GasStation,
        x=x,
        y=y,
        psi=psi,
        speed=speed,
        length=length,
        width=width,
        lr=lr,
        batch_size=1,
        agent_counts=10,
        obs_length=1,
        step_times=1,
        recurrent_states=recurrent_states,
        return_birdviews=True,
    )
    states = drive_response.states
    x = states.x
    y = states.y
    psi = states.psi
    speed = states.speed
    recurrent_states = np.array(drive_response.recurrent_states)[:, :, 0, ...].tolist()
    birdview = np.array(drive_response.bird_view, dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("test.gif", np.array(frames), format="GIF-PIL")
