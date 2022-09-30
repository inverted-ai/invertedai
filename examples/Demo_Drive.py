#!/usr/bin/env ipython
import os
import sys
from PIL import Image as PImage
import imageio
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()
if os.environ.get("DEV", False):
    sys.path.append("../")
import invertedai as iai

# logger.setLevel(10)

parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default="")
parser.add_argument("--location", type=str, default="CARLA:Town03:Roundabout")
args = parser.parse_args()

iai.add_apikey("")
response = iai.initialize(
    location=args.location,
    agent_count=10,
    batch_size=1,
)
# breakpoint()
agent_attributes = response["attributes"]
frames = []
for i in tqdm(range(50)):
    response = iai.drive(
        agent_attributes=agent_attributes,
        states=response["states"],
        recurrent_states=response["recurrent_states"],
        get_birdviews=True,
        location=args.location,
        obs_length=1,
        steps=1,
        traffic_states_id=response["traffic_states_id"]
        # traffic_timer=response["traffic_timer"],
    )

    birdview = np.array(response["bird_view"], dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
