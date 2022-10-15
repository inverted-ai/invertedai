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

response = iai.available_locations("carla", "roundabout")
response = iai.location_info(location=args.location)

file_name = args.location.replace(":", "_")
if response.lanelet_map_source is not None:
    file_path = f"{file_name}.osm"
    with open(file_path, "w") as f:
        f.write(response.lanelet_map_source)
if response.rendered_map is not None:
    file_path = f"{file_name}.jpg"
    rendered_map = np.array(response.rendered_map, dtype=np.uint8)
    image = cv2.imdecode(rendered_map, cv2.IMREAD_COLOR)
    cv2.imwrite(file_path, image)

response = iai.initialize(
    location=args.location,
    agent_count=10,
)
agent_sizes = response.agent_sizes
frames = []
pbar = tqdm(range(50))
for i in pbar:
    response = iai.drive(
        agent_sizes=agent_sizes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdviews=True,
        location=args.location,
        steps=1,
        traffic_states_id=response.traffic_states_id,
        get_infractions=True,
    )
    pbar.set_description(
        f"Collision rate: {100*np.array(response.collision)[-1, :].mean():.2f}% | "
        + f"Off-road rate: {100*np.array(response.offroad)[-1, :].mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array(response.wrong_way)[-1, :].mean():.2f}%"
    )

    birdview = np.array(response.bird_view, dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
