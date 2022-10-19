#!/usr/bin/env ipython
import os
import sys
from PIL import Image as PImage
import imageio
import numpy as np
import cv2
from tqdm import tqdm
import argparse

os.environ["IAI_MOCK_API"] = "0"
os.environ["IAI_DEV"] = "1"
# os.environ["IAI_DEV_URL"] = "http://localhost:8888"

if os.environ.get("IAI_DEV", False):
    sys.path.append("../")
import invertedai as iai

# logger.setLevel(10)

parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default="")
parser.add_argument("--location", type=str, default="iai:ubc_roundabout")
args = parser.parse_args()

iai.add_apikey(args.api_key)

# response = iai.available_locations("carla", "roundabout")
response = iai.location_info(location=args.location)

file_name = args.location.replace(":", "_")
if response.osm_map is not None:
    file_path = f"{file_name}.osm"
    with open(file_path, "w") as f:
        f.write(response.osm_map[0])
if response.birdview_image is not None:
    file_path = f"{file_name}.jpg"
    rendered_map = np.array(response.birdview_image, dtype=np.uint8)
    image = cv2.imdecode(rendered_map, cv2.IMREAD_COLOR)
    cv2.imwrite(file_path, image)
simulation = iai.Simulation(
    location=args.location,
    agent_count=10,
    monitor_infractions=True,
    render_birdview=True,
)
frames = []
pbar = tqdm(range(50))
for i in pbar:
    simulation.step(current_ego_agent_states=[])
    collision, offroad, wrong_way = simulation.infractions
    pbar.set_description(
        f"Collision rate: {100*np.array(collision).mean():.2f}% | "
        + f"Off-road rate: {100*np.array(offroad).mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array(wrong_way).mean():.2f}%"
    )

    birdview = np.array(simulation.birdview, dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
