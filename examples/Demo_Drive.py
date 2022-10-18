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
parser.add_argument("--location", type=str, default="iai:ubc_roundabout")
args = parser.parse_args()

api_key = args.api_key or os.environ.get('IAI_API_KEY', None)
if api_key:
    iai.add_apikey(api_key)
else:
    print("Running mock API - specify IAI_API_KEY to obtain real results")
    iai.use_mock_api()

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
response = iai.initialize(
    location=args.location,
    agent_count=10,
    agent_attributes=None,
)
agent_attributes = response.agent_attributes
frames = []
pbar = tqdm(range(50))
for i in pbar:
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdviews=True,
        location=args.location,
        get_infractions=True,
    )
    pbar.set_description(
        f"Collision rate: {100*np.array(response.infractions.collisions).mean():.2f}% | "
        + f"Off-road rate: {100*np.array(response.infractions.offroad).mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array(response.infractions.wrong_way).mean():.2f}%"
    )

    birdview = np.array(response.bird_view, dtype=np.uint8)
    image = cv2.imdecode(birdview, cv2.IMREAD_COLOR)
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
