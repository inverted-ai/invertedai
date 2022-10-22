from PIL import Image as PImage
import imageio
import numpy as np
from tqdm import tqdm
import argparse
import invertedai as iai

parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--location", type=str, default="canada:vancouver:ubc_roundabout")
args = parser.parse_args()

if args.api_key is not None:
    iai.add_apikey(args.api_key)

response = iai.location_info(location=args.location)

file_name = args.location.replace(":", "_")
if response.osm_map is not None:
    file_path = f"{file_name}.osm"
    with open(file_path, "w") as f:
        response.osm_map.save_osm_file(file_path)
if response.birdview_image is not None:
    file_path = f"{file_name}.jpg"
    response.birdview_image.decode_and_save(file_path)
response = iai.initialize(
    location=args.location,
    agent_count=10,
)
agent_attributes = response.agent_attributes
frames = []
pbar = tqdm(range(50))
for i in pbar:
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdview=True,
        location=args.location,
        get_infractions=True,
    )
    pbar.set_description(
        f"Collision rate: {100*np.array([inf.collisions for inf in response.infractions]).mean():.2f}% | "
        + f"Off-road rate: {100*np.array([inf.offroad for inf in response.infractions]).mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array([inf.wrong_way for inf in response.infractions]).mean():.2f}%"
    )

    image = response.bird_view.decode()
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
