import invertedai as iai
import argparse
from tqdm import tqdm
import numpy as np
import imageio
from PIL import Image as PImage

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
    response.osm_map.save_osm_file(file_path)
if response.birdview_image is not None:
    file_path = f"{file_name}.jpg"
    response.birdview_image.decode_and_save(file_path)
simulation = iai.BasicCosimulation(
    location=args.location,
    agent_count=10,
    monitor_infractions=True,
    ego_agent_mask=[False] * 10,
    get_birdview=True,
)
frames = []
pbar = tqdm(range(50))
for i in pbar:
    simulation.step(current_ego_agent_states=[])
    collision, offroad, wrong_way = simulation.infraction_rates
    pbar.set_description(
        f"Collision rate: {100*collision:.2f}% | "
        + f"Off-road rate: {100*offroad:.2f}% | "
        + f"Wrong-way rate: {100*wrong_way:.2f}%"
    )

    image = simulation.birdview.decode()
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
