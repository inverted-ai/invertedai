from PIL import Image as PImage
import imageio
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import invertedai as iai

parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--location", type=str, default="canada:vancouver:victoria_drive_and_marine_drive")
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

df = pd.read_csv('canada:vancouver:victoria_drive_and_marine_drive.csv', index_col=0)
df.columns = df.columns.map(int)
traffic_light_counter = 1
response = iai.initialize(
    location=args.location,
    agent_count=10,
    get_birdview=True,
    get_infractions=True,
    traffic_light_state_history=[df[traffic_light_counter].to_dict()]
)
print(
    f"Initialize:\n"
    + f"Collision: {sum([inf.collisions for inf in response.infractions])}/{len(response.infractions)} | "
    + f"Off-road: {sum([inf.offroad for inf in response.infractions])}/{len(response.infractions)} |"
    + f"Wrong-way: {sum([inf.wrong_way for inf in response.infractions])}/{len(response.infractions)}"
)
agent_attributes = response.agent_attributes
frames = []
pbar = tqdm(range(50))
for i in pbar:
    traffic_light_counter += 1
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        traffic_lights_states=df[traffic_light_counter%len(df.columns)].to_dict(),
        get_birdview=True,
        location=args.location,
        get_infractions=True,
    )
    pbar.set_description(
        f"Collision rate: {100*np.array([inf.collisions for inf in response.infractions]).mean():.2f}% | "
        + f"Off-road rate: {100*np.array([inf.offroad for inf in response.infractions]).mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array([inf.wrong_way for inf in response.infractions]).mean():.2f}%"
    )

    image = response.birdview.decode()
    frames.append(image)
    im = PImage.fromarray(image)
imageio.mimsave("iai-drive-tl.gif", np.array(frames), format="GIF-PIL")
