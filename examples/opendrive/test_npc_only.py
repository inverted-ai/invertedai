from PIL import Image as PImage
import imageio
import numpy as np
from tqdm import tqdm
import argparse
import invertedai as iai
from examples.opendrive.open_drive import draw_map, ScenePlotter
import matplotlib.pyplot as plt
import cv2

plt.ion()
sim_len = 100
parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--location", type=str,
                    default="carla:Town01")
# default="carla:Town02")
# default="carla:Town01:3way")
# default="carla:Town03")
# default="carla:Town03:Roundabout")
# default="carla:Town04")
# default="carla:Town06")
# default="carla:Town07")
# default="carla:Town03:3way_Protected")

args = parser.parse_args()


open_drive_file_name = f"/home/alireza/iai/drive-sdk/foretelix/examples/opendrive/{args.location.split(':')[1]}.csv"

fig, ax = plt.subplots()
p1 = draw_map(ax, open_drive_file_name)


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

light_response = iai.light(location=args.location)

response = iai.initialize(
    location=args.location,
    agent_count=10,
    get_birdview=True,
    get_infractions=True,
    traffic_light_state_history=[light_response.traffic_lights_states],
)

for agent in response.agent_states:
    ax.plot(agent.center.x, -agent.center.y,
            color='green', marker='o', markersize=12)

print(
    f"Initialize:\n"
    + f"Collision: {sum([inf.collisions for inf in response.infractions])}/{len(response.infractions)} | "
    + f"Off-road: {sum([inf.offroad for inf in response.infractions])}/{len(response.infractions)} |"
    + f"Wrong-way: {sum([inf.wrong_way for inf in response.infractions])}/{len(response.infractions)}"
)
agent_attributes = response.agent_attributes
frames = []
x_frame = []
pbar = tqdm(range(sim_len))
for i in pbar:
    light_response = iai.light(
        location=args.location, recurrent_states=light_response.recurrent_states)
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdview=True,
        location=args.location,
        get_infractions=True,
        # rendering_fov=500,
        traffic_lights_states=light_response.traffic_lights_states,
    )
    pbar.set_description(
        f"Collision rate: {100*np.array([inf.collisions for inf in response.infractions]).mean():.2f}% | "
        + f"Off-road rate: {100*np.array([inf.offroad for inf in response.infractions]).mean():.2f}% | "
        + f"Wrong-way rate: {100*np.array([inf.wrong_way for inf in response.infractions]).mean():.2f}%"
    )
    ax.clear()
    p1 = draw_map(ax, open_drive_file_name)
    for agent in response.agent_states:
        ax.plot(agent.center.x, -agent.center.y,
                color='green', marker='o', markersize=8)
    plt.savefig(f'./img/img_{i}.png',
                transparent=False,
                facecolor='white'
                )
    # plt.pause(0.00001)
    image = response.birdview.decode()
    frames.append(image)
    im = PImage.fromarray(image)

x_frames = []
concat_frames = []
for t, img in enumerate(frames):
    image = imageio.v2.imread(f'./img/img_{t}.png')
    w, d = img.shape[:2]
    res = cv2.resize(image, dsize=(w, d), interpolation=cv2.INTER_CUBIC)
    x_frames.append(res)
    im_v = cv2.hconcat([res[:, :, :3], img])
    concat_frames.append(im_v)

imageio.mimsave("iai-drive.gif", np.array(frames), format="GIF-PIL")
imageio.mimsave("iai-drive_opendrive.gif",
                np.array(x_frames[2:]), format="GIF-PIL")
imageio.mimsave(f"gifs/iai-{''.join(args.location.split(':')[1:])}.gif", np.array(
    concat_frames), format="GIF-PIL", fps=10)

plt.show()
