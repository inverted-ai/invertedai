from PIL import Image as PImage
import imageio
import numpy as np
from tqdm import tqdm
import argparse
import invertedai as iai
from invertedai.common import AgentState, Point
from examples.opendrive.open_drive import draw_map, ScenePlotter
import matplotlib.pyplot as plt
import cv2

#     "carla:Town01:3way": [93.70275115966797, 200.7403106689453],
#     "carla:Town01:Straight": [254.2249755859375, 300.88372802734375],
#     "carla:Town02:3way": [41.620059967041016, 191.54580688476562],
#     "carla:Town02:Straight": [110.15972900390625, 258.85760498046875],
#     "carla:Town03:3way_Protected": [187.63462829589844, -2.8370285034179688],
#     "carla:Town03:3way_Unprotected": [3.7141036987304688, -154.9966278076172],
#     "carla:Town03:4way": [-65.73756408691406, -122.9695053100586],
#     "carla:Town03:5way": [-64.09320068359375, 2.4671669006347656],
#     "carla:Town03:Gas_Station": [-20.871902465820312, 141.3246307373047],
#     "carla:Town03:Roundabout": [0.0, 0.0],
#     "carla:Town04:3way_Large": [-378.30316162109375, -12.988147735595703],
#     "carla:Town04:3way_Small": [127.83306884765625, -167.14236450195312],
#     "carla:Town04:4way_Stop": [197.35690307617188, -192.64134216308594],
#     "carla:Town04:Merging": [92.00399780273438, 5.6601104736328125],
#     "carla:Town04:Parking": [288.290771484375, -212.30825805664062],
#     "carla:Town06:4way_Large": [2.7990493774414062, -17.995826721191406],
#     "carla:Town06:Merge_Double": [-142.56692504882812, 58.08183670043945],
#     "carla:Town06:Merge_Single": [515.815185546875, 29.682167053222656],
#     "carla:Town07:3way": [-95.72276306152344, 25.259380340576172],
#     "carla:Town07:4way": [-101.99176025390625, 40.31178283691406],
#     "carla:Town10HD:3way_Protected": [-32.242706298828125, 87.99059295654297],
#     "carla:Town10HD:3way_Stop": [28.333023071289062, 62.63806915283203],
#     "carla:Town10HD:4way": [-33.782386779785156, 26.179428100585938],
map = "carla:Town01"
map_center = [254.2249755859375, 300.88372802734375]  # "carla:Town01:Straight":
map = "carla:Town02"
map_center = [41.620059967041016, 191.54580688476562]
map = "carla:Town03"
map_center = [187.63462829589844, -2.8370285034179688]  # "carla:Town03:3way_Protected":
# map = "carla:Town04"
# map_center = [288.290771484375, -212.30825805664062]  # "carla:Town04:Parking":
# map = "carla:Town06"
# map_center = [515.815185546875, 29.682167053222656]  # "carla:Town06:Merge_Single":
# map = "carla:Town07"
# map_center = [-95.72276306152344, 25.259380340576172]  # "carla:Town07:3way":
plt.ion()
sim_len = 100
parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--location", type=str,
                    # default="carla:Town01")
                    default=map)
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
# newax = fig.add_axes(ax.get_position(), frameon=True)

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
    location_of_interest=map_center,
)

corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                        state.orientation, state.speed]) for state in response.agent_states]
# scene_plotter = iai.utils.ScenePlotter(open_drive=True)
# scene_plotter.plot_scene(corrected_agents,
#                          response.agent_attributes,
#                          #  traffic_light_states=traffic_light,
#                          numbers=True, velocity_vec=False, direction_vec=True, ax=newax)
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
    corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                             state.orientation, state.speed]) for state in response.agent_states]

    # scene_plotter.plot_scene(corrected_agents,
    #                          agent_attributes,
    #                          #  traffic_light_states=traffic_light,
    #                          numbers=True, velocity_vec=False, direction_vec=True, ax=newax)
    # pass
    # ax.clear()
    p1 = draw_map(ax, open_drive_file_name)
    for agent in response.agent_states:
        ax.plot(agent.center.x, -agent.center.y,
                color='green', marker='o', markersize=8)
    plt.savefig(f'./img/img_{i}.png',
                transparent=False,
                facecolor='white'
                )
    plt.pause(0.00001)
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
