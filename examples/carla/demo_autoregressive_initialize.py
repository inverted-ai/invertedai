from tqdm import tqdm
import argparse
import invertedai as iai
from carla_simulator import PreSets
from invertedai.common import AgentState, StaticMapActor
import pathlib
import matplotlib.pyplot as plt
path = pathlib.Path(__file__).parent.resolve()


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--fov", type=float, default=500)
parser.add_argument("--location", type=str,  default="carla:Town10HD")
parser.add_argument("-l", "--episode_length", type=int, default=300)
args = parser.parse_args()
map_center = PreSets.map_centers[args.location]


if args.api_key is not None:
    iai.add_apikey(args.api_key)

location_info_response = iai.location_info(location=args.location)
file_name = args.location.replace(":", "_")
if location_info_response.osm_map is not None:
    file_path = f"{file_name}.osm"
    with open(file_path, "w") as f:
        location_info_response.osm_map.save_osm_file(file_path)
if location_info_response.birdview_image is not None:
    file_path = f"{file_name}.jpg"
    location_info_response.birdview_image.decode_and_save(file_path)

rendered_static_map = location_info_response.birdview_image.decode()
corrected_static_actors = [StaticMapActor.fromdict(dict(x=state.center.x, y=-state.center.y, actor_id=state.actor_id,
                                                        agent_type=state.agent_type, orientation=-state.orientation, length=state.length, width=state.width, dependant=state.dependant
                                                        ))for state in location_info_response.static_actors]


light_response = iai.light(location=args.location)
response = iai.utils.area_initialization(
    location=args.location, agent_density=6, traffic_lights_states=None, map_center=map_center, width=500, height=500, stride=50, initialize_fov=100, static_actors=corrected_static_actors)


# Carla simulator uses left-hand coordinates and xord map are right-handed, thus, the position of agents require a simple transformation
corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                         -state.orientation, state.speed]) for state in response.agent_states]
open_drive_file_name = f"{path}/data/open_drive/{args.location.split(':')[1]}.csv"
scene_plotter = iai.utils.ScenePlotter(
    fov=args.fov, xy_offset=(map_center[0], -map_center[1]), static_actors=corrected_static_actors, open_drive=open_drive_file_name)

agent_attributes = response.agent_attributes
scene_plotter.initialize_recording(corrected_agents, agent_attributes=agent_attributes)

frames = []
x_frame = []
for i in tqdm(range(args.episode_length), desc=f"Driving {args.location.split(':')[1]}"):
    light_response = iai.light(
        location=args.location, recurrent_states=light_response.recurrent_states)
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        location=args.location,
        traffic_lights_states=light_response.traffic_lights_states,
    )

    corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                             (-state.orientation), state.speed]) for state in response.agent_states]
    scene_plotter.record_step(corrected_agents, traffic_light_states=light_response.traffic_lights_states)

gif_name = f'iai-driving-on-{args.location}.gif'
ani = scene_plotter.animate_scene(output_name=gif_name,
                                  numbers=False, direction_vec=False, velocity_vec=False,
                                  plot_frame_number=False)


ani.save(f'iai-driving-on-{args.location.split(":")[1]}.mp4')
plt.show(block=True)
