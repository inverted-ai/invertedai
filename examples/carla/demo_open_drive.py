from tqdm import tqdm
import argparse
import invertedai as iai
from carla_simulator import PreSets
from invertedai.common import AgentState, StaticMapActor
import pathlib
import subprocess
import tempfile
path = pathlib.Path(__file__).parent.resolve()


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--fov", type=float, default=150)
parser.add_argument("--location", type=str, default="carla:Town03")
parser.add_argument("-l", "--episode_length", type=int, default=30)
parser.add_argument("-xodr", "--opendrive_map_path", type=str, default="data/open_drive/Town03.xodr")
parser.add_argument("-mc", "--map_center", type=str, default=None)
parser.add_argument("-lhc", "--left_hand_coordinate", type=int, default=1)
args = parser.parse_args()

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

if args.map_center:
    map_center = PreSets.map_centers["carla:Town03:Roundabout"]
else:
    map_center = (location_info_response.map_center.x, location_info_response.map_center.y)


rendered_static_map = location_info_response.birdview_image.decode()

try:
    light_response = iai.light(location=args.location)
except:
    light_response = None

response = iai.initialize(
    location=args.location,
    agent_count=10,
    get_birdview=True,
    get_infractions=True,
    traffic_light_state_history=[light_response.traffic_lights_states] if light_response else None,
    location_of_interest=map_center,
)
if args.left_hand_coordinate:
    # Carla xord map coordinate differ from eachother and position of agents require a simple transformation
    corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                            -state.orientation, state.speed]) for state in response.agent_states]
    corrected_static_actors = [StaticMapActor.fromdict(dict(x=state.center.x, y=-state.center.y, actor_id=state.actor_id,
                                                            agent_type=state.agent_type, orientation=-state.orientation,
                                                            length=state.length, width=state.width, dependant=state.dependant
                                                            ))for state in location_info_response.static_actors]
else:
    corrected_agents = response.agent_states
    corrected_static_actors = location_info_response.static_actors

# Convert xord map to csv for rendering
result = subprocess.run(['./data/open_drive/odrplot', args.opendrive_map_path], stdout=subprocess.PIPE)

# open_drive_file_name = f"{path}/data/open_drive/{args.location.split(':')[1]}.csv"
open_drive_file_name = f"track.csv"
scene_plotter = iai.utils.ScenePlotter(
    fov=args.fov, xy_offset=(map_center[0], -map_center[1]), static_actors=corrected_static_actors, open_drive=open_drive_file_name)
scene_plotter.initialize_recording(corrected_agents, agent_attributes=response.agent_attributes)

agent_attributes = response.agent_attributes
frames = []
x_frame = []
for i in tqdm(range(args.episode_length)):
    try:
        light_response = iai.light(
        location=args.location, recurrent_states=light_response.recurrent_states)
    except:
        light_response = None
    response = iai.drive(
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        location=args.location,
        traffic_lights_states=light_response.traffic_lights_states if light_response else None,
    )

    if args.left_hand_coordinate:
        corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                             (-state.orientation), state.speed]) for state in response.agent_states]
    else:
        corrected_agents = response.agent_states
    scene_plotter.record_step(corrected_agents, traffic_light_states=light_response.traffic_lights_states if light_response else None)

gif_name = 'new_iai-drive-side-road-green.gif'
scene_plotter.animate_scene(output_name=gif_name,
                            numbers=False, direction_vec=True, velocity_vec=False,
                            plot_frame_number=True)
