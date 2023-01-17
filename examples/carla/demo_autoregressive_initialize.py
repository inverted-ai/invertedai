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
parser.add_argument("--location", type=str,  default="carla:Town03")
parser.add_argument("-l", "--episode_length", type=int, default=30)
args = parser.parse_args()
map_center = PreSets.map_centers["carla:Town03:Roundabout"]


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
    location=args.location, agent_density=15, traffic_lights_states=None, map_center=(0, 0), width=500, height=500, stride=50, initialize_fov=100, static_actors=corrected_static_actors)


# Carla xord map coordinate differ from eachother and position of agents require a simple transformation
corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
                                         -state.orientation, state.speed]) for state in response.agent_states]
open_drive_file_name = f"{path}/data/open_drive/{args.location.split(':')[1]}.csv"
scene_plotter = iai.utils.ScenePlotter(
    fov=args.fov, xy_offset=(map_center[0], -map_center[1]), static_actors=corrected_static_actors, open_drive=open_drive_file_name)

scene_plotter.plot_scene(corrected_agents,
                         agent_attributes=response.agent_attributes,
                         #  traffic_light_states = traffic_light,
                         numbers=False, velocity_vec=False, direction_vec=True)
plt.show(block=True)
"""
scene_plotter.initialize_recording(corrected_agents, agent_attributes=response.agent_attributes)

agent_attributes = response.agent_attributes
frames = []
x_frame = []
for i in tqdm(range(args.episode_length)):
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

gif_name = 'new_iai-drive-side-road-green.gif'
scene_plotter.animate_scene(output_name=gif_name,
                            numbers=False, direction_vec=True, velocity_vec=False,
                            plot_frame_number=True)
"""
