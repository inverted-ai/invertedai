import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.helpers.waypoints import get_default_waypoints
from invertedai.common import AgentType, Point

import matplotlib.pyplot as plt
import os
import time

location = "carla:Town10HD"  # select one of available locations
simulation_length = 100
waypoint_threshold = 5.0
seed = int(time.time())

destination_waypoints = [None,None,Point(x=100.0, y=0.0)]
min_distances = [None,100.0,None]
num_example_agents = len(destination_waypoints)

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey('<INSERT_KEY_HERE>')  # specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location, include_map_source=True)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_properties=get_default_agent_properties({AgentType.car:10}),  # number of NPCs to spawn
    random_seed=seed
)
agent_properties = response.agent_properties  # get dimension and other attributes of NPCs

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors,
    resolution = (2048,2048),
    left_hand_coordinates = location.split(":")[0] == "carla"
)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties,
)

waypoints_list = get_default_waypoints(
    location_info_response = location_info_response,
    agent_states = response.agent_states[0:num_example_agents],
    destination_waypoints = destination_waypoints,
    min_distances = min_distances
)

idx = [0 for _ in range(num_example_agents)] # starting index of the waypoint
waypoints_to_show = 1 # how many waypoints to show to the agent
print("Begin stepping through simulation.")
for _ in range(simulation_length):  # how many simulation steps to execute (10 steps is 1 second)

    for i, wps in enumerate(waypoints_list):
        agent_x, agent_y = response.agent_states[i].center.x, response.agent_states[i].center.y
        if idx[i] < len(wps) and (wps[idx[i]].x - agent_x) ** 2 + (wps[idx[i]].y - agent_y) ** 2 < waypoint_threshold:
            idx[i] += 1 # if within 5m of the waypoint, show the next one
    for i, wps in enumerate(waypoints_list):
        agent_properties[i].waypoints = wps[idx[i]:idx[i]+waypoints_to_show]

    response = iai.drive(
        location=location,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        light_recurrent_states=response.light_recurrent_states,
        random_seed=seed
    )
    
    # save the visualization
    scene_plotter.record_step(
        agent_states=response.agent_states,
        agent_properties=agent_properties, #This is important to capture the new waypoints every time step
        traffic_light_states=response.traffic_lights_states
    )

print("Simulation finished, save visualization.")
# save the visualization to disk
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
gif_name = f'{seed}_waypoint_example.gif'
scene_plotter.animate_scene(
    output_name=gif_name,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers = list(range(num_example_agents))
)
print("Done")