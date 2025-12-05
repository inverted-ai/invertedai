import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.helpers.waypoints import generate_waypoints_from_lane_ids, generate_lane_ids_from_lanelet_map
from invertedai.common import AgentType

import matplotlib.pyplot as plt
import os

location = "canada:drake_street_and_pacific_blvd"  # select one of available locations

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
)
agent_properties = response.agent_properties  # get dimension and other attributes of NPCs

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties,
)

lanelet_map = location_info_response.get_lanelet_map()
lane_ids = generate_lane_ids_from_lanelet_map(response.agent_states[0], lanelet_map) # Sequence of lane ids starting from the current state of the agent
# lane_ids = generate_lane_ids_from_lanelet_map(response.agent_states[0], lanelet_map, waypoint=Point(x=0.0, y=0.0)) # Sequence of lane ids starting from the current state of the agent to the waypoint
waypoints = generate_waypoints_from_lane_ids(response.agent_states[0], lanelet_map, lane_ids, 15.0)
agent_properties[0].waypoints = waypoints # Set the first agent's waypoint explicitly

idx = 0 # starting index of the waypoint
waypoints_to_show = 1 # how many waypoints to show to the agent
print("Begin stepping through simulation.")
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)

    agent_properties[0].waypoints = waypoints[idx:idx+waypoints_to_show]
    response = iai.drive(
        location=location,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        light_recurrent_states=response.light_recurrent_states,
    )
    agent_x, agent_y = response.agent_states[0].center.x, response.agent_states[0].center.y
    if idx < len(waypoints) and (waypoints[idx].x - agent_x) ** 2 + (waypoints[idx].y - agent_y) ** 2 < 5.0:
        idx += 1 # if within 5m of the waypoint, show the next one

    # save the visualization
    scene_plotter.record_step(response.agent_states,response.traffic_lights_states)

print("Simulation finished, save visualization.")
# save the visualization to disk
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
xs = [p.x for p in waypoints]
ys = [p.y for p in waypoints]

# visualize the waypoints as a red circle
ax.scatter(
    xs, 
    ys,
    s=10,                
    facecolors='none', 
    edgecolors='red',   
    linewidth=2
)

gif_name = 'minimal_example.gif'
scene_plotter.animate_scene(
    output_name=gif_name,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
)
print("Done")