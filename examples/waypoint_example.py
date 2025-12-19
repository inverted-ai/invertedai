import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import matplotlib.pyplot as plt
import os
import time

location = "carla_xodr:Town10HD"  # select one of available locations
simulation_length = 300
seed = int(time.time())
drive_model = "nBu1"
num_agents = 10
fov = 250

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey('<INSERT_KEY_HERE>')  # specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(
    location=location, 
    include_map_source=True,
    rendering_fov=fov
)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_properties=get_default_agent_properties({AgentType.car:num_agents}),  # number of NPCs to spawn
    random_seed=seed
)

wp_manager = iai.WaypointManager(
    location_info_response = location_info_response,
    cfg = iai.WaypointManagerConfig(
        random_seed=seed,
        fail_soft=True
    )
)
agent_properties = wp_manager.update(
    response = response,
    agent_properties = response.agent_properties,
)

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    map_image = rendered_static_map,
    fov = fov,
    xy_offset = (location_info_response.map_center.x, location_info_response.map_center.y),
    static_actors = location_info_response.static_actors,
    resolution = (2048,2048),
    left_hand_coordinates = location.split(":")[0] == "carla"
)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties,
)

print("Begin stepping through simulation.")
for _ in range(simulation_length):  # how many simulation steps to execute (10 steps is 1 second)
    response = iai.drive(
        location=location,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        light_recurrent_states=response.light_recurrent_states,
        random_seed=seed,
        api_model_version=drive_model
    )
    agent_properties = wp_manager.update(
        response = response,
        agent_properties = agent_properties,
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
    numbers = list(range(num_agents))
)
print("Done")