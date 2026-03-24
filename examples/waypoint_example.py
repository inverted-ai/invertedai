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

#Below shows the method for initializing the WaypointManager.
#The provided LocationInfoResponse object MUST contain the OSM map source by setting the include_map_source flag to True
#   when calling the location_info API.
#The WaypointManager can be configured in several ways:
#1. Pseudo-random seed can be used to make your simulations repeatable if all states remain the same.
#2. Setting the log level can adjust how warnings and messages are displayed and stored for you.
#3. The WaypointManager normally raises an exception if a path cannot be found for an agent. Instead, if the fail_soft parameter is set to True, the
#   the WaypointManager will ignore this exception and leave the agents waypoint list unchanged.
#NOTE: WaypointManager functions best on closed maps where all lanes are reachable. On other maps, this may impact computational performance.
wp_manager = iai.WaypointManager(
    location_info_response = location_info_response,
    cfg = iai.WaypointManagerConfig(
        random_seed=seed,
        fail_soft=False
    )
)

#The update() function must be called to fill in the AgentProperties of every agent. 
#In the most simple case, the update function only needs the InitializeResponse or DriveResponse object.
#The WaypointManager will update and return a list of waypoints in the AgentProperties in 3 different cases depending on the value of the waypoints field:
# 1. waypoints is None: The WaypointManager assumes the agent needs to be initialized with a waypoint route for 2 different cases:
# 1.a. target_paths is not defined: The WaypointManager finds an arbitrary route in the map resembling realistic traffic.
# 1.b. target_paths is defined: The WaypointManager generates a route between the given waypoints. The returned list of waypoints may contain secondary 
#       waypoints between the given points in the target path. If the target_paths field is defined, the list must be the same size as the given list of
#       of agents. If any other agents should not have a target path, set the respective index to a value of None.
# 2. waypoints is an empty list: This is the case where it has achieved its given path. By default, the WaypointManager will generate a new route.
# 3.    waypoints is a non-empty list: The WaypointManager checks if the current waypoint (index 0) has been reached and pops it from the list if so.
# The WaypointManager will generate waypoints based on the above algorithm by default unless specified otherwise by the agents_mask field. This mask,
#   if specified, must be the same length as the list of given agents. A value of False means the agent at that index will be ignored.
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

    #The update() function must be called every time step to ensure the AgentProperties are updated correctly. If the target_paths field is specified,
    #   it must be provided every time step until the target path is completed. At that point, providing the target path is optional. If a new route is
    #   desired, regardless of whether a new target path is provided or an arbitrary realistic path should be generated by the WaypointManager, the waypoints
    #   field must be set to None to reinitialize the waypoints list.
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