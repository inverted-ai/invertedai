import invertedai as iai
from invertedai.utils import get_default_agent_properties, LogWriter, LogReader

import os
from random import randint
import matplotlib.pyplot as plt

LOCATION = "canada:drake_street_and_pacific_blvd"  # select one of available locations
SIMULATION_LENGTH = 100
SIMULATION_LENGTH_EXTEND = 100
SIMULATION_BEGIN_NEW_ROLLOUT = 50

######################################################################################
# Produce a log and write it
print("Producing log...")
location_info_response = iai.location_info(location=LOCATION)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=LOCATION,  # select one of available locations
    agent_properties=get_default_agent_properties({"car":5}),    # number of NPCs to spawn
)
agent_properties = response.agent_properties  # get dimension and other attributes of NPCs

log_writer = LogWriter()
log_writer.initialize(
    location=LOCATION,
    location_info_response=location_info_response,
    init_response=response
)

print("Stepping through simulation...")
for _ in range(SIMULATION_LENGTH): 
    # query the API for subsequent NPC predictions
    response = iai.drive(
        location=LOCATION,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        traffic_lights_states=response.traffic_lights_states,
        random_seed=randint(1,100000)
    )

    log_writer.drive(drive_response=response)


log_path = os.path.join(os.getcwd(),f"scenario_log_example.json")
log_writer.write_scenario_log_to_json(log_path=log_path)
gif_path_original = os.path.join(os.getcwd(),f"scenario_log_example_original.gif")
log_writer.visualize(
    gif_path=gif_path_original,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = False,
    velocity_vec = False,
    plot_frame_number = True
)

######################################################################################
# Replay original log
print("Reading log...")

log_reader = LogReader(log_path)
gif_path_replay = os.path.join(os.getcwd(),f"scenario_log_example_replay.gif")
log_reader.visualize(
    gif_path=gif_path_replay,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = False,
    velocity_vec = False,
    plot_frame_number = True
)

print("Extending read log...")

location_info_response_replay = log_reader.location_info()
response = log_reader.initialize()
agent_properties = response.agent_properties

rendered_static_map = location_info_response_replay.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response_replay.map_fov,
    (location_info_response_replay.map_center.x, location_info_response_replay.map_center.y),
    location_info_response_replay.static_actors
)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties
)

print("Stepping through simulation...")
while True: # Log reader will return None when it has run out of simulation data
    popped_response = log_reader.drive()
    if popped_response is None:
        break
    else:
        response = popped_response
    scene_plotter.record_step(response.agent_states,response.traffic_lights_states)
for _ in range(SIMULATION_LENGTH_EXTEND): 
    response = iai.drive(
        location=log_reader.scenario_log.location,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        traffic_lights_states=response.traffic_lights_states
    )

    scene_plotter.record_step(response.agent_states,response.traffic_lights_states)

gif_path_extended = os.path.join(os.getcwd(),f"scenario_log_example_extended.gif")
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
plt.axis('off')
scene_plotter.animate_scene(
    output_name=gif_path_extended,
    ax=ax,
    direction_vec = False,
    velocity_vec = False,
    plot_frame_number = True
)


######################################################################################
# Re-read the log and choose an earlier timestep from which to branch off
print("Re-reading the log...")
log_reader.reset_log()

location_info_response_replay = log_reader.location_info()
response = log_reader.initialize()
agent_properties = response.agent_properties

rendered_static_map = location_info_response_replay.birdview_image.decode()
scene_plotter_new = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response_replay.map_fov,
    (location_info_response_replay.map_center.x, location_info_response_replay.map_center.y),
    location_info_response_replay.static_actors
)
scene_plotter_new.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties
)

print("Stepping through simulation...")
for _ in range(SIMULATION_BEGIN_NEW_ROLLOUT):
    response = log_reader.drive()
    scene_plotter_new.record_step(response.agent_states,response.traffic_lights_states)
for _ in range(SIMULATION_LENGTH-SIMULATION_BEGIN_NEW_ROLLOUT): 
    response = iai.drive(
        location=log_reader.scenario_log.location,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        traffic_lights_states=response.traffic_lights_states,
        random_seed=randint(1,100000)
    )

    scene_plotter_new.record_step(response.agent_states,response.traffic_lights_states)

gif_path_branched = os.path.join(os.getcwd(),f"scenario_log_example_branched.gif")
fig_new, ax_new = plt.subplots(constrained_layout=True, figsize=(50, 50))
plt.axis('off')
scene_plotter_new.animate_scene(
    output_name=gif_path_branched,
    ax=ax_new,
    direction_vec = False,
    velocity_vec = False,
    plot_frame_number = True
)

