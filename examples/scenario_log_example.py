import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import os
import matplotlib.pyplot as plt

from random import randint
from copy import deepcopy

LOCATION = "canada:drake_street_and_pacific_blvd"  # select one of available locations
SIMULATION_LENGTH = 100
SIMULATION_LENGTH_EXTEND = 100
SIMULATION_BEGIN_NEW_ROLLOUT = 50

NUM_EXTRA_AGENTS = 1
NUM_INITIAL_AGENTS = 5
AGENT_TO_REMOVE = 0
TIMESTEP_ADD_AGENTS = 10
TIMESTEP_REMOVE_AGENTS = 40

######################################################################################
# Produce a log and write it
print("Producing log...")
location_info_response = iai.location_info(location=LOCATION)

# initialize the simulation by spawning NPCs
response_all = iai.initialize(
    location=LOCATION,  # select one of available locations
    agent_properties=get_default_agent_properties({AgentType.car:NUM_INITIAL_AGENTS+NUM_EXTRA_AGENTS}),    # number of NPCs to spawn
)
agent_states_added = response_all.agent_states[NUM_INITIAL_AGENTS:]
agent_properties_added = response_all.agent_properties[NUM_INITIAL_AGENTS:]
recurrent_states_added = response_all.recurrent_states[NUM_INITIAL_AGENTS:]
response = deepcopy(response_all)
response.agent_states = response_all.agent_states[:NUM_INITIAL_AGENTS]
response.agent_properties = response_all.agent_properties[:NUM_INITIAL_AGENTS]
response.recurrent_states = response_all.recurrent_states[:NUM_INITIAL_AGENTS]
agent_properties = response.agent_properties

log_writer = iai.LogWriter()
log_writer.initialize(
    location=LOCATION,
    location_info_response=location_info_response,
    init_response=response
)

print("Stepping through simulation...")
for ts in range(SIMULATION_LENGTH): 
    present_indexes = None
    new_agent_properties = None
    if ts == TIMESTEP_REMOVE_AGENTS:
        present_indexes = log_writer.get_present_agents()
        present_indexes.pop(AGENT_TO_REMOVE)
        
        response.agent_states.pop(AGENT_TO_REMOVE)
        agent_properties.pop(AGENT_TO_REMOVE)
        response.recurrent_states.pop(AGENT_TO_REMOVE)

    if ts == TIMESTEP_ADD_AGENTS:
        response.agent_states.extend(agent_states_added)
        agent_properties.extend(agent_properties_added)
        response.recurrent_states = None

        if present_indexes is None:
            present_indexes = log_writer.get_present_agents()

        num_agent_properties = len(log_writer.get_all_agent_properties())
        present_indexes.extend([num_agent_properties + i for i in range(len(agent_properties_added))])
        new_agent_properties = agent_properties_added

    response = iai.drive(
        location=LOCATION,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        light_recurrent_states = response.light_recurrent_states,
        recurrent_states=response.recurrent_states,
        random_seed=randint(1,100000)
    )

    log_writer.drive(
        drive_response=response,
        present_indexes=present_indexes,
        new_agent_properties=new_agent_properties
    )

log_path = os.path.join(os.getcwd(),f"scenario_log_example.json")
log_writer.export_to_file(log_path=log_path)
gif_path_original = os.path.join(os.getcwd(),f"scenario_log_example_original.gif")
log_writer.visualize(
    gif_path=gif_path_original,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = True,
    velocity_vec = False,
    plot_frame_number = True
)

######################################################################################
# Replay original log
print("Reading log...")

log_reader = iai.LogReader(log_path)
gif_path_replay = os.path.join(os.getcwd(),f"scenario_log_example_replay.gif")
log_reader.visualize(
    gif_path=gif_path_replay,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = True,
    velocity_vec = False,
    plot_frame_number = True
)

print("Extending read log...")

location_info_response_replay = log_reader.location_info_response
log_reader.initialize()
agent_properties = log_reader.agent_properties

rendered_static_map = location_info_response_replay.birdview_image.decode()
scene_plotter_new = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response_replay.map_fov,
    (location_info_response_replay.map_center.x, location_info_response_replay.map_center.y),
    location_info_response_replay.static_actors
)
scene_plotter_new.initialize_recording(
    agent_states=log_reader.agent_states,
    agent_properties=agent_properties
)

print("Stepping through simulation...")
while True: # Log reader will return False when it has run out of simulation data
    is_timestep_populated = log_reader.drive()
    if not is_timestep_populated:
        break
    agent_properties = log_reader.agent_properties
    scene_plotter_new.record_step(
        agent_states=log_reader.agent_states,
        traffic_light_states=log_reader.traffic_lights_states,
        agent_properties=agent_properties
    )

agent_states = log_reader.agent_states
recurrent_states = log_reader.recurrent_states
traffic_lights_states = log_reader.traffic_lights_states
light_recurrent_states = log_reader.light_recurrent_states
for _ in range(SIMULATION_LENGTH_EXTEND): 
    response = iai.drive(
        location=log_reader.location,
        agent_properties=agent_properties,
        agent_states=agent_states,
        recurrent_states=recurrent_states,
        light_recurrent_states=light_recurrent_states
    )

    agent_states = response.agent_states
    recurrent_states = response.recurrent_states
    traffic_lights_states = response.traffic_lights_states
    light_recurrent_states = response.light_recurrent_states

    scene_plotter_new.record_step(
        agent_states=agent_states,
        traffic_light_states=traffic_lights_states
    )

gif_path_extended = os.path.join(os.getcwd(),f"scenario_log_example_extended.gif")
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
plt.axis('off')
scene_plotter_new.animate_scene(
    output_name=gif_path_extended,
    ax=ax,
    direction_vec = True,
    velocity_vec = False,
    plot_frame_number = True
)

######################################################################################
# Re-read the log and choose an earlier timestep from which to branch off
print("Re-reading the log...")
log_reader.reset_log()
log_writer_branched = iai.LogWriter()

location_info_response_replay = log_reader.location_info_response
log_reader.initialize()
agent_properties = log_reader.agent_properties

rendered_static_map = location_info_response_replay.birdview_image.decode()
scene_plotter_branch = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response_replay.map_fov,
    (location_info_response_replay.map_center.x, location_info_response_replay.map_center.y),
    location_info_response_replay.static_actors
)
scene_plotter_branch.initialize_recording(
    agent_states=log_reader.agent_states,
    agent_properties=agent_properties
)
log_writer_branched.initialize(
    scenario_log=log_reader.return_scenario_log(
        timestep_range=(0,SIMULATION_LENGTH-SIMULATION_BEGIN_NEW_ROLLOUT)
    )
)

print("Stepping through simulation...")
for _ in range(SIMULATION_BEGIN_NEW_ROLLOUT):
    log_reader.drive()
    agent_properties = log_reader.agent_properties
    scene_plotter_branch.record_step(
        agent_states=log_reader.agent_states,
        traffic_light_states=log_reader.traffic_lights_states,
        agent_properties=agent_properties
    )

agent_states = log_reader.agent_states
recurrent_states = log_reader.recurrent_states
light_recurrent_states = log_reader.light_recurrent_states
for _ in range(SIMULATION_LENGTH-SIMULATION_BEGIN_NEW_ROLLOUT): 
    response = iai.drive(
        location=log_reader.location,
        agent_properties=agent_properties,
        agent_states=agent_states,
        recurrent_states=recurrent_states,
        light_recurrent_states=light_recurrent_states,
        random_seed=randint(1,100000)
    )

    log_writer_branched.drive(drive_response=response)

    agent_states = response.agent_states
    recurrent_states = response.recurrent_states
    light_recurrent_states = response.light_recurrent_states

    scene_plotter_branch.record_step(
        agent_states=agent_states,
        traffic_light_states=response.traffic_lights_states,
    )

log_path_branched = os.path.join(os.getcwd(),f"scenario_log_example_branched.json")
log_writer_branched.export_to_file(log_path=log_path_branched)
gif_path_branched = os.path.join(os.getcwd(),f"scenario_log_example_branched.gif")
log_writer_branched.visualize(
    gif_path=gif_path_branched,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = True,
    velocity_vec = False,
    plot_frame_number = True
)
