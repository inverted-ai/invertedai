import invertedai as iai
from typing import List
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType, AgentState, RecurrentState
from invertedai.keyed_agent import AgentData, KeyedAgents
import matplotlib.pyplot as plt
import os


location = "canada:drake_street_and_pacific_blvd"
num_agents_to_add = 4 # number of agents initialized
agent_to_remove = "agent_1"

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

agents = KeyedAgents(num_agents=num_agents_to_add)
#keyed_initialize calls initialize/large_intialize under the hood and returns InitializeResponse
response = agents.initialize(location=location)

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)

scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=response.agent_properties,
)

print("Begin stepping through simulation.")
for step in range(100):
    # pop agent 1 at step 30 and reinsert at step 60 
    if step == 30:
        # save the agent data for later
        saved_agent_data = agents.remove_agent(agent_to_remove)
        print(f"Removed {agent_to_remove} at step {step}")

    if step == 60:
        # add into agents dict with key "agent_x"
        agents.add_agent("ego", saved_agent_data)
        print(f"Added agent_x at step {step}")

    # calls drive/large_drive under the hood and returns DriveResponse
    response = agents.drive(location=location, light_recurrent_states=response.light_recurrent_states)

    scene_plotter.record_step(
        agents.get_states(),
        traffic_light_states=response.traffic_lights_states,
        agent_properties=agents.get_properties(),
    )

print("Simulation finished, save visualization.")

fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
scene_plotter.animate_scene(
    output_name="keyed_minimal_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
)

print("Done")