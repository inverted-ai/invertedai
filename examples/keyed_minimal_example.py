import invertedai as iai
from typing import List
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType, AgentState, RecurrentState
from invertedai.keyed_agent import AgentData, KeyedAgents
import matplotlib.pyplot as plt
import os


location = "canada:drake_street_and_pacific_blvd"
num_agents_to_add = 4 # number of agents initialized

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

#create keyed agents dict with default naming "agent_i"
agents = {}
for i in range(num_agents_to_add):
    agent_properties = AgentData(
        properties=get_default_agent_properties({AgentType.car:1})[0]
    )
    agents[f"agent_{i}"] = agent_properties

#keyed_initialize calls initialize/large_intialize under the hood and returns updated agents with states
agents, response = KeyedAgents.keyed_initialize(agents, location=location)

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
state_history: List[List[AgentState]] = []
recurr: RecurrentState
for step in range(100):
    # pop agent 1 at step 50 and re-add as "ego" at step 100
    if step == 30:
        # save the states for addition later
        recurr = agents["agent_1"].recurrent 
        state = agents["agent_1"].state
        prop = agents["agent_1"].properties
        agents.pop("agent_1")
        print(f"Removed agent_1 at step {step}")

    if step == 60:
        # add into agents dict with key "ego"
        agents["agent_x"] = AgentData(
            state=state,
            properties=prop,
            recurrent = recurr,
        )
        print(f"Added agent_x at step {step}")

    # keyed_drive calls drive/large_drive under the hood and returns updated agents with states
    agents, response = KeyedAgents.keyed_drive(
        agents,
        location=location,
        light_recurrent_states=response.light_recurrent_states,
    )

    scene_plotter.record_step(
        [a.state for a in agents.values()],
        traffic_light_states=response.traffic_lights_states,
        agent_properties=[a.properties for a in agents.values()],
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