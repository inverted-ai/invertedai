import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType
from invertedai.keyed_agent import AgentData, AgentWrapperManager
import matplotlib.pyplot as plt
import os


location = "canada:drake_street_and_pacific_blvd"

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

def make_agent_id(i: int) -> str:
    return f"car_{i}"

#create keyed agents 
agents = {}
for i in range(10):
    agent_properties = AgentData(
        properties=get_default_agent_properties({AgentType.car:1})[0]
    )
    agents[make_agent_id(i)] = agent_properties
#keyed_initialize calls initialize/large_intialize under the hood and returns updated agents with states
agents, response = AgentWrapperManager.keyed_initialize(
    agents,
    location=location,
)

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

    if step == 30:
        agents.pop("agent_9")

    # if step == 50:
    #     agents["agent_100"] = AgentData(
    #         properties=get_default_agent_properties({AgentType.car:1})[0]
    #     )

    # keyed_drive calls drive/large_drive under the hood and returns updated agents with states
    agents, response = AgentWrapperManager.keyed_drive( # light recurrent states is not here!!!!
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