import invertedai as iai
from typing import List
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType, AgentState, RecurrentState, Point
from invertedai.keyed_agent import AgentData, KeyedAgents
from invertedai.offroad import OffroadManager
import matplotlib.pyplot as plt
import os
from driving_models.database.interface import MasterVideoProcessingDatabaseConfig, PostgreSQLVideoProcessingDatabaseWithOrchestration
from driving_models.database.data_models import Location

master_db_cfg = MasterVideoProcessingDatabaseConfig()
db = PostgreSQLVideoProcessingDatabaseWithOrchestration(master_db_cfg)
with db:
    locations = db.session.query(Location).all()
    for location in locations:
          latest_maps_per_version = location.latest_maps_per_version
          latest_v00_map = latest_maps_per_version.get((0,1))

location = "aus:the_entrance_road_and_archbold_road_australia"
num_agents_to_add = 4 # number of agents initialized
agent_to_remove = "agent_1"

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")

location_info_response = iai.location_info(location=location, include_map_source=True) # must include_map_source=True to get lanelet maps

agents = KeyedAgents(num_agents=num_agents_to_add)

response = iai.initialize(location=location, keyed_agents= agents)

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
offroad_manager = OffroadManager( # initialize offroad manager with respawn set to True
    location_info_response=location_info_response,
    respawn=True
)
print("Begin stepping through simulation.")
for step in range(200):
    response = iai.drive(
        location=location, 
        light_recurrent_states=response.light_recurrent_states, 
        keyed_agents=agents,
        offroad_manager_config=offroad_manager
    )
    scene_plotter.record_step(
        agents.get_states(),
        traffic_light_states=response.traffic_lights_states,
        agent_properties=agents.get_properties(),
    )

print("Simulation finished, save visualization.")

fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
scene_plotter.animate_scene(
    output_name="offroad_removal_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
)

print("Done")