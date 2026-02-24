import invertedai as iai
from invertedai.simulation_manager import SimulationManager
from invertedai.utils import ScenePlotterConfig
from invertedai.helpers.waypoints import WaypointManagerConfig
from invertedai.logs.logger import LogWriterConfig
import matplotlib.pyplot as plt
import os


location = "carla:Town10HD" 
num_agents_to_add = 4 # number of agents initialized
agent_to_remove = "agent_1"

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
location_info_response = iai.location_info(location=location, include_map_source=True)
scene_plotter_cfg = ScenePlotterConfig(location=location, location_info_response=location_info_response)
waypoint_cfg = WaypointManagerConfig(lanelet_map = location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(log_path="keyed_minimal_example_log.json",location=location, location_info_response=location_info_response)
agents = SimulationManager(num_agents=num_agents_to_add, scene_plotter_cfg=scene_plotter_cfg, waypoint_cfg=waypoint_cfg, log_writer_cfg=log_cfg)
print("initialized agents with ids ", agents.get_agent_ids())
response = agents.initialize(location=location)
rendered_static_map = location_info_response.birdview_image.decode()

print("Begin stepping through simulation.")
for step in range(150):
    response = agents.drive(location=location, light_recurrent_states=response.light_recurrent_states)

print("Simulation finished, save visualization.")

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
agents.animate_scene( # we can try making this a flag in the AgentDataManager class? this way it can animate under the hood?
    output_name="keyed_minimal_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers = list(range(num_agents_to_add))
)
print("Simulation finished, save to json log.")
agents.export_log()
print("Done")