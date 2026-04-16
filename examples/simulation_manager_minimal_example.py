import invertedai as iai
from invertedai import AgentType
from invertedai import WaypointManagerConfig
from invertedai import SimulationManager
from invertedai import SceneVisualizerConfig
from invertedai import LogWriterConfig
from invertedai import RegionsConfig
import matplotlib.pyplot as plt
import os


LOCATION = "carla:Town10HD"
NUM_AGENTS = 4 # number of agents initialized
SIM_LENGTH=150 # number of timesteps

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
location_info_response = iai.location_info(location=LOCATION, include_map_source=True)
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
scene_viz_cfg = SceneVisualizerConfig(
    left_hand_coordinates=LOCATION.split(":")[0] == "carla",
    plot_frame_number=True,
    location=LOCATION,
    fov=location_info_response.map_fov,
    ax=ax,
)
waypoint_cfg = WaypointManagerConfig(lanelet_map = location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(log_path="simulation_manager_minimal_example_log.json",location=LOCATION, location_info_response=location_info_response)
simulation_manager = SimulationManager(location_info_response=location_info_response, scene_visualizer_cfg=scene_viz_cfg, waypoint_cfg=waypoint_cfg, log_writer_cfg=log_cfg)
regions_config = RegionsConfig(location=LOCATION, agent_count_dict={AgentType.car: NUM_AGENTS})
regions = simulation_manager.form_regions(regions_config)
response = simulation_manager.initialize(location=LOCATION, regions=regions)
print("initialized agents with ids ", simulation_manager.get_agent_ids())
rendered_static_map = location_info_response.birdview_image.decode()

print("Begin stepping through simulation.")
for step in range(SIM_LENGTH):
    response = simulation_manager.drive(location=LOCATION, light_recurrent_states=response.light_recurrent_states)

print("Simulation finished, save visualization.")

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
simulation_manager.visualize_data(output_name="simulation_manager_minimal_example.mp4")
print("Simulation finished, save to json log.")
simulation_manager.export_log()
print("Done")
