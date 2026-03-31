import invertedai as iai
from invertedai import AgentType
from invertedai import WaypointManagerConfig
from invertedai import SimulationManager
from invertedai import ScenePlotterConfig
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
scene_plotter_cfg = ScenePlotterConfig(location=LOCATION, location_info_response=location_info_response)
waypoint_cfg = WaypointManagerConfig(lanelet_map = location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(log_path="keyed_minimal_example_log.json",location=LOCATION, location_info_response=location_info_response)
simulation_manager = SimulationManager(scene_plotter_cfg=scene_plotter_cfg, waypoint_cfg=waypoint_cfg, log_writer_cfg=log_cfg)
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
simulation_manager.visualize_data(
    output_name="Simulation_manager_minimal_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers = list(range(NUM_AGENTS))
)
print("Simulation finished, save to json log.")
simulation_manager.export_log()
print("Done")