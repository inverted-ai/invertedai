import invertedai as iai
from invertedai.large.common import Region
from invertedai.simulation_manager import AgentData, SimulationAgentDict, SimulationManager
from invertedai.common import AgentProperties, AgentState, AgentType, Point
from invertedai.utils import ScenePlotterConfig
from invertedai.helpers.waypoints import WaypointManagerConfig
from invertedai.logs.logger import LogWriterConfig
import matplotlib.pyplot as plt
import os


LOCATION = "carla:Town10HD"
NUM_AGENTS = 5
 # number of agents initialized
SIM_LENGTH=150 # number of timesteps

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

print("Begin initialization.")
location_info_response = iai.location_info(
    location=LOCATION, 
    include_map_source=True
)
scene_plotter_cfg = ScenePlotterConfig(
    location=LOCATION, 
    location_info_response=location_info_response
)
waypoint_cfg = WaypointManagerConfig(lanelet_map = location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(
    log_path="keyed_minimal_example_log.json",
    location=LOCATION, 
    location_info_response=location_info_response
)
simulation_manager = SimulationManager(
    scene_plotter_cfg=scene_plotter_cfg, 
    waypoint_cfg=waypoint_cfg, 
    log_writer_cfg=log_cfg
)
regions = iai.get_regions_default(
    agent_count_dict = {AgentType.car: NUM_AGENTS}, 
    location = LOCATION, 
    map_center=tuple([location_info_response.map_center.x, location_info_response.map_center.y])
)
external_agent_data = {
    # please use the Scenario Builder tool to check the validity of the agent states
    "new": AgentData(
        state=AgentState(
            center=Point(x=-45.19154717515613, y=46.50373005906251),
            orientation=4.7,
            speed=1.0
        ),
        properties=AgentProperties(
            agent_type=AgentType.car,
            length=4.0,
            width=2.02,
            rear_axis_offset=1.784,
        )
    )
}
response = simulation_manager.initialize(
    location=LOCATION, 
    regions=regions, 
    external_agent_data=external_agent_data
)
print("initialized agents with ids ", simulation_manager.get_agent_ids())
rendered_static_map = location_info_response.birdview_image.decode()

print("Begin stepping through simulation.")
for step in range(SIM_LENGTH):
    response = simulation_manager.drive(
        location=LOCATION, 
        light_recurrent_states=response.light_recurrent_states
    )

print("Simulation finished, save visualization.")

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
simulation_manager.visualize_data(
    output_name="simulation_manager_external_agent_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers = list(range(NUM_AGENTS))
)
print("Simulation finished, save to json log.")
simulation_manager.export_log()
print("Done")