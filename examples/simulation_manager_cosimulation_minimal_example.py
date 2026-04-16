import invertedai as iai
from invertedai import (
    AgentType,
    AgentData,
)
from invertedai import (
    WaypointManager,
    WaypointManagerConfig,
)
from invertedai import SimulationManager
from invertedai import (
    SceneVisualizerConfig,
    get_default_agent_properties,
)
from invertedai import LogWriterConfig
from invertedai import RegionsConfig
from invertedai.utils import AgentTag
import matplotlib.pyplot as plt
import os
import uuid


LOCATION = "carla:Town10HD"
NUM_AGENTS = 5
SIM_LENGTH=150 # number of timesteps
NUM_EGO_AGENTS = 5

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")
print("Begin initialization.")
location_info_response = iai.location_info(
    location=LOCATION,
    include_map_source=True
)
ego_agent_ids = [f"ego_{i}" for i in range(NUM_EGO_AGENTS)]
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
scene_viz_cfg = SceneVisualizerConfig(
    location=LOCATION,
    left_hand_coordinates=LOCATION.split(":")[0] == "carla",
    direction_vec=False,
    velocity_vec=False,
    display_agent_ids=ego_agent_ids,
    ax=ax,
)
waypoint_cfg = WaypointManagerConfig(lanelet_map = location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(
    log_path="simulation_manager_cosimulation_example.json",
    location=LOCATION, 
    location_info_response=location_info_response
)
simulation_manager = SimulationManager(
    scene_visualizer_cfg=scene_viz_cfg,
    waypoint_cfg=waypoint_cfg,
    log_writer_cfg=log_cfg
)
##########################################################################################################
# INSERT YOUR OWN EGO PREDICTIONS FOR THE INITIALIZATION
ego_waypoint_manager = WaypointManager(cfg=waypoint_cfg)
ego_response = iai.initialize(
    location = LOCATION,
    agent_properties = get_default_agent_properties({AgentType.car:NUM_EGO_AGENTS}),
)
ego_props = ego_response.agent_properties
ego_props = ego_waypoint_manager.update(
    response=ego_response,
    agent_properties=ego_props
)
##########################################################################################################
regions_config = RegionsConfig(
    location=LOCATION,
    agent_count_dict={AgentType.car: NUM_AGENTS},
    map_center=(location_info_response.map_center.x, location_info_response.map_center.y),
)
regions = simulation_manager.form_regions(regions_config)
# set the AgentTag for agents using their ids
simulation_manager.agent_tags = {
    agent_id: AgentTag.ego for agent_id in ego_agent_ids
}
external_agent_data = {
    ego_agent_ids[i]: AgentData(
        state=ego_response.agent_states[i],
        properties=ego_props[i],
        recurrent=None,
    )
    for i in range(NUM_EGO_AGENTS)
}
response = simulation_manager.initialize(
    location=LOCATION,
    regions=regions,
    external_agent_data=external_agent_data
)

print("initialized agents with ids ", simulation_manager.get_agent_ids())
print("Begin stepping through simulation.")
for step in range(SIM_LENGTH):
##########################################################################################################    
    # INSERT YOUR OWN EGO PREDICTIONS FOR THIS TIME STEP
    ego_props = ego_waypoint_manager.update(
        response=ego_response,
        agent_properties=ego_props
    )
    ego_response= iai.drive(
        location=LOCATION,
        agent_states=ego_response.agent_states,
        agent_properties=ego_props,
        recurrent_states=ego_response.recurrent_states, 
    )
    external_agent_data = {
        ego_agent_ids[i]: AgentData(
            state=ego_response.agent_states[i],
            properties=ego_props[i],
            recurrent=None,  # recurrent is always zeroed for external agents in SimulationManager.drive()
        )
        for i in range(NUM_EGO_AGENTS)
    }
 ######################################################################################################
    response = simulation_manager.drive(
        external_agent_data=external_agent_data,
        location=LOCATION, 
        light_recurrent_states=response.light_recurrent_states
    )

print("Simulation finished, save visualization.")
simulation_manager.visualize_data(output_name="simulation_manager_cosimulation_example.mp4")
print("Simulation finished, save to json log.")
simulation_manager.export_log()
print("Done")
