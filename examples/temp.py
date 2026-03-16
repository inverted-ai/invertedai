import invertedai as iai
from invertedai.large.common import Region
from invertedai.simulation_manager import AgentData, SimulationAgentDict, SimulationManager, build_agent_dict_from_lists
from invertedai.common import AgentProperties, AgentState, AgentType, Point
from invertedai.utils import ScenePlotterConfig, get_default_agent_properties
from invertedai.helpers.waypoints import WaypointManagerConfig
from invertedai.logs.logger import LogWriterConfig
import matplotlib.pyplot as plt
import os
import uuid


LOCATION = "carla:Town10HD"
NUM_AGENTS = 5
SIM_LENGTH = 150
NUM_EGO_AGENTS = 1
NUM_CONDITIONAL_AGENTS = 2

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

##########################################################################################################
# INSERT YOUR OWN EGO PREDICTIONS FOR THE INITIALIZATION
ego_response = iai.initialize(
    location=LOCATION,
    agent_properties=get_default_agent_properties({AgentType.car: NUM_CONDITIONAL_AGENTS}),
)
ego_agent_properties = ego_response.agent_properties[:NUM_EGO_AGENTS]
ego_agent_states = ego_response.agent_states[:NUM_EGO_AGENTS]
ego_recurrent_states = ego_response.recurrent_states[:NUM_EGO_AGENTS]

predefined_agent_properties = ego_response.agent_properties[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]
predefined_agent_states = ego_response.agent_states[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]
predefined_recurrent_states = ego_response.recurrent_states[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]
##########################################################################################################

print("Begin initialization.")
location_info_response = iai.location_info(location=LOCATION, include_map_source=True)

scene_plotter_cfg = ScenePlotterConfig(location=LOCATION, location_info_response=location_info_response)
waypoint_cfg = WaypointManagerConfig(lanelet_map=location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(
    log_path="cosimulation_manager_example_log.json",
    location=LOCATION,
    location_info_response=location_info_response
)

simulation_manager = SimulationManager(
    scene_plotter_cfg=scene_plotter_cfg,
    waypoint_cfg=waypoint_cfg,
    log_writer_cfg=log_cfg
)

regions = iai.get_regions_default(
    agent_count_dict={AgentType.car: NUM_AGENTS},
    location=LOCATION,
    map_center=(location_info_response.map_center.x, location_info_response.map_center.y)
)

# ✅ Assign stable IDs for conditional agents ONCE — reused every step
conditional_agent_ids = [str(uuid.uuid4()) for _ in range(NUM_CONDITIONAL_AGENTS)]
ego_agent_ids = conditional_agent_ids[:NUM_EGO_AGENTS]
predefined_agent_ids = conditional_agent_ids[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]

# Build external_agent_data with explicit stable IDs
external_agent_data = {
    conditional_agent_ids[i]: AgentData(
        state=ego_response.agent_states[i],
        properties=(ego_agent_properties + predefined_agent_properties)[i],
        recurrent=ego_response.recurrent_states[i],
    )
    for i in range(NUM_CONDITIONAL_AGENTS)
}

response = simulation_manager.initialize(
    location=LOCATION,
    regions=regions,
    external_agent_data=external_agent_data,
    traffic_light_state_history=[ego_response.traffic_lights_states],
)
print("Initialized agents with ids:", simulation_manager.get_agent_ids())

print("Begin stepping through simulation.")
for step in range(SIM_LENGTH):
##########################################################################################################
    # INSERT YOUR OWN EGO PREDICTIONS FOR THIS TIME STEP
    ego_drive_response = iai.drive(
        location=LOCATION,
        agent_states=ego_agent_states + predefined_agent_states,
        agent_properties=ego_agent_properties + predefined_agent_properties,
        recurrent_states=ego_recurrent_states + predefined_recurrent_states,
        light_recurrent_states=response.light_recurrent_states,
    )
    ego_agent_states = ego_drive_response.agent_states[:NUM_EGO_AGENTS]
    ego_recurrent_states = ego_drive_response.recurrent_states[:NUM_EGO_AGENTS]
    predefined_agent_states = ego_drive_response.agent_states[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]
    predefined_recurrent_states = ego_drive_response.recurrent_states[NUM_EGO_AGENTS:NUM_CONDITIONAL_AGENTS]
##########################################################################################################

    # ✅ Overwrite conditional agents using their stable IDs — no new UUIDs generated
    external_agent_data = {
        conditional_agent_ids[i]: AgentData(
            state=ego_drive_response.agent_states[i],
            properties=(ego_agent_properties + predefined_agent_properties)[i],
            recurrent=ego_drive_response.recurrent_states[i],
        )
        for i in range(NUM_CONDITIONAL_AGENTS)
    }

    response = simulation_manager.drive(
        external_agent_data=external_agent_data,
        location=LOCATION,
        light_recurrent_states=response.light_recurrent_states,
    )

print("Simulation finished, saving visualization.")
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
simulation_manager.visualize_data(
    output_name="cosimulation_manager_example.gif",
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers=list(range(NUM_AGENTS + NUM_CONDITIONAL_AGENTS))
)

print("Saving JSON log.")
simulation_manager.export_log()
print("Done")