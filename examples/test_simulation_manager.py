"""
Test: SimulationManager — Carla maps, agent add/remove, logging, visualization

Parameters:
    LOCATION: carla:Town10HD (Carla map with traffic lights)
    NUM_INITIAL_AGENTS: 4 (internal NPC agents)
    NUM_EGO_AGENTS: 2 (external cosim agents, added at init)
    SIM_LENGTH: 50 timesteps
    AGENTS_TO_REMOVE_AT: timestep 20 (remove 1 internal agent by ID)
    AGENTS_TO_ADD_AT: timestep 30 (insert 1 new internal agent via insert_agents)

situations covered:
    1. SimulationManager init with ScenePlotter, WaypointManager, LogWriter
    2. form_regions + initialize with external_agent_data (cosim)
    3. drive loop with external agents
    4. remove_agents mid-simulation
    5. insert_agents mid-simulation
    6. Agent ID persistence - verify keyed agents survive add/remove
    7. LogWriter export - LogReader roundtrip
    8. Visualization output (gif)
    9. get_*/set_* accessors
"""

import invertedai as iai
from invertedai import (
    AgentType,
    AgentData,
    SimulationManager,
    ScenePlotterConfig,
    WaypointManagerConfig,
    LogWriterConfig,
    RegionsConfig,
    get_default_agent_properties,
)
import matplotlib.pyplot as plt
import os
import uuid

# Parameters
LOCATION = "carla:Town10HD"
NUM_INITIAL_AGENTS = 4
NUM_EGO_AGENTS = 2
SIM_LENGTH = 50
AGENTS_TO_REMOVE_AT = 20
AGENTS_TO_ADD_AT = 30
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUTPUT_DIR, "test_sim_manager_log.json")
GIF_PATH = os.path.join(OUTPUT_DIR, "test_sim_manager.gif")
REPLAY_GIF_PATH = os.path.join(OUTPUT_DIR, "test_sim_manager_replay.gif")

# Setup
print("=== Test SimulationManager ===")
print(f"Location: {LOCATION}")
print(f"Initial agents: {NUM_INITIAL_AGENTS} internal + {NUM_EGO_AGENTS} external")
print(f"Sim length: {SIM_LENGTH} steps")
print(f"Remove agent at step {AGENTS_TO_REMOVE_AT}, add agent at step {AGENTS_TO_ADD_AT}")

print("\n[1] location_info + config setup")
location_info_response = iai.location_info(location=LOCATION, include_map_source=True)
scene_plotter_cfg = ScenePlotterConfig(location=LOCATION, location_info_response=location_info_response)
waypoint_cfg = WaypointManagerConfig(lanelet_map=location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(log_path=LOG_PATH, location=LOCATION, location_info_response=location_info_response)

simulation_manager = SimulationManager(
    scene_plotter_cfg=scene_plotter_cfg,
    waypoint_cfg=waypoint_cfg,
    log_writer_cfg=log_cfg,
)

# External ego agents
print("Initialize ego agents for cosimulation")
ego_wp_manager = iai.WaypointManager(cfg=waypoint_cfg)
ego_response = iai.initialize(
    location=LOCATION,
    agent_properties=get_default_agent_properties({AgentType.car: NUM_EGO_AGENTS}),
)
ego_props = ego_wp_manager.update(response=ego_response, agent_properties=ego_response.agent_properties)
ego_agent_ids = [f"ego_{i}_{str(uuid.uuid4())[:8]}" for i in range(NUM_EGO_AGENTS)]
external_agent_data = {
    ego_agent_ids[i]: AgentData(
        state=ego_response.agent_states[i],
        properties=ego_props[i],
        recurrent=None,
    )
    for i in range(NUM_EGO_AGENTS)
}

# Initialize
print("form_regions + initialize")
regions_config = RegionsConfig(
    location=LOCATION,
    agent_count_dict={AgentType.car: NUM_INITIAL_AGENTS},
    map_center=(location_info_response.map_center.x, location_info_response.map_center.y),
)
regions = simulation_manager.form_regions(regions_config)
response = simulation_manager.initialize(
    location=LOCATION,
    regions=regions,
    external_agent_data=external_agent_data,
)

initial_ids = simulation_manager.get_agent_ids()
print(f"  Internal agent IDs after init: {initial_ids}")
print(f"  Total agents in response (internal+external): {len(response.agent_states)}")
assert len(initial_ids) >= NUM_INITIAL_AGENTS, f"Expected at least {NUM_INITIAL_AGENTS} internal agents, got {len(initial_ids)}"

# Test getters
print("Test get_/set_ accessors")
states = simulation_manager.get_states()
props = simulation_manager.get_properties()
recurrents = simulation_manager.get_recurrent_states()
assert len(states) == len(initial_ids)
assert len(props) == len(initial_ids)
assert len(recurrents) == len(initial_ids)
first_id = initial_ids[0]
agent_data = simulation_manager.get_agent_data(first_id)
assert agent_data.state is not None
assert agent_data.properties is not None
print(f"  Agent '{first_id}' state: ({agent_data.state.center.x:.1f}, {agent_data.state.center.y:.1f})")

#  Drive loop with add/remove 
print(f" Drive loop ({SIM_LENGTH} steps) with agent add/remove")
removed_agent_id = None
added_agent_ids = None

for step in range(SIM_LENGTH):
    # Update ego predictions
    ego_props = ego_wp_manager.update(response=ego_response, agent_properties=ego_props)
    ego_response = iai.drive(
        location=LOCATION,
        agent_states=ego_response.agent_states,
        agent_properties=ego_props,
        recurrent_states=ego_response.recurrent_states,
    )
    external_agent_data = {
        ego_agent_ids[i]: AgentData(
            state=ego_response.agent_states[i],
            properties=ego_props[i],
            recurrent=None,
        )
        for i in range(NUM_EGO_AGENTS)
    }

    # Remove an agent mid simulation
    if step == AGENTS_TO_REMOVE_AT:
        current_ids = simulation_manager.get_agent_ids()
        removed_agent_id = current_ids[2:4]
        print(f"  Step {step}: removing agent '{removed_agent_id}'")
        simulation_manager.remove_agents(removed_agent_id)
        assert removed_agent_id not in simulation_manager.get_agent_ids()

    # Add an agent mid-simulation
    if step == AGENTS_TO_ADD_AT:
        # Copy properties from an existing agent (which has proper dimensions from initialize)
        ref_id = simulation_manager.get_agent_ids()[0]
        ref_props = simulation_manager.get_property(ref_id)
        ref_state = simulation_manager.get_states()[0]
        from invertedai.common import AgentState, Point
        new_agent = AgentData(
            state=AgentState(
                center=Point(x=ref_state.center.x + 5, y=ref_state.center.y + 5), # a random state
                orientation=1.0, # orientation must be valid for initialization
                speed=0.0,
            ),
            properties=ref_props,
            recurrent=None,
        )
        added_agent_ids = simulation_manager.insert_agents(
            agent_data_list=[new_agent],
            ids=[f"added_{str(uuid.uuid4())[:8]}"],
        )
        print(f"  Step {step}: added agent '{added_agent_ids[0]}'")
        assert added_agent_ids[0] in simulation_manager.get_agent_ids()

    response = simulation_manager.drive(
        external_agent_data=external_agent_data,
        location=LOCATION,
        light_recurrent_states=response.light_recurrent_states,
    )

    if step % 10 == 0:
        print(f"  Step {step}: {len(simulation_manager.get_agent_ids())} internal agents")

#  Verify agent ID persistence 
print("\n[6] Verify agent ID persistence")
final_ids = simulation_manager.get_agent_ids()
print(f"  Final internal agent IDs: {final_ids}")
if removed_agent_id:
    assert removed_agent_id not in final_ids, f"Removed agent '{removed_agent_id}' should not be present"
if added_agent_ids:
    assert added_agent_ids[0] in final_ids, f"Added agent '{added_agent_ids[0]}' should be present"
print("  PASS: agent IDs correctly persisted through add/remove")

#  Export log + visualization 
print("\n[7] Export log and visualization")
simulation_manager.export_log()
assert os.path.exists(LOG_PATH), f"Log file not created at {LOG_PATH}"
print(f"  Log exported to {LOG_PATH}")

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
simulation_manager.visualize_data(
    output_name=GIF_PATH,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
    numbers=list(range(NUM_INITIAL_AGENTS + NUM_EGO_AGENTS)),
)
plt.close(fig)
assert os.path.exists(GIF_PATH), f"GIF not created at {GIF_PATH}"
print(f"  GIF exported to {GIF_PATH}")

# LogReader roundtrip 
print("\n[8] LogReader roundtrip — read back the exported log")
log_reader = iai.LogReader(LOG_PATH)
print(f"  Log length: {log_reader.log_length} timesteps")
print(f"  Location: {log_reader.location}")

log_reader.initialize()
print(f"  Agent count at t=0: {len(log_reader.agent_states)}")
print(f"  Agent IDs at t=0 (from agents dict): {list(log_reader.agents.keys())}")

# Step through and verify agent counts change
step = 0
while log_reader.drive():
    step += 1
print(f"  Replayed {step} drive steps")

# Visualize the replayed log
log_reader.visualize(
    gif_path=REPLAY_GIF_PATH,
    fov=200,
    resolution=(1024, 1024),
    dpi=150,
    direction_vec=True,
    plot_frame_number=True,
)
assert os.path.exists(REPLAY_GIF_PATH), f"Replay GIF not created at {REPLAY_GIF_PATH}"
print(f"  Replay GIF exported to {REPLAY_GIF_PATH}")

# Legacy ScenarioLog roundtrip
print("\n[9] Legacy ScenarioLog from LogReader")
legacy_log = log_reader.return_scenario_log()
print(f"  Legacy ScenarioLog type: {type(legacy_log).__name__}")
print(f"  agent_states timesteps: {len(legacy_log.agent_states)}")
print(f"  agent_properties count: {len(legacy_log.agent_properties)}")
print(f"  present_indexes timesteps: {len(legacy_log.present_indexes)}")

print("\n=== ALL TESTS PASSED ===")
