"""
Test: Scenario replay and ego takeover flows

Parameters:
    LOG_PATH: ./assets/carla_Town10HD_example_emergency_scenario.json (pre-existing scenario)
    NUM_BACKGROUND_AGENTS: 2 (spawned around the scenario)
    SIM_LENGTH: 50 timesteps
    TAKEOVER_TIMESTEP: 15 (ego takes over from log at this step)
    EGO_ID: 0 (index of ego agent in log agents)
    FOV: 100 (visualization field of view)

Tests covered:
    1. LogReader reads pre-existing scenario log
    2. LogReader.initialize() + step through with drive()
    3. LogReader.agents property returns SimulationAgentDict
    4. large_initialize with log agents as conditional agents
    5. Ego takeover: replace log agent states with DRIVE predictions at takeover_timestep
    6. LogWriter records the combined simulation
    7. Verify log export and visualization
"""

import invertedai as iai
from invertedai import AgentType, Region, Point

import os
import time
from tqdm import tqdm

# ─── Parameters ───────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "carla_Town10HD_example_emergency_scenario.json")
NUM_BACKGROUND_AGENTS = 2
SIM_LENGTH = 50
TAKEOVER_TIMESTEP = 15
EGO_ID = 0
FOV = 100
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_GIF = os.path.join(OUTPUT_DIR, "test_scenario_replay.gif")
OUTPUT_LOG = os.path.join(OUTPUT_DIR, "test_scenario_replay.json")

# ─── Read scenario log ───────────────────────────────────────────────────────
print("=== Test Scenario Replay + Ego Takeover ===")
print(f"Log: {LOG_PATH}")
print(f"Background agents: {NUM_BACKGROUND_AGENTS}, Sim length: {SIM_LENGTH}")
print(f"Takeover at step {TAKEOVER_TIMESTEP}, ego index {EGO_ID}")

print("\n[1] LogReader — read scenario log")
assert os.path.exists(LOG_PATH), f"Scenario log not found at {LOG_PATH}"
log_reader = iai.LogReader(log_path=LOG_PATH)
location = log_reader.location
print(f"  Location: {location}")
print(f"  Log length: {log_reader.log_length} timesteps")

# ─── Verify LogReader properties ──────────────────────────────────────────────
print("\n[2] LogReader.initialize() + verify properties")
log_reader.initialize()
print(f"  Agent count at t=0: {len(log_reader.agent_states)}")
print(f"  Agent properties count: {len(log_reader.agent_properties)}")
print(f"  Traffic lights present: {log_reader.traffic_lights_states is not None}")

# Test the agents dict property
agents_dict = log_reader.agents
print(f"  Agent IDs from .agents: {list(agents_dict.keys())}")
for aid, data in agents_dict.items():
    assert data.state is not None, f"Agent {aid} has no state"
    assert data.properties is not None, f"Agent {aid} has no properties"
print("  PASS: all agents have state and properties")

NUM_LOG_AGENTS = len(log_reader.agent_properties)
LOG_LENGTH = log_reader.log_length

# ─── Location info ────────────────────────────────────────────────────────────
print("\n[3] location_info + large_initialize with log agents")
location_info_response = iai.location_info(
    location=location,
    rendering_fov=FOV,
    include_map_source=True,
)
scenario_center = (location_info_response.map_center.x, location_info_response.map_center.y)

if NUM_BACKGROUND_AGENTS > 0:
    regions = iai.get_regions_default(
        location=location,
        agent_count_dict={AgentType.car: NUM_BACKGROUND_AGENTS},
        area_shape=(FOV // 2, FOV // 2),
        map_center=scenario_center,
    )
else:
    regions = [
        Region.create_square_region(
            center=Point(x=scenario_center[0], y=scenario_center[1]),
            size=FOV,
        )
    ]

response = iai.large_initialize(
    location=location,
    regions=regions,
    agent_properties=log_reader.agent_properties,
    agent_states=log_reader.agent_states,
    traffic_light_state_history=[log_reader.traffic_lights_states] if log_reader.traffic_lights_states is not None else None,
)
total_num_agents = len(response.agent_states)
print(f"  Total agents after init: {total_num_agents} ({NUM_LOG_AGENTS} from log + background)")

# ─── Setup waypoints + log writer ────────────────────────────────────────────
print("\n[4] Setup WaypointManager + LogWriter")
wp_manager = iai.WaypointManager(
    cfg=iai.WaypointManagerConfig(
        lanelet_map=location_info_response.get_lanelet_map(),
        fail_soft=True,
    )
)
response.agent_properties = wp_manager.update(
    response=response,
    agent_properties=response.agent_properties,
)

log_writer = iai.LogWriter()
log_writer.initialize(
    location=location,
    location_info_response=location_info_response,
    init_response=response,
)

# ─── Drive loop with ego takeover ─────────────────────────────────────────────
print(f"\n[5] Drive loop ({SIM_LENGTH} steps) with ego takeover at step {TAKEOVER_TIMESTEP}")
agent_properties = response.agent_properties
takeover_happened = False

for ts in tqdm(range(SIM_LENGTH), desc="Simulating"):
    is_log_tl = log_reader.traffic_lights_states is not None and ts < LOG_LENGTH
    response = iai.large_drive(
        location=location,
        agent_states=response.agent_states,
        agent_properties=agent_properties,
        recurrent_states=response.recurrent_states,
        traffic_lights_states=log_reader.traffic_lights_states if is_log_tl else None,
        light_recurrent_states=response.light_recurrent_states if not is_log_tl else None,
    )
    agent_properties = wp_manager.update(
        response=response,
        agent_properties=agent_properties,
    )

    # Replay log agents, with ego takeover
    if ts < LOG_LENGTH:
        log_reader.drive()
        agent_states = response.agent_states
        agent_states[:NUM_LOG_AGENTS] = log_reader.agent_states
        if ts >= TAKEOVER_TIMESTEP:
            agent_states[EGO_ID] = response.agent_states[EGO_ID]
            if not takeover_happened:
                print(f"\n  Ego takeover at step {ts}")
                takeover_happened = True
        response.agent_states = agent_states

    log_writer.drive(
        drive_response=response,
        agent_properties=agent_properties,
    )

assert takeover_happened, "Ego takeover should have happened"

# ─── Export and verify ────────────────────────────────────────────────────────
print(f"\n[6] Export log and visualization")
log_writer.export_to_file(log_path=OUTPUT_LOG)
assert os.path.exists(OUTPUT_LOG)
print(f"  Log exported to {OUTPUT_LOG}")

log_writer.visualize(
    gif_path=OUTPUT_GIF,
    fov=FOV,
    resolution=(1024, 1024),
    dpi=150,
    direction_vec=True,
    plot_frame_number=True,
    map_center=scenario_center,
    left_hand_coordinates=location.split(":")[0] == "carla",
)
assert os.path.exists(OUTPUT_GIF)
print(f"  GIF exported to {OUTPUT_GIF}")

# ─── Read back and verify ────────────────────────────────────────────────────
print(f"\n[7] Verify exported log via LogReader")
verify_reader = iai.LogReader(OUTPUT_LOG)
print(f"  Log length: {verify_reader.log_length}")
assert verify_reader.log_length == SIM_LENGTH + 1  # init + SIM_LENGTH drive steps
print("  PASS: log length matches")

print("\n=== ALL TESTS PASSED ===")
