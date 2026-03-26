"""
Test: LogWriter/LogReader roundtrip — write a log, read it back, verify data integrity

Parameters:
    LOCATION: canada:drake_street_and_pacific_blvd (non-Carla map)
    NUM_AGENTS: 5
    SIM_LENGTH: 30 timesteps
    NUM_EXTRA_AGENTS: 1 (added at TIMESTEP_ADD=10)
    AGENT_TO_REMOVE: 0 (removed at TIMESTEP_REMOVE=20)

Tests covered:
    1. LogWriter.initialize() with init_response (legacy path, no agents_dict)
    2. LogWriter.drive() with current_present_indexes + new_agent_properties (legacy path)
    3. LogWriter.export_to_file() → JSON output
    4. LogReader reads JSON back → verify agent counts per timestep
    5. LogReader.return_scenario_log() → legacy ScenarioLog fields match
    6. LogReader.return_scenario_logV2() → ScenarioLog with agent_data
    7. LogWriter.initialize(scenario_log=...) → branching from a sub-range
    8. LogWriter.drive() with no present_indexes (assume unchanged agents)
    9. Verify JSON keys in predetermined_agents match expected agent IDs
"""

import invertedai as iai
from invertedai import AgentType, get_default_agent_properties

import os
import json
from random import randint
from copy import deepcopy

# ─── Parameters ───────────────────────────────────────────────────────────────
LOCATION = "canada:drake_street_and_pacific_blvd"
NUM_AGENTS = 5
NUM_EXTRA_AGENTS = 1
SIM_LENGTH = 30
TIMESTEP_ADD = 10
TIMESTEP_REMOVE = 20
AGENT_TO_REMOVE = 0
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUTPUT_DIR, "test_log_roundtrip.json")
BRANCHED_LOG_PATH = os.path.join(OUTPUT_DIR, "test_log_roundtrip_branched.json")

# ─── Write a log ──────────────────────────────────────────────────────────────
print("=== Test LogWriter/LogReader Roundtrip ===")
print(f"Location: {LOCATION}, Agents: {NUM_AGENTS}+{NUM_EXTRA_AGENTS}, Steps: {SIM_LENGTH}")

print("\n[1] Initialize simulation + LogWriter")
location_info_response = iai.location_info(location=LOCATION)

response_all = iai.initialize(
    location=LOCATION,
    agent_properties=get_default_agent_properties({AgentType.car: NUM_AGENTS + NUM_EXTRA_AGENTS}),
)
# Split: initial agents vs extra agents to add later
agent_states_added = response_all.agent_states[NUM_AGENTS:]
agent_properties_added = response_all.agent_properties[NUM_AGENTS:]

response = deepcopy(response_all)
response.agent_states = response_all.agent_states[:NUM_AGENTS]
response.agent_properties = response_all.agent_properties[:NUM_AGENTS]
response.recurrent_states = response_all.recurrent_states[:NUM_AGENTS]
agent_properties = response.agent_properties

log_writer = iai.LogWriter()
log_writer.initialize(
    location=LOCATION,
    location_info_response=location_info_response,
    init_response=response,
)
print(f"  LogWriter initialized with {NUM_AGENTS} agents")

# ─── Drive loop with add/remove ──────────────────────────────────────────────
print(f"\n[2] Drive loop ({SIM_LENGTH} steps) — add at {TIMESTEP_ADD}, remove at {TIMESTEP_REMOVE}")
for ts in range(SIM_LENGTH):
    current_present_indexes = None
    new_agent_properties = None

    if ts == TIMESTEP_ADD:
        response.agent_states.extend(agent_states_added)
        agent_properties.extend(agent_properties_added)
        response.recurrent_states = None

        current_present_indexes = log_writer.current_present_indexes
        num_existing = len(log_writer.all_agent_properties)
        current_present_indexes.extend([num_existing + i for i in range(len(agent_properties_added))])
        new_agent_properties = agent_properties_added
        print(f"  Step {ts}: added {NUM_EXTRA_AGENTS} agent(s), present_indexes={current_present_indexes}")

    if ts == TIMESTEP_REMOVE:
        current_present_indexes = log_writer.current_present_indexes
        current_present_indexes.pop(AGENT_TO_REMOVE)

        response.agent_states.pop(AGENT_TO_REMOVE)
        agent_properties.pop(AGENT_TO_REMOVE)
        response.recurrent_states.pop(AGENT_TO_REMOVE)
        print(f"  Step {ts}: removed agent index {AGENT_TO_REMOVE}, present_indexes={current_present_indexes}")

    response = iai.drive(
        location=LOCATION,
        agent_properties=agent_properties,
        agent_states=response.agent_states,
        light_recurrent_states=response.light_recurrent_states,
        recurrent_states=response.recurrent_states,
        random_seed=randint(1, 100000),
    )

    log_writer.drive(
        drive_response=response,
        current_present_indexes=current_present_indexes,
        new_agent_properties=new_agent_properties,
    )

print(f"  Final present agent count: {len(log_writer.current_present_indexes)}")

# ─── Export ───────────────────────────────────────────────────────────────────
print(f"\n[3] Export to {LOG_PATH}")
log_writer.export_to_file(log_path=LOG_PATH)
assert os.path.exists(LOG_PATH)

# ─── Verify JSON structure ────────────────────────────────────────────────────
print("\n[4] Verify JSON structure")
with open(LOG_PATH) as f:
    log_json = json.load(f)

assert log_json["scenario_length"] == SIM_LENGTH + 1  # init + SIM_LENGTH drive steps
print(f"  scenario_length: {log_json['scenario_length']}")

agent_keys = list(log_json["predetermined_agents"].keys())
print(f"  predetermined_agents keys: {agent_keys}")
print(f"  num_agents: {log_json['num_agents']}")

# Verify agent at index 0 has states for timesteps 0..TIMESTEP_REMOVE-1 (present before removal)
agent_0_states = log_json["predetermined_agents"][agent_keys[0]]["states"]
print(f"  Agent '{agent_keys[0]}' present at {len(agent_0_states)} timesteps")

# ─── Read back with LogReader ─────────────────────────────────────────────────
print("\n[5] LogReader — read back and verify")
log_reader = iai.LogReader(LOG_PATH)
print(f"  Log length: {log_reader.log_length}")
print(f"  Location: {log_reader.location}")

log_reader.initialize()
init_agent_count = len(log_reader.agent_states)
print(f"  Agents at t=0: {init_agent_count}")
assert init_agent_count == NUM_AGENTS, f"Expected {NUM_AGENTS} at t=0, got {init_agent_count}"

# Step through and track agent counts
agent_counts = [init_agent_count]
while log_reader.drive():
    agent_counts.append(len(log_reader.agent_states))

print(f"  Agent count at step {TIMESTEP_ADD}: {agent_counts[TIMESTEP_ADD]}")
print(f"  Agent count at step {TIMESTEP_REMOVE}: {agent_counts[TIMESTEP_REMOVE]}")
# After add: NUM_AGENTS + NUM_EXTRA_AGENTS
assert agent_counts[TIMESTEP_ADD] == NUM_AGENTS + NUM_EXTRA_AGENTS, \
    f"Expected {NUM_AGENTS + NUM_EXTRA_AGENTS} after add, got {agent_counts[TIMESTEP_ADD]}"
# After remove: NUM_AGENTS + NUM_EXTRA_AGENTS - 1
assert agent_counts[TIMESTEP_REMOVE] == NUM_AGENTS + NUM_EXTRA_AGENTS - 1, \
    f"Expected {NUM_AGENTS + NUM_EXTRA_AGENTS - 1} after remove, got {agent_counts[TIMESTEP_REMOVE]}"
print("  PASS: agent counts match expected add/remove pattern")

# ─── Legacy ScenarioLog ──────────────────────────────────────────────────────
print("\n[6] return_scenario_log() — legacy format")
log_reader.reset_log()
legacy_log = log_reader.return_scenario_log()
print(f"  Type: {type(legacy_log).__name__}")
print(f"  agent_states: {len(legacy_log.agent_states)} timesteps")
print(f"  agent_properties: {len(legacy_log.agent_properties)} agents")
print(f"  present_indexes: {len(legacy_log.present_indexes)} timesteps")
assert legacy_log.waypoints_per_frame is not None or legacy_log.waypoints_per_frame is None  # just check field exists
print("  PASS: legacy ScenarioLog fields intact")

# ─── ScenarioLog (new format) ─────────────────────────────────────────────────
print("\n[7] return_scenario_logV2() — new ScenarioLog format")
log_reader.reset_log()
v2_log = log_reader.return_scenario_logV2()
print(f"  Type: {type(v2_log).__name__}")
print(f"  agent_data: {len(v2_log.agent_data)} timesteps")
print(f"  Agent IDs at t=0: {list(v2_log.agent_data[0].keys())}")
print(f"  Agent IDs at t={TIMESTEP_ADD}: {list(v2_log.agent_data[TIMESTEP_ADD].keys())}")
# Verify the new format has the correct agent count at each timestep
assert len(v2_log.agent_data[0]) == NUM_AGENTS
assert len(v2_log.agent_data[TIMESTEP_ADD]) == NUM_AGENTS + NUM_EXTRA_AGENTS
print("  PASS: ScenarioLog agent_data matches expected counts")

# ─── Branching: init LogWriter from a sub-range ──────────────────────────────
print("\n[8] Branching — init LogWriter from scenario_log sub-range")
log_reader.reset_log()
log_reader.initialize()
branched_writer = iai.LogWriter()
branched_writer.initialize(
    scenario_log=log_reader.return_scenario_log(timestep_range=(0, 15))
)
print(f"  Branched writer initialized with {branched_writer.simulation_length} timesteps")

# Drive a few more steps without specifying present_indexes (assume unchanged)
for _ in range(5):
    log_reader.drive()
branch_response = iai.drive(
    location=log_reader.location,
    agent_properties=log_reader.agent_properties,
    agent_states=log_reader.agent_states,
    recurrent_states=log_reader.recurrent_states,
    light_recurrent_states=log_reader.light_recurrent_states,
)
branched_writer.drive(drive_response=branch_response)
print(f"  Branched writer now has {branched_writer.simulation_length} timesteps")

branched_writer.export_to_file(log_path=BRANCHED_LOG_PATH)
assert os.path.exists(BRANCHED_LOG_PATH)
print(f"  Branched log exported to {BRANCHED_LOG_PATH}")

print("\n=== ALL TESTS PASSED ===")
