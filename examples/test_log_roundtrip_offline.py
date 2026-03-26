"""
Offline test: LogWriter/LogReader roundtrip - no API calls required.

Situations covered:
    1. LogWriter.initialize() with agents_dict (new path)
    2. LogWriter.drive() with agents_dict
    3. LogWriter.export_to_file() -> JSON output
    4. LogReader reads JSON back -> verify agent counts per timestep
    5. LogReader.return_scenario_log() -> legacy ScenarioLog fields
    6. LogReader.return_scenario_logV2() -> ScenarioLog with agent_data
    7. Agent add/remove across timesteps
    8. Branching from a sub-range
"""

import os
import sys
import json
import tempfile
from copy import deepcopy
from unittest.mock import patch, MagicMock
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from invertedai.common import (
    AgentData,
    AgentProperties,
    AgentState,
    Point,
    RecurrentState,
    SimulationAgentDict,
    RECURRENT_SIZE,
)
from invertedai.logs.logger import LogWriter, LogReader, ScenarioLog
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse
from invertedai.api.location import LocationResponse, Image

# Helper to create mock agents
def make_agent(x, y, orientation=0.0, speed=1.0, length=4.5, width=2.0, rear_axis_offset=1.5, agent_type="car"):
    return AgentData(
        state=AgentState(center=Point(x=x, y=y), orientation=orientation, speed=speed),
        properties=AgentProperties(length=length, width=width, rear_axis_offset=rear_axis_offset, agent_type=agent_type),
        recurrent=RecurrentState(packed=[0.0] * RECURRENT_SIZE),
    )

def make_mock_location_response():
    return LocationResponse(
        version="1.0",
        max_agent_number=100,
        bounding_polygon=None,
        birdview_image=Image(encoded_image=""),
        osm_map=None,
        map_center=Point(x=0.0, y=0.0),
        map_fov=200,
        static_actors=[],
    )

print("=== Offline Test: LogWriter/LogReader Roundtrip ===")

# Test 1: LogWriter.initialize() with agents_dict
print("\n[1] LogWriter.initialize() with agents_dict")
agents_dict: SimulationAgentDict = {
    "agent_A": make_agent(10.0, 20.0),
    "agent_B": make_agent(30.0, 40.0),
    "agent_C": make_agent(50.0, 60.0),
}
location = "test:offline_location"
location_info_response = make_mock_location_response()

log_writer = LogWriter()
log_writer.initialize(
    location=location,
    location_info_response=location_info_response,
    agents_dict=agents_dict,
    # Provide a minimal InitializeResponse for traffic lights etc.
    init_response=InitializeResponse(
        agent_states=[d.state for d in agents_dict.values()],
        recurrent_states=[d.recurrent for d in agents_dict.values()],
        agent_attributes=[None] * len(agents_dict),
        agent_properties=[d.properties for d in agents_dict.values()],
        birdview=None,
        infractions=None,
        traffic_lights_states=None,
        light_recurrent_states=None,
        api_model_version="test",
    ),
)
assert log_writer.simulation_length == 1
print("  PASS: LogWriter initialized with 3 agents via agents_dict")

# Test 2: LogWriter.drive() with agents_dict — simulate 10 steps
print("\n[2] LogWriter.drive() — 10 steps with agent add at step 5, remove at step 8")
SIM_LENGTH = 10
ADD_STEP = 5
REMOVE_STEP = 8

current_agents = deepcopy(agents_dict)
for step in range(SIM_LENGTH):
    # Move agents slightly
    for aid, data in current_agents.items():
        data.state = AgentState(
            center=Point(x=data.state.center.x + 0.5, y=data.state.center.y + 0.3),
            orientation=data.state.orientation,
            speed=data.state.speed,
        )

    if step == ADD_STEP:
        current_agents["agent_D"] = make_agent(70.0, 80.0, speed=2.0)
        print(f"  Step {step}: added agent_D (now {len(current_agents)} agents)")

    if step == REMOVE_STEP:
        del current_agents["agent_A"]
        print(f"  Step {step}: removed agent_A (now {len(current_agents)} agents)")

    drive_response = DriveResponse(
        agent_states=[d.state for d in current_agents.values()],
        recurrent_states=[d.recurrent for d in current_agents.values()],
        birdview=None,
        infractions=None,
        is_inside_supported_area=[True] * len(current_agents),
        traffic_lights_states=None,
        light_recurrent_states=None,
        api_model_version="test",
    )
    log_writer.drive(
        drive_response=drive_response,
        agents_dict=deepcopy(current_agents),
    )

assert log_writer.simulation_length == SIM_LENGTH + 1  # init + drive steps
print(f"  PASS: {SIM_LENGTH} drive steps recorded (total length: {log_writer.simulation_length})")

# Test 3: Export to file
print("\n[3] Export to JSON")
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=os.path.dirname(__file__)) as tmp:
    log_path = tmp.name

log_writer.export_to_file(log_path=log_path)
assert os.path.exists(log_path)
print(f"  Exported to {log_path}")

# Test 4: Verify JSON structure
print("\n[4] Verify JSON structure")
with open(log_path) as f:
    log_json = json.load(f)

assert log_json["scenario_length"] == SIM_LENGTH + 1
print(f"  scenario_length: {log_json['scenario_length']}")

agent_keys = list(log_json["predetermined_agents"].keys())
print(f"  predetermined_agents keys: {agent_keys}")
assert "agent_A" in agent_keys
assert "agent_B" in agent_keys
assert "agent_C" in agent_keys
assert "agent_D" in agent_keys
print("  PASS: all agent keys present in JSON")

# Verify agent_A has states for timesteps 0 through REMOVE_STEP (before removal)
agent_A_states = log_json["predetermined_agents"]["agent_A"]["states"]
print(f"  agent_A present at {len(agent_A_states)} timesteps (expected {REMOVE_STEP + 1})")
assert len(agent_A_states) == REMOVE_STEP + 1

# Verify agent_D appears at step ADD_STEP+1 (drive step ADD_STEP = timestep ADD_STEP+1 in log)
agent_D_states = log_json["predetermined_agents"]["agent_D"]["states"]
print(f"  agent_D present at {len(agent_D_states)} timesteps")
assert len(agent_D_states) == SIM_LENGTH - ADD_STEP
print("  PASS: agent state presence matches add/remove pattern")

# Test 5: LogReader reads it back
print("\n[5] LogReader — read back and verify")
with patch("invertedai.logs.logger.location_info", return_value=location_info_response):
    log_reader = LogReader(log_path)

print(f"  Log length: {log_reader.log_length}")
print(f"  Location: {log_reader.location}")
assert log_reader.location == location
assert log_reader.log_length == SIM_LENGTH + 1

log_reader.initialize()
init_agent_count = len(log_reader.agent_states)
print(f"  Agents at t=0: {init_agent_count}")
assert init_agent_count == 3  # agent_A, B, C

# Step through and track agent counts
agent_counts = [init_agent_count]
while log_reader.drive():
    agent_counts.append(len(log_reader.agent_states))

print(f"  Agent counts over time: {agent_counts}")
# After add at step 5 (timestep 6 in log): should be 4
assert agent_counts[ADD_STEP + 1] == 4, f"Expected 4 after add, got {agent_counts[ADD_STEP + 1]}"
# After remove at step 8 (timestep 9 in log): should be 3
assert agent_counts[REMOVE_STEP + 1] == 3, f"Expected 3 after remove, got {agent_counts[REMOVE_STEP + 1]}"
print("  PASS: agent counts match expected add/remove pattern")

# Test the .agents property
log_reader.reset_log()
log_reader.initialize()
agents = log_reader.agents
assert isinstance(agents, dict)
assert "agent_A" in agents
assert agents["agent_A"].state is not None
assert agents["agent_A"].properties is not None
print("  PASS: .agents property returns correct dict with state and properties")

# Test 6: Legacy ScenarioLog
print("\n[6] return_scenario_log() — legacy format")
log_reader.reset_log()
legacy_log = log_reader.return_scenario_log()
print(f"  Type: {type(legacy_log).__name__}")
print(f"  agent_states: {len(legacy_log.agent_states)} timesteps")
print(f"  agent_properties: {len(legacy_log.agent_properties)} agents")
print(f"  present_indexes: {len(legacy_log.present_indexes)} timesteps")
assert len(legacy_log.agent_states) == SIM_LENGTH + 1
assert len(legacy_log.present_indexes) == SIM_LENGTH + 1
print("  PASS: legacy ScenarioLog fields intact")

# Test 7: New format ScenarioLog
print("\n[7] return_scenario_logV2() — new format")
log_reader.reset_log()
v2_log = log_reader.return_scenario_logV2()
print(f"  Type: {type(v2_log).__name__}")
print(f"  agent_data: {len(v2_log.agent_data)} timesteps")
assert len(v2_log.agent_data[0]) == 3  # A, B, C at t=0
assert len(v2_log.agent_data[ADD_STEP + 1]) == 4  # A, B, C, D after add
print("  PASS: ScenarioLog agent_data matches expected counts")

# Test 8: Branching from sub-range
print("\n[8] Branching — return_scenario_log with timestep_range")
log_reader.reset_log()
sub_log = log_reader.return_scenario_log(timestep_range=(0, 5))
print(f"  Sub-log agent_states: {len(sub_log.agent_states)} timesteps")
assert len(sub_log.agent_states) == 5
print("  PASS: sub-range log has correct length")

# Cleanup
os.remove(log_path)
print(f"\n  Cleaned up {log_path}")

print("\n=== ALL TESTS PASSED ===")
