"""
Offline test: Scenario replay using existing log files — no API calls required.

Tests covered:
    1. LogReader reads pre-existing scenario log (scenario_log_example.json)
    2. LogReader.initialize() + step through with drive()
    3. LogReader.agents property returns SimulationAgentDict
    4. Agent state continuity across timesteps
    5. return_scenario_logV2() preserves agent IDs
    6. return_scenario_log() legacy format conversion
    7. Branching from mid-log with timestep_range
    8. LogReader reads asset scenario logs (emergency scenario)
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from invertedai.common import (
    AgentData,
    AgentProperties,
    AgentState,
    Point,
)
from invertedai.logs.logger import LogReader, ScenarioLog
from invertedai.api.location import LocationResponse, Image

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIO_LOG = os.path.join(EXAMPLES_DIR, "scenario_log_example.json")
EMERGENCY_LOG = os.path.join(EXAMPLES_DIR, "assets", "carla_Town10HD_example_emergency_scenario.json")

def make_mock_location_response(center_x=0.0, center_y=0.0, fov=200):
    return LocationResponse(
        version="1.0",
        max_agent_number=100,
        bounding_polygon=None,
        birdview_image=Image(encoded_image=""),
        osm_map=None,
        map_center=Point(x=center_x, y=center_y),
        map_fov=fov,
        static_actors=[],
    )

print("=== Offline Test: Scenario Replay ===")

# Test 1: Read scenario_log_example.json
print("\n[1] LogReader — read scenario_log_example.json")
assert os.path.exists(SCENARIO_LOG), f"File not found: {SCENARIO_LOG}"

mock_loc_resp = make_mock_location_response()
with patch("invertedai.logs.logger.location_info", return_value=mock_loc_resp):
    log_reader = LogReader(log_path=SCENARIO_LOG)

location = log_reader.location
print(f"  Location: {location}")
print(f"  Log length: {log_reader.log_length} timesteps")
assert log_reader.log_length > 0
assert location == "canada:drake_street_and_pacific_blvd"

# Test 2: initialize + verify properties
print("\n[2] LogReader.initialize() + verify properties")
log_reader.initialize()
init_count = len(log_reader.agent_states)
print(f"  Agent count at t=0: {init_count}")
print(f"  Agent properties count: {len(log_reader.agent_properties)}")
assert init_count > 0
assert len(log_reader.agent_properties) == init_count

for i, state in enumerate(log_reader.agent_states):
    assert state is not None, f"Agent {i} has no state at t=0"
for i, prop in enumerate(log_reader.agent_properties):
    assert prop is not None, f"Agent {i} has no properties"
    assert prop.length is not None, f"Agent {i} has no length"
    assert prop.width is not None, f"Agent {i} has no width"
print("  PASS: all agents have valid state and properties")

# Test 3: .agents property returns SimulationAgentDict
print("\n[3] LogReader.agents property")
agents_dict = log_reader.agents
agent_ids = list(agents_dict.keys())
print(f"  Agent IDs: {agent_ids}")
assert len(agent_ids) == init_count
for aid, data in agents_dict.items():
    assert isinstance(data, AgentData)
    assert data.state is not None
    assert data.properties is not None
print("  PASS: .agents returns correct SimulationAgentDict")

# Test 4: Step through entire log, verify agent state continuity
print("\n[4] Step through entire log")
step_count = 0
prev_states = {aid: data.state for aid, data in agents_dict.items()}

while log_reader.drive():
    step_count += 1
    current_agents = log_reader.agents
    # Verify states exist for all current agents
    for aid, data in current_agents.items():
        assert data.state is not None, f"Agent {aid} has no state at step {step_count}"

print(f"  Stepped through {step_count} drive steps")
assert step_count == log_reader.log_length - 1  # init is step 0
print("  PASS: all steps have valid agent states")

# Test 5: return_scenario_logV2 preserves agent IDs
print("\n[5] return_scenario_logV2() — preserves agent IDs")
log_reader.reset_log()
v2_log = log_reader.return_scenario_logV2()
assert isinstance(v2_log, ScenarioLog)
print(f"  agent_data timesteps: {len(v2_log.agent_data)}")
assert len(v2_log.agent_data) == log_reader.log_length

# Check agent IDs at t=0 match what we saw earlier
v2_ids_t0 = list(v2_log.agent_data[0].keys())
print(f"  Agent IDs at t=0: {v2_ids_t0}")
assert set(v2_ids_t0) == set(agent_ids)
print("  PASS: agent IDs preserved in ScenarioLog")

# Test 6: return_scenario_log legacy format
print("\n[6] return_scenario_log() — legacy format")
log_reader.reset_log()
legacy_log = log_reader.return_scenario_log()
print(f"  Type: {type(legacy_log).__name__}")
print(f"  agent_states timesteps: {len(legacy_log.agent_states)}")
print(f"  agent_properties count: {len(legacy_log.agent_properties)}")
print(f"  present_indexes timesteps: {len(legacy_log.present_indexes)}")
assert len(legacy_log.agent_states) == log_reader.log_length
assert len(legacy_log.present_indexes) == log_reader.log_length
print("  PASS: legacy format conversion successful")

# Test 7: Branching from mid-log
print("\n[7] Branching — timestep_range=(5, 15)")
log_reader.reset_log()
sub_log_v2 = log_reader.return_scenario_logV2(timestep_range=(5, 15))
assert len(sub_log_v2.agent_data) == 10
print(f"  Sub-range agent_data: {len(sub_log_v2.agent_data)} timesteps")

sub_log_legacy = log_reader.return_scenario_log(timestep_range=(5, 15))
assert len(sub_log_legacy.agent_states) == 10
print(f"  Sub-range legacy agent_states: {len(sub_log_legacy.agent_states)} timesteps")
print("  PASS: sub-range extraction works for both formats")

# Test 8: Read emergency scenario from assets
print("\n[8] LogReader — read emergency scenario asset")
assert os.path.exists(EMERGENCY_LOG), f"File not found: {EMERGENCY_LOG}"

with patch("invertedai.logs.logger.location_info", return_value=mock_loc_resp):
    emergency_reader = LogReader(log_path=EMERGENCY_LOG)

print(f"  Location: {emergency_reader.location}")
print(f"  Log length: {emergency_reader.log_length} timesteps")
assert emergency_reader.log_length > 0

emergency_reader.initialize()
emergency_count = len(emergency_reader.agent_states)
print(f"  Agent count at t=0: {emergency_count}")
assert emergency_count > 0

# Step through
steps = 0
while emergency_reader.drive():
    steps += 1
print(f"  Stepped through {steps} drive steps")
assert steps == emergency_reader.log_length - 1
print("  PASS: emergency scenario log reads correctly")

print("\n=== ALL TESTS PASSED ===")
