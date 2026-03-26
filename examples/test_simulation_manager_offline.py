"""
Offline test: SimulationManager unit tests — no API calls required.

Tests covered:
    1. SimulationManager._unpack() handles None recurrent states (the bug fix)
    2. SimulationManager._pack() / _unpack() roundtrip
    3. insert_agents + remove_agents
    4. Per-agent getters: get_state(), get_property(), get_recurrent_state()
    5. Per-agent setters: set_state(), set_property(), set_recurrent_state()
    6. Bulk getters: get_states(), get_properties(), get_recurrent_states()
    7. Bulk setters: set_states(), set_properties(), set_recurrent_states()
    8. get_agent_ids(), get_agent_data(), get_agent_dict()
    9. Error handling: KeyError on missing agents, ValueError on wrong list lengths
"""

import os
import sys
import uuid

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
from invertedai.helpers.simulation_manager import SimulationManager

# Helpers
def make_state(x, y, orientation=0.0, speed=1.0):
    return AgentState(center=Point(x=x, y=y), orientation=orientation, speed=speed)

def make_props(length=4.5, width=2.0, rear_axis_offset=1.5, agent_type="car"):
    return AgentProperties(length=length, width=width, rear_axis_offset=rear_axis_offset, agent_type=agent_type)

def make_recurrent(size=RECURRENT_SIZE, val=0.0):
    return RecurrentState(packed=[val] * size)

def make_agent(x, y, with_recurrent=True):
    return AgentData(
        state=make_state(x, y),
        properties=make_props(),
        recurrent=make_recurrent() if with_recurrent else None,
    )

print("=== Offline Test: SimulationManager Unit Tests ===")

# Create a SimulationManager without any configs (no plotter, no waypoints, no log)
sm = SimulationManager()

# Seed agents_dict directly
sm.agents_dict = {
    "agent_1": make_agent(10.0, 20.0),
    "agent_2": make_agent(30.0, 40.0),
    "agent_3": make_agent(50.0, 60.0),
}

# Test 1: _unpack handles None recurrent states
print("\n[1] _unpack() with None recurrent states")
# Add an agent with no recurrent state (simulates insert_agents with recurrent=None)
sm.agents_dict["agent_no_recur"] = make_agent(70.0, 80.0, with_recurrent=False)

ids, states, props, recurrents = sm._unpack()
assert len(ids) == 4
assert len(states) == 4
assert len(props) == 4
assert len(recurrents) == 4

# The None recurrent should have been replaced with a zero-filled RecurrentState
for r in recurrents:
    assert isinstance(r, RecurrentState), f"Expected RecurrentState, got {type(r)}"
    assert len(r.packed) == RECURRENT_SIZE

# Verify it matched the size of existing agents
existing_size = len(sm.agents_dict["agent_1"].recurrent.packed)
no_recur_idx = ids.index("agent_no_recur")
assert len(recurrents[no_recur_idx].packed) == existing_size
print(f"  PASS: None recurrent replaced with zero-filled RecurrentState (size={existing_size})")

# Remove the test agent for subsequent tests
del sm.agents_dict["agent_no_recur"]

# Test 2: _pack / _unpack roundtrip
print("\n[2] _pack() / _unpack() roundtrip")
ids, states, props, recurrents = sm._unpack()
repacked = sm._pack(ids, states, props, recurrents)
assert set(repacked.keys()) == set(sm.agents_dict.keys())
for aid in ids:
    assert repacked[aid].state.center.x == sm.agents_dict[aid].state.center.x
    assert repacked[aid].properties.length == sm.agents_dict[aid].properties.length
print("  PASS: pack/unpack roundtrip preserves data")

# Test 3: insert_agents + remove_agents
print("\n[3] insert_agents + remove_agents")
new_agents = [
    make_agent(100.0, 200.0),
    make_agent(150.0, 250.0),
]
new_ids = sm.insert_agents(agent_data_list=new_agents, ids=["new_A", "new_B"])
assert "new_A" in sm.agents_dict
assert "new_B" in sm.agents_dict
assert len(sm.agents_dict) == 5
print(f"  Inserted: {new_ids}, total agents: {len(sm.agents_dict)}")

sm.remove_agents(["new_A"])
assert "new_A" not in sm.agents_dict
assert len(sm.agents_dict) == 4
print(f"  Removed new_A, total agents: {len(sm.agents_dict)}")

sm.remove_agents(["new_B"])
assert len(sm.agents_dict) == 3
print("  PASS: insert/remove works correctly")

# Test insert with auto-generated IDs
auto_ids = sm.insert_agents(agent_data_list=[make_agent(0, 0)], ids=None)
assert len(auto_ids) == 1
assert auto_ids[0] in sm.agents_dict
sm.remove_agents(auto_ids)
print("  PASS: insert with auto-generated IDs works")

# Test insert with duplicate ID raises error
sm.insert_agents(agent_data_list=[make_agent(0, 0)], ids=["dup_test"])
try:
    sm.insert_agents(agent_data_list=[make_agent(0, 0)], ids=["dup_test"])
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"  PASS: duplicate insert raises ValueError: {e}")
sm.remove_agents(["dup_test"])

# Test insert with overwrite=True
sm.insert_agents(agent_data_list=[make_agent(0, 0)], ids=["overwrite_test"])
sm.insert_agents(agent_data_list=[make_agent(99, 99)], ids=["overwrite_test"], overwrite=True)
assert sm.agents_dict["overwrite_test"].state.center.x == 99
sm.remove_agents(["overwrite_test"])
print("  PASS: insert with overwrite=True works")

# Test 4: Per-agent getters
print("\n[4] Per-agent getters: get_state(), get_property(), get_recurrent_state()")
state = sm.get_state("agent_1")
assert state.center.x == 10.0
assert state.center.y == 20.0
print(f"  get_state('agent_1'): ({state.center.x}, {state.center.y})")

prop = sm.get_property("agent_1")
assert prop.length == 4.5
assert prop.width == 2.0
print(f"  get_property('agent_1'): length={prop.length}, width={prop.width}")

recur = sm.get_recurrent_state("agent_1")
assert isinstance(recur, RecurrentState)
assert len(recur.packed) == RECURRENT_SIZE
print(f"  get_recurrent_state('agent_1'): size={len(recur.packed)}")

# Test missing agent raises KeyError
for getter_name in ["get_state", "get_property", "get_recurrent_state"]:
    try:
        getattr(sm, getter_name)("nonexistent")
        assert False, f"{getter_name} should have raised KeyError"
    except KeyError:
        pass
print("  PASS: all per-agent getters work, KeyError on missing agent")

# Test 5: Per-agent setters
print("\n[5] Per-agent setters: set_state(), set_property(), set_recurrent_state()")
new_state = make_state(99.0, 88.0, orientation=1.5, speed=5.0)
sm.set_state("agent_1", new_state)
assert sm.get_state("agent_1").center.x == 99.0
assert sm.get_state("agent_1").speed == 5.0

new_prop = make_props(length=6.0, width=2.5)
sm.set_property("agent_1", new_prop)
assert sm.get_property("agent_1").length == 6.0

new_recur = make_recurrent(val=1.0)
sm.set_recurrent_state("agent_1", new_recur)
assert sm.get_recurrent_state("agent_1").packed[0] == 1.0

# Restore original
sm.agents_dict["agent_1"] = make_agent(10.0, 20.0)
print("  PASS: all per-agent setters work")

# Test 6: Bulk getters
print("\n[6] Bulk getters: get_states(), get_properties(), get_recurrent_states()")
all_states = sm.get_states()
assert len(all_states) == 3
assert all(isinstance(s, AgentState) for s in all_states)

all_props = sm.get_properties()
assert len(all_props) == 3
assert all(isinstance(p, AgentProperties) for p in all_props)

all_recur = sm.get_recurrent_states()
assert len(all_recur) == 3
assert all(isinstance(r, RecurrentState) for r in all_recur)
print("  PASS: all bulk getters return correct types and counts")

# Test 7: Bulk setters
print("\n[7] Bulk setters: set_states(), set_properties(), set_recurrent_states()")
new_states = [make_state(i * 10, i * 20) for i in range(3)]
sm.set_states(new_states)
assert sm.get_states()[0].center.x == 0.0
assert sm.get_states()[2].center.x == 20.0

new_props = [make_props(length=i + 3) for i in range(3)]
sm.set_properties(new_props)
assert sm.get_properties()[0].length == 3.0
assert sm.get_properties()[2].length == 5.0

new_recurs = [make_recurrent(val=float(i)) for i in range(3)]
sm.set_recurrent_states(new_recurs)
assert sm.get_recurrent_states()[0].packed[0] == 0.0
assert sm.get_recurrent_states()[2].packed[0] == 2.0

# Wrong length raises ValueError
try:
    sm.set_states([make_state(0, 0)])
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    sm.set_properties([make_props()])
    assert False, "Should have raised ValueError"
except ValueError:
    pass

try:
    sm.set_recurrent_states([make_recurrent()])
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("  PASS: all bulk setters work, ValueError on wrong length")

# Restore
sm.agents_dict = {
    "agent_1": make_agent(10.0, 20.0),
    "agent_2": make_agent(30.0, 40.0),
    "agent_3": make_agent(50.0, 60.0),
}

# Test 8: get_agent_ids, get_agent_data, get_agent_dict
print("\n[8] get_agent_ids(), get_agent_data(), get_agent_dict()")
ids = sm.get_agent_ids()
assert ids == ["agent_1", "agent_2", "agent_3"]
print(f"  Agent IDs: {ids}")

data = sm.get_agent_data("agent_2")
assert isinstance(data, AgentData)
assert data.state.center.x == 30.0
print(f"  get_agent_data('agent_2'): x={data.state.center.x}")

agent_dict = sm.get_agent_dict()
assert set(agent_dict.keys()) == {"agent_1", "agent_2", "agent_3"}
print("  PASS: all accessors return correct data")

# Test missing agent in get_agent_data
try:
    sm.get_agent_data("nonexistent")
    assert False, "Should have raised KeyError"
except KeyError:
    pass
print("  PASS: KeyError on missing agent_data")

# Test 9: _unpack with different recurrent sizes
print("\n[9] _unpack() matches recurrent size from existing agents")
CUSTOM_SIZE = 200
sm.agents_dict["agent_1"].recurrent = RecurrentState(packed=[0.5] * CUSTOM_SIZE)
sm.agents_dict["agent_2"].recurrent = RecurrentState(packed=[0.5] * CUSTOM_SIZE)
sm.agents_dict["agent_3"].recurrent = None  # This one is None

ids, states, props, recurrents = sm._unpack()
# agent_3's recurrent should match CUSTOM_SIZE, not RECURRENT_SIZE
agent_3_idx = ids.index("agent_3")
assert len(recurrents[agent_3_idx].packed) == CUSTOM_SIZE, \
    f"Expected size {CUSTOM_SIZE}, got {len(recurrents[agent_3_idx].packed)}"
print(f"  PASS: None recurrent matched existing size ({CUSTOM_SIZE}), not default ({RECURRENT_SIZE})")

# Test 10: _unpack with ALL None recurrents falls back to RECURRENT_SIZE
print("\n[10] _unpack() with all None recurrents falls back to RECURRENT_SIZE")
sm.agents_dict = {
    "a": AgentData(state=make_state(0, 0), properties=make_props(), recurrent=None),
    "b": AgentData(state=make_state(1, 1), properties=make_props(), recurrent=None),
}
ids, states, props, recurrents = sm._unpack()
assert len(recurrents[0].packed) == RECURRENT_SIZE
assert len(recurrents[1].packed) == RECURRENT_SIZE
print(f"  PASS: all-None recurrents default to RECURRENT_SIZE ({RECURRENT_SIZE})")

print("\n=== ALL TESTS PASSED ===")
