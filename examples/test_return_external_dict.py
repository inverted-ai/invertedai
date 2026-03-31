"""
Tests for the return_external_dict feature on SimulationManager.initialize() and .drive().

Covers:
  1. initialize() with return_external_dict=False  → returns InitializeResponse only (no regression)
  2. initialize() with return_external_dict=True   → returns (InitializeResponse, dict) keyed by external IDs
  3. drive()      with return_external_dict=False  → returns DriveResponse only (no regression)
  4. drive()      with return_external_dict=True   → returns (DriveResponse, dict) keyed by external IDs
  5. External dict keys match the IDs passed in
  6. External dict states match the corresponding slice of response.agent_states
  7. Calling with no external_agent_data and return_external_dict=True → empty dict
  8. Multiple drive steps: external dict is freshly computed each step (no stale state)
"""

import os
import uuid
import invertedai as iai
from invertedai import SimulationManager, RegionsConfig, AgentType
from invertedai.common import AgentData, AgentState, AgentProperties, Point

iai.add_apikey(os.environ.get("IAI_API_KEY"))

LOCATION = "carla:Town10HD"
NUM_IAI_AGENTS = 3
NUM_EXT_AGENTS = 2
SIM_STEPS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_external_agents(n: int):
    """Initialize n external agents via iai.initialize() and return as AgentData dict."""
    response = iai.initialize(
        location=LOCATION,
        agent_properties=iai.utils.get_default_agent_properties({AgentType.car: n}),
    )
    ext_ids = [str(uuid.uuid4()) for _ in range(n)]
    return {
        ext_ids[i]: AgentData(
            state=response.agent_states[i],
            properties=response.agent_properties[i],
            recurrent=None,
        )
        for i in range(n)
    }


def make_simulation_manager():
    location_info_response = iai.location_info(location=LOCATION, include_map_source=True)
    waypoint_cfg = iai.WaypointManagerConfig(
        lanelet_map=location_info_response.get_lanelet_map(),
        fail_soft=True,
    )
    sm = SimulationManager(waypoint_cfg=waypoint_cfg)
    regions_config = RegionsConfig(
        location=LOCATION,
        agent_count_dict={AgentType.car: NUM_IAI_AGENTS},
    )
    regions = sm.form_regions(regions_config)
    return sm, regions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initialize_no_flag_returns_response_only():
    """Test 1: default (return_external_dict=False) returns bare InitializeResponse."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)

    result = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    assert not isinstance(result, tuple), \
        "With return_external_dict=False, initialize() must return InitializeResponse, not a tuple"
    print("PASS test_initialize_no_flag_returns_response_only")


def test_initialize_with_flag_returns_tuple():
    """Test 2: return_external_dict=True returns (InitializeResponse, dict)."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)

    result = sm.initialize(
        location=LOCATION, regions=regions,
        external_agent_data=ext, return_external_dict=True,
    )

    assert isinstance(result, tuple) and len(result) == 2, \
        "With return_external_dict=True, initialize() must return a 2-tuple"
    response, ext_dict = result
    assert hasattr(response, "agent_states"), "First element must be InitializeResponse"
    assert isinstance(ext_dict, dict), "Second element must be a dict"
    print("PASS test_initialize_with_flag_returns_tuple")


def test_initialize_external_dict_keys_match():
    """Test 3: external dict keys match the IDs passed in as external_agent_data."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)

    _, ext_dict = sm.initialize(
        location=LOCATION, regions=regions,
        external_agent_data=ext, return_external_dict=True,
    )

    assert set(ext_dict.keys()) == set(ext.keys()), \
        f"External dict keys {set(ext_dict.keys())} != expected {set(ext.keys())}"
    print("PASS test_initialize_external_dict_keys_match")


def test_initialize_external_dict_states_match_response():
    """Test 4: external dict states are consistent with response.agent_states.

    Ordering note: initialize() merges agents_dict and external_agent_data into one
    dict before calling _unpack, which puts agents-with-states first. Since external
    agents always have states and agents_dict is empty here, external agents land at
    the FRONT of response.agent_states (indices 0..num_ext-1). This differs from
    drive(), which explicitly appends external agents after internal ones.
    """
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)

    response, ext_dict = sm.initialize(
        location=LOCATION, regions=regions,
        external_agent_data=ext, return_external_dict=True,
    )

    # External agents come first in the response for initialize() (not num_internal + i)
    ext_ids_ordered = list(ext.keys())
    for i, aid in enumerate(ext_ids_ordered):
        expected_state = response.agent_states[i]
        actual_state = ext_dict[aid].state
        assert actual_state.center.x == expected_state.center.x and \
               actual_state.center.y == expected_state.center.y, \
            f"State mismatch for external agent '{aid}' at index {i}"
    print("PASS test_initialize_external_dict_states_match_response")


def test_drive_no_flag_returns_response_only():
    """Test 5: default (return_external_dict=False) drive() returns bare DriveResponse."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)
    response = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    result = sm.drive(
        location=LOCATION,
        external_agent_data=ext,
        light_recurrent_states=response.light_recurrent_states,
    )

    assert not isinstance(result, tuple), \
        "With return_external_dict=False, drive() must return DriveResponse, not a tuple"
    print("PASS test_drive_no_flag_returns_response_only")


def test_drive_with_flag_returns_tuple():
    """Test 6: return_external_dict=True drive() returns (DriveResponse, dict)."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)
    response = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    result = sm.drive(
        location=LOCATION,
        external_agent_data=ext,
        light_recurrent_states=response.light_recurrent_states,
        return_external_dict=True,
    )

    assert isinstance(result, tuple) and len(result) == 2, \
        "With return_external_dict=True, drive() must return a 2-tuple"
    response, ext_dict = result
    assert hasattr(response, "agent_states"), "First element must be DriveResponse"
    assert isinstance(ext_dict, dict), "Second element must be a dict"
    print("PASS test_drive_with_flag_returns_tuple")


def test_drive_external_dict_keys_match():
    """Test 7: drive() external dict keys match the IDs passed in."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)
    response = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    _, ext_dict = sm.drive(
        location=LOCATION,
        external_agent_data=ext,
        light_recurrent_states=response.light_recurrent_states,
        return_external_dict=True,
    )

    assert set(ext_dict.keys()) == set(ext.keys()), \
        f"External dict keys {set(ext_dict.keys())} != expected {set(ext.keys())}"
    print("PASS test_drive_external_dict_keys_match")


def test_drive_external_dict_states_match_response():
    """Test 8: drive() external dict states are consistent with response.agent_states."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)
    init_response = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    response, ext_dict = sm.drive(
        location=LOCATION,
        external_agent_data=ext,
        light_recurrent_states=init_response.light_recurrent_states,
        return_external_dict=True,
    )

    num_internal = len(sm.get_agent_ids())
    ext_ids_ordered = list(ext.keys())

    for i, aid in enumerate(ext_ids_ordered):
        expected_state = response.agent_states[num_internal + i]
        actual_state = ext_dict[aid].state
        assert actual_state.center.x == expected_state.center.x and \
               actual_state.center.y == expected_state.center.y, \
            f"State mismatch for external agent '{aid}' at index {i}"
    print("PASS test_drive_external_dict_states_match_response")


def test_drive_no_external_agents_returns_empty_dict():
    """Test 9: return_external_dict=True with no external agents returns empty dict."""
    sm, regions = make_simulation_manager()
    response = sm.initialize(location=LOCATION, regions=regions)

    response, ext_dict = sm.drive(
        location=LOCATION,
        light_recurrent_states=response.light_recurrent_states,
        return_external_dict=True,
    )

    assert ext_dict == {}, f"Expected empty dict, got {ext_dict}"
    print("PASS test_drive_no_external_agents_returns_empty_dict")


def test_drive_external_dict_fresh_each_step():
    """Test 10: external dict reflects the current step's states, not a previous step's."""
    sm, regions = make_simulation_manager()
    ext = make_external_agents(NUM_EXT_AGENTS)
    response = sm.initialize(location=LOCATION, regions=regions, external_agent_data=ext)

    seen_states = []
    for step in range(SIM_STEPS):
        response, ext_dict = sm.drive(
            location=LOCATION,
            external_agent_data=ext,
            light_recurrent_states=response.light_recurrent_states,
            return_external_dict=True,
        )
        # Use updated states from the returned dict as input to the next step
        ext = {
            aid: AgentData(
                state=data.state,
                properties=data.properties,
                recurrent=None,
            )
            for aid, data in ext_dict.items()
        }
        first_id = list(ext_dict.keys())[0]
        seen_states.append(ext_dict[first_id].state)

    # States should differ across steps as agents move
    positions = [(s.center.x, s.center.y) for s in seen_states]
    assert len(set(positions)) > 1, \
        "External agent position did not change across steps — dict may be stale"
    print("PASS test_drive_external_dict_fresh_each_step")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_initialize_no_flag_returns_response_only,
        test_initialize_with_flag_returns_tuple,
        test_initialize_external_dict_keys_match,
        test_initialize_external_dict_states_match_response,
        test_drive_no_flag_returns_response_only,
        test_drive_with_flag_returns_tuple,
        test_drive_external_dict_keys_match,
        test_drive_external_dict_states_match_response,
        test_drive_no_external_agents_returns_empty_dict,
        test_drive_external_dict_fresh_each_step,
    ]

    passed, failed = 0, []
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed.append((test.__name__, e))
            print(f"FAIL {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if failed:
        raise SystemExit(1)
