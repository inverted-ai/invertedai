import sys
import pytest

sys.path.insert(0, "../../")
from invertedai.api.initialize import initialize
from invertedai.api.drive import drive, DriveResponse
from invertedai.api.location import location_info
from invertedai.api.light import light
from invertedai.error import InvalidRequestError

positive_tests = [
    ("carla:Town04",
     None,
     None,
     False, 5),
    ("iai:drake_street_and_pacific_blvd",
     None,
     [dict(agent_type='car'),
      dict(),
      dict(agent_type='car'),
      dict()],
     False, 5),
    ("iai:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      dict(agent_type='car')],
     False, None),
    ("iai:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      dict(agent_type='car'),
      dict(agent_type='car')],
     False, None),
]
negative_tests = [
    ("iai:drake_street_and_pacific_blvd",
     None,
     [dict(agent_type='car'),
      dict(),
      dict(agent_type='pedestrian'),
      dict()],
     False, 5),
]


def test_drive(location, states_history, agent_attributes, get_infractions, agent_count, simulation_length: int = 20):
    location_info_response = location_info(location=location, rendering_fov=200)
    if any(actor.agent_type == "traffic-light" for actor in location_info_response.static_actors):
        scene_has_lights = True
        light_response = light(location=location)
    else:
        light_response = None
        scene_has_lights = False
    initialize_response = initialize(
        location,
        agent_attributes=agent_attributes,
        states_history=states_history,
        traffic_light_state_history=[light_response.traffic_lights_states] if scene_has_lights else None,
        get_birdview=False,
        get_infractions=get_infractions,
        agent_count=agent_count,
    )
    agent_attributes = initialize_response.agent_attributes
    updated_state = initialize_response
    for t in range(simulation_length):
        updated_state = drive(
            agent_attributes=agent_attributes,
            agent_states=updated_state.agent_states,
            recurrent_states=updated_state.recurrent_states,
            traffic_lights_states=light_response.traffic_lights_states if light_response is not None else None,
            get_birdview=False,
            location=location,
            get_infractions=get_infractions,
        )
        assert isinstance(updated_state,
                          DriveResponse) and updated_state.agent_states is not None and updated_state.recurrent_states is not None


def test_negative(location, states_history, agent_attributes, get_infractions, agent_count,
                  simulation_length: int = 20):
    with pytest.raises(InvalidRequestError):
        test_drive(location, states_history, agent_attributes, get_infractions, agent_count)


def run_all(tests):
    i = 0
    for test in tests:
        test_drive(*test)
        # print(f"positive test case {i} passed")
        i += 1
    print(f"all {i} positive tests passed!")


def run_all_negative(tests):
    i = 0
    for test in tests:
        test_negative(*test)
        # print(f"negative test case {i} passed")
        i += 1
    print(f"all {i} negative tests passed!")


run_all(positive_tests)
run_all_negative(negative_tests)
