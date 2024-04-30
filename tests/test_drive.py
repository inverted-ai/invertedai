import sys
import pytest

sys.path.insert(0, "../../")
from invertedai.api.initialize import initialize
from invertedai.api.drive import drive, DriveResponse
from invertedai.api.location import location_info
from invertedai.api.light import light
from invertedai.common import Point, AgentProperties
from invertedai.error import InvalidRequestError


def recurrent_states_helper(states_to_extend):
    result = [0.0] * 128
    result.extend(states_to_extend)
    return result


positive_tests_old = [
    ("carla:Town04",
     None,
     None,
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [dict(agent_type='car'),
      dict(),
      dict(agent_type='car'),
      dict()],
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [dict(agent_type='car'),
      dict(),
      dict(agent_type='pedestrian'),
      dict()],
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [dict(agent_type='pedestrian'),
      dict(agent_type='pedestrian'),
      dict(agent_type='pedestrian'),
      dict(agent_type='pedestrian')],
     False, 5),
    ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      dict(agent_type='car')],
     False, None),
    ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      dict(agent_type='car'),
      dict(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car', waypoint=Point(x=1, y=2)),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='pedestrian', waypoint=Point(x=1, y=2)),
      dict(agent_type='car'),
      dict(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, waypoint=Point(x=1, y=2)),
      dict(length=6.29, width=2.56, rear_axis_offset=2.39,  waypoint=Point(x=1, y=2)),
      dict(agent_type='car'),
      dict(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [dict(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='pedestrian',waypoint=Point(x=1, y=2)),
      dict(length=6.29, width=2.56, rear_axis_offset=None, agent_type='pedestrian', waypoint=Point(x=1, y=2)),
      dict(agent_type='car'),
      dict(agent_type='car')],
     False, None)
]

# Notice parameterization in direct drive flows are different
negative_tests_old = [
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [dict(length=0.97, agent_type="pedestrian"),
      dict(length=4.86, width=2.12, rear_axis_offset=1.85, agent_type='car')],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [dict(length=0.97, width=1.06, rear_axis_offset=None, agent_type="pedestrian"),
      dict(length=4.86, width=2.12, agent_type='car')],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [dict(length=0.97, width=1.06, agent_type="pedestrian"),
      dict(width=2.12, rear_axis_offset=1.85)],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
]


def run_initialize_drive_flow(location, states_history, agent_attributes, agent_properties, get_infractions, agent_count,
                              simulation_length: int = 20):
    location_info_response = location_info(location=location, rendering_fov=200)
    scene_has_lights = any(actor.agent_type == "traffic_light" for actor in location_info_response.static_actors)
    
    initialize_response = initialize(
        location,
        agent_attributes=agent_attributes,
        agent_properties=agent_properties,
        states_history=states_history,
        traffic_light_state_history=None,
        get_birdview=False,
        get_infractions=get_infractions,
        agent_count=agent_count,
    )
    agent_attributes = initialize_response.agent_attributes if agent_attributes is None else None
    agent_properties = initialize_response.agent_properties
    updated_state = initialize_response
    for t in range(simulation_length):
        updated_state = drive(
            agent_attributes=agent_attributes,
            agent_properties=agent_properties,
            agent_states=updated_state.agent_states,
            recurrent_states=updated_state.recurrent_states,
            light_recurrent_states=updated_state.light_recurrent_states if scene_has_lights else None,
            get_birdview=False,
            location=location,
            get_infractions=get_infractions,
        )
        assert isinstance(updated_state,
                          DriveResponse) and updated_state.agent_states is not None and updated_state.recurrent_states is not None
        if scene_has_lights:
            assert updated_state.traffic_lights_states is not None
            assert updated_state.light_recurrent_states is not None

def run_direct_drive(location, agent_states, agent_attributes, agent_properties, recurrent_states, get_infractions):
    drive_response = drive(
        agent_attributes=agent_attributes,
        agent_properties=agent_properties,
        agent_states=agent_states,
        recurrent_states=recurrent_states,
        traffic_lights_states=None,
        get_birdview=False,
        location=location,
        get_infractions=get_infractions
    )
    assert isinstance(drive_response,
                      DriveResponse) and drive_response.agent_states is not None and drive_response.recurrent_states is not None


@pytest.mark.parametrize("location, agent_states, agent_attributes, recurrent_states, get_infractions", negative_tests_old)
def test_negative_old(location, agent_states, agent_attributes, recurrent_states, get_infractions):
    with pytest.raises(InvalidRequestError):
        run_direct_drive(location, agent_states, agent_attributes, None, recurrent_states, get_infractions)


@pytest.mark.parametrize("location, states_history, agent_attributes, get_infractions, agent_count", positive_tests_old)
def test_positive_old(location, states_history, agent_attributes, get_infractions, agent_count,
                  simulation_length: int = 20):
    run_initialize_drive_flow(location, states_history, agent_attributes, None, get_infractions, agent_count,
                              simulation_length)


positive_tests = [
    ("carla:Town04",
     None,
     None,
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [AgentProperties(agent_type='car'),
      AgentProperties(),
      AgentProperties(agent_type='car'),
      AgentProperties()],
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [AgentProperties(agent_type='car'),
      AgentProperties(),
      AgentProperties(agent_type='pedestrian'),
      AgentProperties()],
     False, 5),
    ("canada:drake_street_and_pacific_blvd",
     None,
     [AgentProperties(agent_type='pedestrian'),
      AgentProperties(agent_type='pedestrian'),
      AgentProperties(agent_type='pedestrian'),
      AgentProperties(agent_type='pedestrian')],
     False, 5),
    ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)]],
     [AgentProperties(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      AgentProperties(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      AgentProperties(agent_type='car')],
     False, None),
    ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [AgentProperties(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car'),
      AgentProperties(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='car'),
      AgentProperties(agent_type='car'),
      AgentProperties(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [AgentProperties(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='car', waypoint=Point(x=1, y=2)),
      AgentProperties(length=6.29, width=2.56, rear_axis_offset=2.39, agent_type='pedestrian', waypoint=Point(x=1, y=2)),
      AgentProperties(agent_type='car'),
      AgentProperties(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [AgentProperties(length=5.3, width=2.25, rear_axis_offset=2.01, waypoint=Point(x=1, y=2)),
      AgentProperties(length=6.29, width=2.56, rear_axis_offset=2.39,  waypoint=Point(x=1, y=2)),
      AgentProperties(agent_type='car'),
      AgentProperties(agent_type='car')],
     False, None),
     ("canada:ubc_roundabout",
     [[dict(center=dict(x=60.82, y=1.22), orientation=0.63, speed=11.43,),
       dict(center=dict(x=-36.88, y=-33.93), orientation=-2.64, speed=9.43)],
      [dict(center=dict(x=62.82, y=3.22), orientation=0.63, speed=11.43),
       dict(center=dict(x=-34.88, y=-31.93), orientation=-2.64, speed=9.43)]],
     [AgentProperties(length=5.3, width=2.25, rear_axis_offset=2.01, agent_type='pedestrian',waypoint=Point(x=1, y=2)),
      AgentProperties(length=6.29, width=2.56, rear_axis_offset=None, agent_type='pedestrian', waypoint=Point(x=1, y=2)),
      AgentProperties(agent_type='car'),
      AgentProperties(agent_type='car')],
     False, None)
]

# Notice parameterization in direct drive flows are different
negative_tests = [
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [AgentProperties(length=0.97, agent_type="pedestrian"),
      AgentProperties(length=4.86, width=2.12, rear_axis_offset=1.85, agent_type='car')],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [AgentProperties(length=0.97, width=1.06, rear_axis_offset=None, agent_type="pedestrian"),
      AgentProperties(length=4.86, width=2.12, agent_type='car')],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
    ("carla:Town03",
     [dict(center=dict(x=-21.2, y=-17.11), orientation=4.54, speed=1.8),
      dict(center=dict(x=-5.81, y=-49.47), orientation=1.62, speed=11.4)],
     [AgentProperties(length=0.97, width=1.06, agent_type="pedestrian"),
      AgentProperties(width=2.12, rear_axis_offset=1.85)],
     [dict(packed=recurrent_states_helper(
         [21.203039169311523, -17.10862159729004, -1.3971855640411377, 1.7982733249664307])),
         dict(packed=recurrent_states_helper(
             [5.810295104980469, -49.47068786621094, 1.5232856273651123, 11.404326438903809]))],
     False),
]


@pytest.mark.parametrize("location, agent_states, agent_properties, recurrent_states, get_infractions", negative_tests)
def test_negative(location, agent_states, agent_properties, recurrent_states, get_infractions):
    with pytest.raises(InvalidRequestError):
        run_direct_drive(location, agent_states, None, agent_properties, recurrent_states, get_infractions)


@pytest.mark.parametrize("location, states_history, agent_properties, get_infractions, agent_count", positive_tests)
def test_positive(location, states_history, agent_properties, get_infractions, agent_count,
                  simulation_length: int = 20):
    run_initialize_drive_flow(location, states_history, None, agent_properties, get_infractions, agent_count,
                              simulation_length)
