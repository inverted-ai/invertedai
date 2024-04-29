import sys
import pytest

sys.path.insert(0, "../../")
from invertedai.common import Point, AgentAttributes, AgentState
from invertedai.api.initialize import InitializeResponse
from invertedai.error import InvalidRequestError, InvertedAIError

from invertedai.large.common import Region
from invertedai.large.initialize import large_initialize


positive_tests = [
    ( # Single region
        "canada:ubc_roundabout",
        [Region.create_square_region(center=Point.fromlist([-31.1,-24.36]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(1)])],
        True
    ),
    ( # Multiple regions
        "carla:Town03",
        [
            Region.create_square_region(center=Point.fromlist([79.8,-4.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(1)]),
            Region.create_square_region(center=Point.fromlist([-82.7,-122.1]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(1)])
        ],
        True
    ),
    ( # More than regular initialize maximum load
        "carla:Town03",
        [
            Region.create_square_region(center=Point.fromlist([79.8,-4.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
            Region.create_square_region(center=Point.fromlist([-82.7,-122.1]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
            Region.create_square_region(center=Point.fromlist([-82.7,136.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
            Region.create_square_region(center=Point.fromlist([-2.2,132.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
            Region.create_square_region(center=Point.fromlist([-80.5,-3.2]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)])
        ],
        True
    ),
    ( # Turn off return exact number of agents
        "canada:ubc_roundabout",
        [
            Region.create_square_region(
                center=Point.fromlist([82.0,-127.1]),
                agent_states=[AgentState.fromlist([82.0,-127.1,1.55,0.11])],
                agent_attributes=[AgentAttributes.fromlist([4.55,1.94,1.4,'car'])]+[AgentAttributes.fromlist(["car"]) for _ in range(50)]
            )
        ],
        False
    )
]

negative_tests_general = [
    (
        "canada:ubc_roundabout",
        [Region.create_square_region(center=Point.fromlist([-31.1,-24.36]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(101)])],
        True
    )
]

def run_large_initialize(location, regions, return_exact_agents):
    response = large_initialize(
      location=location,
      regions=regions,
      traffic_light_state_history=None,
      return_exact_agents=return_exact_agents
  )
    assert isinstance(response,InitializeResponse) and response.agent_attributes is not None and response.agent_states is not None
    if response.traffic_lights_states is not None:
      assert response.light_recurrent_states is not None


@pytest.mark.parametrize("location, regions, return_exact_agents", negative_tests_general)
def test_negative_general(location, regions, return_exact_agents):
    with pytest.raises(InvertedAIError):
        run_large_initialize(location, regions, return_exact_agents)


@pytest.mark.parametrize("location, regions, return_exact_agents", positive_tests)
def test_positive(location, regions, return_exact_agents):
    run_large_initialize(location, regions, return_exact_agents)
