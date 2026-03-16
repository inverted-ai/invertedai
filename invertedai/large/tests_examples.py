from invertedai.large.large_initialize import large_initialize
from invertedai.large.common import Region
from invertedai.common import AgentProperties, AgentState, Point, InitializeResponse, RecurrentState

# single region, single Agent
# test create_square_region
regions = [Region.create_square_region(center=Point(0,0))]
assert len(regions) == 1
assert regions[0].center == Point(0,0)
assert regions[0].size == 100.0
agent_properties = [AgentProperties(agent_type = "car")]
# check it is initialized on the region

# two regions, two agents, can be anywhere on the map
# check that there are no duplicate agent ids
regions = [
    Region.create_square_region(center=Point(0,0), agent_properties=agent_properties),
    Region.create_square_region(center=Point(100,0), agent_properties=agent_properties)
]
agent_properties = [AgentProperties(agent_type="car"), AgentProperties(agent_type="pedestrian")]
assert len(regions) == 2
assert regions[0].center == Point(0,0)
assert regions[1].center == Point(100,0)
assert agent_properties[0].agent_type == "car"
assert agent_properties[1].agent_type == "pedestrian"
assert len(agent_properties) == 2
# check it is initialized on the region
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties)
assert len(response.agents) == 2
agent_ids = [agent.id for agent in response.agents]
assert len(agent_ids) == len(set(agent_ids))  # check for duplicates
# check that the agents are within the bounds of the regions
for agent in response.agents:
    assert any(region.contains_point(agent.state.center) for region in regions)
# check that the agent properties are correctly mapped to the input agents by id
for agent, prop in zip(response.agents, agent_properties):
    assert agent.properties.agent_type == prop.agent_type
# check that the agent states are correctly mapped to the input agents by id
for agent, state in zip(response.agents, response.agent_states):
    assert agent.state == state
# check that the region map is correct
assert response.region_map == [(0,0), (1,0)]  # first agent in region 0, second agent in region 1
# check that the agent properties are correctly mapped to the input agents by id
for (region_idx, agent_idx), prop in zip(response.region_map, agent_properties):
    assert response.agents[region_idx * len(agent_properties) + agent_idx].properties.agent_type == prop.agent_type


# multiple regions 200x200 square, multiple agents
# check that there are no duplicate agent ids
regions = [
    Region.create_square_region(center=Point(0,0), agent_properties=agent_properties),
    Region.create_square_region(center=Point(100,0), agent_properties=agent_properties),
    Region.create_square_region(center=Point(0,100), agent_properties=agent_properties),
    Region.create_square_region(center=Point(100,100), agent_properties=agent_properties)
]
agent_properties = [AgentProperties(agent_type="car"), AgentProperties(agent_type="pedestrian"), AgentProperties(agent_type="truck")]


# nearby agents to other regions
# input: 2 regions close enough that one agent is inside the other's FOV
# expected: that agent gets passed into the neighbour regions's iai.initialize call
regions = [
    Region.create_square_region(center=Point(0,0), agent_properties=agent_properties),
    Region.create_square_region(center=Point(80,0), agent_properties=agent_properties)
]


# far away agents to other regions
# input: 2 regions far enough that no agent is inside the other's FOV
# expected: that no agent gets passed into the neighbour regions's iai.initialize call
regions = [
    Region.create_square_region(center=Point(0,0), agent_properties=agent_properties),
    Region.create_square_region(center=Point(200,0), agent_properties=agent_properties)
]
agent_properties=["car", "pedestrian", "pedestrian"]

agent_states = [
    AgentState(center=Point(-10,0)),  # near region 0
    AgentState(center=Point(10,0)),   # near region 0
    AgentState(center=Point(90,0)),   # near region 1
    AgentState(center=Point(110,0))   # near region 1
]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
# check that the agents in the region_map are correctly mapped to the input agents
[(0, 0),   # first agent placed in region 0, index 0
 (0, 1),   # second agent placed in region 0, index 1
 (1, 0),   # third agent placed in region 1, index 0
 (1, 1)]   # fourth agent placed in region 1, index 1




# Infractions+Recurrent States
regions = [Region.create_square_region(center=Point(0,0), agent_properties=agent_properties)]
agent_properties = [AgentProperties(agent_type="car"), AgentProperties(agent_type="pedestrian")]
agent_states = [AgentState(center=Point(0,0)), AgentState(center=Point(10,0))]
recurrent_states = [RecurrentState(), RecurrentState()]
get_infractions = True # in consolidate_all_responses
response: InitializeResponse = large_initialize(
    regions=regions, 
    agent_properties=agent_properties, 
    agent_states=agent_states, 
    recurrent_states=recurrent_states, 
    get_infractions = get_infractions)
# check that the response contains 2 recurrent states
assert len(response.recurrent_states) == 2
# check that the response contains 2 infractions
assert len(response.infractions) == 2
# check that the infractions are correctly mapped to the input agents by id
for infraction in response.infractions:
    assert infraction.agent_id in [agent.id for agent in response.agents]
response.agent_properties  # → [car, pedestrian]
response.agent_states      # → [Point(0,0), Point(10,0)]
response.recurrent_states  # → [RecurrentState(), RecurrentState()]
response.infractions       # → one entry per agent (get_infractions=True) [{"collision": False}, {"collision": False}]




#Only Infractions
large_initialize(
    regions=[Region.create_square_region(center=Point(0,0))],
    agent_properties=[AgentProperties(agent_type="bike")],
    agent_states=None,
    traffic_light_state_history=None,
    get_infractions=True,
    random_seed=42,
    api_model_version=None,
    return_exact_agents=False
)
#expected output:
agent_properties = ["bike"]
agent_states = [Point(0,0)]
recurrent_states = [RecurrentState(...)]
infractions = [{"collision": True}] # example infraction

# no Infractions
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states, recurrent_states=recurrent_states, get_infractions = false)
# Expected: infractions=[] in final response


# edge case: no regions
regions = []
agent_properties = [AgentProperties(agent_type="car")]
agent_states = [AgentState(center=Point(0,0))]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
#expected:
assert len(response.agents) == 0

# edge case: no agent properties
regions = [Region.create_square_region(center=Point(0,0))]
agent_properties = []
agent_states = [AgentState(center=Point(0,0))]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
assert len(response.agents) == 0
#expected: exception due to mismatched lengths

# edge case: no agent states
regions = [Region.create_square_region(center=Point(0,0))]
agent_properties = [AgentProperties(agent_type="car")]
agent_states = []
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
assert len(response.agents) == 0
#expected: exception due to mismatched lengths

# edge case: mismatched lengths of agent_properties and agent_states
regions = [Region.create_square_region(center=Point(0,0))]
agent_properties = [AgentProperties(agent_type="car")]
agent_states = [AgentState(center=Point(0,0)), AgentState(center=Point(10,0))]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
#expected: exception due to mismatched lengths


# retry logic
regions = [Region.create_square_region(center=Point(0,0))]
agent_properties = [AgentProperties(agent_type="car")]
agent_states = [AgentState(center=Point(0,0))]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states, max_retries=3)
#hopefully it retries? not sure how to enforce a retry

# stress test: large number of regions and agents
regions = [Region.create_square_region(center=Point(x*150, y*150), agent_properties=agent_properties) for x in range(5) for y in range(5)]
agent_properties = [AgentProperties(agent_type="car"), AgentProperties(agent_type="pedestrian"), AgentProperties(agent_type="truck")]
agent_states = [AgentState(center=Point(x*10, y*10)) for x in range(50) for y in range(50)]
response: InitializeResponse = large_initialize(regions=regions, agent_properties=agent_properties, agent_states=agent_states)
# expected: all agents are initialized without error
assert len(response.agents) == len(agent_states)
# check that all agents are within the bounds of the regions
for agent in response.agents:
    assert any(region.contains_point(agent.state.center) for region in regions)
# 2500 agents distributed across 25 regions, response has exactly 2500 agents, all within region bounds, no errors

# strict mode: ensure that all agents are initialized
_consolidate_all_responses(
    regions=[Region(center=(0,0), size=100)]
    agent_properties=[car]
    agent_states=[Point(0,0)]
    return_exact_agents=True
    region_map=[(0,5)] )
# expected behvaiour: get IndexError -> InvertedAIError

# if return_exact_agents = false, should only produce warning messages