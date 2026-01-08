import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import itertools
import os
import time
import tqdm

LOCATIONS_TO_TEST = ["carla:Town10HD", "carla:Town03"]
SIM_LENGTHS = [25, 50, 100]
NUM_AGENTS = [5, 10, 20]
API_WARMUP = 0
REPETITIONS = 10
FOV = 250
DRIVE_MODEL = "nBu1"
seed = int(time.time())
SAVE_CSV_PATH = "./results"
os.makedirs(SAVE_CSV_PATH, exist_ok=True)

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey('<INSERT_KEY_HERE>')

configurations = list(itertools.product(LOCATIONS_TO_TEST, SIM_LENGTHS, NUM_AGENTS))

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return end - start
    wrapper.__name__ = func.__name__
    return wrapper

@time_function
def with_wp(location, sim_length, num_agents):
    location_info_response = iai.location_info(
        location=location, 
        include_map_source=True,
        rendering_fov=FOV
    )
    response = iai.initialize(
        location=location,
        agent_properties=get_default_agent_properties({AgentType.car: num_agents}),
        random_seed=seed
    )
    wp_manager = iai.WaypointManager(
        location_info_response = location_info_response,
        cfg = iai.WaypointManagerConfig(
            random_seed=seed,
            fail_soft=False
        )
    )
    agent_properties = wp_manager.update(
        response = response,
        agent_properties = response.agent_properties,
    )
    for _ in range(sim_length):
        response = iai.drive(
            location=location,
            agent_properties=agent_properties,
            agent_states=response.agent_states,
            recurrent_states=response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states,
            random_seed=seed,
            api_model_version=DRIVE_MODEL
        )
        agent_properties = wp_manager.update(
            response = response,
            agent_properties = agent_properties,
        )

@time_function
def without_wp(location, sim_length, num_agents):
    initialize_response = iai.initialize(
        location=location,
        agent_properties=get_default_agent_properties({AgentType.car: num_agents}),
        random_seed=seed
    )
    response = initialize_response
    for _ in range(sim_length):
        response = iai.drive(
            location=location,
            agent_properties=initialize_response.agent_properties,
            agent_states=response.agent_states,
            recurrent_states=response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states,
            random_seed=seed,
            api_model_version=DRIVE_MODEL
        )

FUNCTIONS_UNDER_TEST = [with_wp, without_wp]

for f in FUNCTIONS_UNDER_TEST:
    ret = ""
    for _ in range(REPETITIONS):
        for iteration, (location, sim_length, num_agents) in enumerate(tqdm.tqdm(configurations)):
            if iteration % (len(SIM_LENGTHS) * len(NUM_AGENTS)) == 0:
                for _ in range(API_WARMUP):
                    f(location, sim_length, num_agents) # Run warmup round per location...
            elapsed = f(location, sim_length, num_agents)
            ret += f"{location},{sim_length},{num_agents},{elapsed}\n"
    with open(os.path.join(SAVE_CSV_PATH, f"{f.__name__}.csv"), "w+") as f:
        f.write(ret)