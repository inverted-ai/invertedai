from invertedai.error import TryAgain
import math
from typing import List, Optional
import time
import invertedai as iai


TIMEOUT = 10


def initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=1,
    batch_size=1,
    min_speed=None,
    max_speed=None,
) -> dict:
    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            params = {
                "location": location,
                "num_agents_to_spawn": agent_count,
                "num_samples": batch_size,
                "spawn_min_speed": min_speed and int(math.ceil(min_speed / 3.6)),
                "spawn_max_speed": max_speed and int(math.ceil(max_speed / 3.6)),
            }
            initial_states = iai.session.request(model="initialize", params=params)
            response = {
                "states": initial_states["initial_condition"]["agent_states"],
                "recurrent_states": initial_states["recurrent_states"],
                "attributes": initial_states["initial_condition"]["agent_sizes"],
                "traffic_light_state": initial_states["traffic_light_state"],
                "traffic_states_id": initial_states["traffic_states_id"],
            }
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def drive(
    states: dict,
    agent_attributes: dict,
    recurrent_states: Optional[List] = None,
    get_birdviews: bool = False,
    location="CARLA:Town03:Roundabout",
    steps: int = 1,
    get_infractions: bool = False,
    traffic_states_id: str = "000:0",
    exclude_ego_agent: bool = True,
) -> dict:
    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    recurrent_states = (
        _tolist(recurrent_states) if recurrent_states is not None else None
    )  # Bx(num_predictions)xAxTx2x64

    model_inputs = dict(
        location=location,
        past_observations=dict(
            agent_states=states,
            agent_sizes=agent_attributes,
        ),
        recurrent_states=recurrent_states,
        # Expand from BxA to BxAxT_total for the API interface
        steps=steps,
        get_birdviews=get_birdviews,
        get_infractions=get_infractions,
        traffic_states_id=traffic_states_id,
        exclude_ego_agent=exclude_ego_agent,
    )

    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            return iai.session.request(model="drive", data=model_inputs)
        except Exception as e:
            # TODO: Add logger
            iai.logger.warning("Retrying")
            if timeout is not None and time.time() > start + timeout:
                raise e


def get_map(location="CARLA:Town03:Roundabout", include_map_source=0):
    params = {"location": location, "include_map_source": include_map_source}
    return iai.session.request(model="map", params=params)
