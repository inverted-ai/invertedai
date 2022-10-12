"""
Python SDK for

Functions
---------
.. autosummary::
   :toctree: generated/
    available_locations
    drive
    get_map
    initialize
"""
from invertedai.error import TryAgain
import math
from typing import List, Optional
import time
import invertedai as iai


TIMEOUT = 10


def available_locations(*args: str):
    """
    Searching the available locations using the provided keywords as *args

    Parameters
    ----------
    *args: str
        Variable length argument list of keywords.

    Returns
    -------
    out : List[str]
        A list of "available locations" to your account (api-key)

    See Also
    --------
    invertedai.get_map

    Notes
    -----
    Providing more than three keywords is uncessary

    Examples
    --------
    >>> iai.available_locations("carla", "roundabout")
    ["CARLA:Town03:Roundabout"]
    """
    start = time.time()
    timeout = TIMEOUT
    keywords = "+".join(list(args))
    while True:
        try:
            params = {
                "keywords": keywords,
            }
            response = iai.session.request(model="available_locations", params=params)
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def get_map(
    location: str = "CARLA:Town03:Roundabout", include_map_source: bool = True
) -> dict:
    """
    Providing map information, i.e., rendered image, map in OSM format,
    dictionary of static agents (traffic lights and traffic signs).

    Parameters
    ----------
    location: str
        Name of the location.

    include_map_source: bool
        Flag for requesting the map in Lanelet-OSM format.

    Returns
    -------
    response : Dict
        <rendered_map> : List[int]
            Rendered image of the amp encoded in jpeg format.
            use cv2.imdecode(response["rendered_map"], cv2.IMREAD_COLOR)
            to decode the image
        <lanelet_map_source>
        <static_actors>


    See Also
    --------
    invertedai.get_map

    Notes
    -----
    Providing more than three keywords is uncessary

    Examples
    --------
    >>> iai.available_maps("carla", "roundabout")
    ["CARLA:Town03:Roundabout"]
    """

    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "include_map_source": include_map_source}
    while True:
        try:
            response = iai.session.request(model="get_map", params=params)
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


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
            include_recurrent_states = (
                False if location.split(":")[0] == "huawei" else True
            )
            params = {
                "location": location,
                "num_agents_to_spawn": agent_count,
                "num_samples": batch_size,
                "spawn_min_speed": min_speed
                and int(math.ceil(min_speed / 3.6))
                and not include_recurrent_states,
                "spawn_max_speed": max_speed
                and int(math.ceil(max_speed / 3.6))
                and not include_recurrent_states,
                "include_recurrent_states": include_recurrent_states,
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
    location: str = "CARLA:Town03:Roundabout",
    states: dict = {},
    agent_attributes: dict = {},
    recurrent_states: Optional[List] = None,
    get_birdviews: bool = False,
    steps: int = 1,
    get_infractions: bool = False,
    traffic_states_id: str = "000:0",
    exclude_ego_agent: bool = True,
    present_mask: Optional[List] = None,
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
        present_mask=present_mask,
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
