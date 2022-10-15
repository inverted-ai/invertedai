"""
Python SDK for

Functions
---------
.. autosummary::
   :toctree: generated/
    available_locations
    drive
    location_info
    initialize
"""
from invertedai.error import TryAgain
from typing import List, Optional
import time
import invertedai as iai
from invertedai.models import LocationResponse, InitializeResponse, DriveResponse

TIMEOUT = 10


def available_locations(*args: str) -> List[str]:
    """
    Searching the available locations using the provided keywords as *args

    Parameters
    ----------
    *args : str
        Variable length of keywords.
        Provide up to three strings.

    Returns
    -------
    response : List[str]
        A list of "available locations" to your account (api-key)

    See Also
    --------
    invertedai.location_info

    Notes
    -----

    Examples
    --------
    >>> import invertedai as iai
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


def location_info(
    location: str = "CARLA:Town03:Roundabout", include_map_source: bool = True
) -> LocationResponse:
    """
    Providing map information, i.e., rendered image, map in OSM format,
    dictionary of static agents (traffic lights and traffic signs).

    Parameters
    ----------
    location : str
        Name of the location.

    include_map_source: bool
        Flag for requesting the map in Lanelet-OSM format.

    Returns
    -------
    Response : LocationResponse


    See Also
    --------
    invertedai.available_locations

    Notes
    -----

    Examples
    --------
    >>> import invertedai as iai
    >>> response = iai.location_info(location="")
    >>> if response["lanelet_map_source"] is not None:
    >>>     file_path = "map.osm"
    >>>     with open(file_path, "w") as f:
    >>>         f.write(response["lanelet_map_source"])
    >>> if response["rendered_map"] is not None:
    >>>     file_path = "map.jpg"
    >>>     rendered_map = np.array(response["rendered_map"], dtype=np.uint8)
    >>>     image = cv2.imdecode(rendered_map, cv2.IMREAD_COLOR)
    >>>     cv2.imwrite(file_path, image)
    """

    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "include_map_source": include_map_source}
    while True:
        try:
            response = iai.session.request(model="location_info", params=params)
            return LocationResponse(**response)
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=1,
) -> InitializeResponse:
    """
    Parameters
    ----------
    location : str
        Name of the location.

    agent_count : int
        Number of cars to spawn on the map

    min_speed : Optional[int]
        Not available yet, (for setting the minimum speed of spawned cars)

    max_speed : Optional[int]
        Not available yet, (for setting the minimum speed of spawned cars)

    Returns
    -------
    Response : InitializeResponse

    See Also
    --------
    invertedai.drive

    Notes
    -----

    Examples
    --------
    >>> import invertedai as iai
    >>> response = iai.initialize(location="CARLA:Town03:Roundabout", agent_count=10)
    """

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
                "include_recurrent_states": include_recurrent_states,
            }
            initial_states = iai.session.request(model="initialize", params=params)
            agents_spawned = len(initial_states["agent_states"])
            if agents_spawned != agent_count:
                iai.logger.warning(
                    f"Unable to spawn a scenario for {agent_count} agents,  {agents_spawned} spawned instead."
                )
            response = InitializeResponse(**initial_states)
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def drive(
    location: str = "CARLA:Town03:Roundabout",
    agent_states: list = [],
    agent_sizes: list = [],
    recurrent_states: Optional[List] = None,
    get_birdviews: bool = False,
    steps: int = 1,
    get_infractions: bool = False,
    traffic_states_id: str = "000:0",
    exclude_ego_agent: bool = True,
    present_mask: Optional[List] = None,
) -> DriveResponse:
    """
    Parameters
    ----------
    location : str
        Name of the location.

    states : List[List[Tuple[(float,) * 4]]] (AxTx4)
        List of positions and speeds of agents.
        List of A (number of actors) lists,
        each element is of T (number of time steps) list,
        each element is a list of 4 floats (x,y,speed, orientation)

    agent_attributes : List[Tuple[(float,) * 3]]  (Ax3)
        List of agent attributes
        List of A (number of actors) lists,
        each element is a list of x floats (width, length, lr)

    recurrent_states : List[Tuple[(Tuple[(float,) * 64],) * 2]] (Ax2x64)
        Internal state of simulation, which must be fedback to continue simulation
        This should have been obtained either from iai.drive or iai.initialize.

    get_birdviews: bool = False
        If True, a rendered bird's-eye view of the map with agents is returned

    steps: int
        Number of time-steps to run the simulation

    get_infractions: bool = False
        If True, 'collision', 'offroad', 'wrong_way' infractions of each agent
        is returned.

    traffic_states_id: str = "000:0"
        An id to set the state of the traffic-lights.
        If Traffic-lights are controlled by Inverted-AI
        This parameter should be set using "traffic_states_id" returned by
        "invertedia.initialize" or "invertedai.drive"

    exclude_ego_agent: bool = True,
        This parameter will be deprecated soon.

    present_mask: Optional[List] = None
        A list of booleans of size A (number of agents), which is false when
        an agent has crossed the boundary of the map.
        Set to None, or use "present_mask" returned by previous calls to DRIVE.


    Returns
    -------
    Response : DriveResponse

    See Also
    --------
    invertedai.initialize

    Notes
    -----

    Examples
    --------
    >>> import invertedai as iai
    >>> response = iai.drive(
            location="CARLA:Town03:Roundabout",
            agent_attributes=response["attributes"],
            states=response["states"],
            recurrent_states=response["recurrent_states"],
            traffic_states_id=response["traffic_states_id"],
            steps=1,
            get_birdviews=True,
            get_infractions=True,)
    """

    def _tolist(input_data: List):
        if not isinstance(input_data, list):
            return input_data.tolist()
        else:
            return input_data

    recurrent_states = (
        _tolist(recurrent_states) if recurrent_states is not None else None
    )  # AxTx2x64

    model_inputs = dict(
        location=location,
        agent_states=agent_states,
        agent_sizes=agent_sizes,
        recurrent_states=recurrent_states,
        # Expand from A to AxT_total for the API interface
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
            response = iai.session.request(model="drive", data=model_inputs)
            out = DriveResponse(**response)
            return out
        except Exception as e:
            iai.logger.warning("Retrying")
            if timeout is not None and time.time() > start + timeout:
                raise e


from dataclasses import dataclass
from typing import Tuple
