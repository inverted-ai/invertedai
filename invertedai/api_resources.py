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
    *args : str
        Variable length argument list of keywords.

    Returns
    -------
    response : List[str]
        A list of "available locations" to your account (api-key)

    See Also
    --------
    invertedai.location_info

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


def location_info(
    location: str = "CARLA:Town03:Roundabout", include_map_source: bool = True
) -> dict:
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
    response : Dict
        A dictionary of the json payload from the server
        <rendered_map> : List[int]
            Rendered image of the amp encoded in jpeg format.
            use cv2.imdecode(response["rendered_map"], cv2.IMREAD_COLOR)
            to decode the image
        <lanelet_map_source> : str
            Serialized XML file of the OSM map.
            save the map by write(response["lanelet_map_source"])
        <static_actors> : List[Dict]
            A list of static actors of the location, i.e, traffic signs and lights
                <track_id> : int
                     A unique ID of the actor, used to track and change state of the actor
                <agent_type> : str
                    Type of the agent, either "traffic-light", or "stop-sign"
                <x> : float
                    The x coordinate of the agent on the map
                <y> : float
                    The y coordinate of the agent on the map
                <psi_rad> : float
                    The orientation of the agent
                <length> : float
                    The lenght of the actor
                <width> : float
                    The width of the actor

    See Also
    --------
    invertedai.available_locations

    Notes
    -----

    Examples
    --------
    >>> response = iai.location_info(location=args.location)
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
    """
    Parameters
    ----------
    location : str
        Name of the location.

    agent_count : int
        Number of cars to spawn on the map

    batch_size : int
        Batch size

    min_speed : Optional[int]
        Not available yet, (for setting the minimum speed of spawned cars)

    max_speed : Optional[int]
        Not available yet, (for setting the minimum speed of spawned cars)

    Returns
    -------
    Response: Dict
        A dictionary of the json payload from the server
        <states> : List[List[List[Tuple[(float,) * 4]]]] (BxAxTx4)
            List of positions and speeds of agents.
            List of B (batch size) lists,
            each element is of size A (number of agents) lists,
            eeach element is of T (number of time steps) list,
            each elemnt is a list of 4 floats (x,y,speed, orientation)

        <recurrent_states> : List[List[Tuple[(Tuple[(float,) * 64],) * 2]]]
            Internal state of simulation, which must be fedback to continue simulation

        <attributes> : List[List[Tuple[(float,) * 3]]]  (BxAx3)
            List of agent attributes
            List of B (batch size) lists,
            each element is of size A (number of agents) lists,
            each elemnt is a list of x floats (width, lenght, lr)

        <traffic_light_state>: Dict[str, str]
            Dictionary of traffic light states.
            Keys are the traffic-light ids and
            values are light state: 'red', 'green', 'yellow' and 'red'

        <traffic_state_id>: str
            The id of the current stat of the traffic light,
            which must be fedback to get the next state of the traffic light

    See Also
    --------
    invertedai.drive

    Notes
    -----

    Examples
    --------
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
    states: list = [],
    agent_attributes: list = [],
    recurrent_states: Optional[List] = None,
    get_birdviews: bool = False,
    steps: int = 1,
    get_infractions: bool = False,
    traffic_states_id: str = "000:0",
    exclude_ego_agent: bool = True,
    present_mask: Optional[List] = None,
) -> dict:
    """
        Parameters
        ----------
        location : str
            Name of the location.

        states : List[List[List[Tuple[(float,) * 4]]]] (BxAxTx4)
            List of positions and speeds of agents.
            List of B (batch size) lists,
            each element is of size A (number of agents) lists,
            eeach element is of T (number of time steps) list,
            each elemnt is a list of 4 floats (x,y,speed, orientation)

        agent_attributes : List[List[Tuple[(float,) * 3]]]  (BxAx3)
            List of agent attributes
            List of B (batch size) lists,
            each element is of size A (number of agents) lists,
            each elemnt is a list of x floats (width, lenght, lr)

        recurrent_states : List[List[Tuple[(Tuple[(float,) * 64],) * 2]]]
            Internal state of simulation, which must be fedback to continue simulation
            This should have been obtained either from iai.drive or iai.initialize.

        get_birdviews: bool
            If True returns bird's-eye render of the map with agents

        steps: int = 1,
        get_infractions: bool = False,
        traffic_states_id: str = "000:0",
        exclude_ego_agent: bool = True,
        present_mask: Optional[List] = None,
    )

        Returns
        -------
        Response: Dict
            A dictionary of the json payload from the server
            <states> : List[List[List[Tuple[(float,) * 4]]]] (BxAxTx4)
                List of positions and speeds of agents.
                List of B (batch size) lists,
                each element is of size A (number of agents) lists,
                eeach element is of T (number of time steps) list,
                each elemnt is a list of 4 floats (x,y,speed, orientation)

            <recurrent_states> : List[List[Tuple[(Tuple[(float,) * 64],) * 2]]]
                Internal state of simulation, which must be fedback to continue simulation

            <attributes> : List[List[Tuple[(float,) * 3]]]  (BxAx3)
                List of agent attributes
                List of B (batch size) lists,
                each element is of size A (number of agents) lists,
                each elemnt is a list of x floats (width, lenght, lr)

            <traffic_light_state>: Dict[str, str]
                Dictionary of traffic light states.
                Keys are the traffic-light ids and
                values are light state: 'red', 'green', 'yellow' and 'red'

            <traffic_state_id>: str
                The id of the current stat of the traffic light,
                which must be fedback to get the next state of the traffic light

        See Also
        --------
        invertedai.drive

        Notes
        -----

        Examples
        --------
        >>> response = iai.initialize(location="CARLA:Town03:Roundabout", agent_count=10)
    """

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
