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
from typing import List, Optional, Dict
import time
import invertedai as iai
from invertedai.models import (
    LocationResponse,
    InitializeResponse,
    DriveResponse,
    AgentState,
    AgentAttributes,
    TrafficLightId,
    TrafficLightState,
    InfractionIndicators,
    StaticMapActors,
    RecurrentStates,
    TrafficLightStates,
)

TIMEOUT = 10


def location_info(
    location: str = "iai:ubc_roundabout", include_map_source: bool = True
) -> LocationResponse:
    """
    Providing map information, i.e., rendered Bird's-eye view image, map in OSM format,
    list of static agents (traffic lights and traffic signs).

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
    invertedai.initialize

    Notes
    -----

    Examples
    --------
    >>> import invertedai as iai
    >>> response = iai.location_info(location="")
    >>> if response.osm_map is not None:
    >>>     file_path = f"{file_name}.osm"
    >>>     with open(file_path, "w") as f:
    >>>         f.write(response.osm_map[0])
    >>> if response.birdview_image is not None:
    >>>     file_path = f"{file_name}.jpg"
    >>>     rendered_map = np.array(response.birdview_image, dtype=np.uint8)
    >>>     image = cv2.imdecode(rendered_map, cv2.IMREAD_COLOR)
    >>>     cv2.imwrite(file_path, image)
    """

    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "include_map_source": include_map_source}
    while True:
        try:
            response = iai.session.request(model="location_info", params=params)
            if response["static_actors"] is not None:
                response["static_actors"] = [
                    StaticMapActors(**actor) for actor in response["static_actors"]
                ]
            return LocationResponse(**response)
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def initialize(
    location: str = "iai:ubc_roundabout",
    agent_count: int = 1,
    agent_attributes: List[AgentAttributes] = [],
    states_history: Optional[List[List[AgentState]]] = [],
    traffic_light_state_history: Optional[TrafficLightStates] = None,
) -> InitializeResponse:
    """
    Parameters
    ----------
    location : str
        Name of the location.

    agent_count : int
        Number of cars to spawn on the map

    agent_attributes : List[AgentAttributes]
        List of agent attributes

    states_history: [List[List[AgentState]]]
       History of agent states

    traffic_light_state_history: Optional[List[Dict[TrafficLightId, TrafficLightState]]]
       History of traffic light states

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
    >>> response = iai.initialize(location="iai:ubc_roundabout", agent_count=10)
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
            response = InitializeResponse(
                agent_states=[
                    AgentState(*state) for state in initial_states["agent_states"]
                ],
                agent_attributes=[
                    AgentAttributes(*attr)
                    for attr in initial_states["agent_attributes"]
                ],
                recurrent_states=initial_states["recurrent_states"],
            )
            return response
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))


def drive(
    location: str = "iai:ubc_roundabout",
    agent_states: List[AgentState] = [],
    agent_attributes: List[AgentAttributes] = [],
    recurrent_states: Optional[RecurrentStates] = None,
    get_birdviews: bool = False,
    get_infractions: bool = False,
    traffic_lights_states: Optional[TrafficLightStates] = None,
) -> DriveResponse:
    """
    Parameters
    ----------
    location : str
        Name of the location.

    agent_states : List[AgentState]
        List of agent states.

    agent_attributes : List[AgentAttributes]
        List of agent attributes

    recurrent_states : List[RecurrentStates]
        Internal simulation state

    get_birdviews: bool = False
        If True, a rendered bird's-eye view of the map with agents is returned

    get_infractions: bool = False
        If True, 'collision', 'offroad', 'wrong_way' infractions of each agent
        is returned.

    traffic_light_state_history: Optional[List[TrafficLightStates]]
       Traffic light states

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
    >>> response = iai.initialize(location="iai:ubc_roundabout", agent_count=10)
    >>> agent_attributes = response.agent_attributes
    >>> for _ in range(10):
    >>>     response = iai.drive(
                location="iai:ubc_roundabout",
                agent_attributes=agent_attributes,
                agent_states=response.agent_states,
                recurrent_states=response.recurrent_states,
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
        agent_states=[state.tolist() for state in agent_states],
        agent_attributes=[state.tolist() for state in agent_attributes],
        recurrent_states=recurrent_states,
        traffic_lights_states=traffic_lights_states,
        get_birdviews=get_birdviews,
        get_infractions=get_infractions,
    )

    start = time.time()
    timeout = TIMEOUT

    while True:
        try:
            response = iai.session.request(model="drive", data=model_inputs)

            out = DriveResponse(
                agent_states=[AgentState(*state) for state in response["agent_states"]],
                recurrent_states=response["recurrent_states"],
                bird_view=response["bird_view"],
                infractions=InfractionIndicators(
                    collisions=response["collision"],
                    offroad=response["offroad"],
                    wrong_way=response["wrong_way"],
                ),
                present_mask=response["present_mask"],
            )

            return out
        except Exception as e:
            iai.logger.warning("Retrying")
            if timeout is not None and time.time() > start + timeout:
                raise e
