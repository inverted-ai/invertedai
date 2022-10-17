from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

AgentStates = List[List[Tuple[(float,) * 4]]]  # AxT_obs x4
AgentSizes = List[Tuple[(float,) * 3]]  # Ax3
RecurrentStates = List[Tuple[Tuple]]  # Ax2x64


class ControlStateType(Enum):
    none = "none"
    green = "green"
    yellow = "yellow"
    red = "red"


@dataclass
class TrafficLightStates:
    id: str
    states: List[ControlStateType]


@dataclass
class AgentStatesWithSample:
    x: List[List[List[float]]]  # BxAxT_obs
    y: List[List[List[float]]]  # BxAxT_obs
    psi: List[List[List[float]]]  # BxAxT_obs
    speed: List[List[List[float]]]  # BxAxT_obs


@dataclass
class DriveResponse:
    agent_states: AgentStates
    present_mask: List[float]  # A
    recurrent_states: RecurrentStates  # Ax2x64
    bird_view: List[int]
    traffic_light_state: Optional[Dict[str, ControlStateType]]
    collision: Optional[List[List[float]]] = None  # AxT_obs
    offroad: Optional[List[List[float]]] = None  # AxT_obs
    wrong_way: Optional[List[List[float]]] = None  # AxT_obs
    traffic_states_id: Optional[str] = None


@dataclass
class DrivePayload:
    """
    agent_states : List[List[Tuple[(float,) * 4]]] (AxTx4)
        List of positions and speeds of agents.
        List of A (number of actors) lists,
        each element is of T (number of time steps) list,
        each element is a list of 4 floats (x,y,speed, orientation)

    present_mask : List[int]
        A list of booleans of size A (number of agents), which is false when
        an agent has crossed the boundary of the map.

    recurrent_states : List[Tuple[(Tuple[(float,) * 64],) * 2]] (Ax2x64)
        Internal state of simulation, which must be fedback to continue simulation

    attributes : List[Tuple[(float,) * 3]]  (Ax3)
        List of agent attributes
        List of A (number of actors) lists,
        each element is a list of x floats (width, length, lr)

    traffic_light_state: Dict[str, str]
        Dictionary of traffic light states.
        Keys are the traffic-light ids and
        values are light state: 'red', 'green', 'yellow' and 'red'

    traffic_state_id: str
        The id of the current stat of the traffic light,
        which must be fedback to get the next state of the traffic light

    bird_view : List[int]
        Rendered image of the amp with agents encoded in JPEG format,
        (for decoding use JPEG decoder
        e.g., cv2.imdecode(response["rendered_map"], cv2.IMREAD_COLOR) ).

    collision : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
        List of collision infraction for each of the agents.
        List of A (number of actors) lists,
        each element is a list of size T_obs+T (number of time steps)
        floats (intersection over union).

    offroad : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
        List of offroad infraction for each of the agents.
        List of A (number of actors) lists,
        each element is a list of size T_obs+T (number of time steps) floats.

    wrong_way : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
        List of wrong_way infraction for each of the agents.
        List of A (number of actors) lists,
        each element is a list of size T_obs+T (number of time steps) floats.
    """

    location: str
    agent_states: AgentStates
    agent_sizes: AgentSizes
    steps: int
    recurrent_states: Optional[RecurrentStates]
    get_birdviews: bool = False
    get_infractions: bool = False
    include_traffic_controls: bool = False
    traffic_lights_states: Optional[List[TrafficLightStates]] = None
    exclude_ego_agent: bool = False
    traffic_states_id: Optional[str] = None
    present_mask: Optional[List[bool]] = None  # xA


@dataclass
class InitializeResponse:
    """
    agent_states : List[List[Tuple[(float,) * 4]]] (AxTx4)
        List of positions and speeds of agents.
        List of A (number of actors) lists,
        each element is a list of size T (number of time steps),
        each element is a list of 4 floats (x,y,speed, orientation)

    recurrent_states : List[Tuple[(Tuple[(float,) * 64],) * 2]] (Ax2x64)
        Internal state of simulation, which must be fedback to continue simulation

    agent_sizes : List[Tuple[(float,) * 3]]  (Ax3)
        List of agent attributes
        List of A (number of actors) lists,
        each element is a list of x floats (width, length, lr)

    traffic_light_state: Dict[str, str]
        Dictionary of traffic light states.
        Keys are the traffic-light ids and
        values are light state: 'red', 'green', 'yellow' and 'red'

    traffic_state_id: str
        The id of the current stat of the traffic light,
        which must be fedback to get the next state of the traffic light

    """

    agent_states: AgentStates
    agent_sizes: AgentSizes
    recurrent_states: Optional[RecurrentStates]
    traffic_light_state: Optional[Dict[str, ControlStateType]]
    traffic_states_id: Optional[str]
    present_mask: Optional[List[bool]] = None  # xA


@dataclass
class LocationResponse:
    """
    rendered_map : List[int]
        Rendered image of the amp encoded in JPEG format
        (for decoding use JPEG decoder
        e.g., cv2.imdecode(response["rendered_map"], cv2.IMREAD_COLOR) ).

    lanelet_map_source : str
        Serialized XML file of the OSM map.
        save the map by write(response["lanelet_map_source"])
    static_actors : List[Dict]
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
                The length of the actor
            <width> : float
                The width of the actor

    """

    rendered_map: List[int]
    lanelet_map_source: Optional[str]
    static_actors: Optional[List[Dict]]
