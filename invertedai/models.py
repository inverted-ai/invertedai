from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

RecurrentStates = List[float]  # Recurrent Dim
TrafficLightId = str


class TrafficLightState(Enum):
    none = "none"
    green = "green"
    yellow = "yellow"
    red = "red"


@dataclass
class Location:
    name: str
    version: Optional[str]


@dataclass
class AgentAttributes:
    length: float
    width: float
    rear_axis_offset: float

    def tolist(self):
        return [self.length, self.width, self.rear_axis_offset]


@dataclass
class AgentState:
    x: float
    y: float
    orientation: float  # in radians with 0 pointing along x and pi/2 pointing along y
    speed: float  # in m/s

    def tolist(self):
        return [self.x, self.y, self.orientation, self.speed]


@dataclass
class TrafficLightStates:
    id: str
    states: List[TrafficLightState]


@dataclass
class InfractionIndicators:
    collisions: List[bool]
    offroad: List[bool]
    wrong_way: List[bool]


@dataclass
class DriveResponse:
    agent_states: List[AgentState]
    present_mask: List[float]  # A
    recurrent_states: List[RecurrentStates]  # Ax2x64
    bird_view: List[int]
    infractions: Optional[InfractionIndicators]


# @dataclass
# class DrivePayload:
#     """
#     agent_states : List[List[Tuple[(float,) * 4]]] (AxTx4)
#         List of positions and speeds of agents.
#         List of A (number of actors) lists,
#         each element is of T (number of time steps) list,
#         each element is a list of 4 floats (x,y,speed, orientation)

#     present_mask : List[int]
#         A list of booleans of size A (number of agents), which is false when
#         an agent has crossed the boundary of the map.

#     recurrent_states : List[Tuple[(Tuple[(float,) * 64],) * 2]] (Ax2x64)
#         Internal state of simulation, which must be fedback to continue simulation

#     attributes : List[Tuple[(float,) * 3]]  (Ax3)
#         List of agent attributes
#         List of A (number of actors) lists,
#         each element is a list of x floats (width, length, lr)

#     traffic_light_state: Dict[str, str]
#         Dictionary of traffic light states.
#         Keys are the traffic-light ids and
#         values are light state: 'red', 'green', 'yellow' and 'red'

#     traffic_state_id: str
#         The id of the current stat of the traffic light,
#         which must be fedback to get the next state of the traffic light

#     bird_view : List[int]
#         Rendered image of the amp with agents encoded in JPEG format,
#         (for decoding use JPEG decoder
#         e.g., cv2.imdecode(response["rendered_map"], cv2.IMREAD_COLOR) ).

#     collision : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
#         List of collision infraction for each of the agents.
#         List of A (number of actors) lists,
#         each element is a list of size T_obs+T (number of time steps)
#         floats (intersection over union).

#     offroad : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
#         List of offroad infraction for each of the agents.
#         List of A (number of actors) lists,
#         each element is a list of size T_obs+T (number of time steps) floats.

#     wrong_way : List[Tuple[(float,) * T_obs+T]] (AxT_obs+T)
#         List of wrong_way infraction for each of the agents.
#         List of A (number of actors) lists,
#         each element is a list of size T_obs+T (number of time steps) floats.
#     """

#     location: str
#     agent_states: AgentStates
#     agent_attributes: AgentSizes
#     steps: int
#     recurrent_states: Optional[RecurrentStates]
#     get_birdviews: bool = False
#     get_infractions: bool = False
#     include_traffic_controls: bool = False
#     traffic_lights_states: Optional[List[TrafficLightStates]] = None
#     exclude_ego_agent: bool = False
#     traffic_states_id: Optional[str] = None
#     present_mask: Optional[List[bool]] = None  # xA


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

    agent_attributes : List[Tuple[(float,) * 3]]  (Ax3)
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

    agent_states: List[AgentState]
    agent_attributes: List[AgentAttributes]
    recurrent_states: List[RecurrentStates]


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
