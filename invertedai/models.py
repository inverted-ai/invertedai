from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Literal
from enum import Enum

RecurrentStates = List[float]  # Recurrent Dim
TrafficLightId = str
Point = Tuple[float, float]
Origin = Tuple[
    float, float
]  # lat/lon of the origin point use to project the OSM map to UTM
Map = Tuple[str, Origin]  # serialized OSM file and the associated origin point
Image = List[int]  # Map birdview  encoded in JPEG format
# (for decoding use a JPEG decoder
# such as cv2.imdecode(birdview_image: Image, cv2.IMREAD_COLOR) ).


class TrafficLightState(Enum):
    none = "0"
    green = "1"
    yellow = "2"
    red = "3"


@dataclass
class Location:
    name: str
    version: Optional[str]

    def __str__(self):
        if self.version is not None:
            return f"{self.name}:{self.version}"
        else:
            return f"{self.name}"


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
    present_mask: List[bool]  # A
    recurrent_states: List[RecurrentStates]  # Ax2x64
    bird_view: Optional[Image]
    infractions: Optional[InfractionIndicators]


@dataclass
class InitializeResponse:
    agent_states: List[AgentState]
    agent_attributes: List[AgentAttributes]
    recurrent_states: List[RecurrentStates]


@dataclass
class StaticMapActors:
    track_id: int
    agent_type: Literal["traffic-light", "stop-sign"]
    x: float
    y: float
    psi_rad: float
    length: float
    width: float


@dataclass
class LocationResponse:
    version: str
    max_agent_number: int
    birdview_image: Image
    bounding_polygon: Optional[
        List[Point]
    ]  # “inner” polygon – the map may extend beyond this
    # birdview_image:
    osm_map: Optional[Map]
    static_actors: Optional[List[StaticMapActors]]
