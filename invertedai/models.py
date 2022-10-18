from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Literal
from enum import Enum

RecurrentState = List[float]  # Recurrent Dim
TrafficLightId = int
Point = Tuple[float, float]
Origin = Tuple[
    float, float
]  # lat/lon of the origin point use to project the OSM map to UTM
Map = Tuple[str, Origin]  # Serialized OSM file and the associated origin point
Image = List[int]  # Images  encoded in JPEG format
# for decoding use a JPEG decoder
# such as cv2.imdecode(birdview_image: Image, cv2.IMREAD_COLOR).


class TrafficLightState(Enum):
    none = "0"
    green = "1"
    yellow = "2"
    red = "3"


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
class InfractionIndicators:
    collisions: bool
    offroad: bool
    wrong_way: bool


@dataclass
class DriveResponse:
    agent_states: List[AgentState]
    recurrent_states: List[RecurrentState]  # Ax2x64
    bird_view: Optional[Image]
    infractions: Optional[List[InfractionIndicators]]
    is_inside_supported_area: List[bool]  # A


@dataclass
class InitializeResponse:
    recurrent_states: List[RecurrentState]
    agent_states: List[AgentState]
    agent_attributes: List[AgentAttributes]


@dataclass
class StaticMapActor:
    track_id: int
    agent_type: Literal["traffic-light"]  # Kept for possible changes in the future
    x: float
    y: float
    psi_rad: float
    length: float
    width: float


@dataclass
class LocationResponse:
    version: str
    max_agent_number: int
    bounding_polygon: Optional[
        List[Point]
    ]  # “inner” polygon – the map may extend beyond this
    birdview_image: Image
    osm_map: Optional[Map]
    static_actors: List[StaticMapActor]
