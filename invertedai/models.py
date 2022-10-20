from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Literal
from enum import Enum

@dataclass
class RecurrentState:
    """
    Recurrent state used in :func:`iai.drive`.
    It should not be modified, but rather passed along as received.
    """
    packed: List[float]  #: Internal representation of the recurrent state.

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
    """
    Dynamic state of a traffic light.

    See Also
    --------
    StaticMapActor
    """
    none = "0"  #: The light is off and will be ignored.
    green = "1"
    yellow = "2"
    red = "3"


@dataclass
class AgentAttributes:
    """
    Static attributes of the agent, which don't change over the course of a simulation.
    We assume every agent is a rectangle obeying a kinematic bicycle model.

    See Also
    --------
    AgentState
    """
    length: float  #: Longitudinal extent of the agent, in meters.
    width: float  #: Lateral extent of the agent, in meters.
    rear_axis_offset: float  #: Distance from the agent's center to its rear axis. Determines motion constraints.

    def tolist(self):
        return [self.length, self.width, self.rear_axis_offset]


@dataclass
class AgentState:
    """
    The current or predicted state of a given agent at a given point.

    See Also
    --------
    AgentAttributes
    """
    x: float  #: Agent's center in the locations designated coordinate frame, in meters.
    y: float  #: Agent's center in the locations designated coordinate frame, in meters.
    orientation: float  #: The direction the agent is facing, in radians with 0 pointing along x and pi/2 pointing along y.
    speed: float  #: In meters per second, negative if the agent is reversing.

    def tolist(self):
        return [self.x, self.y, self.orientation, self.speed]


@dataclass
class InfractionIndicators:
    """
    Infractions committed by a given agent, as returned from :func:`iai.drive`.
    """
    collisions: bool  #: True if the agent's bounding box overlaps with another agent's bounding box.
    offroad: bool  #: True if the agent is outside the designated driveable area specified by the map.
    wrong_way: bool  #: True if the cross product of the agent's and its lanelet's directions is negative.


@dataclass
class DriveResponse:
    """
    Response returned from an API call to :func:`iai.drive`.
    """
    agent_states: List[AgentState]  #: Predicted states for all agents at the next time step.
    recurrent_states: List[RecurrentState]  #: To pass to :func:`iai.drive` at the subsequent time step.
    bird_view: Optional[Image]  #: If `get_birdview` was set, this contains the resulting image.
    infractions: Optional[List[InfractionIndicators]]  #: If `get_infractions` was set, they are returned here.
    is_inside_supported_area: List[bool]  #: For each agent, indicates whether the predicted state is inside supported area.


@dataclass
class InitializeResponse:
    """
    Response returned from an API call to :func:`iai.initialize`.
    """
    recurrent_states: List[RecurrentState]  #: To pass to :func:`iai.drive` at the first time step.
    agent_states: List[AgentState]  #: Initial states of all initialized agents.
    agent_attributes: List[AgentAttributes]  #: Static attributes of all initialized agents.


@dataclass
class StaticMapActor:
    """
    Specifies a traffic light placement. We represent traffic lights as rectangular bounding boxes
    of the associated stop lines, with orientation matching the direction of traffic
    going through it.

    See Also
    --------
    TrafficLightState
    """
    track_id: int  #: ID as used in :func:`iai.initialize` and :func:`iai.drive`.
    agent_type: Literal["traffic-light"]  #: Not currently used, there may be more traffic signals in the future.
    x: float  #: Stop line's center in the location's coordinate frame.
    y: float  #: Stop line's center in the location's coordinate frame.
    orientation: float  #: Natural direction of traffic going through the stop line, in radians like in :class:`AgentState`.
    length: float  #: Size of the stop line, in meters, along its `orientation`.
    width: float  #: Size of the stop line, in meters, across its `orientation`.


@dataclass
class LocationResponse:
    """
    Response returned from an API call to :func:`iai.location_info`.
    """
    version: str  #: Map version. Matches the version in the input location string, if one was specified.
    max_agent_number: int  #: Maximum number of agents recommended in the location. Use more at your own risk.
    bounding_polygon: Optional[
        List[Point]
    ]  #: Convex polygon denoting the boundary of the supported area within the location.
    birdview_image: Image  #: Visualization of the location.
    osm_map: Optional[Map]  #: Underlying map annotation, returned if `include_map_source` was set.
    static_actors: List[StaticMapActor]  #: Lists traffic lights with their IDs and locations.
