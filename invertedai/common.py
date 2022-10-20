from dataclasses import dataclass
from typing import List, Tuple, Literal
from enum import Enum

import invertedai as iai


TrafficLightId = int
Origin = Tuple[
    float, float
]  # lat/lon of the origin point use to project the OSM map to UTM

@dataclass
class RecurrentState:
    """
    Recurrent state used in :func:`iai.drive`.
    It should not be modified, but rather passed along as received.
    """
    packed: List[float]  #: Internal representation of the recurrent state.

@dataclass
class Point:
    """
    2D coordinates of a point in a given location.
    Each location comes with a canonical coordinate system, where
    the distance units are meters.
    """
    x: float
    y: float


class LocationMap:
    """
    Serializable representation of a Lanelet2 map and the corresponding origin.
    To reconstruct the map locally, save the OSM file to disk and load it
    with the UTM projector using the origin provided here.
    This projection defines the canonical coordinate frame of the map.
    """
    def __init__(self, encoded_map: str, origin: Origin):
        self._encoded_map = encoded_map
        self._origin = origin

    @property
    def origin(self) -> Origin:
        """
        Origin of the map, specified as a pair of latitude and longitude coordinates.
        Allows for geolocation of the map and can be used with a UTM projector to
        construct the Lanelet2 map object in the canonical coordinate frame.
        """
        return self._origin

    def save_osm_file(self, path: str):
        """
        Save the OSM file to disk.
        """
        with open(path, "w") as f:
            f.write(self._encoded_map)


class Image:
    """
    Images sent through the API in their encoded format.
    Decoding the images requires additional dependencies on top of what invertedai uses.
    """
    def __init__(self, encoded_image: List[int]):
        self._encoded_image = encoded_image

    def decode(self):
        try:
            import numpy as np
            import cv2
        except ImportError as e:
            iai.logger.error("Decoding images requires numpy and cv2, which were not found.")
            raise e
        array = np.array(self._encoded_image, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return image

    def decode_and_save(self, path):
        """
        Decode the image and save it to the specified path.
        Requires numpy and cv2 to be available, otherwise raises `ImportError`.
        """
        image = self.decode()
        import cv2
        cv2.imwrite(path, image)



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
    rear_axis_offset: float  #: Distance from the agent's center to its rear axis in meters. Determines motion constraints.

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
    center: Point  #: The center point of the agent's bounding box.
    orientation: float  #: The direction the agent is facing, in radians with 0 pointing along x and pi/2 pointing along y.
    speed: float  #: In meters per second, negative if the agent is reversing.

    def tolist(self):
        return [self.center.x, self.center.y, self.orientation, self.speed]

    @classmethod
    def fromlist(cls, l):
        x, y, psi, v = l
        return cls(center=Point(x=x, y=y), orientation=psi, speed=v)


@dataclass
class InfractionIndicators:
    """
    Infractions committed by a given agent, as returned from :func:`iai.drive`.
    """
    collisions: bool  #: True if the agent's bounding box overlaps with another agent's bounding box.
    offroad: bool  #: True if the agent is outside the designated driveable area specified by the map.
    wrong_way: bool  #: True if the cross product of the agent's and its lanelet's directions is negative.


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
    track_id: TrafficLightId  #: ID as used in :func:`iai.initialize` and :func:`iai.drive`.
    agent_type: Literal["traffic-light"]  #: Not currently used, there may be more traffic signals in the future.
    center: Point  #: The center of the stop line.
    orientation: float  #: Natural direction of traffic going through the stop line, in radians like in :class:`AgentState`.
    length: float  #: Size of the stop line, in meters, along its `orientation`.
    width: float  #: Size of the stop line, in meters, across its `orientation`.

    @classmethod
    def fromdict(cls, d):
        d = d.copy()
        d['center'] = Point(d['x'], d['y'])
        del d['center']
        return cls(**d)
