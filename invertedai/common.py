from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, root_validator
import math

import invertedai as iai
from invertedai.error import InvalidInputType, InvalidInput

RECURRENT_SIZE = 132
TrafficLightId = int


class RecurrentState(BaseModel):
    """
    Recurrent state used in :func:`iai.drive`.
    It should not be modified, but rather passed along as received.
    """

    packed: List[float] = [0.0] * RECURRENT_SIZE

    #: Internal representation of the recurrent state.

    @root_validator
    @classmethod
    def check_recurrentstate(cls, values):
        if len(values.get("packed")) == RECURRENT_SIZE:
            return values
        else:
            raise InvalidInput("Incorrect Recurrentstate Size.")

    @classmethod
    def fromval(cls, val):
        return cls(packed=val)


class Point(BaseModel):
    """
    2D coordinates of a point in a given location.
    Each location comes with a canonical coordinate system, where
    the distance units are meters.
    """

    x: float
    y: float

    @classmethod
    def fromlist(cls, l):
        x, y = l
        return cls(x=x, y=y)

    def __sub__(self, other):
        return math.sqrt(((self.x - other.x) ** 2) + ((self.y - other.y) ** 2))


class Origin(Point):
    # lat/lon of the origin point use to project the OSM map to UTM
    pass


class LocationMap(BaseModel):
    """
    Serializable representation of a Lanelet2 map and the corresponding origin.
    To reconstruct the map locally, save the OSM file to disk and load it
    with the UTM projector using the origin provided here.
    This projection defines the canonical coordinate frame of the map.
    Origin of the map, specified as a pair of latitude and longitude coordinates.
    Allows for geolocation of the map and can be used with a UTM projector to
    construct the Lanelet2 map object in the canonical coordinate frame.
    """
    encoded_map: str
    origin: Origin

    def save_osm_file(self, path: str):
        """
        Save the OSM file to disk.
        """
        with open(path, "w") as f:
            f.write(self.encoded_map)


class Image(BaseModel):
    """
    Images sent through the API in their encoded format.
    Decoding the images requires additional dependencies on top of what invertedai uses.
    """
    encoded_image: List[int]

    def decode(self):
        """
        Decode and return the image.
        Requires numpy and cv2 to be available, otherwise raises `ImportError`.
        """
        try:
            import numpy as np
            import cv2
        except ImportError as e:
            iai.logger.error(
                "Decoding images requires numpy and cv2, which were not found."
            )
            raise e
        array = np.array(self.encoded_image, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return image

    @classmethod
    def fromval(cls, val):
        return cls(encoded_image=val)

    def decode_and_save(self, path):
        """
        Decode the image and save it to the specified path.
        Requires numpy and cv2 to be available, otherwise raises `ImportError`.
        """
        image = self.decode()
        import cv2
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class TrafficLightState(str, Enum):
    """
    Dynamic state of a traffic light.

    See Also
    --------
    StaticMapActor
    """

    none = "none"  #: The light is off and will be ignored.
    green = "green"
    yellow = "yellow"
    red = "red"


class AgentType(str, Enum):
    car = "car"
    pedestrian = "pedestrian"


class AgentAttributes(BaseModel):
    """
    Static attributes of the agent, which don't change over the course of a simulation.
    We assume every agent is a rectangle obeying a kinematic bicycle model.

    See Also
    --------
    AgentState
    """

    length: Optional[float] = None  #: Longitudinal extent of the agent, in meters.
    width: Optional[float] = None  #: Lateral extent of the agent, in meters.
    #: Distance from the agent's center to its rear axis in meters. Determines motion constraints.
    rear_axis_offset: Optional[float] = None
    agent_type: Optional[str] = 'car'  #: Valid types are those in `AgentType`, but we use `str` here for extensibility.
    waypoints: Optional[Point] = None  #: Target waypoint of the agent. If provided the agent will attempt to reach it.

    @classmethod
    def fromlist(cls, l):
        if len(l) == 5:
            length, width, rear_axis_offset, agent_type, waypoints = l
            return cls(length=length, width=width, rear_axis_offset=rear_axis_offset, agent_type=agent_type, waypoints=waypoints)
        if len(l) == 4:
            if type(l[3]) == list:
                if type(l[2]) == str:
                    length, width, agent_type, waypoints = l
                    return cls(length=length, width=width, agent_type=agent_type, waypoints=waypoints)
                else:
                    length, width, rear_axis_offset, waypoints = l
                    return cls(length=length, width=width, rear_axis_offset=rear_axis_offset, waypoints=waypoints)
            else:
                length, width, rear_axis_offset, agent_type = l
                return cls(length=length, width=width, rear_axis_offset=rear_axis_offset, agent_type=agent_type)
        elif len(l) == 3:
            if type(l[2]) == list:
                length, width, waypoints = l
                return cls(length=length, width=width, waypoints=waypoints)
            elif type(l[2]) == str:
                length, width, agent_type = l
                return cls(length=length, width=width, agent_type=agent_type)
            else:
                length, width, rear_axis_offset = l
                return cls(length=length, width=width, rear_axis_offset=rear_axis_offset)
        else:
            assert len(l) == 1
            agent_type, = l
            return cls(agent_type=agent_type)

    def tolist(self):
        """
        Convert AgentAttributes to a flattened list of agent attributes
        in this order: [length, width, rear_axis_offset, agent_type]
        """
        attr_list = []
        if self.length is not None:
            attr_list.append(self.length)
        if self.width is not None:
            attr_list.append(self.width)
        if self.rear_axis_offset is not None:
            attr_list.append(self.rear_axis_offset)
        if self.agent_type is not None:
            attr_list.append(self.agent_type)
        if self.waypoints is not None:
            attr_list.append([self.waypoints.x, self.waypoints.y])
        return attr_list


class AgentState(BaseModel):
    """
    The current or predicted state of a given agent at a given point.

    See Also
    --------
    AgentAttributes
    """

    center: Point  #: The center point of the agent's bounding box.
    #: The direction the agent is facing, in radians with 0 pointing along x and pi/2 pointing along y.
    orientation: float
    speed: float  #: In meters per second, negative if the agent is reversing.

    def tolist(self):
        """
        Convert AgentState to flattened list of state attributes in this order: [x, y, orientation, speed]
        """
        return [self.center.x, self.center.y, self.orientation, self.speed]

    @classmethod
    def fromlist(cls, l):
        """
        Build AgentState from a list with this order: [x, y, orientation, speed]
        """
        x, y, psi, v = l
        return cls(center=Point(x=x, y=y), orientation=psi, speed=v)


class InfractionIndicators(BaseModel):
    """
    Infractions committed by a given agent, as returned from :func:`iai.drive`.
    """

    collisions: bool  #: True if the agent's bounding box overlaps with another agent's bounding box.
    offroad: bool  #: True if the agent is outside the designated driveable area specified by the map.
    wrong_way: bool  #: True if the cross product of the agent's and its lanelet's directions is negative.

    @classmethod
    def fromlist(cls, l):
        collisions, offroad, wrong_way = l
        return cls(collisions=collisions, offroad=offroad, wrong_way=wrong_way)


class StaticMapActor(BaseModel):
    """
    Specifies a traffic light placement. We represent traffic lights as rectangular bounding boxes
    of the associated stop lines, with orientation matching the direction of traffic
    going through it.

    See Also
    --------
    TrafficLightState
    """

    actor_id: TrafficLightId  #: ID as used in :func:`iai.initialize` and :func:`iai.drive`.
    agent_type: str  #: Not currently used, there may be more traffic signals in the future.
    center: Point  #: The center of the stop line.
    #: Natural direction of traffic going through the stop line, in radians like in :class:`AgentState`.
    orientation: float
    length: Optional[float]  #: Size of the stop line, in meters, along its `orientation`.
    width: Optional[float]  #: Size of the stop line, in meters, across its `orientation`.
    dependant: Optional[List[int]]  # : List of ID's of other actors that are dependant to this actor.

    @classmethod
    def fromdict(cls, d):
        """
        Build StaticMapActor from a dictionary
        with keys: `actor_id`, `agent_type`, `orientation`, `length`, `width`, `x`, `y`, `dependant`
        """
        d = d.copy()
        d["center"] = Point.fromlist([d.pop("x"), d.pop("y")])
        return cls(**d)


TrafficLightStatesDict = Dict[TrafficLightId, TrafficLightState]
