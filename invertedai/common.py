from typing import List, Optional, Dict, Tuple
from enum import Enum
from pydantic import BaseModel, model_validator
import math
from PIL import Image as PImage
import numpy as np
import io
import json

import invertedai as iai
from invertedai.error import InvalidInputType, InvalidInput

RECURRENT_SIZE = 152
TrafficLightId = int


class RecurrentState(BaseModel):
    """
    Recurrent state used in :func:`iai.drive`.
    It should not be modified, but rather passed along as received.
    """

    packed: List[float] = [0.0] * RECURRENT_SIZE

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
        """
        self.encoded_image = bytes(self.encoded_image)
        img_stream = io.BytesIO(self.encoded_image)
        img = PImage.open(img_stream)
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]
        return img_array


    @classmethod
    def fromval(cls, val):
        return cls(encoded_image=val)

    def decode_and_save(self, path):
        """
        Decode the image and save it to the specified path.
        """
        image = self.decode()
        image_pil = PImage.fromarray(image)
        image_pil.save(path)


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


class LightRecurrentState(BaseModel):
    """
    Recurrent state of all the traffic lights in one light group (one intersection).
    """
    state: float
    time_remaining: float
    
    def tolist(self):
        """
        Convert LightRecurrentState to a list in this order: [state, time_remaining]
        """
        return [self.state, self.time_remaining]
    
    
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
    waypoint: Optional[Point] = None  #: Target waypoint of the agent. If provided the agent will attempt to reach it.

    @classmethod
    def fromlist(cls, l):
        length, width, rear_axis_offset, agent_type, waypoint = None, None, None, None, None    
        if len(l) == 5:
            length, width, rear_axis_offset, agent_type, waypoint = l
        elif len(l) == 4:
            if type(l[3]) == list:
                if type(l[2]) == str:
                    length, width, agent_type, waypoint = l
                else:
                    length, width, rear_axis_offset, waypoint = l
            else:
                length, width, rear_axis_offset, agent_type = l
        elif len(l) == 3:
            if type(l[2]) == list:
                length, width, waypoint = l
            elif type(l[2]) == str:
                length, width, agent_type = l
            else:
                length, width, rear_axis_offset = l
        elif len(l) == 2:
            agent_type, waypoint = l
        else:
            assert len(l) == 1, "Only a single item (agent_type or waypoint) is allowed when the size of the provided list is neither 3, 4 nor 5."
            if type(l[0]) == list:
                waypoint, = l
            else:
                agent_type, = l        
        assert type(waypoint) is list if waypoint is not None else True, "waypoint must be a list of two floats"
        return cls(length=length, width=width, rear_axis_offset=rear_axis_offset, agent_type=agent_type, waypoint=Point(x=waypoint[0], y=waypoint[1]) if waypoint is not None else None)

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
        if self.waypoint is not None:
            attr_list.append([self.waypoint.x, self.waypoint.y])
        return attr_list

class AgentProperties(BaseModel):
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
    waypoint: Optional[Point] = None  #: Target waypoint of the agent. If provided the agent will attempt to reach it.
    max_speed: Optional[float] = None  #: Maximum speed limit of the agent in m/s.

    @classmethod
    def deserialize(cls, val):
        return cls(length=val['length'], width=val['width'], rear_axis_offset=val['rear_axis_offset'], agent_type=val['agent_type'], 
                   waypoint=Point(x=val['waypoint'][0], y=val['waypoint'][1]) if val['waypoint'] else None, max_speed=val['max_speed'])
    
    def serialize(self):
        """
        Convert AgentProperties to a valid request format in json
        """
        return {"length": self.length, "width": self.width, "rear_axis_offset": self.rear_axis_offset, "agent_type": self.agent_type, 
                 "waypoint": [self.waypoint.x, self.waypoint.y] if self.waypoint else None, "max_speed": self.max_speed}
    
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
    wrong_way: bool  #: CURRENTLY DISABLED. True if the cross product of the agent's and its lanelet's directions is negative.

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
    agent_type: str  #: Supported types are "traffic_light" and "stop_sign" and "yield_sign".
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

class Scenario(BaseModel):
    name: str


class LeaderFollow(Scenario):
    leader_id: int
    follower_id: int
    gap_in_seconds: float = 1.5

class DenseRightMerge(Scenario):
    merger_id: int



TrafficLightStatesDict = Dict[TrafficLightId, TrafficLightState]
LightRecurrentStates = List[LightRecurrentState]
