import sys
import os

sys.path.append("../")
os.environ["DEV"] = "1"

import collections
from invertedai_drive import Drive, Config
from dataclasses import dataclass
import carla
import math
import numpy as np
from collections import namedtuple
import socket
from typing import Dict

cord = namedtuple("cord", ["x", "y"])
# TODO: Refactor: actor.get_trasform is called several times can be done in the
#                 most outer loop
#
@dataclass
class simulation_config:
    roi_center: cord = cord(x=0, y=0)  # region of interest center
    map_name: str = "Town03"
    fps: int = 30
    traffic_count: int = 100
    simulation_time: int = 10  # In Seconds
    proximity_threshold: int = 50
    entrance_interval: int = 2  # In Seconds
    slack: int = 3


def npcs_in_roi(npcs, ego, config, world, flag_npcs=True):
    actor_geo_centers = []
    actors = []
    states = []
    dimenstions = []
    recurrent_states = []
    debug = world.debug
    for actor, recurrent_state in zip(npcs["actors"], npcs["recurrent_states"]):
        actor_geo_center = actor.get_location()

        distance = math.sqrt(
            ((actor_geo_center.x - config.roi_center["x"]) ** 2)
            + ((actor_geo_center.y - config.roi_center["y"]) ** 2)
        )
        if distance < config.proximity_threshold + config.slack:
            length, width, lr = get_actor_dimensions(actor)
            xs, ys, psis, vs = get_actor_state(actor)
            states.append([xs, ys, psis, vs])
            dimenstions.append([length, width, lr])
            actor_geo_centers.append(actor_geo_center)
            actors.append(actor)
            recurrent_states.append(recurrent_state)
            if flag_npcs:
                flag_npc(actor_geo_center, debug)
        else:
            try:
                actor.set_autopilot(False)
            except:
                pass
            actor.destroy()

    return dict(
        actors=actors,
        actor_geo_centers=actor_geo_centers,
        states=states,
        dimenstions=dimenstions,
        recurrent_states=recurrent_states,
    )


def get_actor_dimensions(actor):
    bb = actor.bounding_box.extent
    length = max(
        2 * bb.x, 1.0
    )  # provide minimum value since CARLA returns 0 for some agents
    width = max(2 * bb.y, 0.2)
    physics_control = actor.get_physics_control()
    # Wheel position is in centimeter: https://github.com/carla-simulator/carla/issues/2153
    rear_left_wheel_position = physics_control.wheels[2].position / 100
    rear_right_wheel_position = physics_control.wheels[3].position / 100
    real_mid_position = 0.5 * (rear_left_wheel_position + rear_right_wheel_position)
    actor_geo_center = actor.get_location()
    lr = actor_geo_center.distance(real_mid_position)
    # front_left_wheel_position = physics_control.wheels[0].position / 100
    # lf = front_left_wheel_position.distance(rear_left_wheel_position) - lr
    # max_steer_angle = math.radians(physics_control.wheels[0].max_steer_angle)
    # vehicles_stats.extend([lr, lf, length, width, max_steer_angle])
    return (length, width, lr)


def get_actor_state(actor):
    t = actor.get_transform()
    loc, rot = t.location, t.rotation
    xs = loc.x
    ys = loc.y
    psis = np.radians(rot.yaw)
    v = actor.get_velocity()
    vs = np.sqrt(v.x**2 + v.y**2)
    return (xs, ys, psis, vs)


def iai_drive(npcs, drive):
    pass


def get_available_port(subsequent_ports: int = 2) -> int:
    """
    Finds an open port such that the given number of subsequent ports are also available.
    The default number of two ports corresponds to what is required by the CARLA server.

    :param subsequent_ports: How many more subsequent ports need to be free.
    """

    # CARLA server needs three consecutive ports.
    # It is possible for some other process to grab the sockets
    # between finding them here and starting the server,
    # but it's generally unlikely.
    limit = 1000
    for attempt in range(limit):
        first = socket.socket()
        subsequent = [socket.socket() for i in range(subsequent_ports)]
        try:
            first.bind(("", 0))
            port = first.getsockname()[1]
            for i in range(len(subsequent)):
                subsequent[i].bind(("", port + i + 1))
            return port
        except OSError as e:
            if attempt + 1 == limit:
                raise RuntimeError(
                    "Failed to find required ports in %d attempts" % limit
                ) from e
        finally:
            first.close()
            for s in subsequent:
                s.close()
    assert False  # this line should be unreachable


def flag_npc(loc, debug):
    loc.z += 3
    debug.draw_point(
        location=loc,
        size=0.1,
        color=carla.Color(0, 255, 0, 0),
        life_time=0.2,
    )


def to_transform(poses: list) -> list:
    t = []
    for pos in poses:
        loc = carla.Location(x=pos[0][0], y=pos[0][1])
        rot = carla.Rotation(yaw=pos[0][2])
        t.append(carla.Transform(loc, rot))
    return t


def from_transform(t: carla.Transform) -> Dict[str, float]:
    loc = t.location
    rot = t.rotation
    d = {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "yaw": rot.yaw,
    }
    return d


def get_entrance(spawn_points, config):
    slack = 1
    entrance = []
    for sp in spawn_points:
        distance = math.sqrt(
            ((sp.location.x - config.roi_center["x"]) ** 2)
            + ((sp.location.y - config.roi_center["y"]) ** 2)
        )
        if (
            config.proximity_threshold - slack
            < distance
            < config.proximity_threshold + slack
        ):
            entrance.append(sp)
    return entrance


def get_roi_spawn_points(spawn_points, config):
    roi_spawn_points = []
    for sp in spawn_points:
        distance = math.sqrt(
            ((sp.location.x - config.roi_center["x"]) ** 2)
            + ((sp.location.y - config.roi_center["y"]) ** 2)
        )
        if distance < config.proximity_threshold:
            roi_spawn_points.append(sp)
    return roi_spawn_points
