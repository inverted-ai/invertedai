import collections
from dataclasses import dataclass
import carla
import math
import numpy as np
from collections import namedtuple

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


def npcs_in_roi(npcs, config):
    actor_geo_centers = []
    actors = []
    states = []
    dimenstions = []
    for actor in npcs:
        actor_geo_center = actor.get_location()

        distance = math.sqrt(
            ((actor_geo_center.x - config.roi_center["x"]) ** 2)
            + ((actor_geo_center.y - config.roi_center["y"]) ** 2)
        )
        if distance < config.proximity_threshold:
            length, width, lr = get_actor_dimensions(actor)
            xs, ys, psis, vs = get_actor_state(actor)
            states.append([xs, ys, psis, vs])
            dimenstions.append([length, width, lr])
            actor_geo_centers.append(actor_geo_center)
            actors.append(actor)
    return dict(
        actors=actors,
        actor_geo_centers=actor_geo_centers,
        states=states,
        dimenstions=dimenstions,
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
