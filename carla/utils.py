import sys
import os

sys.path.append("../")
os.environ["DEV"] = "1"

import collections
from invertedai_drive import Drive, Config
from dataclasses import dataclass
import carla
from carla import Location, Rotation, Transform
import math
import numpy as np
from collections import namedtuple
import socket
import random
from typing import Dict
import gym


TOWN03_ROUNDABOUT_DEMO_LOCATIONS = [
    Transform(
        Location(x=-54.5, y=-0.1, z=0.5), Rotation(pitch=0.0, yaw=1.76, roll=0.0)
    ),
    Transform(
        Location(x=-1.6, y=-87.4, z=0.5), Rotation(pitch=0.0, yaw=91.0, roll=0.0)
    ),
    Transform(Location(x=1.5, y=78.6, z=0.5), Rotation(pitch=0.0, yaw=-83.5, roll=0.0)),
    Transform(
        Location(x=68.1, y=-4.1, z=0.5), Rotation(pitch=0.0, yaw=178.7, roll=0.0)
    ),
]


cord = namedtuple("cord", ["x", "y"])


@dataclass
class carla_simulation_config:
    roi_center: cord = cord(x=0, y=0)  # region of interest center
    map_name: str = "Town03"
    fps: int = 30
    traffic_count: int = 100
    episode_lenght: int = 20  # In Seconds
    proximity_threshold: int = 50
    entrance_interval: int = 2  # In Seconds
    follow_ego: bool = False
    slack: int = 3


class carla_env(gym.Env):
    def __init__(
        self,
        config: carla_simulation_config,
        ego_spawn_point=None,
        npc_roi_spawn_points=None,
        npc_entrance_spawn_points=None,
        spectator_transform=None,
    ) -> None:
        world_settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1 / float(config.fps),
        )
        client = carla.Client("localhost", 2000)
        traffic_manager = client.get_trafficmanager(
            get_available_port(subsequent_ports=0)
        )
        world = client.load_world(config.map_name)
        self.original_settings = client.get_world().get_settings()
        world.apply_settings(world_settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)
        if spectator_transform is None:
            camera_loc = carla.Location(
                config.roi_center["x"], config.roi_center["y"], z=110
            )
            camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator_transform = carla.Transform(camera_loc, camera_rot)
        if npc_roi_spawn_points is None:
            spawn_points = world.get_map().get_spawn_points()
            npc_roi_spawn_points = get_roi_spawn_points(spawn_points, config)
        if npc_entrance_spawn_points is None:
            spawn_points = world.get_map().get_spawn_points()
            npc_entrance_spawn_points = get_entrance(spawn_points, config)
        if ego_spawn_point is None:
            ego_spawn_point = random.choice(TOWN03_ROUNDABOUT_DEMO_LOCATIONS)

        self.spectator = world.get_spectator()
        self.spectator.set_transform(spectator_transform)
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager
        self.roi_spawn_points = npc_roi_spawn_points
        self.entrance_spawn_points = npc_entrance_spawn_points
        self.ego_spawn_point = ego_spawn_point

    def reset(self):
        pass

    def step(self, action):
        self.simulator.step(action)
        return self.get_obs(), self.get_reward(), self.is_done(), self.get_info()

    def get_obs(self):
        pass

    def get_reward(self):
        pass

    def is_done(self):
        pass

    def get_info(self):
        x = self.simulator.get_state()[..., 0]
        info = dict(
            invasion=self.simulator.compute_offroad() > self.offroad_threshold,
            collision=self.simulator.compute_collision() > self.collision_threshold,
            gear=torch.ones_like(x, dtype=torch.long),
            expert_action=torch.zeros_like(self.prev_action),
            outcome=None,
        )
        return info

    def seed(self, seed=None):
        pass

    def render(self, mode="human"):
        pass

    @classmethod
    def from_data(cls, config):
        return cls(config)

    def _spawn_vehicle(self, spawn_points, vehicle_blue_print=None):
        if vehicle_blue_print is None:
            bp_lib = self.world.get_blueprint_library()
            vehicle_blue_print = random.choice(bp_lib.filter("vehicle"))
        return self.world.try_spawn_actor(vehicle_blue_print, spawn_points)


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
