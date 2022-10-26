"""
This module provides definitions of classes encapsulating a CARLA simulation.
"""
from invertedai.common import AgentAttributes, Point, AgentState, RecurrentState
from dataclasses import dataclass, field
import carla
from carla import Location, Rotation, Transform
import math
import numpy as np
from collections import deque
import socket
import random
import time
from typing import Tuple, List
from data.static_carla import (
    MAP_CENTERS,
    DEMO_LOCATIONS,
    NPC_BPS,
    EGO_FLAG_COLOR,
    NPC_FLAG_COLOR,
    cord,
)


@dataclass
class CarlaSimulationConfig:
    """
    A collection of static configuration options for the CARLA simulation.
    """
    ego_bp: str = "vehicle.tesla.model3"  #: blueprint name for the ego vehicle
    npc_bps: Tuple[str] = NPC_BPS  #: blueprint names for NPC vehicles
    location: str = "CARLA:Town03:Roundabout"  #: in format recognized by Inverted AI API
    fps: int = 10  #: 10 is the only value currently compatible with Inverted AI API
    traffic_count: int = 20  #: total number of vehicles to place in simulation
    episode_length: int = 20  #: in seconds
    follow_ego: bool = False  #: whether the spectator camera follows the ego vehicle
    slack: int = 3  #: in meters, how far outside supported area to track NPCs
    seed: float = time.time()  #: random number generator seed to control stochastic behavior
    flag_npcs: bool = True  #: whether to display a blue dot on Inverted AI NPCs
    flag_ego: bool = True  #: whether to display a red dot on the ego vehicle
    ego_autopilot: bool = True  #: wheter to use traffic manager to control ego vehicle
    npcs_autopilot: bool = False  #: ???
    npc_population_interval: int = 1  #: in seconds, ???
    non_roi_npc_mode: str = (
        "carla_handoff"
    )  #: ???, select from ["no_non_roi_npc", "spawn_at_entrance", "carla_handoff"]
    max_cars_in_map: int = 100  #: upper bound on how many vehicles total are allowed in simulation


class Car:
    def __init__(
        self,
        actor: carla.Actor,
        speed: float = 0.0,
        recurrent_state: RecurrentState = RecurrentState(),
    ) -> None:
        self.actor = actor
        self.recurrent_state = recurrent_state
        self._dimension = None
        self._states = deque(maxlen=10)
        self.speed = speed

    def update_dimension(self):
        self._dimension = self._get_actor_dimensions()

    def update_speed(self):
        v = self.actor.get_velocity()
        vs = np.sqrt(v.x**2 + v.y**2)
        self.speed = vs

    @property
    def dims(self):
        return self._dimension

    def get_state(self, from_carla=False):
        self.transform, state = self._get_actor_state(from_carla)
        self._states.append(state)
        return dict(
            transform=self.transform,
            states=list(self._states),
            recurrent_state=self.recurrent_state,
        )

    def set_state(self, state=None, recurrent_state=None):
        self.recurrent_state = recurrent_state
        if state is not None:
            # NOTE: state is of size 4 : [x, y, angle, speed]
            loc = carla.Location(
                state.center.x, state.center.y, self.transform.location.z
            )
            rot = carla.Rotation(
                yaw=np.degrees(state.orientation),
                pitch=self.transform.rotation.pitch,
                roll=self.transform.rotation.roll,
            )
            next_transform = carla.Transform(loc, rot)
            self.actor.set_transform(next_transform)
            self.speed = state.speed

    def _get_actor_dimensions(self):
        bb = self.actor.bounding_box.extent
        length = max(
            2 * bb.x, 1.0
        )  # provide minimum value since CARLA returns 0 for some agents
        width = max(2 * bb.y, 0.2)
        physics_control = self.actor.get_physics_control()
        # Wheel position is in centimeter: https://github.com/carla-simulator/carla/issues/2153
        rear_left_wheel_position = physics_control.wheels[2].position / 100
        rear_right_wheel_position = physics_control.wheels[3].position / 100
        real_mid_position = 0.5 * (rear_left_wheel_position + rear_right_wheel_position)
        actor_geo_center = self.actor.get_location()
        lr = actor_geo_center.distance(real_mid_position)
        return (length, width, lr)

    def _get_actor_state(self, from_carla=False):
        t = self.actor.get_transform()
        loc, rot = t.location, t.rotation
        xs = loc.x
        ys = loc.y
        psis = np.radians(rot.yaw)
        if from_carla:
            v = self.actor.get_velocity()
            vs = np.sqrt(v.x**2 + v.y**2)
        else:
            vs = self.speed
        return t, (xs, ys, psis, vs)


class CarlaEnv:
    """
    A class encapsulating a CARLA simulation, handling all the logic
    of spawning and controlling vehicles, as well as connecting to
    the server and setting simulation parameters.
    """
    def __init__(
        self,
        cfg: CarlaSimulationConfig,
        ego_spawn_point=None,
        initial_states=None,
        initial_recurrent_states=None,
        npc_entrance_spawn_points=None,
        spectator_transform=None,
    ) -> None:

        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # assemble information about area where Inverted AI NPCs will be deployed
        centers = MAP_CENTERS[cfg.location]
        self.roi_center = cord(x=centers[0], y=centers[1])
        self.proximity_threshold = (
            50
            if cfg.location not in DEMO_LOCATIONS.keys()
            else DEMO_LOCATIONS[cfg.location]["proximity_threshold"]
        )

        # connect to CARLA server and set simulation parameters
        client = carla.Client("localhost", 2000)
        traffic_manager = client.get_trafficmanager(
            get_available_port(subsequent_ports=0)
        )
        world_settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1 / float(cfg.fps),
        )
        world = client.load_world(cfg.location.split(":")[1])
        self.original_settings = client.get_world().get_settings()
        world.apply_settings(world_settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        # pick spawn points for NPCs
        if initial_states is None:
            # initial state not provided - create one
            spawn_points = world.get_map().get_spawn_points()
            (
                npc_roi_spawn_points,
                initial_speed,
                initial_recurrent_states,
            ) = self.get_roi_spawn_points(
                spawn_points, speed=np.zeros_like(spawn_points)
            )
        else:
            # initial state provided - use it
            spawn_points, speed = self._to_transform(initial_states)
            if initial_recurrent_states is not None:
                assert len(initial_recurrent_states) == len(initial_states)
            (
                npc_roi_spawn_points,
                initial_speed,
                initial_recurrent_states,
            ) = self.get_roi_spawn_points(
                spawn_points, speed, initial_recurrent_states=initial_recurrent_states
            )
        # pick a spawn point for the ego vehicle
        if (ego_spawn_point is None) or (cfg.location not in DEMO_LOCATIONS.keys()):
            # pick random spawn point for the ego vehicle
            ego_spawn_point, ego_rs, _ = (
                npc_roi_spawn_points.pop(),
                initial_recurrent_states.pop(),
                initial_speed.pop(),
            )
        elif ego_spawn_point == "demo":
            # pick one of designated spawn points for the location for the ego vehicle
            locs = DEMO_LOCATIONS[cfg.location]
            ego_spawn_point = self.rng.choice(locs["spawning_locations"])
            ego_rs = RecurrentState()
        else:
            # use the spawn point provided
            ego_rs = RecurrentState()
            assert isinstance(
                ego_spawn_point, carla.Transform
            ), "ego_spawn_point must be a Carla.Transform"

        # spawn vehicles
        if cfg.non_roi_npc_mode == "spawn_at_entrance":
            self.nroi_npc_mode = 0
            # TODO: use enum to combine self.nroi_npc_mode and cfg.non_roi_npc_mode
            if npc_entrance_spawn_points is None:
                spawn_points = world.get_map().get_spawn_points()
                npc_entrance_spawn_points = self.get_entrance(spawn_points)
            else:
                spawn_points = self._to_transform(npc_roi_spawn_points)
                npc_entrance_spawn_points, _, _ = self.get_roi_spawn_points(
                    spawn_points
                )
        elif cfg.non_roi_npc_mode == "carla_handoff":
            self.nroi_npc_mode = 1
            spawn_points = world.get_map().get_spawn_points()
            self.non_roi_spawn_points, _, _ = self.get_roi_spawn_points(
                spawn_points, roi=False
            )
        else:
            self.nroi_npc_mode = 2

        # set the spectator camera
        if spectator_transform is None:
            camera_loc = carla.Location(self.roi_center.x, self.roi_center.y, z=100)
            camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator_transform = carla.Transform(camera_loc, camera_rot)
            self.spectator_mode = "birdview"
        elif spectator_transform == "follow_ego":
            spectator_transform = carla.Transform(
                ego_spawn_point.transform(carla.Location(x=-6, z=2.5)),
                ego_spawn_point.rotation,
            )
            self.spectator_mode = "follow_ego"
        else:
            assert isinstance(
                spectator_transform, carla.Transform
            ), "spectator_transform must be a Carla.Transform"
            self.spectator_mode = "user_defined"
        self.spectator = world.get_spectator()
        self.spectator.set_transform(spectator_transform)

        # store some variables
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager
        self.roi_spawn_points = npc_roi_spawn_points
        self.initial_speed = initial_speed
        self.entrance_spawn_points = npc_entrance_spawn_points
        self.ego_spawn_point = ego_spawn_point
        self.npc_rs = initial_recurrent_states
        self.ego_rs = ego_rs

        # compute how many steps to warm up NPCs for
        self.populate_step = self.cfg.fps * self.cfg.npc_population_interval
        self.npcs = []
        self.non_roi_npcs = []

    def _initialize(self):
        """
        Initialize the simulation state by spawning all vehicles and setting
        their controllers.
        """
        # First spawn ego to ensure no NPC is spawned there
        self.ego = self._spawn_npcs(
            [self.ego_spawn_point], [0], [self.cfg.ego_bp], [self.ego_rs]
        ).pop()

        # Check that it's possible to spawn the requested number of NPCs.
        if len(self.roi_spawn_points) < self.cfg.traffic_count:
            print("Number of roi_spawn_points is less than traffic_count")
            # TODO: Add logger
        num_npcs = min(len(self.roi_spawn_points), self.cfg.traffic_count)

        # Spawn NPCs
        self.npcs.extend(
            self._spawn_npcs(
                self.roi_spawn_points[:num_npcs],
                self.initial_speed,
                self.cfg.npc_bps,
                self.npc_rs,
            )
        )

        # Spawn more NPCs outside the supported area
        if self.cfg.non_roi_npc_mode == "carla_handoff":
            num_npcs = min(len(self.non_roi_spawn_points), self.cfg.max_cars_in_map)
            self.non_roi_npcs.extend(
                self._spawn_npcs(
                    self.non_roi_spawn_points[:num_npcs],
                    [0 for _ in range(num_npcs)],
                    self.cfg.npc_bps,
                )
            )

        # Set traffic manager to control all NPCs initially
        self.set_npc_autopilot(self.npcs, True)

        # Allow the vehicles to drop to the ground
        for _ in range(10):
            self.world.tick()

        # Set controllers for all vehicles
        for npc in self.npcs:
            npc.update_dimension()
        self.ego.update_dimension()
        self.set_npc_autopilot(self.npcs, self.cfg.npcs_autopilot)
        self.set_ego_autopilot(self.cfg.ego_autopilot)
        if self.cfg.non_roi_npc_mode == "carla_handoff":
            self.set_npc_autopilot(self.non_roi_npcs, True)
        self.step_counter = 0

    def reset(self, include_ego=True):
        """
        Re-initialize simulation with the same parameters.
        """
        try:
            self.destroy(npcs=True, ego=True, world=False)
        except:
            pass
        self._initialize()
        return self.get_obs(include_ego=include_ego)

    def step(self, ego="autopilot", npcs=None, include_ego=True):
        """
        Advance the simulation using supplied NPC predictions.
        """
        self.step_counter += 1
        self._set_state_and_filter_npcs(ego, npcs, include_ego)
        if self.cfg.flag_ego:
            self._flag_npc([self.ego], EGO_FLAG_COLOR)
        if self.cfg.flag_npcs:
            self._flag_npc(self.npcs, NPC_FLAG_COLOR)
        if self.spectator_mode == "follow_ego":
            ego_transform = self.ego.transform
            spectator_transform = carla.Transform(
                ego_transform.transform(carla.Location(x=-6, z=2.5)),
                ego_transform.rotation,
            )
            self.spectator.set_transform(spectator_transform)

        self.world.tick()
        return self.get_obs()

    def destroy(self, npcs=True, ego=True, world=True):
        """
        Finish the simulation, destroying agents and optionally
        releasing the server.
        """
        if npcs:
            self._destory_npcs(self.npcs)
            self._destory_npcs(self.non_roi_npcs)
            self.npcs = []
            self.non_roi_npcs = []
        if ego:
            self._destory_npcs([self.ego])
            self.ego = None
        if world:
            self.client.get_world().apply_settings(self.original_settings)
            self.traffic_manager.set_synchronous_mode(False)

    def get_obs(self, obs_len=1, include_ego=True, warmup=False):
        """
        Obtain agent information as required by Inverted AI `drive`.
        """
        if self.cfg.non_roi_npc_mode == "spawn_at_entrance":
            if len(self.non_roi_npcs) > 0:
                for npc in self.non_roi_npcs:
                    npc.update_dimension()
                self.npcs.extend(self.non_roi_npcs)
                self.non_roi_npcs = []
                self.set_npc_autopilot(self.npcs, self.cfg.npcs_autopilot)
        states = []
        rec_state = []
        dims = []
        for npc in self.npcs:
            obs = npc.get_state(from_carla=warmup)
            states.append(obs["states"][-obs_len:])
            rec_state.append(obs["recurrent_state"])
            dims.append(npc.dims)
        if include_ego:
            obs = self.ego.get_state(from_carla=True)
            states.append(obs["states"][-obs_len:])
            rec_state.append(obs["recurrent_state"])
            dims.append(self.ego.dims)
        agent_states = [
            AgentState(
                center=Point(*state[0][:2]), orientation=state[0][2], speed=state[0][3]
            )
            for state in states
        ]
        agent_attributes = [AgentAttributes(*attr) for attr in dims]
        # recurrent_state = [RecurrentState(rs) for rs in rec_state]
        recurrent_state = rec_state
        return agent_states, recurrent_state, agent_attributes

    @staticmethod
    def set_npc_autopilot(npcs, on=True):
        for npc in npcs:
            try:
                npc.actor.set_autopilot(on)
                npc.actor.set_simulate_physics(on)
            except:
                print("Unable to set autopilot")
                # TODO: add logger

    def set_ego_autopilot(self, on=True):
        try:
            self.ego.actor.set_autopilot(on)
        except:
            print("Unable to set autopilot")
            # TODO: add logger

    @classmethod
    def from_preset_data(
        cls,
        ego_spawn_point=None,
        initial_states=None,
        npc_entrance_spawn_points=None,
        spectator_transform=None,
    ):
        cfg = CarlaSimulationConfig()
        return cls(
            cfg,
            ego_spawn_point,
            initial_states,
            npc_entrance_spawn_points,
            spectator_transform,
        )

    def _destory_npcs(self, npcs: List):
        for npc in npcs:
            try:
                npc.actor.set_autopilot(False)
            except:
                print("Unable to set autopilot")
                # TODO: add logger
            npc.actor.destroy()

    def _spawn_npcs(self, spawn_points, speeds, bps, recurrent_states=None):
        npcs = []
        if recurrent_states is None:
            recurrent_states = [RecurrentState()] * len(spawn_points)
        for spawn_point, speed, rs in zip(spawn_points, speeds, recurrent_states):
            blueprint = self.world.get_blueprint_library().find(self.rng.choice(bps))
            # ego_spawn_point = self.roi_spawn_points[i]
            actor = self.world.try_spawn_actor(blueprint, spawn_point)
            if actor is None:
                print(f"Cannot spawn NPC at:{str(spawn_point)}")
            else:
                npc = Car(actor, speed, rs)
                npcs.append(npc)
        return npcs

    def _flag_npc(self, actors, color):
        for actor in actors:
            loc = actor.actor.get_location()
            loc.z += 3
            self.world.debug.draw_point(
                location=loc,
                size=0.1,
                color=color,
                life_time=2 / self.cfg.fps,
            )

    def _set_state_and_filter_npcs(self, ego="autopilot", npcs=None, include_ego=True):
        if npcs is not None:
            states = npcs.agent_states
            recurrent_states = npcs.recurrent_states
            id = -1  # In case all NPCs vanish!
            for id, npc in enumerate(self.npcs):
                # NOTE: states is of size (batch_size x actor x time x state)
                # where state is a list: [x, y, angle, speed]
                rs = None if recurrent_states is None else recurrent_states[id]
                npc.set_state(states[id], rs)

            if include_ego:
                rs = None if recurrent_states is None else recurrent_states[id + 1]
                self.ego.set_state(recurrent_state=rs)
        exit_npcs = []
        remaining_npcs = []
        for npc in self.npcs:
            actor_geo_center = npc.get_state()["transform"].location
            distance = math.sqrt(
                ((actor_geo_center.x - self.roi_center.x) ** 2)
                + ((actor_geo_center.y - self.roi_center.y) ** 2)
            )
            if distance < self.proximity_threshold + self.cfg.slack:
                remaining_npcs.append(npc)
            else:
                exit_npcs.append(npc)
        if self.cfg.non_roi_npc_mode == "carla_handoff":
            for npc in self.non_roi_npcs:
                actor_geo_center = npc.get_state()["transform"].location
                distance = math.sqrt(
                    ((actor_geo_center.x - self.roi_center.x) ** 2)
                    + ((actor_geo_center.y - self.roi_center.y) ** 2)
                )
                if distance < self.proximity_threshold + self.cfg.slack:
                    npc.update_dimension()
                    self.set_npc_autopilot([npc], on=self.cfg.npcs_autopilot)
                    remaining_npcs.append(npc)
                    npc.update_speed()
                else:
                    exit_npcs.append(npc)
            self.non_roi_npcs = exit_npcs
            self.set_npc_autopilot(self.non_roi_npcs, True)
            exit_npcs = []
        elif self.cfg.non_roi_npc_mode == "spawn_at_entrance":
            if not (self.step_counter % self.populate_step):
                self.non_roi_npcs = self._spawn_npcs(
                    self.entrance_spawn_points,
                    (3 * np.ones_like(self.entrance_spawn_points)).tolist(),
                    self.cfg.npc_bps,
                )

        self._destory_npcs(exit_npcs)
        self.npcs = remaining_npcs

    @staticmethod
    def _to_transform(poses: list) -> Tuple[list, list]:
        t = []
        speed = []
        for pos in poses:
            loc = carla.Location(x=pos.center.x, y=pos.center.y, z=1)
            rot = carla.Rotation(yaw=np.degrees(pos.orientation))
            t.append(carla.Transform(loc, rot))
            speed.append(pos.speed)
        return (t, speed)

    def get_entrance(self, spawn_points):
        entrance = []
        for sp in spawn_points:
            distance = math.sqrt(
                ((sp.location.x - self.roi_center.x) ** 2)
                + ((sp.location.y - self.roi_center.y) ** 2)
            )
            if (
                self.proximity_threshold - self.cfg.slack
                < distance
                < self.proximity_threshold + self.cfg.slack
            ):
                entrance.append(sp)
        return entrance

    def get_roi_spawn_points(
        self, spawn_points, speed=None, roi=True, initial_recurrent_states=None
    ):
        roi_spawn_points = []
        initial_speed = []
        keep_initial_recurrent_states = []
        for ind, sp in enumerate(spawn_points):
            distance = math.sqrt(
                ((sp.location.x - self.roi_center.x) ** 2)
                + ((sp.location.y - self.roi_center.y) ** 2)
            )
            if roi & (distance < self.proximity_threshold):
                roi_spawn_points.append(sp)
                if speed is not None:
                    initial_speed.append(speed[ind])
                if initial_recurrent_states is not None:
                    keep_initial_recurrent_states.append(initial_recurrent_states[ind])
            elif (not roi) & (distance > self.proximity_threshold):
                roi_spawn_points.append(sp)
        if len(keep_initial_recurrent_states) == 0:
            keep_initial_recurrent_states = [RecurrentState()] * len(roi_spawn_points)
        return roi_spawn_points, initial_speed, keep_initial_recurrent_states


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