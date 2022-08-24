from dataclasses import dataclass
import carla
from carla import Location, Rotation, Transform
import math
import numpy as np
from collections import namedtuple, deque
import socket
import random
import time
from typing import Tuple, List


TOWN03_ROUNDABOUT_DEMO_LOCATIONS = [
    Transform(Location(x=-54.5, y=-0.1, z=0.5), Rotation(pitch=0.0, yaw=1.76, roll=0.0))
]

NPC_BPS: Tuple[str] = (
    "vehicle.audi.a2",
    "vehicle.audi.etron",
    "vehicle.audi.tt",
    "vehicle.bmw.grandtourer",
    "vehicle.citroen.c3",
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020",
    "vehicle.ford.mustang",
    "vehicle.ford.crown",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.nissan.micra",
    "vehicle.nissan.patrol_2021",
    "vehicle.seat.leon",
    "vehicle.toyota.prius",
    "vehicle.volkswagen.t2_2021",
)
EGO_FLAG_COLOR = carla.Color(255, 0, 0, 0)
NPC_FLAG_COLOR = carla.Color(0, 0, 255, 0)
RS = np.zeros([2, 64]).tolist()

cord = namedtuple("cord", ["x", "y"])


@dataclass
class CarlaSimulationConfig:
    npc_bps: Tuple[str] = NPC_BPS
    roi_center: cord = cord(x=0, y=0)  # region of interest center
    map_name: str = "Town03"
    fps: int = 10
    traffic_count: int = 20
    episode_lenght: int = 20  # In Seconds
    proximity_threshold: int = 50
    entrance_interval: int = 2  # In Seconds
    follow_ego: bool = False
    slack: int = 5
    ego_bp: str = "vehicle.tesla.model3"
    seed: float = time.time()
    flag_npcs: bool = True
    flag_ego: bool = True
    ego_autopilot: bool = True
    npcs_autopilot: bool = False
    populate_npcs: bool = True
    npc_population_interval: int = 1  # In Seconds


class Car:
    def __init__(self, actor: carla.Actor, speed: float = 0.0) -> None:
        self.actor = actor
        self.recurrent_state = RS
        self._dimension = None
        self._states = deque(maxlen=10)
        self.speed = speed

    def update_dimension(self):
        self._dimension = self._get_actor_dimensions()

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
            loc = carla.Location(state[0], state[1], self.transform.location.z)
            rot = carla.Rotation(
                yaw=np.degrees(state[2]),
                pitch=self.transform.rotation.pitch,
                roll=self.transform.rotation.roll,
            )
            next_transform = carla.Transform(loc, rot)
            self.actor.set_transform(next_transform)
            self.speed = state[3]

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
    def __init__(
        self,
        config: CarlaSimulationConfig,
        ego_spawn_point=None,
        initial_states=None,
        npc_entrance_spawn_points=None,
        spectator_transform=None,
    ) -> None:
        self.rng = random.Random(config.seed)
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
            camera_loc = carla.Location(config.roi_center.x, config.roi_center.y, z=100)
            camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator_transform = carla.Transform(camera_loc, camera_rot)
        if initial_states is None:
            spawn_points = world.get_map().get_spawn_points()
            npc_roi_spawn_points, initial_speed = self.get_roi_spawn_points(
                config, spawn_points, speed=np.zeros_like(spawn_points)
            )
        else:
            spawn_points, speed = self._to_transform(initial_states)
            npc_roi_spawn_points, initial_speed = self.get_roi_spawn_points(
                config, spawn_points, speed
            )
        if npc_entrance_spawn_points is None:
            spawn_points = world.get_map().get_spawn_points()
            npc_entrance_spawn_points = self.get_entrance(config, spawn_points)
        else:
            spawn_points = self._to_transform(npc_roi_spawn_points)
            npc_entrance_spawn_points = self.get_roi_spawn_points(config, spawn_points)
        if ego_spawn_point is None:
            ego_spawn_point = self.rng.choice(TOWN03_ROUNDABOUT_DEMO_LOCATIONS)

        self.config = config
        self.spectator = world.get_spectator()
        self.spectator.set_transform(spectator_transform)
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager
        self.roi_spawn_points = npc_roi_spawn_points
        self.initial_speed = initial_speed
        self.entrance_spawn_points = npc_entrance_spawn_points
        self.ego_spawn_point = ego_spawn_point
        self.npcs = []
        self.new_npcs = []

    def _initialize(self):
        # Keep the order of first spawining ego then NPCs
        # to avoid spawning npc in ego location
        self.ego = self._spawn_npcs(
            [self.ego_spawn_point],
            [0],
            [self.config.ego_bp],
        ).pop()
        if len(self.roi_spawn_points) < self.config.traffic_count:
            print("Number of roi_spawn_points is less than traffic_count")
            # TODO: Add logger
        num_npcs = min(len(self.roi_spawn_points), self.config.traffic_count)
        self.npcs.extend(
            self._spawn_npcs(
                self.roi_spawn_points[:num_npcs],
                self.initial_speed,
                self.config.npc_bps,
            )
        )
        self.world.tick()
        for npc in self.npcs:
            npc.update_dimension()
        self.ego.update_dimension()
        self.set_npc_autopilot(self.config.npcs_autopilot)
        self.set_ego_autopilot(self.config.ego_autopilot)
        self.step_counter = 0

    def reset(self, include_ego=True):
        try:
            self.destroy()
        except:
            pass
        self._initialize()
        return self.get_obs(include_ego=include_ego)

    def step(self, ego="autopilot", npcs=None, time_step=0, include_ego=True):
        if npcs is not None:
            states = npcs["states"]
            recurrent_states = npcs["recurrent_states"]
            for id, npc in enumerate(self.npcs):
                # NOTE: states is of size (batch_size x actor x time x state)
                # state is of size 4 : [x, y, angle, speed]
                rs = None if recurrent_states is None else recurrent_states[0][id]
                npc.set_state(states[0][id][time_step], rs)

            if include_ego:
                rs = None if recurrent_states is None else recurrent_states[0][id + 1]
                self.ego.set_state(recurrent_state=rs)
        self._filter_npcs()
        self.step_counter += 1
        if self.config.flag_ego:
            self._flag_npc([self.ego], EGO_FLAG_COLOR)
        if self.config.flag_npcs:
            self._flag_npc(self.npcs, NPC_FLAG_COLOR)
        if self.config.populate_npcs & (
            not (
                self.step_counter
                % (self.config.fps * self.config.npc_population_interval)
            )
        ):
            self.new_npcs = self._spawn_npcs(
                self.entrance_spawn_points,
                (1.5 * np.ones_like(self.entrance_spawn_points)).tolist(),
                self.config.npc_bps,
            )

        self.world.tick()
        if len(self.new_npcs) > 0:
            for npc in self.new_npcs:
                npc.update_dimension()
            self.npcs.extend(self.new_npcs)
            self.new_npcs = []
            self.set_npc_autopilot(self.config.npcs_autopilot)

        return self.get_obs()

    def destroy(self, npcs=True, ego=True, world=True):
        if npcs:
            self._destory_npcs(self.npcs)
            self.npcs = []
        if ego:
            self._destory_npcs([self.ego])
            self.ego = None
        if world:
            self.client.get_world().apply_settings(self.original_settings)
            self.traffic_manager.set_synchronous_mode(False)

    def get_obs(self, obs_len=1, include_ego=True, warmup=False):
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
        return states, rec_state, dims

    def get_infractions(self):
        pass

    def get_reward(self):
        pass

    def is_done(self):
        pass

    def seed(self, seed=None):
        pass

    def render(self, mode="human"):
        pass

    def set_npc_autopilot(self, on=True):
        for npc in self.npcs:
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
        config = CarlaSimulationConfig()
        return cls(
            config,
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

    def _spawn_npcs(self, spawn_points, speeds, bps):
        npcs = []
        for spawn_point, speed in zip(spawn_points, speeds):
            blueprint = self.world.get_blueprint_library().find(self.rng.choice(bps))
            # ego_spawn_point = self.roi_spawn_points[i]
            actor = self.world.try_spawn_actor(blueprint, spawn_point)
            if actor is None:
                print(f"Cannot spawn NPC at:{str(spawn_point)}")
            else:
                npc = Car(actor, speed)
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
                life_time=2 / self.config.fps,
            )

    def _filter_npcs(self):
        exit_npcs = []
        remaining_npcs = []
        for npc in self.npcs:
            actor_geo_center = npc.get_state()["transform"].location
            distance = math.sqrt(
                ((actor_geo_center.x - self.config.roi_center.x) ** 2)
                + ((actor_geo_center.y - self.config.roi_center.y) ** 2)
            )
            if distance < self.config.proximity_threshold + self.config.slack:
                remaining_npcs.append(npc)
            else:
                exit_npcs.append(npc)
        self._destory_npcs(exit_npcs)
        self.npcs = remaining_npcs

    @staticmethod
    def _to_transform(poses: list) -> Tuple[list, list]:
        t = []
        speed = []
        for pos in poses:
            loc = carla.Location(x=pos[0][0], y=pos[0][1], z=1.5)
            rot = carla.Rotation(yaw=np.degrees(pos[0][2]))
            t.append(carla.Transform(loc, rot))
            speed.append(pos[0][3])
        return (t, speed)

    @staticmethod
    def get_entrance(config, spawn_points):
        slack = 1
        entrance = []
        for sp in spawn_points:
            distance = math.sqrt(
                ((sp.location.x - config.roi_center.x) ** 2)
                + ((sp.location.y - config.roi_center.y) ** 2)
            )
            if (
                config.proximity_threshold - slack
                < distance
                < config.proximity_threshold + slack
            ):
                entrance.append(sp)
        return entrance

    @staticmethod
    def get_roi_spawn_points(config, spawn_points, speed):
        roi_spawn_points = []
        initial_speed = []
        for ind, sp in enumerate(spawn_points):
            distance = math.sqrt(
                ((sp.location.x - config.roi_center.x) ** 2)
                + ((sp.location.y - config.roi_center.y) ** 2)
            )
            if distance < config.proximity_threshold:
                roi_spawn_points.append(sp)
                initial_speed.append(speed[ind])
        return roi_spawn_points, initial_speed


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
