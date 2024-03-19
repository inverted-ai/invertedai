from typing import Tuple, Optional, List
from collections import deque
from pygame.math import Vector2
from dataclasses import dataclass
import numpy as np
import asyncio
import pygame

import invertedai as iai
from invertedai.common import Point
from simulation.utils import MAX_HISTORY_LEN, Rectangle, get_pygame_convertors, DEBUG
from simulation.regions import QuadTree
from simulation.car import Car

Color1 = (1, 1, 1)


@dataclass
class SimulationConfig:
    """
    A collection of static configuration options for the simulation.
    """
    location: str = "carla:Town10HD"  #: in format recognized by Inverted AI API center map coordinate of the selected carla town
    map_center: Tuple[float] = (0.0, 0.0)
    map_width: float = 100
    map_height: float = 100
    map_fov: float = 100
    initialize_center: Optional[Tuple[float]] = None
    rendered_static_map: Optional[np.ndarray] = None
    agent_density: float = 10
    initialize_stride: float = 100
    fps: int = 10  #: 10 is the only value currently compatible with Inverted AI API
    traffic_count: int = 20  #: total number of vehicles to place in simulation
    episode_length: int = 20  #: in seconds
    random_seed: Optional[int] = None
    re_initialization_period: Optional[int] = None
    quadtree_reconstruction_period: int = 10
    quadtree_capacity: int = 10
    async_call: bool = True
    pygame_window: bool = True
    pygame_resolution: Tuple[int] = (1200, 1200)
    use_traffic_lights: bool = True
    external_traffic_lights_states: bool = False


class Simulation:
    """
    Stateful simulation for large maps with async calls to Inverted AI API for simultaneously driving npcs
    in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(self,
                 cfg: SimulationConfig,
                 traffic_lights_states=None
                 ):
        self.cfg = cfg
        self.location = cfg.location
        self.center = Point(x=cfg.map_center[0], y=cfg.map_center[1])
        self.initialize_center = Point(
            x=cfg.initialize_center[0], y=cfg.initialize_center[1]) if cfg.initialize_center else self.center
        self.width = cfg.map_width
        self.height = cfg.map_height
        self.agent_per_region = cfg.agent_density
        self.random_seed = cfg.random_seed
        self.initialize_stride = cfg.initialize_stride
        self.re_initialization_period = cfg.re_initialization_period
        self.quad_re_initialization = cfg.quadtree_reconstruction_period
        self.timer = 1
        self.screen = None
        self.async_call = cfg.async_call
        self.show_quadtree = False
        self.location_info = iai.location_info(location=self.location)
        self.map_fov = cfg.map_fov
        self.map_image = pygame.surfarray.make_surface(cfg.rendered_static_map)
        self.light_history = deque([], maxlen=MAX_HISTORY_LEN)
        self.light_response = None
        self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales = get_pygame_convertors(
            self.center.x - self.map_fov / 2, self.center.x + self.map_fov / 2,
            self.center.y - self.map_fov / 2, self.center.y + self.map_fov / 2,
            cfg.pygame_resolution[0], cfg.pygame_resolution[1])
        self.boundary = Rectangle(Vector2(self.cfg.map_center[0] - (self.map_fov / 2),
                                          self.cfg.map_center[1] - (self.map_fov / 2)),
                                  Vector2((self.map_fov, self.map_fov)),
                                  convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
        if cfg.pygame_window:
            self.top_left = cfg.convert_to_pygame_coords(
                self.center.x - (self.map_fov / 2), self.center.y - (self.map_fov / 2))
            self.x_scale, self.y_scale = cfg.convert_to_pygame_scales(self.map_fov, self.map_fov)

            self.screen = pygame.display.set_mode(cfg.pygame_resolution)
            pygame.display.set_caption("Quadtree")
        if self.cfg.external_traffic_lights_states and traffic_lights_states:
            self.traffic_light_states = traffic_lights_states
        self._initialize_regions()
        self.create_quadtree()

    def _initialize_regions(self, reintialize: bool = False):
        if DEBUG:
            save_birdviews_to = f"img/debug/initialize/{self.timer}"
        else:
            save_birdviews_to = None

        if reintialize:
            if self.cfg.use_traffic_lights and (len(self.light_history) > 0):
                traffic_lights_states = self.light_history
            else:
                traffic_lights_states = None

            agent_attributes = [car.agent_attributes for car in self.npcs]
            states_history = [list(car.agent_states_history) for car in self.npcs]

            initialize_response = asyncio.run(iai.utils.async_area_re_initialization(
                self.location,
                agent_attributes=agent_attributes,
                states_history=states_history,
                traffic_lights_states=traffic_lights_states,
                random_seed=self.random_seed,
                map_center=(self.initialize_center.x,self.initialize_center.y),
                width=self.width, 
                height=self.height,
                save_birdviews_to=save_birdviews_to))
        else:

            if self.cfg.use_traffic_lights:
                if not self.cfg.external_traffic_lights_states:
                    try:
                        light_response = iai.light(location=self.location)
                        self.light_response = light_response
                        traffic_lights_states = [light_response.traffic_lights_states]
                        self.traffic_light_states = light_response.traffic_lights_states
                    except BaseException:
                        traffic_lights_states = None
                else:
                    traffic_lights_states = [self.traffic_light_states]

            else:
                traffic_lights_states = None
            initialize_response = iai.utils.area_initialization(
                self.location, 
                self.agent_per_region,
                traffic_lights_states=traffic_lights_states,
                random_seed=self.random_seed,
                map_center=(self.initialize_center.x,self.initialize_center.y),
                width=self.width, 
                height=self.height, 
                stride=self.initialize_stride,
                save_birdviews_to=save_birdviews_to
            )

            npcs = [Car(agent_attributes=attr, agent_states=state, recurrent_states=rs, screen=self.screen,
                        convertor=self.cfg.convert_to_pygame_coords, cfg=self.cfg) for attr, state,
                    rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states)]

            self.npcs = npcs

        self.initialize_response = initialize_response

    def create_quadtree(self):
        quadtree = QuadTree(cfg=self.cfg, capacity=self.cfg.quadtree_capacity, boundary=self.boundary,
                            convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
        quadtree.lineThickness = 1
        quadtree.color = (0, 87, 146)
        for npc in self.npcs:
            quadtree.insert(npc)
        self.quadtree = quadtree

    async def async_drive(self):
        regions = self.quadtree.get_regions()
        await asyncio.gather(*[region.async_drive(self.traffic_light_states) for region in regions])

    def sync_drive(self):
        regions = self.quadtree.get_regions()
        for region in regions:
            region.sync_drive(self.traffic_light_states)

    @property
    def traffic_light_states(self):
        if self.cfg.use_traffic_lights:
            return self.light_history[-1]
        else:
            return None

    @traffic_light_states.setter
    def traffic_light_states(self, traffic_lights_states):
        if self.cfg.use_traffic_lights:
            self.light_history.append(traffic_lights_states)
            return True
        else:
            return False

    @property
    def agent_states(self):
        states = []
        for npc in self.npcs:
            states.append(npc.agent_states)
        return states

    @property
    def agent_attributes(self):
        attributes = []
        for npc in self.npcs:
            attributes.append(npc.agent_attributes)
        return attributes

    def drive(self):
        if self.cfg.pygame_window:
            self.screen.fill(Color1)
            self.screen.blit(pygame.transform.scale(pygame.transform.flip(
                pygame.transform.rotate(self.map_image, 90), True, False), (self.x_scale, self.y_scale)), self.top_left)
        if (self.re_initialization_period) and not (self.timer % self.re_initialization_period):
            self._initialize_regions(reintialize=True)
            self.create_quadtree()
        elif not (self.timer % self.quad_re_initialization):
            self.create_quadtree()
        self.timer += 1

        if self.cfg.use_traffic_lights and (not self.cfg.external_traffic_lights_states):
            try:
                light_response = iai.light(
                    location=self.location, recurrent_states=self.light_response.recurrent_states if self.light_response else None)
                self.light_response = light_response
                self.traffic_light_states = light_response.traffic_lights_states
            except BaseException:
                pass

        self.update_agents_in_fov()
        if self.async_call:
            asyncio.run(self.async_drive())
        else:
            self.sync_drive()
        if self.show_quadtree:
            self._show_quadtree()

        if self.cfg.pygame_window:
            pygame.display.flip()

    def update_agents_in_fov(self):
        for car in self.npcs:
            car.fov_agents = self.quadtree.queryRange(car.fov_range())

    def _show_quadtree(self):
        self.quadtree.Show(self.screen)

    def destroy_npc(self, ids: List[int]):
        self.npcs = list(filter(lambda x: x.id not in ids, self.npcs))

    def keep_npcs(self, ids: List[int]):
        self.npcs = list(filter(lambda x: x.id in ids, self.npcs))

    @property
    def agent_ids(self):
        return [npc.id for npc in self.npcs]
