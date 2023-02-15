from typing import List, Tuple, Optional, Union
from pydantic import BaseModel, validate_arguments
from pygame.math import Vector2
from dataclasses import dataclass
from itertools import product
import numpy as np
import asyncio
from itertools import chain
import pygame

import invertedai as iai
from invertedai import initialize, location_info, light, drive, async_drive, async_initialize
from invertedai.common import (AgentState, InfractionIndicators, Image,
                               TrafficLightStatesDict, AgentAttributes, RecurrentState, Point)
from invertedai.simulation.utils import Rotations, RE_INITIALIZATION_PERIOD, Rectangle, QUAD_RE_INITIALIZATION_PERIOD, get_pygame_convertors
from invertedai.simulation.regions import QuadTree
from invertedai.simulation.car import Car

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
    pygame_resolution: Tuple[int] = (1400, 1400)


class Simulation:
    """
    Stateful simulation for large maps with async calls to Inverted AI API for simultaneously driving npcs
    in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(self,
                 cfg: SimulationConfig,
                 ):
        self.cfg = cfg
        self.location = cfg.location
        self.center = Point(x=cfg.map_center[0], y=cfg.map_center[1])
        self.width = cfg.map_width
        self.height = cfg.map_height
        self.agent_per_region = cfg.agent_density
        self.random_seed = cfg.random_seed
        self.initialize_stride = cfg.initialize_stride
        self.re_initialization = cfg.re_initialization_period
        self.quad_re_initialization = cfg.quadtree_reconstruction_period
        self.timer = 0
        self.screen = None
        self.async_call = cfg.async_call
        self.show_quadtree = False
        self.location_info = iai.location_info(location=self.location)
        self.map_fov = cfg.map_fov
        self.map_image = pygame.surfarray.make_surface(cfg.rendered_static_map)
        self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales = get_pygame_convertors(
            self.center.x-self.map_fov/2, self.center.x+self.map_fov/2,
            self.center.y-self.map_fov/2, self.center.y+self.map_fov/2,
            cfg.pygame_resolution[0], cfg.pygame_resolution[1])
        self.boundary = Rectangle(Vector2(self.cfg.map_center[0] - (self.map_fov / 2),
                                          self.cfg.map_center[1] - (self.map_fov / 2)),
                                  Vector2((self.map_fov, self.map_fov)),
                                  convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
        if cfg.pygame_window:
            self.top_left = cfg.convert_to_pygame_coords(self.center.x-(self.map_fov/2), self.center.y-(self.map_fov/2))
            self.x_scale, self.y_scale = cfg.convert_to_pygame_scales(self.map_fov, self.map_fov)

            self.screen = pygame.display.set_mode(cfg.pygame_resolution)
            pygame.display.set_caption("Quadtree")
        self._initialize_regions()

    def _initialize_regions(self):
        try:
            light_response = iai.light(location=self.location)
            traffic_lights_states = [light_response.traffic_lights_states]
        except BaseException:
            traffic_lights_states = None
        initialize_response = iai.utils.area_initialization(self.location, self.agent_per_region,
                                                            traffic_lights_states=traffic_lights_states,
                                                            random_seed=self.random_seed,
                                                            map_center=(self.center.x, self.center.y),
                                                            width=self.width, height=self.height, stride=self.initialize_stride)

        npcs = [Car(agent_attributes=attr, agent_states=state, recurrent_states=rs, screen=self.screen,
                    convertor=self.cfg.convert_to_pygame_coords, cfg=self.cfg) for attr, state,
                rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states)]
        quadtree = None
        quadtree = QuadTree(cfg=self.cfg, capacity=self.cfg.quadtree_capacity, boundary=self.boundary,
                            convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
        quadtree.lineThickness = 1
        quadtree.color = (0, 87, 146)
        for npc in npcs:
            quadtree.insert(npc)

        self.npcs = npcs
        self.quadtree = quadtree
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
        if self.timer > self.quad_re_initialization:
            self.create_quadtree()
            self.timer = 0
        else:
            self.timer += 1
        regions = self.quadtree.get_regions()
        await asyncio.gather(*[region.async_drive() for region in regions])

    def sync_drive(self):
        if self.timer > self.quad_re_initialization:
            self.create_quadtree()
        else:
            self.timer += 1
        regions = self.quadtree.get_regions()
        for region in regions:
            region.sync_drive()

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
            # pygame.display.set_caption("QuadTree Fps: " + str(int(clock.get_fps())))
            self.screen.blit(pygame.transform.scale(pygame.transform.flip(
                pygame.transform.rotate(self.map_image, 90), True, False), (self.x_scale, self.y_scale)), self.top_left)

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
