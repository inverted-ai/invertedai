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
from invertedai.simulation.utils import Rotations, RE_INITIALIZATION_PERIOD, Rectangle
from invertedai.simulation.regions import QuadTree, UniformGrid
from invertedai.simulation.car import Car


@dataclass
class SimulationConfig:
    """
    A collection of static configuration options for the simulation.
    """
    location: str = "carla:Town10HD"  #: in format recognized by Inverted AI API center map coordinate of the selected carla town
    map_center: Tuple[float] = (0.0, 0.0)
    fps: int = 10  #: 10 is the only value currently compatible with Inverted AI API
    traffic_count: int = 20  #: total number of vehicles to place in simulation
    episode_length: int = 20  #: in seconds
    map_fov: float = 200


class Simulation:
    """
    Stateful simulation for large maps with async calls to Inverted AI API for simultaneously driving npcs
    in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(self,
                 cfg: SimulationConfig,
                 location: str,
                 center: Tuple[float],
                 width: float,
                 height: float,
                 agent_per_region: float,
                 random_seed: Optional[float] = None,
                 region_fov: Optional[float] = 120,
                 initialize_stride: Optional[float] = 60,
                 screen=None,
                 convertor=None,
                 use_quadtree: bool = False,
                 async_call: bool = True,
                 ):
        self.cfg = cfg
        self.location = cfg.location
        self.center = Point(x=center[0], y=center[1])
        self.width = width
        self.height = height
        self.agent_per_region = agent_per_region
        self.random_seed = random_seed
        self.region_fov = region_fov
        self.initialize_stride = initialize_stride
        self.location_info = iai.location_info(location=self.location)
        self.re_initialization = RE_INITIALIZATION_PERIOD  # :
        self.screen = screen
        self.convertor = convertor
        self.async_call = async_call
        self.use_quadtree = use_quadtree
        self.boundary = Rectangle(Vector2(self.cfg.map_center[0] - (self.cfg.map_fov / 2),
                                          self.cfg.map_center[1] - (self.cfg.map_fov / 2)),
                                  Vector2((self.cfg.map_fov, self.cfg.map_fov)),
                                  convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))

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

        npcs = [Car(agent_attributes=attr, agent_states=state, recurrent_states=rs, screen=self.screen, convertor=self.convertor) for attr, state,
                rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states)]
        quadtree = None
        regions = None
        if self.use_quadtree:
            quadtree = QuadTree(cfg=self.cfg, capacity=self.cfg.node_capacity, boundary=self.boundary,
                                convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
            quadtree.lineThickness = 1
            quadtree.color = (0, 87, 146)
            for npc in npcs:
                quadtree.insert(npc)
        else:
            h_start, h_end = self.center.x - (self.height / 2) + (self.region_fov /
                                                                  2), self.center.x + (self.height / 2) - (self.region_fov / 2) + 1
            w_start, w_end = self.center.y - (self.width / 2) + (self.region_fov /
                                                                 2), self.center.y + (self.width / 2) - (self.region_fov / 2) + 1
            region_centers = product(np.arange(h_start, h_end, self.region_fov / 2),
                                     np.arange(w_start, w_end, self.region_fov / 2))
            regions = []
            for region_center in map(Point.fromlist, region_centers):
                region_npcs = list(filter(lambda x: UniformGrid.inside_fov(region_center, self.region_fov, x), npcs))
                regions.append(UniformGrid(location=self.location, center=region_center,
                                           region_fov=self.region_fov, npcs=region_npcs))

        self.regions = regions
        self.npcs = npcs
        self.quadtree = quadtree
        self.initialize_response = initialize_response

    def re_create_quadtree(self):
        quadtree = QuadTree(cfg=self.cfg, capacity=self.cfg.node_capacity, boundary=self.boundary,
                            convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))
        quadtree.lineThickness = 1
        quadtree.color = (0, 87, 146)
        for npc in self.npcs:
            quadtree.insert(npc)
        self.quadtree = quadtree

    async def async_drive(self):
        if self.use_quadtree:
            regions = self.quadtree.get_regions()
        else:
            regions = self.regions
        outgoings = await asyncio.gather(*[region.async_drive() for region in regions])
        return outgoings

    def sync_drive(self):
        if self.use_quadtree:
            regions = self.quadtree.get_regions()
        else:
            regions = self.regions
        outgoings = [region.sync_drive() for region in regions]
        return outgoings

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
        if self.async_call:
            asyncio.run(self.async_drive())
        else:
            self.sync_drive()

    # def async_drive(self):

    #     if self.use_quadtree:
    #         pass
    #     else:
    #         outgoing_npcs = asyncio.run(self.drive_regions())
    #         outgoing_npcs = list(chain.from_iterable(outgoing_npcs))
    #         for region in self.regions:
    #             region_npcs = list(filter(lambda x: UniformGrid.inside_fov(
    #                 region.center, region.region_fov, x), outgoing_npcs))
    #             # region.incomming(region_npcs)

    #     return True

    # def sync_drive(self):

    #     if self.use_quadtree:
    #         pass
    #     else:
    #         outgoing_npcs = self.drive_regions()
    #         outgoing_npcs = list(chain.from_iterable(outgoing_npcs))
    #         for region in self.regions:
    #             region_npcs = list(filter(lambda x: UniformGrid.inside_fov(
    #                 region.center, region.region_fov, x), outgoing_npcs))
    #             # region.incomming(region_npcs)

    #     return True

    def show(self):
        if self.use_quadtree:
            self.quadtree.Show(self.screen)
