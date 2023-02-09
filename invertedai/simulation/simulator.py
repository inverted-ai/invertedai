from typing import List, Tuple, Optional, Union
from pydantic import BaseModel, validate_arguments
from itertools import product
import numpy as np
import asyncio
from itertools import chain
import pygame

import invertedai as iai
from invertedai import initialize, location_info, light, drive, async_drive, async_initialize
from invertedai.common import (AgentState, InfractionIndicators, Image,
                               TrafficLightStatesDict, AgentAttributes, RecurrentState, Point)
from invertedai.simulation.utils import Rotations, RE_INITIALIZATION_PERIOD
from invertedai.simulation.regions import QuadTree, UniformGrid
from invertedai.simulation.car import Car


class Simulation:
    """
    Stateful simulation for large maps with async calls to Inverted AI API for simultaneously driving npcs
    in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(self,
                 location: str,
                 center: Tuple[float],
                 width: float,
                 height: float,
                 agent_per_region: float,
                 random_seed: Optional[float] = None,
                 region_fov: Optional[float] = 120,
                 screen=None,
                 convertor=None,
                 quadtree: bool = False,
                 async_call: bool = True,
                 ):
        self.location = location
        self.center = Point(x=center[0], y=center[1])
        self.width = width
        self.height = height
        self.agent_per_region = agent_per_region
        self.random_seed = random_seed
        self.region_fov = region_fov
        self.location_info = iai.location_info(location=self.location)
        self.re_initialization = RE_INITIALIZATION_PERIOD  # :
        self.screen = screen
        self.convertor = convertor
        self.async_call = async_call
        self.regions, self.npcs, self.initialize = self._initialize_regions()

    def _initialize_regions(self):
        try:
            light_response = iai.light(location=self.location)
            traffic_lights_states = [light_response.traffic_lights_states]
        except:
            traffic_lights_states = None
        initialize_response = iai.utils.area_initialization(self.location, self.agent_per_region,
                                                            traffic_lights_states=traffic_lights_states,
                                                            random_seed=self.random_seed,
                                                            map_center=(self.center.x, self.center.y),
                                                            width=self.width, height=self.height, stride=self.region_fov/2)

        npcs = [Car(agent_attributes=attr, agent_states=state, recurrent_states=rs, screen=self.screen, convertor=self.convertor) for attr, state,
                rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states)]

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

        return regions, npcs, initialize_response

    async def drive_regions(self):
        outgoings = await asyncio.gather(*[region.drive() for region in self.regions])
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
            self.async_drive()
        else:
            self.sync_drive()

    def async_drive(self):
        outgoing_npcs = asyncio.run(self.drive_regions())
        outgoing_npcs = list(chain.from_iterable(outgoing_npcs))
        for region in self.regions:
            region_npcs = list(filter(lambda x: UniformGrid.inside_fov(
                region.center, region.region_fov, x), outgoing_npcs))
            # region.incomming(region_npcs)

        return True

    def sync_drive(self):
        outgoing_npcs = asyncio.run(self.drive_regions())
        outgoing_npcs = list(chain.from_iterable(outgoing_npcs))
        for region in self.regions:
            region_npcs = list(filter(lambda x: UniformGrid.inside_fov(
                region.center, region.region_fov, x), outgoing_npcs))
            # region.incomming(region_npcs)

        return True
