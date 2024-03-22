from typing import Tuple, Optional, List
from collections import deque
from pygame.math import Vector2
from dataclasses import dataclass
import numpy as np
import pygame
import asyncio

import invertedai as iai
from invertedai.common import Point
from area_drive.utils import MAX_HISTORY_LEN, Rectangle, get_pygame_convertors, DEBUG
from area_drive.regions import QuadTree
from area_drive.car import Car

from invertedai.api.location import LocationResponse
from invertedai.api.initialize import InitializeResponse

Color1 = (1, 1, 1)


@dataclass
class AreaDriverConfig:
    """
    A collection of static configuration options for driving vehicles in an area.
    """
    #Simulation parameters
    location: str  #: in format recognized by Inverted AI API
    map_center: Tuple[float] = (0.0, 0.0) #: The center of the square area over which the quadtree operates and drives the agents
    map_fov: float = 100  #: The size of the square area over which the quadtree operates and drives the agents
    quadtree_reconstruction_period: int = 10 #: After how many timesteps the quadtree will update its leaves
    quadtree_capacity: int = 10 #: The maximum number of agents permitted in a quadtree leaf before splitting
    async_call: bool = True #: Whether to call drive asynchronously
    # Optional parameters if initialize or location info response are not provided
    initialize_center: Optional[Tuple[float]] = None #: The center of the area to be initialized
    map_width: Optional[float] = 100 #: width of the area to be initialized
    map_height: Optional[float] = 100 #: height of the area to be initialized
    agent_density: Optional[float] = 10 #: The number of agents to spawn in each 100x100m region
    initialize_stride: Optional[float] = 50 #: The space between regions to be initialized
    random_seed: Optional[int] = None
    #Visualization parameters
    rendered_static_map: Optional[np.ndarray] = None #: Map image from location info to render vehicles upon
    pygame_window: bool = False
    pygame_resolution: Tuple[int] = (1200, 1200)
    

class AreaDriver:
    """
    Stateful simulation for large maps with calls to Inverted AI API for simultaneously driving npcs in smaller regions.

    :param location: Location name as expected by :func:`initialize`.
    """

    def __init__(
        self, 
        cfg: AreaDriverConfig, 
        location_response: Optional[LocationResponse] = None,
        initialize_response: Optional[InitializeResponse] = None
    ):
        """
        A utility function to drive agents in a simulation with more than 100 agents. The agents are
        placed into a quadtree structure based on a given capacity parameter for agents within a 
        physical region, then runs :func:`drive` on each region. Agents in neighbouring regions are
        passed into each :func:`drive` call to prevent collisions across boundaries. An initialization
        may be passed into the simulation or an initialization can be executed by the simulation itself 
        by providing the appropriate information in the configuration argument. The configuration 
        argument can also be used to specify other parameters such as the capacity of the leaves of the 
        quadtree structure and async_call which controls whether synchronous or asynchronous :func:`drive`
        calls are made.

        Arguments
        ----------
        cfg:
            A data class containing information to configure the simulation
        location_response:
            If a location response already exists, it may be optionally passed in, otherwise a location
            response is acquired during setup.
        initialize_response:
            If an initialize response already exists, it may be optionally passed in, otherwise a initialize
            response is acquired during setup.
        
        See Also
        --------
        :func:`drive`
        """

        self.cfg = cfg
        self.location = cfg.location
        self.center = Point(x=cfg.map_center[0], y=cfg.map_center[1])
        self.initialize_center = Point(x=cfg.initialize_center[0], y=cfg.initialize_center[1]) if cfg.initialize_center else self.center
        self.width = cfg.map_width
        self.height = cfg.map_height
        self.agent_per_region = cfg.agent_density
        self.random_seed = cfg.random_seed
        self.initialize_stride = cfg.initialize_stride
        self.quad_re_initialization = cfg.quadtree_reconstruction_period
        self.timer = 1
        self.screen = None
        self.show_quadtree = False
        self.async_call = cfg.async_call
        self.map_fov = cfg.map_fov
        self.map_image = pygame.surfarray.make_surface(cfg.rendered_static_map)
        self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales = get_pygame_convertors(
            self.center.x - self.map_fov / 2, self.center.x + self.map_fov / 2,
            self.center.y - self.map_fov / 2, self.center.y + self.map_fov / 2,
            cfg.pygame_resolution[0], 
            cfg.pygame_resolution[1]
        )

        self.boundary = Rectangle(
            Vector2(
                self.cfg.map_center[0] - (self.map_fov / 2),
                self.cfg.map_center[1] - (self.map_fov / 2)
            ),
            Vector2((self.map_fov, self.map_fov)),
            convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales)
        )

        if cfg.pygame_window:
            self.top_left = cfg.convert_to_pygame_coords(
                self.center.x - (self.map_fov / 2), self.center.y - (self.map_fov / 2))
            self.x_scale, self.y_scale = cfg.convert_to_pygame_scales(self.map_fov, self.map_fov)
            self.screen = pygame.display.set_mode(cfg.pygame_resolution)
            pygame.display.set_caption("Quadtree")
        
        self.location_response = location_response
        self.initialize_response = initialize_response

        self._initialize_regions(self.location_response,self.initialize_response)
        self.create_quadtree()

    def _initialize_regions(self, location_response=None, initialize_response=None):
        
        if location_response is None:
            self.location_info = iai.location_info(
                location=self.location,
                rednering_center=self.center,
                rendering_fov=self.map_fov
            )

        if DEBUG:
            save_birdviews_to = f"img/debug/initialize/{self.timer}"
        else:
            save_birdviews_to = None

        if initialize_response is None:
            self.initialize_response = iai.utils.area_initialization(
                self.location, 
                self.agent_per_region,
                traffic_lights_states=None,
                random_seed=self.random_seed,
                map_center=(self.initialize_center.x,self.initialize_center.y),
                width=self.width, 
                height=self.height, 
                stride=self.initialize_stride,
                save_birdviews_to=save_birdviews_to
            )

        self.traffic_light_states = self.initialize_response.traffic_lights_states
        self.light_recurrent_states = self.initialize_response.light_recurrent_states

        self.npcs = [Car(
            agent_attributes=attr, 
            agent_states=state, 
            recurrent_states=rs, 
            screen=self.screen,
            convertor=self.cfg.convert_to_pygame_coords, 
            cfg=self.cfg) for attr, state, rs in zip(initialize_response.agent_attributes, initialize_response.agent_states, initialize_response.recurrent_states
        )]

    def create_quadtree(self):
        self.quadtree = QuadTree(
            cfg=self.cfg, 
            capacity=self.cfg.quadtree_capacity, 
            boundary=self.boundary,
            convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales)
        )
        self.quadtree.lineThickness = 1
        self.quadtree.color = (0, 87, 146)
        for npc in self.npcs:
            is_inserted = self.quadtree.insert(npc)

    async def async_drive(self):
        regions = self.quadtree.get_regions()
        results = await asyncio.gather(*[region.async_drive(self.light_recurrent_states) for region in regions])
        for result in results:
            if result[0] is not None:
                self.traffic_lights_states = result[0]
                self.light_recurrent_states = result[1]
                break

    def sync_drive(self):
        regions = self.quadtree.get_regions()
        is_new_traffic_lights = True
        current_light_recurrent_states = self.light_recurrent_states
        for region in regions:
            traffic_lights_states, light_recurrent_states = region.sync_drive(current_light_recurrent_states)

            if is_new_traffic_lights and traffic_lights_states is not None:
                self.traffic_lights_states = traffic_lights_states
                self.light_recurrent_states = light_recurrent_states
                is_new_traffic_lights = False

    @property
    def traffic_light_states(self):
        return self.traffic_lights_states


    @traffic_light_states.setter
    def traffic_light_states(self, traffic_lights_states):
        self.traffic_lights_states = traffic_lights_states

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
        if not (self.timer % self.quad_re_initialization):
            self.create_quadtree()
        self.timer += 1

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


