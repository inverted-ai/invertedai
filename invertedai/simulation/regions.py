import pygame
from pygame.math import Vector2
from invertedai.simulation.utils import Rectangle, Circle, RE_INITIALIZATION_PERIOD, DEBUG
from typing import List, Optional, Callable
from invertedai import initialize, location_info, light, drive, async_drive, async_initialize
from invertedai.simulation.car import Car
from invertedai.common import (AgentState, InfractionIndicators, Image,
                               TrafficLightStatesDict, AgentAttributes, RecurrentState, Point)


class Region:
    def __init__(self, cfg, boundary: Rectangle, npcs: Optional[List[Car]] = [], query_neighbors: Optional[Callable] = None,
                 re_initialization: Optional[int] = RE_INITIALIZATION_PERIOD) -> None:
        self.cfg = cfg
        self.location = cfg.location
        self.npcs = npcs
        self.query_neighbors = query_neighbors  # : Function that accepts npc and returns a list of neighbors in fov of npc
        self.re_initialization_period = re_initialization
        self.time_to_reinitialize = self.re_initialization_period
        self.bounary = boundary
        self.timer = 0

    def sync_drive(self):
        """_summary_
        updates the state of all NPCs inside the region and returns a list of NPCs that are not longer int the region
        """
        outside_npcs_in_fov = []
        for npc in self.npcs:
            fov_npcs = self.query_neighbors(npc)

        if self.time_to_reinitialize == 0:
            # TODO: Initialization
            self.time_to_reinitialize = self.re_initialization_period
        else:
            self.time_to_reinitialize -= 1

        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        drive_response = drive(location=self.location,
                               agent_attributes=agent_attributes,
                               agent_states=agent_states,
                               recurrent_states=recurrent_states,
                               get_birdview=DEBUG)

        outgoing_npcs = []
        remaining_npcs = []
        for npc, state, rs in zip(self.npcs, drive_response.agent_states, drive_response.recurrent_states):
            npc.update(state, rs)
            if self.bounary.containsParticle(npc):
                remaining_npcs.append(npc)
            else:
                outgoing_npcs.append(npc)

        self.npcs = remaining_npcs
        return outgoing_npcs

    async def drive(self):
        """_summary_
        updates the state of all NPCs inside the region and returns a list of NPCs that are not longer int the region
        """
        if self.time_to_reinitialize == 0:
            # TODO: Initialization
            self.time_to_reinitialize = self.re_initialization_period
            pass
        else:
            self.time_to_reinitialize -= 1
        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        drive_response = await async_drive(location=self.location,
                                           agent_attributes=agent_attributes,
                                           agent_states=agent_states,
                                           recurrent_states=recurrent_states,
                                           get_birdview=DEBUG)
        if DEBUG:
            file_path = f"img/debug/{self.center}-{self.timer}.jpg"
            drive_response.birdview.decode_and_save(file_path)
            self.timer += 1

        outgoing_npcs = []
        remaining_npcs = []
        for npc, state, rs in zip(self.npcs, drive_response.agent_states, drive_response.recurrent_states):
            npc.update(state, rs)
            if self.bounary.containsParticle(npc):
                remaining_npcs.append(npc)
            else:
                outgoing_npcs.append(npc)

        self.npcs = remaining_npcs
        return outgoing_npcs

    def insert(self, npc):
        self.npcs.append(npc)

    @property
    def empty(self):
        return not bool(self.npcs)


class UniformGrid:
    def __init__(self, location, center: Point, region_fov: float, npcs: Optional[List[Car]] = [],
                 re_initialization: Optional[int] = RE_INITIALIZATION_PERIOD) -> None:
        self.center = center
        self.npcs = npcs
        self.region_fov = region_fov
        self.re_initialization_period = re_initialization
        self.time_to_reinitialize = self.re_initialization_period
        self.location = location
        self.incoming = []  # List of incomming NPCs
        self.timer = 0

    def incoming(self, incoming_npcs):
        self.incoming = incoming_npcs

    def sync_drive(self):
        """_summary_
        updates the state of all NPCs inside the region and returns a list of NPCs that are not longer int the region
        """
        if self.incoming:
            # TODO: something with the incomming or just apppend them to
            pass
        if self.empty:
            return []
        if self.time_to_reinitialize == 0:
            # TODO: Initialization
            self.time_to_reinitialize = self.re_initialization_period
            pass
        else:
            self.time_to_reinitialize -= 1
        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        drive_response = drive(location=self.location,
                               agent_attributes=agent_attributes,
                               agent_states=agent_states,
                               recurrent_states=recurrent_states,
                               get_birdview=DEBUG)

        outgoing_npcs = []
        remaining_npcs = []
        for npc, state, rs in zip(self.npcs, drive_response.agent_states, drive_response.recurrent_states):
            npc.update(state, rs)

            remaining_npcs.append(npc)

        self.npcs = remaining_npcs
        return outgoing_npcs

    async def drive(self):
        """_summary_
        updates the state of all NPCs inside the region and returns a list of NPCs that are not longer int the region
        """
        if self.incoming:
            # TODO: something with the incomming or just apppend them to
            pass
        if self.empty:
            return []
        if self.time_to_reinitialize == 0:
            # TODO: Initialization
            self.time_to_reinitialize = self.re_initialization_period
            pass
        else:
            self.time_to_reinitialize -= 1
        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        drive_response = await async_drive(location=self.location,
                                           agent_attributes=agent_attributes,
                                           agent_states=agent_states,
                                           recurrent_states=recurrent_states,
                                           get_birdview=DEBUG)
        if DEBUG:
            file_path = f"img/debug/{self.center}-{self.timer}.jpg"
            drive_response.birdview.decode_and_save(file_path)
            self.timer += 1

        outgoing_npcs = []
        remaining_npcs = []
        for npc, state, rs in zip(self.npcs, drive_response.agent_states, drive_response.recurrent_states):
            npc.update(state, rs)

            remaining_npcs.append(npc)

            # if self.inside_fov(self.center, self.region_fov, npc):
            #     remaining_npcs.append(npc)
            # else:
            #     outgoing_npcs.append(npc)

        self.npcs = remaining_npcs
        return outgoing_npcs

    @staticmethod
    def inside_fov(center: Point, region_fov: float, npc: Car) -> bool:
        return ((center.x - (region_fov / 2) < npc.center.x < center.x + (region_fov / 2)) and
                (center.y - (region_fov / 2) < npc.center.y < center.y + (region_fov / 2)))

    @property
    def empty(self):
        return not bool(self.npcs)


class QuadTree:
    def __init__(self, capacity: int, boundary: Rectangle, color=(140, 255, 160), thickness=1, convertors=None, cfg=None):
        self.capacity = capacity
        self.boundary = boundary
        self.particles = []
        self.color = color
        self.lineThickness = thickness
        self.leaf = True
        self.northWest = None
        self.northEast = None
        self.southWest = None
        self.southEast = None
        self.convertors = convertors
        self.cfg = cfg
        self.region = Region(boundary=boundary, cfg=cfg)

    def subdivide(self):
        parent = self.boundary

        boundary_nw = Rectangle(
            Vector2(
                parent.position.x,
                parent.position.y
            ),
            parent.scale/2,
            convertors=self.convertors
        )
        boundary_ne = Rectangle(
            Vector2(
                parent.position.x + parent.scale.x/2,
                parent.position.y
            ),
            parent.scale/2,
            convertors=self.convertors
        )
        boundary_sw = Rectangle(
            Vector2(
                parent.position.x,
                parent.position.y + parent.scale.y/2
            ),
            parent.scale/2,
            convertors=self.convertors
        )
        boundary_se = Rectangle(
            Vector2(
                parent.position.x + parent.scale.x/2,
                parent.position.y + parent.scale.y/2
            ),
            parent.scale/2,
            convertors=self.convertors
        )

        self.northWest = QuadTree(self.capacity, boundary_nw, self.color,
                                  self.lineThickness, convertors=self.convertors, cfg=self.cfg)
        self.northEast = QuadTree(self.capacity, boundary_ne, self.color,
                                  self.lineThickness, convertors=self.convertors, cfg=self.cfg)
        self.southWest = QuadTree(self.capacity, boundary_sw, self.color,
                                  self.lineThickness, convertors=self.convertors, cfg=self.cfg)
        self.southEast = QuadTree(self.capacity, boundary_se, self.color,
                                  self.lineThickness, convertors=self.convertors, cfg=self.cfg)

        for particle in self.particles:
            if self.northWest.insert(particle):
                continue
            elif self.northEast.insert(particle):
                continue
            elif self.southWest.insert(particle):
                continue
            else:
                self.southEast.insert(particle)
        self.particles = []
        self.region = None
        self.leaf = False

    def soft_insert(self, particle):
        # This method inserts a particle without subdividing and violates the max capacity
        # This is used to not recompute the tree on everycall
        if self.boundary.containsParticle(particle) == False:
            return False

        if self.leaf:
            self.particles.append(particle)
            self.region.insert(particle)
            return True
        else:
            if self.northWest.insert(particle):
                return True
            if self.northEast.insert(particle):
                return True
            if self.southWest.insert(particle):
                return True
            if self.southEast.insert(particle):
                return True
            return False

    def insert(self, particle):
        if self.boundary.containsParticle(particle) == False:
            return False

        if len(self.particles) < self.capacity and self.leaf:
            self.particles.append(particle)
            self.region.insert(particle)
            return True
        else:
            if self.northWest == None:
                self.subdivide()

            if self.northWest.insert(particle):
                return True
            if self.northEast.insert(particle):
                return True
            if self.southWest.insert(particle):
                return True
            if self.southEast.insert(particle):
                return True
            return False

    def get_regions(self):
        if self.region:
            return [self.region]
        else:
            return self.northWest.get_regions() + self.northEast.get_regions() + self.southWest.get_regions() + self.southEast.get_regions()

    # def queryRange(self, _range):
    #     particlesInRange = []

    #     if _range.name == "circle":
    #         if _range.intersects(self.boundary) == False:
    #             return particlesInRange
    #     else:
    #         if _range.intersects(self.boundary) == True:
    #             return particlesInRange

    #     for particle in self.particles:
    #         if _range.containsParticle(particle):
    #             particlesInRange.append(particle)
    #     if self.northWest != None:
    #         particlesInRange += self.northWest.queryRange(_range)
    #         particlesInRange += self.northEast.queryRange(_range)
    #         particlesInRange += self.southWest.queryRange(_range)
    #         particlesInRange += self.southEast.queryRange(_range)
    #     return particlesInRange

        # if self.boundary.intersects(_range):
        #     return particlesInRange
        # else:
        #     for particle in self.particles:
        #         if _range.containsParticle(particle):
        #             particlesInRange.append(particle)
        #
        #     if self.northWest != None:
        #         particlesInRange += self.northWest.queryRange(_range)
        #         particlesInRange += self.northEast.queryRange(_range)
        #         particlesInRange += self.southWest.queryRange(_range)
        #         particlesInRange += self.southEast.queryRange(_range)
        #
        #     return particlesInRange

    def Show(self, screen):
        self.boundary.color = self.color
        self.boundary.lineThickness = self.lineThickness
        self.boundary.Draw(screen)
        if self.northWest != None:
            self.northWest.Show(screen)
            self.northEast.Show(screen)
            self.southWest.Show(screen)
            self.southEast.Show(screen)
