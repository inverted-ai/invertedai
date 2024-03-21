from pygame.math import Vector2
from area_drive.utils import Rectangle, RE_INITIALIZATION_PERIOD, DEBUG
from typing import List, Optional, Callable
from random import randint
from invertedai.api.drive import drive, async_drive
from area_drive.car import Car


class Region:
    def __init__(
        self, 
        cfg, 
        boundary: Rectangle, 
        npcs: Optional[List[Car]] = None, 
        query_neighbors: Optional[Callable] = None,
        re_initialization: Optional[int] = RE_INITIALIZATION_PERIOD
    ) -> None:
        self.cfg = cfg
        self.location = cfg.location
        self.npcs = npcs or []
        self.query_neighbors = query_neighbors  # : Function that accepts npc and returns a list of neighbors in fov of npc
        self.re_initialization_period = re_initialization
        self.time_to_reinitialize = self.re_initialization_period
        self.boundary = boundary
        self.timer = 0
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def pre_drive(self):
        agents_in_fov = []
        for car in self.npcs:
            agents_in_fov.extend(filter(lambda x: x not in self.npcs, car.fov_agents))

        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for npc in self.npcs:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)

        for npc in agents_in_fov:
            agent_states.append(npc.agent_states)
            agent_attributes.append(npc.agent_attributes)
            recurrent_states.append(npc.recurrent_states)
        return agent_attributes, agent_states, recurrent_states

    def post_drive(self, drive_response):
        if DEBUG:
            file_path = f"img/debug/{self.boundary.position}-{self.timer}.jpg"
            drive_response.birdview.decode_and_save(file_path)
            self.timer += 1

        remaining_npcs = []
        for npc, state, rs in zip(
                self.npcs, drive_response.agent_states[:self.size], drive_response.recurrent_states[:self.size]):
            npc.update(state, rs)
            remaining_npcs.append(npc)
        self.npcs = remaining_npcs

    def sync_drive(self, light_recurrent_states = None):
        """_summary_
        updates the state of all NPCs inside the region (agents outside the region that are visible to inside NPCs are included to the call to drive but their state is not changed)
        """
        if self.empty:
            return None, None
        else:
            agent_attributes, agent_states, recurrent_states = self.pre_drive()
            drive_response = drive(
                location=self.location,
                agent_attributes=agent_attributes,
                agent_states=agent_states,
                recurrent_states=recurrent_states,
                light_recurrent_states=light_recurrent_states,
                get_birdview=DEBUG
            )

            self.post_drive(drive_response=drive_response)

            return drive_response.traffic_lights_states, drive_response.light_recurrent_states

    def insert(self, npc):
        self.npcs.append(npc)
        npc.color = self.color

    @property
    def size(self):
        return len(self.npcs)

    @property
    def empty(self):
        return not bool(self.npcs)


class QuadTree:
    def __init__(
        self, 
        capacity: int, 
        boundary: Rectangle, 
        color=(140, 255, 160), 
        thickness=1, 
        convertors=None, 
        cfg=None
    ):
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
            parent.scale / 2,
            convertors=self.convertors
        )
        boundary_ne = Rectangle(
            Vector2(
                parent.position.x + parent.scale.x / 2,
                parent.position.y
            ),
            parent.scale / 2,
            convertors=self.convertors
        )
        boundary_sw = Rectangle(
            Vector2(
                parent.position.x,
                parent.position.y + parent.scale.y / 2
            ),
            parent.scale / 2,
            convertors=self.convertors
        )
        boundary_se = Rectangle(
            Vector2(
                parent.position.x + parent.scale.x / 2,
                parent.position.y + parent.scale.y / 2
            ),
            parent.scale / 2,
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
                pass
            elif self.northEast.insert(particle):
                pass
            elif self.southWest.insert(particle):
                pass
            else:
                self.southEast.insert(particle)
        self.particles = []
        self.region = None
        self.leaf = False

    def insert(self, particle):
        if self.boundary.containsParticle(particle) == False:
            return False

        if len(self.particles) < self.capacity and self.leaf:
            self.particles.append(particle)
            self.region.insert(particle)
            particle.region = self.region
            return True
        else:
            if self.leaf:
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
        if self.leaf:
            return [self.region]
        else:
            return self.northWest.get_regions() + self.northEast.get_regions() + \
                self.southWest.get_regions() + self.southEast.get_regions()

    def queryRange(self, query_range):
        particlesInRange = []
        if self.leaf:
            if query_range.intersects(self.boundary):
                for particle in self.particles:
                    if query_range.containsParticle(particle):
                        particlesInRange.append(particle)
        else:
            particlesInRange += self.northWest.queryRange(query_range)
            particlesInRange += self.northEast.queryRange(query_range)
            particlesInRange += self.southWest.queryRange(query_range)
            particlesInRange += self.southEast.queryRange(query_range)
        return particlesInRange

    def Show(self, screen):
        self.boundary.color = self.color
        self.boundary.lineThickness = self.lineThickness
        self.boundary.Draw(screen)
        if self.northWest is not None:
            self.northWest.Show(screen)
            self.northEast.Show(screen)
            self.southWest.Show(screen)
            self.southEast.Show(screen)
