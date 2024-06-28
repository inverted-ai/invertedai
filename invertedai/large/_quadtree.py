from pydantic import BaseModel

import invertedai as iai
from invertedai.large.common import Region
from invertedai.common import Point, AgentState, AgentAttributes, RecurrentState

BUFFER_FOV = 35
QUADTREE_SIZE_BUFFER = 1

class QuadTreeAgentInfo(BaseModel):
    """
    All information relevant to a single agent.

    See Also
    --------
    AgentState
    AgentAttributes
    RecurrentState
    """

    agent_state: AgentState
    agent_attributes: AgentAttributes
    recurrent_state: RecurrentState
    agent_id: int

    def tolist(self):
        return [self.agent_state, self.agent_attributes, self.recurrent_state, self.agent_id]

    @classmethod
    def fromlist(cls, l):
        agent_state, agent_attributes, recurrent_state, agent_id = l
        return cls(agent_state=agent_state, agent_attributes=agent_attributes, recurrent_state=recurrent_state, agent_id=agent_id)

class QuadTree:
    def __init__(
        self, 
        capacity: int, 
        region: Region, 
    ):
        self.capacity = capacity
        self.region = region
        self.leaf = True
        self.northWest = None
        self.northEast = None
        self.southWest = None
        self.southEast = None

        self.region_buffer = Region.create_square_region(
            center=self.region.center,
            size=self.region.size+2*BUFFER_FOV
        )
        self.particles = []
        self.particles_buffer = []

    def subdivide(self):
        parent = self.region
        new_size = self.region.size/2
        new_center_dist = new_size/2
        parent_x = parent.center.x
        parent_y = parent.center.y

        region_nw = Region.create_square_region(
            center=Point.fromlist([parent_x-new_center_dist,parent_y+new_center_dist]),
            size=new_size
        )
        region_ne = Region.create_square_region(
            center=Point.fromlist([parent_x+new_center_dist,parent_y+new_center_dist]),
            size=new_size
        )
        region_sw = Region.create_square_region(
            center=Point.fromlist([parent_x-new_center_dist,parent_y-new_center_dist]),
            size=new_size
        )
        region_se = Region.create_square_region(
            center=Point.fromlist([parent_x+new_center_dist,parent_y-new_center_dist]),
            size=new_size
        )

        self.northWest = QuadTree(self.capacity,region_nw)
        self.northEast = QuadTree(self.capacity,region_ne)
        self.southWest = QuadTree(self.capacity,region_sw)
        self.southEast = QuadTree(self.capacity,region_se)

        self.leaf = False
        self.region.clear_agents()
        self.region_buffer.clear_agents()

        for particle in self.particles:
            is_inserted = self.insert_particle_in_leaf_nodes(particle,False)
        for particle in self.particles_buffer:
            is_inserted = self.insert_particle_in_leaf_nodes(particle,True)
        self.particles = []
        self.particles_buffer = []
        
    def insert_particle_in_leaf_nodes(self,particle,is_inserted):
        is_inserted_in_this_branch = self.northWest.insert(particle,is_inserted)
        is_inserted_in_this_branch = self.northEast.insert(particle,is_inserted_in_this_branch or is_inserted) or is_inserted_in_this_branch
        is_inserted_in_this_branch = self.southWest.insert(particle,is_inserted_in_this_branch or is_inserted) or is_inserted_in_this_branch
        is_inserted_in_this_branch = self.southEast.insert(particle,is_inserted_in_this_branch or is_inserted) or is_inserted_in_this_branch

        return is_inserted_in_this_branch

    def insert(self, particle, is_particle_placed=False):
        is_in_region = self.region.is_inside(particle.agent_state.center)
        is_in_buffer = self.region_buffer.is_inside(particle.agent_state.center)

        if (not is_in_region) and (not is_in_buffer):
            return False

        if (len(self.particles) + len(self.particles_buffer)) < self.capacity and self.leaf:
            if is_in_region and not is_particle_placed:
                self.particles.append(particle)
                self.region.insert_all_agent_details(*particle.tolist()[:-1])
                return True

            else: # Particle is within the buffer region of this leaf node
                self.particles_buffer.append(particle)
                self.region_buffer.insert_all_agent_details(*particle.tolist()[:-1])
                return False

        else:
            if self.leaf:
                self.subdivide()

            is_inserted = self.insert_particle_in_leaf_nodes(particle,is_particle_placed)

            return is_inserted

    def get_regions(self):
        if self.leaf:
            return [self.region]
        else:
            return self.northWest.get_regions() + self.northEast.get_regions() + \
                self.southWest.get_regions() + self.southEast.get_regions()

    def get_leaf_nodes(self):
        if self.leaf:
            return [self]
        else:
            return self.northWest.get_leaf_nodes() + self.northEast.get_leaf_nodes() + \
                self.southWest.get_leaf_nodes() + self.southEast.get_leaf_nodes()

    def get_number_of_agents_in_node(self):
        return len(self.particles)

def _flatten_and_sort(nested_list,index_list):
    flat_list = [x for sublist in nested_list for x in sublist]
    sorted_list = [x[1] for x in sorted(zip(index_list, flat_list))]
    
    return sorted_list