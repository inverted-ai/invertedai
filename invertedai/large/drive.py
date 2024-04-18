from typing import Tuple, Optional, List
from pydantic import validate_call
import asyncio

import invertedai as iai
from invertedai.large.common import Region
from invertedai.common import Point, AgentState, AgentAttributes, RecurrentState
from invertedai.api.drive import DriveResponse

BUFFER_FOV = 35

class Particle(BaseModel):
    """
    All information relevant to a particular agent

    See Also
    --------
    AgentState
    AgentAttributes
    RecurrentState
    """

    agent_state: AgentState
    agent_attributes: AgentAttributes
    recurrent_state: RecurrentState

    def tolist(self):
        return [agent_state, agent_attributes, recurrent_state]

    @classmethod
    def fromlist(cls, l):
        agent_state, agent_attributes, recurrent_state = l
        return cls(agent_state=agent_state, agent_attributes=agent_attributes, recurrent_state=recurrent_state)

class QuadTree(BaseModel):
    def __init__(
        self, 
        capacity: int, 
        region: Region, 
    ):
        self.capacity = capacity
        self.region = region
        self.particles = []
        self.leaf = True
        self.northWest = None
        self.northEast = None
        self.southWest = None
        self.southEast = None

        self.region_buffer = Region.fromlist([self.region.center,self.region.fov+BUFFER_FOV])
        self.particles_buffer = []

    def subdivide(self):
        parent = self.region
        new_fov = self.region.fov/2

        boundary_nw = Region.fromlist([Point.fromlist([parent.center.x-new_fov/2,parent.center.y+new_fov/2]),new_fov])
        boundary_ne = Region.fromlist([Point.fromlist([parent.center.x+new_fov/2,parent.center.y+new_fov/2]),new_fov])
        boundary_sw = Region.fromlist([Point.fromlist([parent.center.x-new_fov/2,parent.center.y-new_fov/2]),new_fov])
        boundary_se = Region.fromlist([Point.fromlist([parent.center.x+new_fov/2,parent.center.y-new_fov/2]),new_fov])

        self.northWest = QuadTree(self.capacity,boundary_nw)
        self.northEast = QuadTree(self.capacity,boundary_ne)
        self.southWest = QuadTree(self.capacity,boundary_sw)
        self.southEast = QuadTree(self.capacity,boundary_se)

        self.leaf = False

        for particle in self.particles:
            is_inserted = self.insert_particle_in_leaf_nodes(particle)
        for particle in self.particles_buffer:
            is_inserted = self.insert_particle_in_leaf_nodes(particle)
        self.particles = []
        self.particles_buffer = []
        
    def insert_particle_in_leaf_nodes(self,particle):
        is_inserted = False
        is_inserted = self.northWest.insert(particle,is_inserted) or is_inserted
        is_inserted = self.northEast.insert(particle,is_inserted) or is_inserted
        is_inserted = self.southWest.insert(particle,is_inserted) or is_inserted
        is_inserted = self.southEast.insert(particle,is_inserted) or is_inserted

        return is_inserted

    def insert(self, particle, is_particle_placed=False):
        is_in_region = self.region.check_point_in_bounding_box(particle.agent_state.center)
        is_in_buffer = self.region_buffer.check_point_in_bounding_box(particle.agent_state.center)

        if (not is_in_region) and (not is_in_buffer):
            return False

        if (len(self.particles) + len(self.particles_buffer)) < self.capacity and self.leaf:
            if is_in_region and not is_particle_placed:
                self.particles.append(particle)
                if self.region.agent_states is None:
                    self.region.insert_all_agent_details(*particle.tolist())
                else:
                    self.region.insert_all_agent_details(*particle.tolist())
                return True

            else: # Particle is within the buffer region of this leaf node
                self.particles_buffer.append(particle)
                if self.region_buffer.agent_states is None:
                    self.region_buffer.insert_all_agent_details(*particle.tolist())
                else:
                    self.region_buffer.insert_all_agent_details(*particle.tolist())
                return False

        else:
            if self.leaf:
                self.subdivide()

            is_inserted = self.insert_particle_in_leaf_nodes(particle)

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

@validate_call
def region_drive(
    location: str,
    agent_states: List[AgentState],
    agent_attributes: List[AgentAttributes],
    recurrent_states: Optional[List[RecurrentState]] = None,
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    light_recurrent_states: Optional[LightRecurrentStates] = None,
    get_birdview: bool = False,
    rendering_center: Optional[Tuple[float, float]] = None,
    rendering_fov: Optional[float] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None,
    capacity: Optional[int] = 100,
    is_async: Optional[bool] = True
) -> DriveResponse:
    """
    Parameters
    ----------
    location:
        Location name in IAI format.

    agent_states:
        Current states of all agents.
        The state must include x: [float], y: [float] coordinate in meters
        orientation: [float] in radians with 0 pointing along x and pi/2 pointing along y and
        speed: [float] in m/s.

    agent_attributes:
        Static attributes of all agents.
        List of agent attributes. Each agent requires, length: [float]
        width: [float] and rear_axis_offset: [float] all in meters. agent_type: [str],
        currently supports 'car' and 'pedestrian'.
        waypoint: optional [Point], the target waypoint of the agent.

    recurrent_states:
        Recurrent states for all agents, obtained from the previous call to
        :func:`drive` or :func:`initialize`.

    get_birdview:
        Whether to return an image visualizing the simulation state.
        This is very slow and should only be used for debugging.

    rendering_center:
        Optional center coordinates for the rendered birdview.

    rendering_fov:
        Optional fov for the rendered birdview.

    get_infractions:
        Whether to check predicted agent states for infractions.
        This introduces some overhead, but it should be relatively small.

    traffic_lights_states:
       If the location contains traffic lights within the supported area,
       their current state should be provided here. Any traffic light for which no
       state is provided will have a state generated by iai.

    light_recurrent_states:
       Light recurrent states for all agents, obtained from the previous call to
        :func:`drive` or :func:`initialize`.
       Specifies the state and time remaining for each light group in the map.
       If manual control of individual traffic lights is desired, modify the relevant state(s) 
       in traffic_lights_states, then pass in light_recurrent_states as usual.

    random_seed:
        Controls the stochastic aspects of agent behavior for reproducibility.

    api_model_version:
        Optionally specify the version of the model. If None is passed which is by default, 
        the best model will be used.
    
    capacity:
        The number of agents allowed in a region before it must subdivide. Currently this 
        value represents the capacity of a quadtree leaf node that will subdivide if the 
        number of vehicles in the region passes this threshold.

    is_async:
        A flag to control whether to use asynchronous DRIVE calls.

    See Also
    --------
    :func:`drive`
    """
    
    num_agents = len(agent_states)
    agent_x = [None]*num_agents
    agent_y = [None]*num_agents
    for i, agent in enumerate(agent_states):
        agent_x[i] = agent.center.x
        agent_y[i] = agent.center.y
    max_x, min_x, max_y, min_y = max(agent_x), min(agent_x), max(agent_y), min(agent_y)
    region_fov = max(max_x,max_y) - min(min_x,min_y)
    region_center = tuple((max_x-min_x)/2,(max_y-min_y)/2)

    quadtree = QuadTree(
        capacity=capacity,
        boundary=Region.fromlist([Point.fromlist(list(region_center)),region_fov]),
    )
    for agent, attrs, recurr_state in zip(agent_states,agent_attributes,recurrent_states):
        particle = Particle.fromlist([agent, attrs, recurr_state])
        is_inserted = quadtree.insert(particle)

    all_leaf_nodes = quadtree.get_leaf_nodes()
    async_input_params = []
    all_responses = []
    non_empty_regions = []
    for i, leaf_node in enumerate(all_leaf_nodes):
        region, region_buffer = leaf_node.region, leaf_node.region_buffer
        
        if len(region.agent_states) == 0:
            # Region is empty, do not call DRIVE
            continue
        
        else: 
            non_empty_regions.append(region)
            input_params = {
                "location":location,
                "agent_attributes":region.agent_attributes+region_buffer.agent_attributes,
                "agent_states":region.agent_states+region_buffer.agent_states,
                "recurrent_states":region.recurrent_states+region_buffer.recurrent_states,
                "light_recurrent_states":light_recurrent_states,
                "traffic_lights_states":traffic_lights_states,
                "get_birdview":get_birdview,
                "rendering_center":rendering_center,
                "rendering_fov":rendering_fov,
                "get_infractions":get_infractions,
                "random_seed":random_seed,
                "api_model_version":api_model_version
            }
            if not is_async:
                all_responses.append(iai.drive(**input_params))
            else:
                async_input_params.append(input_params)

    if is_async:
        all_responses = await asyncio.gather([iai.async_drive(**input_params) for input_params in async_input_params])

    zip_iterable = zip(all_responses,non_empty_regions)
    response = DriveResponse(
        agent_states = [*response.agent_states[:len(region.agent_attributes)] for response, region in zip_iterable],
        recurrent_states = [*response.recurrent_states[:len(region.agent_attributes)] for response, region in zip_iterable],
        infractions = [*response.infractions[:len(region.agent_attributes)] for response, region in zip_iterable],
        is_inside_supported_area = [*response.is_inside_supported_area[:len(region.agent_attributes)] for response, region in zip_iterable],
        api_model_version = '' if len(all_responses) == 0 else all_responses[0].api_model_version,
        birdview = None if len(all_responses) == 0 or not get_birdview else all_responses[0].birdview,
        traffic_lights_states = traffic_lights_states if len(all_responses) == 0 else all_responses[0].traffic_lights_states,
        light_recurrent_states = light_recurrent_states if len(all_responses) == 0 else all_responses[0].light_recurrent_states
    )

    return response