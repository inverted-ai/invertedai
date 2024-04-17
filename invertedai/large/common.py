from typing import List, Optional, Tuple
from pydantic import BaseModel, model_validator

from invertedai.common import AgentAttributes, AgentState, RecurrentState, Point

class Region(BaseModel):
    """
    A data structure to contain information about a region in a map to be used in large simulations.

    See Also
    --------
    AgentAttributes
    """

    center: Optional[Point] = None # The center of the region if such a concept is relevant (e.g. center of a square, center of a rectangle)
    fov: Optional[float] = 100 # Side length of the region for the default interpretation of a region as a square
    agent_states: Optional[AgentState] = None # A list of existing agents within the region
    agent_attributes: Optional[AgentAttributes] = None # The attributes of agents that exist within the region or that will be initialized within the region
    recurrent_states: Optional[RecurrentState] = None
    region_type = 'square' # Geometric category of the region. As of now, only 'square' is supported.
    vertices: List[Tuple[float,float]] # An ordered list of x-y coordinates of the region defined clockwise around the perimeter of the region

    @classmethod
    def fromlist(cls, l, agent_states = None, agent_attributes = None):
        center = None
        fov = 100

        if region_type == 'square':
            if len(l) == 2:
                center, fov = l

            elif len(l) == 1:
                center = l[0]

            vertices = self.define_square_vertices(center,fov)

        for agent in agent_states:
            assert self.check_point_in_region(agent.center), "Existing agent states must be located within the region."

        return cls(center=center, fov=fov, agent_states=agent_states, agent_attributes=agent_attributes, region_type=region_type, vertices=vertices)


    def define_square_vertices(self,center,fov):
        assert center is not None, "Square region must contain valid center Point"
        vertices = [
            tuple(center.x-fov/2,center.y+fov/2), # Top left
            tuple(center.x+fov/2,center.y+fov/2), # Top right
            tuple(center.x+fov/2,center.y-fov/2), # Bottom right
            tuple(center.x-fov/2,center.y-fov/2) # Bottom left
        ]

        return vertices

    def check_point_in_region(self,point):
        # Use horizontal raycast method to check if point is in region defined by vertices
        # If a ray from the point to positive infinity towards the right crosses edges of
        # the polygon an odd number of times, the point is within the polygon.
        x, y = point.x, point.y
        is_inside = False
     
        p1 = self.vertices[0]
        num_vertices = len(self.vertices)
        for i in range(1, num_vertices + 1):
            p2 = self.vertices[i % num_vertices]

            p1x, p1y, p2x, p2y = p1.x, p1.y, p2.x, p2.y
            if x == p2x and y == p2y:
                # The point is equal to a vertex therefore is within the region
                return True

            # Eliminate edges whose coordinates indicate crossing the ray is impossible
            if y >= min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                        
                        if x_intersection == x:
                            # The point is on the edge between p1 and p2 therefore is within the region
                            return True

                        if p1x == p2x or x <= x_intersection:
                            is_inside = not is_inside
            p1 = p2