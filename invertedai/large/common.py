from typing import List, Optional, Tuple
from pydantic import BaseModel

from invertedai.common import AgentAttributes, AgentState, RecurrentState, Point

class Region(BaseModel):
    """
    A data structure to contain information about a region in a map to be used in large simulations.

    See Also
    --------
    AgentAttributes
    """

    center: Optional[Point] = None # The center of the region if such a concept is relevant (e.g. center of a square, center of a rectangle)
    fov: Optional[float] = None # Side length of the region for the default interpretation of a region as a square
    agent_states: Optional[List[AgentState]] = None # A list of existing agents within the region
    agent_attributes: Optional[List[AgentAttributes]] = None # The attributes of agents that exist within the region or that will be initialized within the region
    recurrent_states: Optional[List[RecurrentState]] = None # Recurrent states of the agents contained within the region
    region_type: str = 'square' # Geometric category of the region. As of now, only 'square' is supported.
    vertexes: List[Point] # An ordered list of x-y coordinates of the region defined clockwise around the perimeter of the region

    @classmethod
    def fromlist(cls, l, agent_states = [], agent_attributes = [], recurrent_states = [], region_type = 'square'):
        center = None
        fov = 100

        if region_type == 'square':
            if len(l) == 2:
                center, fov = l

            elif len(l) == 1:
                center = l[0]

            vertexes = cls.define_square_vertices(cls,center,fov)

        for agent in agent_states:
            assert cls.check_point_in_region(cls,agent.center), f"Existing agent states at position {agent.center} must be located within the region."

        return cls(
            center=center, 
            fov=fov,
            region_type=region_type, 
            vertexes=vertexes,
            agent_states=agent_states, 
            agent_attributes=agent_attributes, 
            recurrent_states=recurrent_states 
        )

    def clear_agents(self):
        self.agent_states = None
        self.agent_attributes = None
        self.recurrent_states = None

    def insert_all_agent_details(self,agent_state,agent_attributes,recurrent_state):
        if self.agent_states is None:
            self.agent_states = [agent_state]
        else:
            self.agent_states.append(agent_state)

        if self.agent_attributes is None:
            self.agent_attributes = [agent_attributes]
        else:
            self.agent_attributes.append(agent_attributes)

        if self.recurrent_states is None:
            self.recurrent_states = [recurrent_state]
        else:
            self.recurrent_states.append(recurrent_state)

    def define_square_vertices(self,center,fov):
        assert center is not None, f"Square region must contain valid center Point."
        fov_split = fov/2
        center_x = center.x
        center_y = center.y
        vertexes = [
            Point.fromlist([center_x-fov_split,center_y+fov_split]), # Top left
            Point.fromlist([center_x+fov_split,center_y+fov_split]), # Top right
            Point.fromlist([center_x+fov_split,center_y-fov_split]), # Bottom right
            Point.fromlist([center_x-fov_split,center_y-fov_split]) # Bottom left
        ]

        return vertexes

    def check_point_in_region(self,point):
        # Use horizontal raycast method to check if point is in region defined by vertexes
        # If a ray from the point to positive infinity towards the right crosses edges of
        # the polygon an odd number of times, the point is within the polygon (works with 
        # both convex and concave polygons).
        x, y = point.x, point.y
        is_inside = False
     
        p1 = self.vertexes[0]
        num_vertices = len(self.vertexes)
        for i in range(1, num_vertices + 1):
            p2 = self.vertexes[i % num_vertices]

            p1x, p1y, p2x, p2y = p1.x, p1.y, p2.x, p2.y
            if x == p2x and y == p2y:
                # The point is equal to the p2 vertex therefore is within the region.
                return True

            # Only consider edges whose coordinates indicate crossing the ray is possible.
            if y >= min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1x == p2x:
                            # This indicates the edge is a horizontal line with its maximum x value
                            # to the right of the point and satisfying the y-value conditions indicates
                            # the ray is parallel and inline with this edge, therefore crosses this edge.
                            is_inside = not is_inside
                            continue

                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                        
                        if x_intersection == x:
                            # The point is on the edge between p1 and p2 therefore is within the region
                            return True

                        if x <= x_intersection:
                            # Since the largest x value of the edge is to the right of the point and the intercept of the edge is 
                            # also to the right, the ray crosses the edge. 
                            is_inside = not is_inside
            p1 = p2

    def check_point_in_bounding_box(self,point):
        # Helper function to check if a point is within an X-Y axis aligned bounding box around the region.
        # This function should be faster but equivalent in result to the other checking function if the 
        # region is rectangular.

        if self.fov is not None:
            fov = self.fov
        else:
            fov = self.get_region_fov()

        x, y = point.x, point.y
        region_x, region_y = self.center.x, self.center.y
        if region_x - fov/2 <= x and x <= region_x + fov/2 and region_y - fov/2 <= y and y <= region_y + fov/2:
            return True
        else:
            return False

    def get_region_fov(self):
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        for vertex in self.vertexes:
            min_x = min(min_x,vertex.x)
            max_x = max(max_x,vertex.x)
            min_y = min(min_y,vertex.y)
            max_y = max(max_y,vertex.y)

        return max(max_x-min_x,max_y-min_y)