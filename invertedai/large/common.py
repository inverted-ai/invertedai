from typing import List, Optional, Tuple, Union
from pydantic import BaseModel

from invertedai.common import AgentAttributes, AgentProperties, AgentState, RecurrentState, Point
from invertedai.utils import convert_attributes_to_properties

REGION_MAX_SIZE = 100.0

class Region(BaseModel):
    """
    A region in a map used to divide a large simulation into smaller parts.

    See Also
    --------
    AgentProperties
    """

    center: Point #: The center of the region if such a concept is relevant (e.g. center of a square, center of a rectangle)
    size: float #: Side length of the region for the default interpretation of a region as a square
    agent_states: Optional[List[AgentState]] = [] #: A list of existing agents within the region
    agent_properties: Optional[List[AgentProperties]] = [] #: The static parameters of agents that exist within the region or that will be initialized within the region
    recurrent_states: Optional[List[RecurrentState]] = [] #: Recurrent states of the agents eixsting within the region

    @classmethod
    def create_square_region(
        cls, 
        center: Point, 
        size: Optional[float] = REGION_MAX_SIZE, 
        agent_states: Optional[List[AgentState]] = [], 
        agent_properties: Optional[List[Union[AgentAttributes,AgentProperties]]] = [], 
        recurrent_states: Optional[List[RecurrentState]] = []
    ):
        cls.center = center
        cls.size = size
        for agent in agent_states:
            assert cls.is_inside(cls,agent.center), f"Existing agent states at position {agent.center} must be located within the region."

        
        agent_properties_new = []
        for properties in agent_properties:
            properties_new = properties
            if isinstance(properties,AgentAttributes):
                properties_new = convert_attributes_to_properties(properties_new)
            agent_properties_new.append(properties_new)
        agent_properties = agent_properties_new

        return cls(
            center=center, 
            size=size,
            agent_states=agent_states, 
            agent_properties=agent_properties, 
            recurrent_states=recurrent_states 
        )

    @classmethod
    def copy(
        cls,
        region
    ):

        return cls(
            center=region.center, 
            size=region.size,
            agent_states=region.agent_states, 
            agent_properties=region.agent_properties, 
            recurrent_states=region.recurrent_states 
        )

    def clear_agents(self):

        self.agent_states = []
        self.agent_properties = []
        self.recurrent_states = []

    def insert_all_agent_details(
        self,
        agent_state: AgentState,
        agent_properties: Union[AgentAttributes,AgentProperties],
        recurrent_state: RecurrentState
    ):
        if isinstance(agent_properties,AgentAttributes):
            agent_properties = convert_attributes_to_properties(agent_properties)

        self.agent_states.append(agent_state)
        self.agent_properties.append(agent_properties)
        self.recurrent_states.append(recurrent_state)

    def is_inside(
        self,
        point: Point
    ) -> bool:
        """
        Helper function to check if a point is within an X-Y axis aligned bounding box around the region.
        This function should be faster but equivalent in result to the other checking function if the 
        region is rectangular.

        """

        x, y = point.x, point.y
        region_x, region_y = self.center.x, self.center.y
        return region_x - self.size/2 <= x and x <= region_x + self.size/2 and region_y - self.size/2 <= y and y <= region_y + self.size/2
