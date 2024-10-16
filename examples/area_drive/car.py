from typing import Optional
try:
    import pygame
except ImportError:
    pygame = None
from pydantic import validate_call
from invertedai.common import AgentState, AgentAttributes, RecurrentState
from collections import deque
from area_drive.utils import MAX_HISTORY_LEN, AGENT_FOV, Rectangle
from uuid import uuid1 as UUID


class Car:
    @validate_call
    def __init__(
        self,
        agent_attributes: Optional[AgentAttributes],
        agent_states: Optional[AgentState],
        recurrent_states: Optional[RecurrentState],
        screen=None,
        convertor_coords=None,
        convertor_scales=None,
        id=None
    ):
        self._agent_attributes = agent_attributes
        self._recurrent_states = recurrent_states
        self.id = id if id else UUID().int
        self.hue = 0
        self.color = (255, 0, 0)
        self.stroke = 1
        self.fov = AGENT_FOV
        self._agents_in_fov = []
        self.show_agent_neighbors = False
        self._region = None

        self._states_history = deque([agent_states], maxlen=MAX_HISTORY_LEN)
        self.screen = screen
        self.convertor_coords = convertor_coords
        self.convertor_scales = convertor_scales

    def update(
        self,
        agent_states: Optional[AgentState],
        recurrent_states: Optional[RecurrentState]
    ):
        self._states_history.append(agent_states)
        self._recurrent_states = recurrent_states
        if self.screen:
            self.render()

    def fov_range(self):
        return Rectangle(
            (self.position.x-(self.fov/2), self.position.y-(self.fov/2)),
            ((self.fov, self.fov)),
            convertors=(self.convertor_coords, self.convertor_scales)
        )

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region):
        self._region = region

    @property
    def fov_agents(self):
        return self._agents_in_fov

    @fov_agents.setter
    def fov_agents(self, others):
        self._agents_in_fov = others

    @property
    def recurrent_states(self):
        return self._recurrent_states

    @property
    def agent_attributes(self):
        return self._agent_attributes

    @ property
    def position(self):
        return self.agent_states.center

    @ property
    def orientation(self):
        return self.agent_states.orientation

    @ property
    def agent_states_history(self):
        return self._states_history

    @ property
    def agent_states(self):
        return self._states_history[-1]

    @ agent_states.setter
    def agent_states(self, agent_states: AgentState):
        self._states_history[-1] = agent_states

    def render(self):
        px, py = self.convertor_coords(self.position.x, self.position.y)
        pygame.draw.circle(self.screen, self.color, (px, py), 5)

        if self.show_agent_neighbors:
            for neighbor in self.fov_agents:
                if neighbor not in self.region.npcs:
                    nx, ny = self.convertor_coords(neighbor.position.x, neighbor.position.y)
                    pygame.draw.line(self.screen, self.color, (px, py), (nx, ny))
