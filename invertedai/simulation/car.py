from typing import Optional
import pygame
from pygame.math import Vector2
from pydantic import validate_arguments
from invertedai.common import AgentState, AgentAttributes, RecurrentState
from collections import deque
from invertedai.simulation.utils import MAX_HISTORY_LEN, AGENT_FOV, Rectangle


class Car:
    @validate_arguments
    def __init__(self,
                 agent_attributes: Optional[AgentAttributes],
                 agent_states: Optional[AgentState],
                 recurrent_states: Optional[RecurrentState],
                 screen=None,
                 convertor=None,
                 cfg=None):

        self._agent_attributes = agent_attributes
        self._recurrent_states = recurrent_states
        self.hue = 0
        self.color = (255, 0, 0)
        self.stroke = 1
        self.fov = AGENT_FOV
        self._agents_in_fov = []
        self.cfg = cfg
        self.show_agent_neighbors = False
        self._region = None

        self._states_history = deque([agent_states], maxlen=MAX_HISTORY_LEN)
        self.screen = screen
        self.convertor = convertor

    def update(self,
               agent_states: Optional[AgentState],
               recurrent_states: Optional[RecurrentState],):
        self._states_history.append(agent_states)
        self._recurrent_states = recurrent_states
        if self.screen:
            self.render()

    def fov_range(self):
        return Rectangle(Vector2(self.position.x-(self.fov/2), self.position.y-(self.fov/2)),
                         Vector2((self.fov, self.fov)),
                         convertors=(self.cfg.convert_to_pygame_coords, self.cfg.convert_to_pygame_scales))

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
    def center(self):
        return self.agent_states.center

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
        # Setting the sate of the ego agent
        self._states_history[-1] = agent_states

    def render(self):
        px, py = self.convertor(self.position.x, self.position.y)
        pygame.draw.circle(self.screen, self.color, (px, py), 5)

        if self.show_agent_neighbors:
            for neighbor in self.fov_agents:
                if neighbor not in self.region.npcs:
                    nx, ny = self.convertor(neighbor.position.x, neighbor.position.y)
                    pygame.draw.line(self.screen, self.color, (px, py), (nx, ny))
