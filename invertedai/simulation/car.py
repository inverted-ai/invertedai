from typing import Optional
import pygame
from pydantic import validate_arguments
from invertedai.common import (AgentState, InfractionIndicators, Image,
                               TrafficLightStatesDict, AgentAttributes, RecurrentState, Point)
from collections import deque
from invertedai.simulation.utils import MAX_HISTORY_LEN


class Car:
    @validate_arguments
    def __init__(self,
                 agent_attributes: Optional[AgentAttributes],
                 agent_states: Optional[AgentState],
                 recurrent_states: Optional[RecurrentState],
                 screen=None,
                 convertor=None):

        self._agent_attributes = agent_attributes
        self._recurrent_states = recurrent_states
        self.hue = 0
        self.stroke = 1

        # TODO: Must check if the order is correct
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
    def agent_states(self):
        return self._states_history[-1]

    @ agent_states.setter
    def agent_states(self, agent_states: AgentState):
        # Setting the sate of the ego agent
        self._states_history[-1] = agent_states

    def render(self):

        px, py = self.convertor(self.position.x, self.position.y)
        pygame.draw.circle(self.screen, (255, 0, 0), (px, py), 3)
        # rect = pygame.Rect((px, py), (5, 5))
        # rotated = pygame.transform.rotate(rect, self.orientation)
        # pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # hsv color
        # c = pygame.Color(0, 0, 0)
        # c.hsva = (self.hue % 360, 100, 100, 100)
        # color = c

        # distance = 5
        # scale = 5
        # ps = []
        # points = [None for _ in range(4)]

        # self.radius = self.agent_attributes.width
        # points[0] = [[-self.agent_attributes.width//1], [self.agent_attributes.length//2], [0]]
        # points[1] = [[self.agent_attributes.width//1], [self.agent_attributes.length//2], [0]]
        # points[2] = [[self.agent_attributes.width//1], [-self.agent_attributes.length//2], [0]]
        # points[3] = [[-self.agent_attributes.width//1], [-self.agent_attributes.length//2], [0]]
        # # points[3] = [[0], [0], [0]]

        # px, py = self.convertor(self.position.x, self.position.y)

        # for point in points:
        #     # rotated = matrix_multiplication(rotationZ(self.angle), point)
        #     rotated = np.array(Rotations.rotationZ(self.orientation*180/np.pi)) @ np.array(point)
        #     z = 1/(distance - rotated[2][0])

        #     projection_matrix = [[z, 0, 0], [0, z, 0]]
        #     # projected_2d = matrix_multiplication(projection_matrix, rotated)
        #     projected_2d = np.array(projection_matrix) @ np.array(rotated)

        #     x = int(projected_2d[0][0] * scale) + px
        #     y = int(projected_2d[1][0] * scale) + py
        #     ps.append((x, y))

        # pygame.draw.polygon(self.screen, color, ps[:])
        # pygame.draw.polygon(self.screen, color, ps[:], self.stroke)
