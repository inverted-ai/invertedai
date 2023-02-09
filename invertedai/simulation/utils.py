import pygame
import math
from invertedai.common import Point
from typing import Tuple

MAX_HISTORY_LEN = 10
RE_INITIALIZATION_PERIOD = 200  # : In frames
DEBUG = False
AGENT_FOV = 35  # : In meters


class Rotations:

    @staticmethod
    def rotationX(angle):
        return [[1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]]

    @staticmethod
    def rotationY(angle):
        return [[math.cos(angle), 0, -math.sin(angle)],
                [0, 1, 0],
                [math.sin(angle), 0, math.cos(angle)]]

    @staticmethod
    def rotationZ(angle):
        return [[math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]]


class Rectangle:
    def __init__(self, position: Tuple[float, float], scale: Tuple[float, float], convertors=None):
        self.position = position
        self.scale = scale
        self.color = (255, 255, 255)
        self.lineThickness = 1
        self.name = "rectangle"
        self.convertors = convertors

    def containsParticle(self, particle):
        x, y = particle
        # x, y = particle.position
        bx, by = self.position
        w, h = self.scale
        if x > bx and x < bx+w and y > by and y < by+h:
            return True
        else:
            return False

    def intersects(self, other):
        x, y = self.position
        w, h = self.scale
        xr, yr = other.position
        wr, hr = other.scale
        if xr > x + w or xr+wr < x-w or yr > y + h or yr+hr < y-h:
            return True
        else:
            return False

    def Draw(self, screen):

        if self.convertors:
            x, y = self.convertors[0](*self.position)
            w, h = self.convertors[1](*self.scale)
        else:
            x, y = self.position
            w, h = self.scale
        pygame.draw.rect(screen, self.color, [x, y, w, h], self.lineThickness)


class Circle:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
        self.sqradius = self.radius * self.radius
        self.scale = None
        self.color = (255, 255, 255)
        self.lineThickness = 1
        self.name = "circle"

    def containsParticle(self, particle):
        x1, y1 = self.position
        x2, y2 = particle.position
        dist = math.pow(x2-x1, 2) + math.pow(y2-y1, 2)
        if dist <= self.sqradius:
            return True
        else:
            return False

    def intersects(self, _range):
        x1, y1 = self.position
        x2, y2 = _range.position
        w, h = _range.scale
        r = self.radius
        dist_x, dist_y = abs(x2-x1), abs(y2-y1)

        edges = pow(dist_x-w, 2) + pow(dist_y-h, 2)

        if dist_x > (r+w) or dist_y > (r+h):
            return False

        if dist_x <= w or dist_y <= h:
            return True

        return (edges <= self.sqradius)

    def Draw(self, screen):
        pygame.draw.circle(screen, self.color, self.position, self.radius, self.lineThickness)
