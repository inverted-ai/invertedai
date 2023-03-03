import pygame
import math
from invertedai.common import Point
from typing import Tuple
import os


DEBUG = os.environ.get("IAI_SIMULATOR_DEBUG", False)
MAX_HISTORY_LEN = 10
RE_INITIALIZATION_PERIOD = 200  # : In frames
AGENT_FOV = 35  # : In meters
QUAD_RE_INITIALIZATION_PERIOD = 10


def get_pygame_convertors(x_min, x_max, y_min, y_max, H, W):
    def convert_to_pygame_coords(x, y):
        x_range = x_max - x_min
        y_range = y_max - y_min
        pygame_x = int((x - x_min) * W / x_range)
        pygame_y = int((y - y_min) * H / y_range)
        return (pygame_x, pygame_y)

    def convert_to_pygame_scales(w, h):
        x_range = x_max - x_min
        y_range = y_max - y_min
        pygame_w = int((w) * W / x_range)
        pygame_h = int((h) * H / y_range)
        return (pygame_w, pygame_h)

    return convert_to_pygame_coords, convert_to_pygame_scales


class Rotations:

    @ staticmethod
    def rotationX(angle):
        return [[1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]]

    @ staticmethod
    def rotationY(angle):
        return [[math.cos(angle), 0, -math.sin(angle)],
                [0, 1, 0],
                [math.sin(angle), 0, math.cos(angle)]]

    @ staticmethod
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
        x, y = particle.position.x, particle.position.y
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
        if ((x < xr < x+w) and (y < yr < y+h)) or ((xr < x < xr+wr) and (yr < y < yr+hr)):
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
