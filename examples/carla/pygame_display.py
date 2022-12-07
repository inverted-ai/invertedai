import pygame
import numpy as np
import carla


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height, headless: bool = False):
        init_image = np.zeros((height, width, 3), dtype='uint8')
        self.headless = headless
        if not headless:
            self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    # Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
    def callback(self, data):
        self.headless_callback(data)
        if not self.headless:
            self.pygame_callback()

    def pygame_callback(self):
        self.surface = pygame.surfarray.make_surface(self._img.swapaxes(0, 1))

    def headless_callback(self, data):
        data.convert(carla.ColorConverter.Raw)
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        self._img = img

    @property
    def image(self):
        return self._img
