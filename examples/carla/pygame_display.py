import pygame
import numpy as np
import carla


# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.zeros((height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))

    # Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
    @staticmethod
    def pygame_callback(data, obj):
        data.convert(carla.ColorConverter.Raw)
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
