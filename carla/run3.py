import carla
import time

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
while True:
    world_snapshot = world.wait_for_tick()
