import carla
import math
import time
import random

client = carla.Client("localhost", 2000)
world = client.get_world()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
spectator = world.get_spectator()
transform = carla.Transform(
    vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
    vehicle.get_transform().rotation,
)
spectator.set_transform(transform)

for i in range(100):
    vehicle_bp = random.choice(bp_lib.filter("vehicle"))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

for v in world.get_actors().filter("*vehicle*"):
    v.set_autopilot(True)
