import carla
import math
import time

actorList = []
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    # world = client.get_world()
    world = client.load_world("Town02")
    spectator = world.get_spectator()
    actorList.append(spectator)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / 10
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("cybertruck")[0]
    transform = carla.Transform(
        carla.Location(x=130, y=195, z=40), carla.Rotation(yaw=180)
    )
    vehicle = world.spawn_actor(vehicle_bp, transform)
    actorList.append(vehicle)

    while True:
        # spectator_transform = vehicle.get_transform()
        # yaw_rad = math.radians(spectator_transform.rotation.yaw)
        # spectator_transform.location -= carla.Location(
        #     x=10 * math.cos(yaw_rad), y=10 * math.sin(yaw_rad), z=2.5
        # )
        spectator_transform = carla.Transform(
            vehicle.get_transform().transform(carla.Location(x=-6, z=2.5)),
            vehicle.get_transform().rotation,
        )
        spectator.set_transform(spectator_transform)
        world.tick()
finally:
    print("Destroying actors")
    client.apply_batch([carla.command.DestroyActor(x) for x in actorList])
