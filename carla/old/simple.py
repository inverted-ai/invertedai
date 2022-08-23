import carla
import math
import time
import random
import pygame

map_name = "Town03"
fps = 30
traffic_count = 100
simulation_time = 10  # In Seconds
roi_center = dict(x=0, y=0)  # region of interest center
proximity_threshold = 50

world_settings = carla.WorldSettings(
    synchronous_mode=True,
    fixed_delta_seconds=1 / float(fps),
)


def npcs_in_roi(npcs):
    vehicles_stats = []
    actor_geo_centers = []
    actors = []
    for actor in npcs:
        actor_geo_center = actor.get_location()

        distance = math.sqrt(
            ((actor_geo_center.x - roi_center["x"]) ** 2)
            + ((actor_geo_center.y - roi_center["y"]) ** 2)
        )
        if distance < proximity_threshold:
            bb = actor.bounding_box.extent
            length = max(
                2 * bb.x, 1.0
            )  # provide minimum value since CARLA returns 0 for some agents
            width = max(2 * bb.y, 0.2)
            physics_control = actor.get_physics_control()
            # Wheel position is in centimeter: https://github.com/carla-simulator/carla/issues/2153
            rear_left_wheel_position = physics_control.wheels[2].position / 100
            rear_right_wheel_position = physics_control.wheels[3].position / 100
            real_mid_position = 0.5 * (
                rear_left_wheel_position + rear_right_wheel_position
            )
            lr = actor_geo_center.distance(real_mid_position)
            front_left_wheel_position = physics_control.wheels[0].position / 100
            lf = front_left_wheel_position.distance(rear_left_wheel_position) - lr
            max_steer_angle = math.radians(physics_control.wheels[0].max_steer_angle)
            vehicles_stats.append([lr, lf, length, width, max_steer_angle])
            actor_geo_centers.append(actor_geo_center)
            actors.append(actor)
    return (actors, actor_geo_centers, vehicles_stats)


client = carla.Client("localhost", 2000)
traffic_manager = client.get_trafficmanager()
world = client.load_world(map_name)
debug = world.debug
original_settings = client.get_world().get_settings()
world.apply_settings(world_settings)
traffic_manager.set_synchronous_mode(True)

# spectator as a free camera
spectator = world.get_spectator()
# camera_loc = carla.Location(roi_center["x"], roi_center["y"], z=70)
# camera_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
# transform = carla.Transform(camera_loc, camera_rot)
# spectator.set_transform(transform)

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)
npcs = []

for spawn_point in spawn_points[:traffic_count]:
    vehicle_bp = random.choice(bp_lib.filter("vehicle"))
    npc = world.try_spawn_actor(vehicle_bp, spawn_point)
    npc.set_autopilot(True)
    npcs.append(npc)

clock = pygame.time.Clock()

for i in range(simulation_time * fps):
    world.tick()
    actors, actor_geo_centers, vehicles_stats = npcs_in_roi(npcs)
    for npc in actors:
        # npc.set_autopilot(False)
        loc = npc.get_location()
        loc.z += 3
        debug.draw_point(
            location=loc,
            size=0.1,
            color=carla.Color(0, 255, 0, 0),
            life_time=0.1,
        )

    clock.tick_busy_loop(fps)


## Clean up
for npc in npcs:
    try:
        npc.set_autopilot(False)
    except:
        pass
    npc.destroy()
client.get_world().apply_settings(original_settings)
traffic_manager.set_synchronous_mode(False)
