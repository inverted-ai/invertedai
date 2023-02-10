from tqdm import tqdm
import argparse
import invertedai as iai
from carla_simulator import PreSets
from invertedai.common import AgentState, StaticMapActor
from invertedai.simulation.utils import Rectangle
from invertedai.simulation.regions import QuadTree
from invertedai.simulation.simulator import Simulation, SimulationConfig
import pathlib
import matplotlib.pyplot as plt
import pygame
from pygame.math import Vector2
path = pathlib.Path(__file__).parent.resolve()

Width, Height = 1000, 1000
agent_per_region = 20
Resolution = (Width, Height)

NODE_CAPACITY = 5
RADIUS = 10

SIZE = 120

Black, White, Blue, Red, Green = (0, 0, 0), (255, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255)
Color1 = (1, 1, 1)


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--fov", type=float, default=500)
parser.add_argument("--location", type=str,  default="carla:Town10HD")
parser.add_argument("--center", type=str,  default="carla:Town10HD")
parser.add_argument("-l", "--episode_length", type=int, default=300)
parser.add_argument("-sp", "--scene_plotter", type=int, default=0)
args = parser.parse_args()
# map_center = PreSets.map_centers[args.location]


if args.api_key is not None:
    iai.add_apikey(args.api_key)
response = iai.location_info(location=args.location)
if response.birdview_image is not None:
    rendered_static_map = response.birdview_image.decode()
    map_image = pygame.surfarray.make_surface(rendered_static_map)
    map_fov = response.map_fov

map_width, map_height = map_fov, map_fov
scene_plotter = iai.utils.ScenePlotter(rendered_static_map, map_fov,
                                       (response.map_center.x,
                                        response.map_center.y), response.static_actors) if args.scene_plotter == 1 else None

screen = pygame.display.set_mode(Resolution)

pygame.display.set_caption("Quadtree")
clock = pygame.time.Clock()

map_center_x = response.map_center.x
map_center_y = response.map_center.y
(x_min, x_max, y_min, y_max, H, W) = map_center_x-map_width/2, map_center_x +\
    map_width/2, map_center_y-map_height/2, map_center_y+map_height/2, Width, Height


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


top_left = convert_to_pygame_coords(map_center_x-(map_fov/2), map_center_y-(map_fov/2))
bottom_right = convert_to_pygame_coords(map_center_x+(map_fov/2), map_center_y+(map_fov/2))
rect_1 = pygame.Rect(convert_to_pygame_coords(map_center_x, map_center_y), (10, 10))
rect_2 = pygame.Rect(top_left, (10, 10))
rect_3 = pygame.Rect(bottom_right, (10, 10))

x_scale = bottom_right[0] - top_left[0]
y_scale = bottom_right[1] - top_left[1]

simcfg = SimulationConfig(location=args.location, map_center=(map_center_x, map_center_y), map_fov=map_fov)
simcfg.convert_to_pygame_coords = convert_to_pygame_coords
simcfg.convert_to_pygame_scales = convert_to_pygame_scales
simcfg.node_capacity = NODE_CAPACITY
simulation = Simulation(cfg=simcfg, location=args.location, center=(map_center_x, map_center_y), width=map_width+200, height=map_height+200,
                        agent_per_region=agent_per_region, screen=screen, convertor=convert_to_pygame_coords, region_fov=100, use_quadtree=True, initialize_stride=50)

fps = 100
run = True
# fig, ax = plt.subplots(1, 1)
while run:
    screen.fill(Color1)
    pygame.display.set_caption("QuadTree Fps: " + str(int(clock.get_fps())))

    screen.blit(pygame.transform.scale(pygame.transform.flip(
        pygame.transform.rotate(map_image, 90), True, False), (x_scale, y_scale)), top_left)
    pygame.draw.rect(screen, Blue, rect_1)
    pygame.draw.rect(screen, Green, rect_2)
    pygame.draw.rect(screen, White, rect_3)
    # screen.blit(pygame.transform.scale(pygame.transform.flip(
    #     pygame.transform.rotate(map_image, 90), True, True), (x_scale, y_scale)), top_left)

    # ----- HANDLE EVENTS ------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                run = False
            if event.key == pygame.K_q:
                run = False
    # -----------------------------

    simulation.drive()
    simulation.show()

    clock.tick(fps)

    # boundary = Rectangle(Vector2(map_center_x-(map_fov/2), map_center_y-(map_fov/2)), Vector2(
    #     (map_fov, map_fov)), convertors=(convert_to_pygame_coords, convert_to_pygame_scales))

    # quadtree = QuadTree(cfg=simcfg, capacity=NODE_CAPACITY, boundary=boundary,
    #                     convertors=(convert_to_pygame_coords, convert_to_pygame_scales))
    # quadtree.lineThickness = 1
    # quadtree.color = (0, 87, 146)

    # for npc in simulation.npcs:
    #     # quadtree.insert((npc.position.x, npc.position.y))
    #     quadtree.insert(npc)

    # quadtree.insert(convert_to_pygame_coords(npc.position.x, npc.position.y))

    # boundary = Rectangle(Vector2(top_left), Vector2(x_scale, y_scale),
    #                      convertors=None)
    # quadtree = QuadTree(NODE_CAPACITY, boundary, convertors=None)
    # quadtree.lineThickness = 1
    # quadtree.color = (0, 87, 146)

    # for npc in simulation.npcs:
    #     quadtree.insert(convert_to_pygame_coords(npc.position.x, npc.position.y))

    # quadtree.Show(screen)

    # flock.Simulate()

    # if showQuadTree:
    #     quadtree.Show(screen)

    pygame.display.flip()
    if scene_plotter:
        scene_plotter.plot_scene(simulation.agent_states,
                                 simulation.agent_attributes,
                                 numbers=False, velocity_vec=False, direction_vec=True)
        plt.draw()
        plt.pause(0.0001)
    # plt.show()

pygame.quit()

# location_info_response = iai.location_info(location=args.location)
# file_name = args.location.replace(":", "_")
# if location_info_response.osm_map is not None:
#     file_path = f"{file_name}.osm"
#     with open(file_path, "w") as f:
#         location_info_response.osm_map.save_osm_file(file_path)
# if location_info_response.birdview_image is not None:
#     file_path = f"{file_name}.jpg"
#     location_info_response.birdview_image.decode_and_save(file_path)

# rendered_static_map = location_info_response.birdview_image.decode()
# corrected_static_actors = [StaticMapActor.fromdict(dict(x=state.center.x, y=-state.center.y, actor_id=state.actor_id,
#                                                         agent_type=state.agent_type, orientation=-state.orientation, length=state.length, width=state.width, dependant=state.dependant
#                                                         ))for state in location_info_response.static_actors]


# light_response = iai.light(location=args.location)
# response = iai.utils.area_initialization(
#     location=args.location, agent_density=6, traffic_lights_states=None, map_center=map_center, width=500, height=500, stride=50, static_actors=corrected_static_actors)


# # Carla simulator uses left-hand coordinates and xord map are right-handed, thus, the position of agents require a simple transformation
# corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
#                                          -state.orientation, state.speed]) for state in response.agent_states]
# open_drive_file_name = f"{path}/data/open_drive/{args.location.split(':')[1]}.csv"
# scene_plotter = iai.utils.ScenePlotter(
#     fov=args.fov, xy_offset=(map_center[0], -map_center[1]), static_actors=corrected_static_actors, open_drive=open_drive_file_name)

# agent_attributes = response.agent_attributes
# scene_plotter.initialize_recording(corrected_agents, agent_attributes=agent_attributes)

# frames = []
# x_frame = []
# for i in tqdm(range(args.episode_length), desc=f"Driving {args.location.split(':')[1]}"):
#     light_response = iai.light(
#         location=args.location, recurrent_states=light_response.recurrent_states)
#     response = iai.drive(
#         agent_attributes=agent_attributes,
#         agent_states=response.agent_states,
#         recurrent_states=response.recurrent_states,
#         location=args.location,
#         traffic_lights_states=light_response.traffic_lights_states,
#     )

#     corrected_agents = [AgentState.fromlist([state.center.x, -state.center.y,
#                                              (-state.orientation), state.speed]) for state in response.agent_states]
#     scene_plotter.record_step(corrected_agents, traffic_light_states=light_response.traffic_lights_states)

# gif_name = f'iai-driving-on-{args.location}.gif'
# ani = scene_plotter.animate_scene(output_name=gif_name,
#                                   numbers=False, direction_vec=False, velocity_vec=False,
#                                   plot_frame_number=False)


# ani.save(f'iai-driving-on-{args.location.split(":")[1]}.mp4')
# plt.show(block=True)
