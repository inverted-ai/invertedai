import argparse
import invertedai as iai
from invertedai.simulation.simulator import Simulation, SimulationConfig
import pathlib
import pygame
path = pathlib.Path(__file__).parent.resolve()


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--fov", type=float, default=500)
parser.add_argument("--location", type=str,  default="carla:Town10HD")
parser.add_argument("--center", type=str,  default="carla:Town10HD")
parser.add_argument("-l", "--episode_length", type=int, default=300)
parser.add_argument("-cap", "--quadtree_capacity", type=int, default=15)
parser.add_argument("-ad", "--agent_density", type=int, default=10)
parser.add_argument("-ri", "--re_initialization", type=int, default=30)
args = parser.parse_args()


if args.api_key is not None:
    iai.add_apikey(args.api_key)
response = iai.location_info(location=args.location)
if response.birdview_image is not None:
    rendered_static_map = response.birdview_image.decode()

cfg = SimulationConfig(location=args.location, map_center=(response.map_center.x, response.map_center.y),
                       map_fov=response.map_fov, rendered_static_map=rendered_static_map,
                       map_width=response.map_fov+200, map_height=response.map_fov+200, agent_density=args.agent_density,
                       initialize_stride=50, quadtree_capacity=args.quadtree_capacity,
                       re_initialization_period=args.re_initialization)
simulation = Simulation(cfg=cfg)

fps = 100
clock = pygame.time.Clock()
run = True
while run:
    # ----- HANDLE EVENTS ------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYUP:
            if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                run = False
            if event.key == pygame.K_t:
                simulation.show_quadtree = not simulation.show_quadtree
            if event.key == pygame.K_l:
                for npc in simulation.npcs:
                    npc.show_agent_neighbors = not npc.show_agent_neighbors
    # -----------------------------
    simulation.drive()
    clock.tick(fps)

pygame.quit()
