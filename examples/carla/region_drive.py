import sys
sys.path.append('../')
import matplotlib.pyplot as plt

import argparse
import invertedai as iai
from simulation.simulator import Simulation, SimulationConfig
import pathlib
import pygame
from tqdm import tqdm
from time import perf_counter
path = pathlib.Path(__file__).parent.resolve()


parser = argparse.ArgumentParser(description="Simulation Parameters.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--fov", type=float, default=500)
parser.add_argument("--location", type=str,  default="carla:Town10HD")
parser.add_argument("--center", type=str,  default="carla:Town10HD")
parser.add_argument("-l", "--episode_length", type=int, default=300)
parser.add_argument("-cap", "--quadtree_capacity", type=int, default=15)
parser.add_argument("-ad", "--agent_density", type=int, default=20)
parser.add_argument("-ri", "--re_initialization", type=int, default=30)
parser.add_argument("-len", "--simulation_length", type=int, default=600)
args = parser.parse_args()


if args.api_key is not None:
    iai.add_apikey(args.api_key)
response = iai.location_info(location=args.location)
if response.birdview_image is not None:
    rendered_static_map = response.birdview_image.decode()

cfg = SimulationConfig(location=args.location, map_center=(response.map_center.x, response.map_center.y),
                       map_fov=response.map_fov, rendered_static_map=rendered_static_map,
                       map_width=response.map_fov, map_height=response.map_fov, agent_density=args.agent_density,
                       initialize_stride=50, quadtree_capacity=args.quadtree_capacity,
                       re_initialization_period=args.re_initialization)
simulation = Simulation(cfg=cfg)

fps = 60
clock = pygame.time.Clock()
run = True
start = perf_counter()
fram_counter = 0
for _ in tqdm(range(args.simulation_length)):
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
    if not run:
        break
    # -----------------------------
    simulation.drive()
    clock.tick(fps)

pygame.quit()
print(f"Speed: {(perf_counter()-start)/simulation.timer} secs/frame")
