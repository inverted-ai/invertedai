from utils import CarlaEnv, CarlaSimulationConfig

import pygame


sim = CarlaEnv.from_preset_data()
sim.set_npc_autopilot()
sim.set_ego_autopilot()


clock = pygame.time.Clock()

for i in range(10 * sim.config.fps):
    sim.step()
    clock.tick_busy_loop(sim.config.fps)

sim.destroy()
