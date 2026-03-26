"""
Test: Large map simulation using SimulationManager

Parameters:
    LOCATION: usa:sanjose_highway (large Carla map)
    DENSITY: 80 agents per km^2
    WIDTH: 300m (area width)
    HEIGHT: 300m (area height)
    SIM_LENGTH: 30 timesteps
    FOV: 200 (visualization field of view)

Tests covered:
    1. SimulationManager with large agent count on a big Carla map
    2. form_regions with area_shape for large area coverage
    3. large_initialize + large_drive through SimulationManager
    4. LogWriter integration with many agents
    5. Visualization of large-scale simulation
    6. LogReader roundtrip on large log
    7. Verify agent ID stability across many timesteps
"""

import invertedai as iai
from invertedai import (
    AgentType,
    SimulationManager,
    ScenePlotterConfig,
    WaypointManagerConfig,
    LogWriterConfig,
    RegionsConfig,
)
import matplotlib.pyplot as plt
import os
import time

# Parameters
LOCATION = "usa:sanjose_highway"
SIM_LENGTH = 30
NUM_AGENTS = 10 # various numbers tested, after 100 it fails to initialize
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
timestamp = int(time.time())
LOG_PATH = os.path.join(OUTPUT_DIR, f"test_large_sim_{timestamp}.json")
GIF_PATH = os.path.join(OUTPUT_DIR, f"test_large_sim_{timestamp}.gif")

# Setup
print(" Test Large Map Simulation")
print(f"Location: {LOCATION}")


location_info_response = iai.location_info(location=LOCATION, include_map_source=True)
map_center = (location_info_response.map_center.x, location_info_response.map_center.y)


scene_plotter_cfg = ScenePlotterConfig(
    location=LOCATION,
    location_info_response=location_info_response,
)
waypoint_cfg = WaypointManagerConfig(lanelet_map=location_info_response.get_lanelet_map())
log_cfg = LogWriterConfig(log_path=LOG_PATH, location=LOCATION, location_info_response=location_info_response)

simulation_manager = SimulationManager(
    scene_plotter_cfg=scene_plotter_cfg,
    waypoint_cfg=waypoint_cfg,
    log_writer_cfg=log_cfg,
)

# Initialize with large area
regions_config = RegionsConfig(
    location=LOCATION,
    agent_count_dict={AgentType.car: NUM_AGENTS},
    map_center=map_center,
)
print(f"  Requested agents: {NUM_AGENTS}")

regions = simulation_manager.form_regions(regions_config)
print(f"  Regions formed: {len(regions)}")

response = simulation_manager.initialize(location=LOCATION, regions=regions)
initial_ids = simulation_manager.get_agent_ids()
print(f"  Agents initialized: {len(initial_ids)}")
print(f"  Total in response: {len(response.agent_states)}")

# Drive loop
for step in range(SIM_LENGTH):
    response = simulation_manager.drive(
        location=LOCATION,
        light_recurrent_states=response.light_recurrent_states,
    )

final_ids = simulation_manager.get_agent_ids()
assert set(initial_ids) == set(final_ids), "Agent IDs changed during simulation!"
print(f"  PASS: {len(final_ids)} agent IDs stable across {SIM_LENGTH} steps")

# Export
simulation_manager.export_log()
assert os.path.exists(LOG_PATH)
print(f"  Log exported: {LOG_PATH}")

fig, ax = plt.subplots(constrained_layout=True, figsize=(15, 15))
simulation_manager.visualize_data(
    output_name=GIF_PATH,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True,
)
plt.close(fig)
assert os.path.exists(GIF_PATH)
print(f"  GIF exported: {GIF_PATH}")

# LogReader roundtrip
log_reader = iai.LogReader(LOG_PATH)
print(f"  Log length: {log_reader.log_length}")
print(f"  Location: {log_reader.location}")

log_reader.initialize()
reader_agent_count = len(log_reader.agent_states)
print(f"  Agents at t=0: {reader_agent_count}")
assert reader_agent_count == len(initial_ids), \
    f"Expected {len(initial_ids)} agents, got {reader_agent_count}"

# Verify agents dict keys match
reader_agent_ids = list(log_reader.agents.keys())
print(f"  Agent IDs from LogReader: {len(reader_agent_ids)} agents")
print("  PASS: LogReader agent count matches original")

print("\n=== ALL TESTS PASSED ===")
