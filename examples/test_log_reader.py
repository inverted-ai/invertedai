
import invertedai as iai
from invertedai import (
    AgentType,
    AgentData,
    SimulationManager,
    ScenePlotterConfig,
    WaypointManagerConfig,
    LogWriterConfig,
    RegionsConfig,
    get_default_agent_properties,
)
import matplotlib.pyplot as plt
import os
import uuid
LOG_PATH= "usa_sanjose_highway_scenario.json"
log_reader = iai.LogReader(LOG_PATH)
print(f"  Log length: {log_reader.log_length} timesteps")
print(f"  Location: {log_reader.location}")

log_reader.initialize()
print(f"  Agent count at t=0: {len(log_reader.agent_states)}")
print(f"  Agent IDs at t=0 (from agents dict): {list(log_reader.agents.keys())}")

# Step through and verify agent counts change
step = 0
while log_reader.drive():
    step += 1
print(f"  Replayed {step} drive steps")

# Visualize the replayed log
log_reader.visualize(
    gif_path=LOG_PATH,
    fov=200,
    resolution=(1024, 1024),
    dpi=150,
    direction_vec=True,
    plot_frame_number=True,
)
assert os.path.exists(LOG_PATH), f"Replay GIF not created at {LOG_PATH}"
print(f"  Replay GIF exported to {LOG_PATH}")
