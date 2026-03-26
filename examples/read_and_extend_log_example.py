import invertedai as iai
from invertedai import (
    AgentType,
    SimulationManager,
    ScenePlotterConfig,
    LogWriterConfig,
    LogReader,
)
import matplotlib.pyplot as plt
import os

LOG_PATH = "hi.json"  # path to an existing log file
EXTEND_LENGTH = 100  # number of additional timesteps to simulate

api_key = os.environ.get("IAI_API_KEY", None)
if api_key is None:
    iai.add_apikey("<INSERT_KEY_HERE>")

# Read the existing log
print("Reading log...")
log_reader = LogReader(log_path=LOG_PATH)
log_reader.initialize()

# Replay through the entire log to reach the final state
while log_reader.drive():
    pass

location = log_reader.location
location_info_response = log_reader.location_info_response

# Set up SimulationManager with visualization and logging
scene_plotter_cfg = ScenePlotterConfig(location=location, location_info_response=location_info_response)
log_cfg = LogWriterConfig(
    log_path="extended_log.json",
    location=location,
    location_info_response=location_info_response,
)
sim_manager = SimulationManager(scene_plotter_cfg=scene_plotter_cfg, log_writer_cfg=log_cfg)

# Seed the SimulationManager with the final state from the log
agent_dict = log_reader.agents
sim_manager.agents_dict = agent_dict

# Initialize from the log's last state using large_initialize
regions = iai.get_regions_default(agent_count_dict={AgentType.car: 1}, location=location)
response = sim_manager.initialize(
    location=location,
    regions=regions,
)

print(f"Extending simulation by {EXTEND_LENGTH} steps...")
for step in range(EXTEND_LENGTH):
    response = sim_manager.drive(
        location=location,
        light_recurrent_states=response.light_recurrent_states,
    )

print("Saving visualization...")
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
sim_manager.visualize_data(
    output_name="extended_simulation.gif",
    ax=ax,
    direction_vec=True,
    velocity_vec=False,
    plot_frame_number=True,
)

print("Saving extended log...")
sim_manager.export_log("extended_log.json")
print("Done")
