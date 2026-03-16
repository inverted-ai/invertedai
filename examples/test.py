import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import os
import matplotlib.pyplot as plt

from random import randint
from copy import deepcopy

print("Reading log...")

log_reader = iai.LogReader(f"keyed_minimal_example_log.json")
gif_path_replay = os.path.join(os.getcwd(),f"keyed_minimal_example_log.gif")
log_reader.visualize(
    gif_path=gif_path_replay,
    fov = 200,
    resolution = (2048,2048),
    dpi = 300,
    map_center = None,
    direction_vec = True,
    velocity_vec = False,
    plot_frame_number = True
)

# print("Extending read log...")

# location_info_response_replay = log_reader.location_info_response
# log_reader.initialize()
# agent_properties = log_reader.agent_properties

# rendered_static_map = location_info_response_replay.birdview_image.decode()
# scene_plotter_new = iai.utils.ScenePlotter(
#     rendered_static_map,
#     location_info_response_replay.map_fov,
#     (location_info_response_replay.map_center.x, location_info_response_replay.map_center.y),
#     location_info_response_replay.static_actors,
#     left_hand_coordinates = True
# )
# scene_plotter_new.initialize_recording(
#     agent_states=log_reader.agent_states,
#     agent_properties=agent_properties
# )

# print("Stepping through simulation...")
# while True: # Log reader will return False when it has run out of simulation data
#     is_timestep_populated = log_reader.drive()
#     if not is_timestep_populated:
#         break
#     agent_properties = log_reader.agent_properties
#     scene_plotter_new.record_step(
#         agent_states=log_reader.agent_states,
#         traffic_light_states=log_reader.traffic_lights_states,
#         agent_properties=agent_properties
#     )

# agent_states = log_reader.agent_states
# recurrent_states = log_reader.recurrent_states
# traffic_lights_states = log_reader.traffic_lights_states
# light_recurrent_states = log_reader.light_recurrent_states
# for _ in range(150): 
#     response = iai.drive(
#         location=log_reader.location,
#         agent_properties=agent_properties,
#         agent_states=agent_states,
#         recurrent_states=recurrent_states,
#         light_recurrent_states=light_recurrent_states
#     )

#     agent_states = response.agent_states
#     recurrent_states = response.recurrent_states
#     traffic_lights_states = response.traffic_lights_states
#     light_recurrent_states = response.light_recurrent_states

#     scene_plotter_new.record_step(
#         agent_states=agent_states,
#         traffic_light_states=traffic_lights_states,
#         agent_properties=agent_properties
#     )

# gif_path_extended = os.path.join(os.getcwd(),f"keyed_replayed.gif")
# fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
# plt.axis('off')
# scene_plotter_new.animate_scene(
#     output_name=gif_path_extended,
#     ax=ax,
#     direction_vec = True,
#     velocity_vec = False,
#     plot_frame_number = True

# )