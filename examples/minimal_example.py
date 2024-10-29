import numpy as np
import matplotlib.pyplot as plt
import invertedai as iai

location = "canada:drake_street_and_pacific_blvd"  # select one of available locations

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_count=10,    # number of NPCs to spawn
)
agent_attributes = response.agent_attributes  # get dimension and other attributes of NPCs

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)
scene_plotter.initialize_recording(
    response.agent_states,
    agent_attributes,
)

print("Begin stepping through simulation.")
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)

    # query the API for subsequent NPC predictions
    response = iai.drive(
        location=location,
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        light_recurrent_states=response.light_recurrent_states,
    )

    # save the visualization
    scene_plotter.record_step(response.agent_states,response.traffic_lights_states)

print("Simulation finished, save visualization.")
# save the visualization to disk
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
gif_name = 'minimal_example.gif'
scene_plotter.animate_scene(
    output_name=gif_name,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
)
print("Done")