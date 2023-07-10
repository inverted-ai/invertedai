import numpy as np
import asyncio
import imageio

import invertedai as iai

location = "canada:vancouver:drake_street_and_pacific_blvd"  # select one of available locations

# iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

# get traffic light states
light_response = iai.light(location=location)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_count=25,    # number of NPCs to spawn
    get_birdview=True,  # provides simple visualization - don't use in production
    traffic_light_state_history=[light_response.traffic_lights_states]  # provide traffic light states
)
agent_attributes = response.agent_attributes  # get dimension and other attributes of NPCs

images = [response.birdview.decode()]  # images storing visualizations of subsequent states
agent_state_history = []
traffic_light_state_history = []

for _ in range(200):  # how many simulation steps to execute (10 steps is 1 second)

    # get next traffic light state
    light_response = iai.light(location=location, recurrent_states=light_response.recurrent_states)

    # query the API for subsequent NPC predictions
    response = iai.drive(
        location=location,
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdview=True,
        traffic_lights_states=light_response.traffic_lights_states,
        get_infractions=True,
        random_seed=1
    )

    agent_state_history.append(response.agent_states)
    traffic_light_state_history.append(light_response.traffic_lights_states)

blame_response = asyncio.run(iai.async_blame(
    location=location,
    candidate_agents=[0, 0],
    agent_state_history=agent_state_history,
    agent_attributes=agent_attributes,
    traffic_light_state_history=traffic_light_state_history,
    get_birdviews=True,
    detect_collisions=True
))

print("blamed_collisions")
print(blame_response.blamed_collisions)


images = [birdview.decode() for birdview in blame_response.birdviews]

# save the visualization to disk
imageio.mimsave("iai-example.gif", np.array(images), format="GIF-PIL")
