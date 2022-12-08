import numpy as np
import imageio
import invertedai as iai

location = "canada:vancouver:drake_street_and_pacific_blvd"  # select one of available locations

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

# get traffic light states
light_response = iai.light(location=location)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_count=10,    # number of NPCs to spawn
    get_birdview=True,  # provides simple visualization - don't use in production
    traffic_light_state_history=[light_response.traffic_lights_states],  # provide traffic light states
)
agent_attributes = response.agent_attributes  # get dimension and other attributes of NPCs

images = [response.birdview.decode()]  # images storing visualizations of subsequent states
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)

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
    )

    # save the visualization - requires np and cv2
    images.append(response.birdview.decode())

# save the visualization to disk
imageio.mimsave("iai-example.gif", np.array(images), format="GIF-PIL")
