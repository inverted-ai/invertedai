[pypi-badge]: https://badge.fury.io/py/invertedai.svg
[pypi-link]: https://pypi.org/project/invertedai/
[python-badge]: https://img.shields.io/pypi/pyversions/invertedai.svg?color=%2334D058
[ci-badge]: https://github.com/inverted-ai/invertedai/actions/workflows/CI.yml/badge.svg?branch=master
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-link]: https://colab.research.google.com/github/inverted-ai/invertedai/blob/develop/examples/IAI_full_demo.ipynb
[rest-link]: https://app.swaggerhub.com/apis-docs/InvertedAI/InvertedAI
[examples-link]: https://github.com/inverted-ai/invertedai/tree/master/examples

[![Documentation Status](https://readthedocs.org/projects/inverted-ai/badge/?version=latest)](https://inverted-ai.readthedocs.io/en/latest/?badge=latest)
[![PyPI][pypi-badge]][pypi-link]
[![python-badge]][pypi-link]
[![ci-badge]](https://github.com/inverted-ai/invertedai/actions/workflows/CI.yml)
[![Open In Colab][colab-badge]][colab-link]

# InvertedAI

## Overview
<!-- start elevator-pitch -->
Inverted AI provides an API for controlling non-playable characters (NPCs) in autonomous driving simulations,
available as either a [REST API][rest-link] or a [Python SDK](https://docs.inverted.ai/en/latest/pythonapi/index.html), (and [C++ SDK](https://docs.inverted.ai/en/latest/cppapi/index.html)) built on top of it. Using the API requires an access key -
create an account on our [user portal](https://www.inverted.ai/portal/login) to get one.  New users are given keys preloaded with an API access budget; researcher users affiliated to academic institutions generally receive a sufficient amount of credits to conduct their research for free.  This page describes how to get started quickly. For more in-depth understanding,
see the [API usage guide](https://docs.inverted.ai/en/latest/userguide.html), and detailed documentation for the [REST API][rest-link],
the [Python SDK](https://docs.inverted.ai/en/latest/pythonapi/index.html), and the [C++ SDK](https://docs.inverted.ai/en/latest/cppapi/index.html).
To understand the underlying technology and why it's necessary for autonomous driving simulations, visit the
[Inverted AI website](https://www.inverted.ai/).
<!-- end elevator-pitch -->

![](docs/images/top_camera.gif)

# Getting started
<!-- start quickstart -->
## Installation
For installing the Python package from [PyPI][pypi-link]:

```bash
pip install --upgrade invertedai
```

The Python client SDK is [open source](https://github.com/inverted-ai/invertedai),
so you can also download it and build locally.


## Minimal example

``` python
import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import matplotlib.pyplot as plt

location = "canada:drake_street_and_pacific_blvd"  # select one of available locations

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_properties=get_default_agent_properties({AgentType.car:10}),  # number of NPCs to spawn
)
agent_properties = response.agent_properties  # get dimension and other attributes of NPCs

rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_properties=agent_properties,
)

print("Begin stepping through simulation.")
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)

    # query the API for subsequent NPC predictions
    response = iai.drive(
        location=location,
        agent_properties=agent_properties,
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

```


### Stateful Cosimulation
Conceptually, the API is used to establish synchronous co-simulation between your own simulator running locally on
your machine and the NPC engine running on Inverted AI servers. The basic integration in Python looks like this.

```python
import invertedai as iai
from invertedai.common import AgentType
from invertedai import get_regions_default
from invertedai.utils import get_default_agent_properties

import numpy as np
import matplotlib.pyplot as plt

from typing import List

iai.add_apikey('5RE3ho3Yhx7njlN0m5fna32MAKp0FNdR1MQWPZ8Z')  # Specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
LOCATION = "canada:drake_street_and_pacific_blvd"

NUM_EGO_AGENTS = 1
NUM_NPC_AGENTS = 10
NUM_TIME_STEPS = 100

##########################################################################################################
# INSERT YOUR OWN EGO PREDICTIONS FOR THE INITIALIZATION
ego_response = iai.initialize(
    location = LOCATION,
    agent_properties = get_default_agent_properties({AgentType.car:NUM_EGO_AGENTS}),
)
ego_agent_properties = ego_response.agent_properties  # get dimension and other attributes of NPCs
##########################################################################################################

# Generate the region objects for large_initialization
regions = get_regions_default(
    location = LOCATION,
    agent_count_dict = {AgentType.car: NUM_NPC_AGENTS}
)
# Instantiate a stateful wrapper for Inverted AI API
iai_simulation = iai.BasicCosimulation(  
    location = LOCATION,
    ego_agent_properties = ego_agent_properties,
    ego_agent_agent_states = ego_response.agent_states,
    regions = regions,
    traffic_light_state_history = [ego_response.traffic_lights_states]
)

# Initialize the ScenePlotter for scene visualization
location_info_response = iai.location_info(location=LOCATION)
rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)
scene_plotter.initialize_recording(
    agent_states = iai_simulation.agent_states,
    agent_properties = iai_simulation.agent_properties,
    conditional_agents = list(range(NUM_EGO_AGENTS)),
    traffic_light_states = ego_response.traffic_lights_states
)

print("Begin stepping through simulation.")
for _ in range(NUM_TIME_STEPS):  # How many simulation time steps to execute (10 steps is 1 second)
##########################################################################################################    
    # INSERT YOUR OWN EGO PREDICTIONS FOR THIS TIME STEP
    ego_response = iai.drive(
        location = LOCATION,
        agent_properties = ego_agent_properties+iai_simulation.npc_properties,
        agent_states = ego_response.agent_states+iai_simulation.npc_states,
        recurrent_states = ego_response.recurrent_states+iai_simulation.npc_recurrent_states,
        light_recurrent_states = ego_response.light_recurrent_states,
    )
    ego_response.agent_states = ego_response.agent_states[:NUM_EGO_AGENTS]
    ego_response.recurrent_states = ego_response.recurrent_states[:NUM_EGO_AGENTS]
##########################################################################################################

    # Query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step(
        current_ego_agent_states = ego_response.agent_states,
        traffic_lights_states = ego_response.traffic_lights_states
    )

    # Save the visualization with ScenePlotter
    scene_plotter.record_step(iai_simulation.agent_states,iai_simulation.light_states)

# Save the visualization to disk
print("Simulation finished, save visualization.")
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
plt.axis('off')
gif_name = 'cosimulation_minimal_example.gif'
scene_plotter.animate_scene(
    output_name = gif_name,
    ax = ax,
    direction_vec = False,
    velocity_vec = False,
    plot_frame_number = True
)
print("Done")
```
To quickly check out how Inverted AI NPCs
behave, try our
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/IAI_full_demo.ipynb),
where all agents are NPCs, or go to our
[github repository](https://github.com/inverted-ai/invertedai/tree/master/examples) to execute it locally.
When you're ready to try our NPCs with a real simulator, see the example [CARLA integration](https://github.com/inverted-ai/invertedai/tree/master/examples/carla).
The examples are currently only provided in Python, but if you want to use the API from another language,
you can use the [REST API][rest-link] directly.

<!-- end quickstart -->
