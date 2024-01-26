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
import numpy as np
import matplotlib.pyplot as plt
import invertedai as iai

location = "iai:drake_street_and_pacific_blvd"  # select one of available locations

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

print("Begin initialization.")
# get static information about a given location including map in osm
# format and list traffic lights with their IDs and locations.
location_info_response = iai.location_info(location=location)

# initialize the simulation by spawning NPCs
response = iai.initialize(
    location=location,  # select one of available locations
    agent_count=10,    # number of NPCs to spawn
    get_birdview=True,  # provides simple visualization - don't use in production
    traffic_light_state_history=None
)
agent_attributes = response.agent_attributes  # get dimension and other attributes of NPCs

location_info_response = iai.location_info(location=location)
rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(rendered_static_map,
                                       location_info_response.map_fov,
                                       (location_info_response.map_center.x, location_info_response.map_center.y),
                                       location_info_response.static_actors)
scene_plotter.initialize_recording(
    agent_states=response.agent_states,
    agent_attributes=agent_attributes,
)

print("Begin stepping through simulation.")
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)

    # query the API for subsequent NPC predictions
    response = iai.drive(
        location=location,
        agent_attributes=agent_attributes,
        agent_states=response.agent_states,
        recurrent_states=response.recurrent_states,
        get_birdview=True,
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
from typing import List
import numpy as np
import invertedai as iai
import matplotlib.pyplot as plt

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable


class LocalSimulator:
    """
    Mock up of a local simulator, where you control the ego vehicle. This example only supports single ego vehicle.
    """

    def __init__(self, ego_state: iai.common.AgentState, npc_states: List[iai.common.AgentState]):
        self.ego_state = ego_state
        self.npc_states = npc_states

    def _step_ego(self):
        """
        The simple motion model drives forward with constant speed.
        The ego agent ignores the map and NPCs for simplicity.
        """
        dt = 0.1
        dx = self.ego_state.speed * dt * np.cos(self.ego_state.orientation)
        dy = self.ego_state.speed * dt * np.sin(self.ego_state.orientation)

        self.ego_state = iai.common.AgentState(
            center=iai.common.Point(x=self.ego_state.center.x + dx, y=self.ego_state.center.y + dy),
            orientation=self.ego_state.orientation,
            speed=self.ego_state.speed,
        )

    def step(self, predicted_npc_states):
        self._step_ego()  # ego vehicle moves first so that it doesn't see future NPC movement
        self.npc_states = predicted_npc_states
        return self.ego_state

print("Begin initialization.")
location = 'iai:ubc_roundabout'
iai_simulation = iai.BasicCosimulation(  # instantiate a stateful wrapper for Inverted AI API
    location=location,  # select one of available locations
    agent_count=5,  # how many vehicles in total to use in the simulation
    ego_agent_mask=[True, False, False, False, False],  # first vehicle is ego, rest are NPCs
    get_birdview=False,  # provides simple visualization - don't use in production
    traffic_lights=True,  # gets the traffic light states and used for initialization and steping the simulation
)

location_info_response = iai.location_info(location=location)
rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(rendered_static_map,
                                       location_info_response.map_fov,
                                       (location_info_response.map_center.x, location_info_response.map_center.y),
                                       location_info_response.static_actors)
scene_plotter.initialize_recording(
    agent_states=iai_simulation.agent_states,
    agent_attributes=iai_simulation.agent_attributes,
)

print("Begin stepping through simulation.")
local_simulation = LocalSimulator(iai_simulation.ego_states[0], iai_simulation.npc_states)
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)
    # query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step([local_simulation.ego_state])
    # collect predictions for the next time step
    predicted_npc_behavior = iai_simulation.npc_states
    # execute predictions in your simulator, using your actions for the ego vehicle
    updated_ego_agent_state = local_simulation.step(predicted_npc_behavior)
    # save the visualization with ScenePlotter
    scene_plotter.record_step(iai_simulation.agent_states)

print("Simulation finished, save visualization.")
# save the visualization to disk
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
gif_name = 'cosimulation_minimal_example.gif'
scene_plotter.animate_scene(
    output_name=gif_name,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
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
