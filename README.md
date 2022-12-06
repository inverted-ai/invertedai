[pypi-badge]: https://badge.fury.io/py/invertedai.svg
[pypi-link]: https://pypi.org/project/invertedai/
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-link]: https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only_colab.ipynb
[rest-link]: https://app.swaggerhub.com/apis/swaggerhub59/Inverted-AI
[examples-link]: https://github.com/inverted-ai/invertedai/tree/master/examples

[![Documentation Status](https://readthedocs.org/projects/inverted-ai/badge/?version=latest)](https://inverted-ai.readthedocs.io/en/latest/?badge=latest)
[![PyPI][pypi-badge]][pypi-link]
[![Open In Colab][colab-badge]][colab-link]

# InvertedAI

## Overview
<!-- start elevator-pitch -->
Inverted AI provides an API for controlling non-playable characters (NPCs) in autonomous driving simulations,
available as either a [REST API][rest-link] or a [Python SDK](https://docs.inverted.ai/en/latest/pythonapi/index.html), (and [C++ SDK](https://docs.inverted.ai/en/latest/cppapi/index.html)) built on top of it. Using the API requires an access key -
[contact us](mailto:sales@inverted.ai) to get yours. This page describes how to get started quickly. For more in-depth understanding,
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

Conceptually, the API is used to establish synchronous co-simulation between your own simulator running locally on
your machine and the NPC engine running on Inverted AI servers. The basic integration in Python looks like this.

```python
from typing import List
import numpy as np
import imageio
import invertedai as iai

# iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable


class LocalSimulator:
    """
    Mock up of a local simulator, where you control the ego vehicle. This example only supports single ego vehicle.
    """
    def __init__(self, ego_state: iai.AgentState, npc_states: List[iai.AgentState]):
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

        self.ego_state = iai.AgentState(
            center=iai.Point(x=self.ego_state.center.x + dx, y=self.ego_state.center.y + dy),
            orientation=self.ego_state.orientation,
            speed=self.ego_state.speed,
        )

    def step(self, predicted_npc_states):
        self._step_ego()  # ego vehicle moves first so that it doesn't see future NPC movement
        self.npc_states = predicted_npc_states
        return self.ego_state


iai_simulation = iai.BasicCosimulation(  # instantiate a stateful wrapper for Inverted AI API
    location='canada:vancouver:ubc_roundabout',  # select one of available locations
    agent_count=5,  # how many vehicles in total to use in the simulation
    ego_agent_mask=[True, False, False, False, False],  # first vehicle is ego, rest are NPCs
    get_birdview=True,  # provides simple visualization - don't use in production
    traffic_lights=True,  # gets the traffic light states and used for initialization and steping the simulation
)
local_simulation = LocalSimulator(iai_simulation.ego_states[0], iai_simulation.npc_states)
images = [iai_simulation.birdview.decode()]  # images storing visualizations of subsequent states
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)
    # query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step([local_simulation.ego_state])
    # collect predictions for the next time step
    predicted_npc_behavior = iai_simulation.npc_states
    # execute predictions in your simulator, using your actions for the ego vehicle
    updated_ego_agent_state = local_simulation.step(predicted_npc_behavior)
    # save the visualization - requires np and cv2
    images.append(iai_simulation.birdview.decode())
# save the visualization to disk
imageio.mimsave("iai-example.gif", np.array(images), format="GIF-PIL")

```
To quickly check out how Inverted AI NPCs
behave, try our
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only_colab.ipynb),
where all agents are NPCs, or go to our
[github repository](https://github.com/inverted-ai/invertedai/tree/master/examples) to execute it locally.
When you're ready to try our NPCs with a real simulator, see the example [CARLA integration](https://github.com/inverted-ai/invertedai/tree/master/examples/carla).
The examples are currently only provided in Python, but if you want to use the API from another language,
you can use the [REST API][rest-link] directly.

<!-- end quickstart -->
