---
hide-toc: true
---

```{eval-rst}
.. image:: ../images/banner.svg
  :width: 600
  :alt: Alternative text
```

[pypi-badge]: https://badge.fury.io/py/invertedai.svg
[pypi-link]: https://pypi.org/project/invertedai/
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-link]: https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb

[![PyPI][pypi-badge]][pypi-link]
[![Open In Colab][colab-badge]][colab-link]

Inverted AI provides an API for controlling non-playable characters (NPCs) in autonomous driving simulations,
available as either a REST API or a Python library built on top of it. Using the API requires an access key -
[contact us]() to get yours. This page describes how to get started quickly. For more in-depth understanding,
see the [API usage guide](userguide.md), and detailed documentation for the [REST API](apireference.md) and the
[Python library](pythonapi/index.md).
To understand the underlying technology and why it's necessary for autonomous driving simulations, visit
[Inverted AI website](https://www.inverted.ai/).


![](../images/top_camera.gif)


## Installation

### Packaged version
For installing the Python package from [PyPI]([pypi-link]) :

```bash
pip install invertedai
```

The `invertedai` package has minimal dependencies on other python packages.
However, it is always recommended to install the package in a virtual environment. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install invertedai
```


### From source
The Python client library is open source. To download the source and build locally, go to the
[github repository](https://github.com/inverted-ai/invertedai).


## Minimal example

Conceptually, the API is used to establish synchronous co-simulation between your own simulator running locally on
your machine and the NPC engine running on Inverted AI servers. The basic integration in Python looks like this.

```python
import invertedai as iai

iai.add_apikey('')  # specify your key here or through IAI_API_KEY variable

agent_count = 15  #  how many vehicles in total to use in the simulation
iai_simulation = iai.Simulation(  # instantiate a stateful wrapper for Inverted AI API
    location='canada:vancouver:ubc_roundabout',  # select one of available locations
    agent_count=agent_count,
    ego_agent_mask=[True] + [False] * (agent_count-1)  # first vehicle is ego, rest are NPCs
)
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)
    # collect predictions for the next time step
    predicted_npc_behavior = iai_simulation.npc_states()
    # execute predictions in your simulator, using your actions for the ego vehicle
    updated_ego_agent_state = step_local_simulator(predicted_npc_behavior)
    # query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step(updated_ego_agent_state)
```

In order to execute this code, you need to connect a simulator locally. To quickly check out how Inverted AI NPCs
behave, try our
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb),
where all agents are NPCs, or go to our
[github repository](https://github.com/inverted-ai/invertedai/examples) to execute it locally.
When you're ready to try our NPCs with a real simulator, see the example [CARLA integration](examples/carlasim.md).
The examples are currently only provided in Python, but if you want to use the API from another language,
you can use the [REST API](apireference.md) directly.

## Further resources

The following pages will help you integrate the Inverted AI API with your own simulator.

```{toctree}
:maxdepth: 2

userguide
pythonapi/index
apireference
examples/index
```

<!-- ```{toctree} -->
<!-- :caption: Python SDK Library -->
<!-- :maxdepth: 2 -->

<!-- modules/modules -->
<!-- ``` -->



```{eval-rst}
REFERENCES
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
