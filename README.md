[pypi-badge]: https://badge.fury.io/py/invertedai.svg
[pypi-link]: https://pypi.org/project/invertedai/
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-link]: https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only_colab.ipynb


[![Documentation Status](https://readthedocs.org/projects/inverted-ai/badge/?version=latest)](https://inverted-ai.readthedocs.io/en/latest/?badge=latest)
[![PyPI][pypi-badge]][pypi-link]
[![Open In Colab][colab-badge]][colab-link]

# InvertedAI

## Overview
<!-- start elevator-pitch -->
Inverted AI provides an API for controlling non-playable characters (NPCs) in autonomous driving simulations,
available as either a REST API or a Python library built on top of it. Using the API requires an access key -
[contact us](mailto:sales@inverted.ai) to get yours. This page describes how to get started quickly. For more in-depth understanding,
see the [API usage guide](userguide.md), and detailed documentation for the [REST API](apireference.md) and the
[Python library](pythonapi/index.md).
To understand the underlying technology and why it's necessary for autonomous driving simulations, visit the
[Inverted AI website](https://www.inverted.ai/).
<!-- end elevator-pitch -->

![](docs/images/top_camera.gif)

# Get Started
<!-- start quickstart -->
## Installation
For installing the Python package from [PyPI][pypi-link]:

```bash
pip install invertedai
```

The Python client library is [open source](https://github.com/inverted-ai/invertedai),
so you can also download it and build locally.


## Minimal example

Conceptually, the API is used to establish synchronous co-simulation between your own simulator running locally on
your machine and the NPC engine running on Inverted AI servers. The basic integration in Python looks like this.

```{eval-rst}
.. literalinclude:: ../../examples/minimal_example.py
   :language: python
```

To quickly check out how Inverted AI NPCs
behave, try our
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only_colab.ipynb),
where all agents are NPCs, or go to our
[github repository](https://github.com/inverted-ai/invertedai/examples) to execute it locally.
When you're ready to try our NPCs with a real simulator, see the example [CARLA integration](examples/carlasim.md).
The examples are currently only provided in Python, but if you want to use the API from another language,
you can use the [REST API](apireference.md) directly.

<!-- end quickstart -->
