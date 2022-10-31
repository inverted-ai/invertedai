# Examples

This folder contains examples demonstrating how to use the Inverted AI API in Python.
To run the examples locally, first build the virtual environment.
```commandline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*If running into package not found issue like`ERROR: No matching distribution found for carla==0.9.13`, try update 
the pip: `pip install upgrade pip `.  

Then, once you obtain an API key, you can run the examples.
```commandline
python npc_only.py --api_key $IAI_API_KEY
```
There are currently three different examples available.

## NPC only

This demonstration script runs without a local simulator and the API is used to control
all vehicles, meaning all vehicles in the simulation are NPCs. To minimize client-side
complexity, the visualizations are provided through the API, which is very inefficient
and only used for demonstration purposes. The purpose of those demonstrations is to
quickly give you an idea for how the underlying NPCs behave. This example is available
in a few different versions, one calling the underlying REST API directly, and others
using the wrapper provided as a part of our library, the latter version also being
available as a Jupyter notebook and
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only_colab.ipynb).

## Minimal example

The minimal example, as shown on the front page, mocks up the local simulator with
a class that implements trivial control logic for the ego vehicle. This example is
meant to provide an illustration of the basic logic for performing co-simulation
using Inverted AI API and give you a sandbox to experiment with it. It does not 
provide visualization, which can be obtained in a fashion similar to NPC-only examples.

## CARLA

Finally, as a realistic example, we provide a basic integration with CARLA.
This example is meant to provide a comprehensive illustration of co-simulation logic
and to be a starting point for creating custom scenarios with Inverted AI NPCs,
both in CARLA and in other simulators. Running CARLA requires additional setup,
which is documented within the corresponding subfolder.
