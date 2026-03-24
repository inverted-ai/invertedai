# Examples

This folder contains examples demonstrating how to use the Inverted AI API in Python. 
<!-- start exampels -->
[Click here](https://download-directory.github.io/?url=https://github.com/inverted-ai/invertedai/tree/master/examples) to download the folder as a zip-file.
To run the examples locally, first build the virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```
Then, once you obtain an API key, you can run the examples.
```bash
python minimal_example.py --api_key $IAI_API_KEY
```
## Minimal Example

This demonstration script runs without a local simulator and the API is used to control
all vehicles, meaning all vehicles in the simulation are NPCs. To minimize client-side
complexity, the visualizations are provided through the API, which is very inefficient
and only used for demonstration purposes ([click here](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/response_time.ipynb) for response time comparison). The purpose of those demonstrations is to
quickly give you an idea for how the underlying NPCs behave. This example is available
in a few different versions, one calling the underlying REST API directly, and others
using the wrapper provided as a part of our library, the latter version also being
available as a Jupyter notebook and
[Colab](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/npc_only.ipynb).

## Waypoint Example

This demonstration script works similarly to the minimal example, except that it shows 
setting the waypoints to an agent expliclty.

## Cosimulation Minimal Example

The minimal example, as shown on the front page, mocks up the local simulator with
a class that implements trivial control logic for the ego vehicle. This example is
meant to provide an illustration of the basic logic for performing co-simulation
using Inverted AI API and give you a sandbox to experiment with it. It will save 
the generated gif as `iai-example.gif` in the current directory.

## Large Map Example

This example demonstrates another powerful tool in the Inverted AI API sandbox. The 
`large_initialize` and `large_drive` functions are analogous to `initialize` and `drive`
but can handle simulatins cover larger areas and larger number of agents than the default 
maximum. A key difference in concepts between the large and standard versions of `initialize` 
is the addition of "regions" which organizes the specific location and number of agents on which 
to make individual `initialize` API calls.

## Scenario Log Example

This example demonstrates how to use the Inverted AI logging tool. This feature is useful for users
who wish to capture a simulation or scenario that can be replayed and modified at will. This example
file shows 3 key features: capturing a log, replaying an original log, and how to use the log to 
initialize a different simulation.

## CARLA

Please go to the following link to see an example of how the Inverted AI API can integrate with the Carla SDK: [Carla Python SDK Github](https://github.com/carla-simulator/carla/blob/ue5-dev/PythonAPI/examples/invertedai_traffic.py)

<!-- end exampels -->
