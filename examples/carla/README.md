# Inverted AI CARLA simulator integration

This folder provides a basic example for using Inverted AI NPCs in CARLA.
[Click here](https://download-directory.github.io/?url=https://github.com/inverted-ai/invertedai/tree/master/examples) to download the example folder as a zip-file.
The entry script is `carla_demo.py`, while `carla_simulator.py` encapsulates
the basic simulation logic for controlling CARLA.

## Quick start

Make sure Docker is installed and running.
This can be done by running `docker info` in the terminal.

Run the following command to start the Carla server.

```sh
docker compose up
```

    - NOTE: You may need to run the above command with `sudo`

Create a python virtual environment and install dependencies.
Requires Python version between `3.6` and `3.8`, inclusive,
otherwise you'll need to install
[CARLA](https://carla.readthedocs.io/en/0.9.13/start_quickstart/) from source.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Run the simulation script.

```sh
python carla_demo.py
```

You'll see an overhead view, in the CARLA server window,
where Inverted AI NPCs are marked with blue dots.
The red dot indicates an ego vehicle and vehicles without dots are NPCs outside
the supported area, controlled by CARLA's traffic manager.

## Key considerations for integrating Inverted AI NPCs

If you're using this example as a starting point to build your own
integration, either with CARLA or with another simulator, here are
the key considerations to keep in mind.

### Executing NPC predictions

`iai.drive` predicts the subsequent state of NPCs, but not all physics
engines allow arbitrarily setting the vehicle state. This is in particular
not possible in CARLA, so instead we resort to teleporting the NPCs.
This is mostly satisfactory, but certain details like wheel and suspension
movement are not faithful, and collision dynamics are generally not realistic.
In principle a custom controller could be used to steer the NPC vehicles
to follow given predictions, but the deviations it introduces would likely
negatively impact subsequent predictions.

### Handling NPCs outside the supported area

`iai.drive` is designed to provide predictions only in a small area, while
the full simulation world can be much larger. As such, there is a question
of how to handle the NPCs as they enter and exit the supported area.
There are two natural solutions, both of which are implemented in this example:
1. Destroy NPCs exiting the supported area, potentially spawning their replacements
at designated entry points.
2. Hand off the control over NPCs to another controller, such as CARLA's
traffic manager, when they exit the supported area, and take it back when they enter.
Note that when teleportation is used within the supported area, this approach typically
results in some discontinuity agent speed when exiting.

### Handling areas with varying elevation

Inverted AI API currently assumes the world is flat and therefore will only perform
well out of the box in (approximately) flat areas. Where significant slopes are present,
custom client-side adjustments would be required. This is particularly important
when using teleportation, where vehicle's elevation and orientation would have to be
adjusted to align it with the road, rather than having it float in the air or
submerge into the road. If you see either of those behaviors, it's likely that you're
using `iai.drive` outside of the supported area.
