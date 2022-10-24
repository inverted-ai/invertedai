# Inverted AI CARLA simulator integration

This folder provides a basic example for using Inverted AI NPCs in CARLA.
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
