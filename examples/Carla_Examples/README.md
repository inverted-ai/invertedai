# InverteAI-Drive CARLA simulator integration

## Quick start

- Make sure Docker is installed and running
  - This can be done by running `docker info` in the terminal
- Run the following command to start the Carla server

```sh
docker compose up
```

    - NOTE: You may need to run the above command with `sudo`

- Create a python virtual environment and install dependencies

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- To run the simulation in the notebook

```sh
.venv/bin/jupyter notebook Carla-Demo.ipynb
```

- To run the simulation script

```sh
python Carla-Demo-Script.py
```

## TODO:

- [x] Change the naming convention of the maps, carla:town03:roundabout
- [x] Handoff the NPCs inside and outside roi to iai-drive and caral traffic-manager
  - [ ] Issue: vehicles stop when handed to carla:
    - [ ] Possible solution: Turn on physics before handing to carla
- [x] Complete and move static data to a JSON file
  - [x] roi_center of each scene
  - [x] Suggested (or demo) Ego spawn location of for each scene
  - [ ] Suggested (or demo) entry points for each scene
- [x] Setup Sphinx and autodoc
  - [x] IAI-Drive
  - [x] Carla Simulator config
- [ ] Finish documentation
  - [ ] Guide
  - [ ] API
  - [ ] SDK
  - [ ] Examples
- [ ] Readthedocs
- [ ] Ego manual driving
- [ ] Scenarios
- [ ] Calculate infractions or get from server
- [ ] Traffic light and signs
  - [ ] Communicate data with Server (simple drive demo)
  - [ ] Get data from and pass to carla (carla drive demo)

## development

- Add the following environment variables to _.evn_ file

```
DEV=1
DEV_URL=http://localhost:8888
```

- Change the port mapping in iai-drive-server

```yaml
ports:
  - "8888:8000"
```
