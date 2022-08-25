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

- [ ] Change the naming convention of the maps, carla:town03:roundabout
- [x] Complete and move static data to a JSON file
  - [x] roi_center of each scene
  - [ ] Ego spawn location of for each scene
- [ ] Calculate infractions
- [ ] Handoff the NPCs inside and outside roi to iai-drive and caral traffic-manager

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
