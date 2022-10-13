# Get Started

## Quick Start

Import `Inverted AI` python package and set the apikey:
```python
import invertedai as iai
iai.add_apikey("XXXXXXXXXXXXXX")
```
Initialize the agents:
```python
response = iai.initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=10,
)
```
Drive the agents:
```python
response = iai.drive(
    location="CARLA:Town03:Roundabout",
    agent_attributes=response["attributes"],
    states=response["states"],
    recurrent_states=response["recurrent_states"],
    traffic_states_id=response["traffic_states_id"],
    steps=1,
    get_birdviews=True,
    get_infractions=True,
)
```


## Demo
- **Goolge Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb)

- **Locally**: Download the 
<a href="https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/inverted-ai/invertedai/tree/master/examples" target="_blank">examples directory</a>, navigate to the unzipped directory in terminal and run 

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/jupyter notebook Drive-Demo.ipynb
```



```{toctree}
:maxdepth: 2


installation
initdrive
mapinfo
```


