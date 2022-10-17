# Get Started
```{toctree}
:maxdepth: 2


gs-installation
gs-drive
gs-initialize
gs-mapinfo
```


---
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

**Goolge Colab Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb)


