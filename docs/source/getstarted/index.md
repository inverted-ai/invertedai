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
    location="iai:ubc_roundabout",
    agent_count=10,
)
```
Drive the agents:
```python
response = iai.drive(
    location="iai:ubc_roundabout",
    agent_attributes=agent_attributes,
    agent_states=response.agent_states,
    recurrent_states=response.recurrent_states,
    get_birdviews=True,
    get_infractions=True,
)
```

**Goolge Colab Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inverted-ai/invertedai-drive/blob/develop/examples/Colab-Demo.ipynb)


