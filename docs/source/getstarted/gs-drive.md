# Driving Agents

**{ref}`DRIVE`** is Inverted AI's cutting-edge realistic driving model trained on millions of miles of traffic data.
This model can drive all the agents with only the current state of the environment, i.e., one step observations (which could be obtained from **{ref}`INITIALIZE`**) or with multiple past observations.
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
>For convenience and to reduce data overhead, ***drive** also returns _recurrent-states_ which can be feedbacked to the model instead of providing all the past observations.\
>Furthermore, **drive** drive all the agents for $steps\times \frac{1}{FPS}$ where by default $FPS=10[frames/sec]$, should you require other time resolutions [contact us](mailto:info@inverted.ai).

---
{ref}`DRIVE`: more information about the python SDK.\
{ref}`REST API`: more information about the REST API and other programming languages.

