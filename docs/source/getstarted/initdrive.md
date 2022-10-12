# Simulation

## INITIALIZE
To run the simulation, the map must be first populated with agents.
Inverted AI provides the **INITIALIZE**, a state-of-the-art model trained with real-life driving scenarios which can generate realistic positions for the initial state of the simulation.\
Having realistic, complicated and diverse initial conditions are particularly crucial to observer interesting and informative interaction between the agents, i.e., the ego vehicle and NPCs (non-player characters).

You can use **INITIALIZE** in two modes:
- _Initialize all agents_: generates initial conditions (position and speed) for all the agents including the ego vehicle
```python
response = iai.initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=10,
)
```
- _Initialize NPCs_: generates initial conditions (position and speed) only for the NPCs according to the provided state of the ego vehicle.
```python
response = iai.initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=10,
)
```
> _response_ is a dictionary of _states_, and _agent-attribute_  (_recurrent-states_ is also returned for compatibility with **drive**)\
> _response["states"]_ is a list of agent states, by default the first on the list is always the ego vehicle.

## DRIVE
**DRIVE** is Inverted AI's cutting-edge realistic driving model trained on millions of miles of traffic data.
This model can drive all the agents with only the current state of the environment, i.e., one step observations (which could be obtained from **INITIALIZE**) or with multiple past observations.
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


