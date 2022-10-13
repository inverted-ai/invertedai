# Spawning Agents
To run the simulation, the map must be first populated with agents.
Inverted AI provides the {ref}`INITIALIZE`, a state-of-the-art model trained with real-life driving scenarios which can generate realistic positions for the initial state of the simulation.
<!-- Having realistic, complicated and diverse initial conditions are particularly crucial to observer interesting and informative <!-- interaction  -\->between the agents, i.e., the ego vehicle and NPCs (non-player characters). -->

<!-- You can use **INITIALIZE** in two modes: -->
## Initialize all agents: 
generates initial conditions (position and speed) for all the agents including the ego vehicle
```python
response = iai.initialize(
    location="CARLA:Town03:Roundabout",
    agent_count=10,
)
```
<!-- - _Initialize NPCs_: generates initial conditions (position and speed) only for the NPCs according to the provided state of the ego vehicle. -->
<!-- ```python -->
<!-- response = iai.initialize( -->
<!--     location="CARLA:Town03:Roundabout", -->
<!--     agent_count=10, -->
<!-- ) -->
<!-- ``` -->
> _response_ is a dictionary of _states_, and _agent-attribute_  (_recurrent-states_ is also returned for compatibility with **drive**)\
> _response["states"]_ is a list of agent states, by default the first on the list is always the ego vehicle.

---
{ref}`INITIALIZE`: more information about the python SDK.\
{ref}`REST API`: more information about the REST API and other programming languages.
