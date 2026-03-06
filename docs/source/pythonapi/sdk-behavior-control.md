[waypoint-manager-example-link]: https://github.com/inverted-ai/invertedai/blob/develop/examples/waypoint_example.py

# Agent Behavior Control

The Inverted AI Python SDK provides several features for modifying and controlling the behavior of individual agents. These parameters are located within the AgentProperties data structure associated with each agent.

```{eval-rst}
.. autoclass:: invertedai.common.AgentProperties
   :members:
```

## Aggressiveness

Aggressiveness is a parameter to can modify the behavior of particular agents. As the name suggests, this parameter varies how "aggressive" or "passive" an agent behaves as it navigates through a scene. Examples of aggressive behavior include, but are not limited to, higher general speed and a lower probability to yielding courteously to other agents.

## Waypoint Management

Inverted AI DRIVE API provides an option to direct the NPC behavior by setting their target waypoints. This is in particular useful when instructing the NPCs to follow a specific route, but also helps maintain global coherence of paths taken by free roaming agents. While low level access to target waypoints allows the user finer control over the NPC behavior, "prompting" NPCs with waypoints takes some practice and for convenience we provide a helper that abstracts low-level waypoint generation in the typical use case of having the NPCs follow a specific route. This computation is performed and cached client side, in order to allow the API to be stateless, and encapsulated inside the Waypoint Manager. Check out the [Waypoint Manager example script][waypoint-manager-example-link] for how to integrate this feature into your code. In short, it translates high-level waypoints placed along a user-defined reachable path into low-level waypoints that can be inserted into DRIVE calls with desired effects.

```{eval-rst}
.. autoclass:: invertedai.helpers.waypoints.WaypointManager
   :members:
```

### Assigned Waypoints

This is the easiest and recommended method of using the Waypoint Manager in your own code. However this simplicity is achieved through complexity under-the-hood. 

If this method is selected for an agent, a goal waypoint will be randomly sampled. This point will be reachable to the vehicle in its current state (i.e. there exists at least one sequence of lanes connected sequentially or laterally for which the beginning and end lanes contain the start and end points respectively) and will meet a list of additional criteria of reasonability (e.g. not too close to the agent's current state). 

Once this goal waypoint is sampled, a set of intermediate waypoints are sampled using similar criteria creating a path for the vehicle to execute. Every time step the Waypoint Manager must be updated with the current state of the agent of interest, and will then track if the current waypoint of interest in the list has been reached. If so, the Waypoint Manager then automatically updates the current waypoint of interest to be the next in the list. If this list is exhausted (i.e. the goal waypoint has been reached), the Waypoint Manager will sample a new goal and repeat the process. 

![](../images/waypoint_Town10HD_frame.png)
Simple example of all traffic agents being assigned waypoints and navigating through the map. Vehicles are dark blue rectangles while their current waypoint is the brown circle with corresponding number ID. 

### User-Defined Waypoints

Additionally, the user can provide a list of waypoints per agent that they want executed. This list of waypoints can include, but is not limited to, a specific path of reasonably spaced points or a sparse set of loose goals very far apart within a given map. Typically, this method is used if a user desires to create a specific scenario. For instance an adversarial agent cutting off an ego vehicle but may also include ensuring traffic stays within their respective lanes and reacts to the ego executing a maneuver. 

![](../images/waypoint_Town06_scenario.png)
Example of configuring an aggressive on-ramp merging scenario.

Regardless of the use case, if the user defines a set of sequential waypoints for an agent, a similar process occurs within the Waypoint Manager. Intermediate waypoints are iteratively sampled between the given goal waypoints given a tunable spacing parameter. Once every user-defined waypoint is reached, the agent will be assigned waypoints following the process outlined above.

![](../images/waypoint_Town06_example_0.gif)
Example of executing the previously configured aggressive on-ramp merging scenario.

### No Waypoints

This is the simplest case for an agent. No waypoints are sampled or passed to the DRIVE API calls.

![](../images/waypoint_Town02_frame.png)