[waypoint-manager-example-link]: https://github.com/inverted-ai/invertedai/blob/develop/examples/waypoint_example.py

# Waypoint Management

Inverted AI provides the most sophisticated NPC behavior of any traffic simulation tool currently available in the market for AV/ADAS development. Inverted AI has widened that gap with a line of even more realistic, reactive, and diverse traffic behavioral simulation models based on waypoints. These waypoints can be used to manually direct agents during V&V testing, scenario-based data generation, and so forth or to automatically enhance the diversity of NPC behavior. In an effort to make this feature as accessible as possible, a Waypoint Manager tool is available to handle the details of using waypoints so users can focus on the aspects of developing AV/ADAS simulations that are important to them using the Inverted AI SDK. This tutorial will discuss how the Waypoint Manager tool works to enable rapid improvement of the quality of your AV/ADAS simulations today.

Within the Inverted AI SDK domain, waypoints are simple 2D points on a drivable region of a map that an agent is meant to reach. An agent given a waypoint will progress towards this goal every time step, while maintaining the characteristic realistic, reactive, and diverse behavior of Inverted AI agents. However, the waypoint must be reachable by the agent from its current state (i.e. the waypoint is on a drivable road surface with no gap in drivable lanes between itself and the agent). 

The Waypoint Manager is designed to manage these waypoints under-the-hood while being as unintrusive to your code as possible. The main features of the Waypoint Manager are to format a list of user-defined waypoints to direct the agents through the API or to generate a realistic and diverse path if none is provided. Check out the [Waypoint Manager example script][waypoint-manager-example-link] for how to integrate this feature into your code.

```{eval-rst}
.. autoclass:: invertedai.helpers.waypoints.WaypointManager
   :members:
```

## Assigned Waypoints

This is the easiest and recommended method of using the Waypoint Manager in your own code. However this simplicity is achieved through complexity under-the-hood. 

If this method is selected for an agent, a goal waypoint will be randomly sampled. This point will be reachable to the vehicle in its current state (i.e. there exists at least one sequence of lanes connected sequentially or laterally for which the beginning and end lanes contain the start and end points respectively) and will meet a list of additional criteria of reasonability (e.g. not too close to the agent's current state). 

Once this goal waypoint is sampled, a set of intermediate waypoints are sampled using similar criteria creating a path for the vehicle to execute. Every time step the Waypoint Manager must be updated with the current state of the agent of interest, and will then track if the current waypoint of interest in the list has been reached. If so, the Waypoint Manager then automatically updates the current waypoint of interest to be the next in the list. If this list is exhausted (i.e. the goal waypoint has been reached), the Waypoint Manager will sample a new goal and repeat the process. 

![](../images/waypoint_Town10HD_frame.png)
Simple example of all traffic agents being assigned waypoints and navigating through the map. Vehicles are dark blue rectangles while their current waypoint is the brown circle with corresponding number ID. 

## User-Defined Waypoints

Additionally, the user can provide a list of waypoints per agent that they want executed. This list of waypoints can include, but is not limited to, a specific path of reasonably spaced points or a sparse set of loose goals very far apart within a given map. Typically, this method is used if a user desires to create a specific scenario. For instance an adversarial agent cutting off an ego vehicle but may also include ensuring traffic stays within their respective lanes and reacts to the ego executing a maneuver. 

![](../images/waypoint_Town06_scenario.png)
Example of configuring an aggressive on-ramp merging scenario.

Regardless of the use case, if the user defines a set of sequential waypoints for an agent, a similar process occurs within the Waypoint Manager. Intermediate waypoints are iteratively sampled between the given goal waypoints given a tunable spacing parameter. Once every user-defined waypoint is reached, the agent will be assigned waypoints following the process outlined above.

![](../images/waypoint_Town06_example_0.gif)
Example of executing the previously configured aggressive on-ramp merging scenario.

## No Waypoints

This is the simplest case for an agent. No waypoints are sampled and the vehicle travels realistically, reactively, and diversely as directed by the Inverted AI DRIVE model.

![](../images/waypoint_Town02_frame.png)
