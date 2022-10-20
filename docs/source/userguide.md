# User guide

Inverted AI API provides a service that controls non-playable characters (NPCs) in driving simulations. The two main
functions are INITIALIZE, called at the beginning of the simulation, and DRIVE, called at each time step. Typically, the
user runs their simulator locally, controlling the actions of the ego vehicle, and querying the API to obtain the
behavior of NPCs. This page describes the high level concepts governing the interaction with the API. Please refer to
specific pages for {ref}`Python SDK`, {ref}`REST API`, {ref}`Getting started`, and {ref}`Examples`.

We follow the continuous space, discrete time approach used in most driving simulators. In the current version, the API
only supports the time step of 100 ms, corresponding to 10 frames per second, and expects to run in a synchronous
fashion. The latency of API calls varies with physical location of the client server and its network configuration,
but generally the API should not be relied upon to provide real-time simulation. For optimal resource utilization,
we recommend that you run multiple simulations in parallel, so that one can execute when another is waiting for the
API reply. The technology underlying the API is based on [ITRA](https://arxiv.org/abs/2104.11212) and was optimized to
handle simulations of up to
20 seconds (200 time steps) contained within an area of roughly 300 meters in diameter. The API backend has been
provisioned to accommodate a large number of agents, where the maximum allowed varies per location.

## Programming language support
The core interface is a {ref}`REST API`, that can be called from any programming language. This is a low-level,
bare-bones access mode that offers maximum flexibility to deploy in any environment.
For convenience, we also provide a {ref}`Python SDK`, freely available on PyPI with minimal dependencies, which
provides an abstraction layer on top of the REST API. In the future we intend to release a similar library in C++ and
potentially other languages.

## Maps and geofencing
The API operates on a pre-defined collection of maps and currently there is no programmatic way to add additional
locations. For each location there is a map, represented internally in the
[Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) format, which specifies
lanelets, traffic lights, and a selection of static traffic signs (along with their relationship to specific lanelets).
Each map comes with a canonical Euclidean coordinate frame in meters, which for OSM files is obtained by applying a
specific UTM projector defined by lat/lon, and everything sent across the API is always specified in terms of this
coordinate frame. To be able to perform co-simulation, you need to have the same map available in your simulator. For
convenience, the map used on our end can be downloaded through {ref}`LOCATION_INFO`.
The maps must be flat (we assume the world is 2D) and generally cover relatively small regions (a few hundred meters).
For each map there is a designated supported area, defined as the interior of a convex polygon represented as a closed
linestring, outside of which the realism of NPCs may significantly deteriorate. It’s valid to query the API outside of
the supported area, but predictions obtained in this way may be unsatisfactory.
The maps for each location are versioned using the standard semantic versioning scheme “major.minor.patch”, starting
from “1.0.0” (or “0.1.0” if location is experimental).
Note that different API keys may allow access to different locations. For a location that a given API key is allowed to
access, LOCATION_INFO provides all the relevant information. Please contact us with requests to include additional
locations.

## Agent types and representations
At the moment the API only supports vehicles, but future releases will also support pedestrians, bicycles, etc.. We
assume that each vehicle is a rigid rectangle with a fixed length and width. The motion of each vehicle is constrained
by the kinematic bicycle model, which further requires specifying the rear axis offset, that is the distance between the
center of the vehicle and its rear axis. Front axis offset is not relevant, because it can not be fit from observational
data, so we omit it. The three static agent attributes are: length, width, and rear offset.
We represent the instantaneous state of each vehicle as four numbers: x and y position, orientation angle, and speed. We
do not consider lateral velocity, vehicle lights, or any other information about vehicle state.
{ref}`DRIVE` predicts the next state for each vehicle, rather than an action that can be executed in the local simulator
and run through its dynamics model. The predicted motion is consistent with the kinematic bicycle model and
accelerations are constrained to a reasonable range observed in real world traffic, but there is no guarantee that the
corresponding motion could be realized through some action given a particular dynamics model in the local simulator. We
recommend teleporting the NPCs to their new positions, since any discrepancies between predicted and realized states for
NPCs may negatively affect the quality of subsequent predictions.

## Traffic lights and other control signals
Static traffic signals form a part of the map description and influence NPC predictions, but they are not exposed in the
interface. Traffic light placement, in particular regarding which traffic light applies to which lanelet, forms a part
of the map as well. Traffic light state changes dynamically and is controlled exclusively by the client when calling the
API. Each traffic light can be green, yellow, or red at any given point. Traffic light IDs are fixed and can be derived
from the map, but for convenience we also provide traffic light IDs and the corresponding locations in LOCATION_INFO.
For maps with traffic lights, the client is responsible for specifying their state on each call to INITIALIZE and DRIVE.
If no state is provided for any particular light, it will be considered absent.

## Handling agents and NPCs
In the API, there is no distinction between agents, controlled by you, and NPCs, controlled by us, so we refer to them
collectively as agents. In any simulation there can be zero or more characters of either kind. When calling DRIVE, the
client needs to list all agents in simulation and we predict the next states for all of them. It is up to the client to
decide which of those agents are NPCs and use the corresponding predictions in the local simulator. However, it is
important to specify all agents when calling the API, since otherwise NPCs will not be able to react to omitted agents.
Due to the recurrent nature of ITRA, we generally recommend that the customer is consistent about this choice throughout
the simulation - predictions for agents whose state is updated differently from ITRA predictions may not be as good as
when ITRA fully controls them.

## Consistent simulation with a stateless API
The API is stateless, so each call to DRIVE requires specifying both the static attributes and the dynamic state of each
agent. However, ITRA is a recurrent model that uses the simulation’s history to make predictions, which we facilitate
through the stateless API by passing around a recurrent state, which is a vector with unspecified semantics from the
client’s perspective. Each call to DRIVE returns a new recurrent state for each agent, which must be passed for this
agent to DRIVE on the subsequent call. Providing an incorrect recurrent state may silently lead to deteriorating
performance, and in order to obtain valid values for the initial recurrent state, the simulation must always start with
INITIALIZE. To initialize the simulation to a specific state, you can provide a sequence of historical states for all
agents that will be used to construct the matching recurrent state. For best performance, at least 10 time steps should
be provided.
To simplify the process of passing the recurrent states around, we provide a stateful [Simulator]() wrapper in the
Python library that handles this internally.

## Entering and exiting simulation
In the simple case there is a fixed number of agents present throughout the entire simulation. However, it is also
possible to dynamically introduce and remove agents, which is typically done when they enter and exit the supported
area. Removing agents is easy, all it takes is removing the information for a given agent from the lists of agent
attributes, agent states, and recurrent states. For convenience, DRIVE returns a boolean vector indicating which agents
are within the supported area after the predicted step.
Introducing agents into a running simulation is more complicated, due to the requirement to construct their recurrent
state. When predictions for the new agents are not going to be consumed, its state can simply be appended to the
relevant lists, with the recurrent state set to zeros. To obtain good predictions for such an agent, another call to
INITIALIZE needs to be made, providing the recent history of all agents, including the new agent. This correctly
initializes the recurrent state and DRIVE can be called from that point on normally. For best performance, each agent
should initially be controlled by the client for at least 10 time steps before being handed off to ITRA as an NPC by
calling INITIALIZE.

## Reproducibility and control over predictions
INITIALIZE and DRIVE optionally accept a random seed, which controls their stochastic behavior. With the same seed and
the same inputs, the outputs will be approximately the same with high accuracy.
Other than for the random seed, there is currently no mechanism to influence the behavior of predicted agents, such as
by directing them to certain exits or setting their speed, but such mechanisms will be included in future releases.

## Validation and debugging
To facilitate development of integration without incurring the costs of API calls, we provide a mock API version that
returns locally computed simple responses in the correct format. This mock API also performs validation of message
formats, including checking lengths of lists and bounds for numeric values, and those checks can also be optionally
performed on the client side before paid API calls. All those features are only available in the Python library and not
in the REST API.
For further debugging and visualization, both INITIALIZE and DRIVE optionally return a rendered birdview image showing
the simulation state after the call to them. This significantly increases the payload size and latency, so it should not
be done in real integrations.
