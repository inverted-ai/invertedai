import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

from typing import List
import numpy as np
import matplotlib.pyplot as plt

iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable


class LocalSimulator:
    """
    Mock up of a local simulator, where you control the ego vehicle. This example only supports single ego vehicle.
    """

    def __init__(self, ego_state: iai.common.AgentState, npc_states: List[iai.common.AgentState]):
        self.ego_state = ego_state
        self.npc_states = npc_states

    def _step_ego(self):
        """
        The simple motion model drives forward with constant speed.
        The ego agent ignores the map and NPCs for simplicity.
        """
        dt = 0.1
        dx = self.ego_state.speed * dt * np.cos(self.ego_state.orientation)
        dy = self.ego_state.speed * dt * np.sin(self.ego_state.orientation)

        self.ego_state = iai.common.AgentState(
            center=iai.common.Point(x=self.ego_state.center.x + dx, y=self.ego_state.center.y + dy),
            orientation=self.ego_state.orientation,
            speed=self.ego_state.speed,
        )

    def step(self, predicted_npc_states):
        self._step_ego()  # ego vehicle moves first so that it doesn't see future NPC movement
        self.npc_states = predicted_npc_states
        return self.ego_state

print("Begin initialization.")
location = "canada:drake_street_and_pacific_blvd"
iai_simulation = iai.BasicCosimulation(  # instantiate a stateful wrapper for Inverted AI API
    location=location,  # select one of available locations
    agent_properties=get_default_agent_properties({AgentType.car:5}),  # how many vehicles in total to use in the simulation
    ego_agent_mask=[True, False, False, False, False],  # first vehicle is ego, rest are NPCs
    get_birdview=False,  # provides simple visualization - don't use in production
    traffic_lights=True,  # gets the traffic light states and used for initialization and steping the simulation
)

location_info_response = iai.location_info(location=location)
rendered_static_map = location_info_response.birdview_image.decode()
scene_plotter = iai.utils.ScenePlotter(
    rendered_static_map,
    location_info_response.map_fov,
    (location_info_response.map_center.x, location_info_response.map_center.y),
    location_info_response.static_actors
)
scene_plotter.initialize_recording(
    agent_states=iai_simulation.agent_states,
    agent_properties=iai_simulation.agent_properties,
)

print("Begin stepping through simulation.")
local_simulation = LocalSimulator(iai_simulation.ego_states[0], iai_simulation.npc_states)
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)
    # query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step([local_simulation.ego_state])
    # collect predictions for the next time step
    predicted_npc_behavior = iai_simulation.npc_states
    # execute predictions in your simulator, using your actions for the ego vehicle
    updated_ego_agent_state = local_simulation.step(predicted_npc_behavior)
    # save the visualization with ScenePlotter
    scene_plotter.record_step(iai_simulation.agent_states,iai_simulation.light_states)

print("Simulation finished, save visualization.")
# save the visualization to disk
fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
gif_name = 'cosimulation_minimal_example.gif'
scene_plotter.animate_scene(
    output_name=gif_name,
    ax=ax,
    direction_vec=False,
    velocity_vec=False,
    plot_frame_number=True
)
print("Done")