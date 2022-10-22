import math
from typing import List
import invertedai as iai

# iai.add_apikey('')  # specify your key here or through the IAI_API_KEY variable

class LocalSimulator:
    """
    Mock up of a local simulator, where you control the ego vehicle.
    """
    def __init__(self, ego_state: iai.AgentState, npc_states: List[iai.AgentState]):
        self.ego_state = ego_state
        self.npc_states = npc_states

    def _step_ego(self):
        """
        The simple motion model drives forward with constant speed.
        The ego agent ignores the map and NPCs for simplicity.
        """
        dt = 0.1
        dx = self.ego_state.speed * dt * math.cos(self.ego_state.orientation)
        dy = self.ego_state.speed * dt * math.sin(self.ego_state.orientation)

        self.ego_state = iai.AgentState(
            center=iai.Point(x=self.ego_state.center.x + dx, y=self.ego_state.center.y + dy),
            orientation=self.ego_state.orientation,
            speed=self.ego_state.speed,
        )

    def step(self, predicted_npc_states):
        self._step_ego()  # ego vehicle moves first so that it doesn't see future NPC movement
        self.npc_states = predicted_npc_states
        return self.ego_state


iai_simulation = iai.BasicCosimulation(  # instantiate a stateful wrapper for Inverted AI API
    location='canada:vancouver:ubc_roundabout',  # select one of available locations
    agent_count=5,  #  how many vehicles in total to use in the simulation
    ego_agent_mask=[True, False, False, False, False]  # first vehicle is ego, rest are NPCs
)
local_simulation = LocalSimulator(iai_simulation.ego_states[0], iai_simulation.npc_states)
for _ in range(100):  # how many simulation steps to execute (10 steps is 1 second)
    # query the API for subsequent NPC predictions, informing it how the ego vehicle acted
    iai_simulation.step([local_simulation.ego_state])
    # collect predictions for the next time step
    predicted_npc_behavior = iai_simulation.npc_states
    # execute predictions in your simulator, using your actions for the ego vehicle
    updated_ego_agent_state = local_simulation.step(predicted_npc_behavior)
