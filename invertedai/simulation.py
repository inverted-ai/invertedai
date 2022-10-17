from typing import Optional, List

from invertedai.api_resources import initialize, drive


class Simulation:
    """
    Stateful wrapper around Inverted AI API to simplify co-simulation.
    Typically, it's sufficient to call `ego_agent_states` and `step`.
    """
    def __init__(
            self,
            # all arguments to initialize are also given to this constructor
            location: str = "CARLA:Town03:Roundabout",
            agent_count: int = 1,
            # some further configuration options
            monitor_infractions:bool = False,
            ego_agent_mask: Optional[List[bool]] = None,
    ):
        self._location = location
        self._agent_count = agent_count
        response = initialize(location=location, agent_count=agent_count)
        self._agent_attributes = response.agent_sizes
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        # etc.
        self._monitor_infractions = monitor_infractions
        self._infractions = None
        if ego_agent_mask is None:
            self._ego_agent_mask = [False] * self.agent_count
        else:
            self._ego_agent_mask = ego_agent_mask
        self._time_step = 0

    @property
    def location(self):
        return self._location

    @property
    def agent_count(self):
        return self._agent_count

    @property
    def agent_states(self):
        return self._agent_states

    @property
    def ego_agent_mask(self):
        return self._ego_agent_mask

    @ego_agent_mask.setter
    def ego_agent_mask(self, value):
        self.ego_agent_mask = value

    def npc_states(self):
        """
        Returns the predicted states of NPCs (non-ego agents) only in order.
        """
        ego_states = []
        for (i, s) in self._agent_states:
            if not self._ego_agent_mask[i]:
                ego_states.append(s)
        return ego_states

    def step(self, current_ego_agent_states):
        """
        Call the API to advance the simulation by one time step.
        :param current_ego_agent_states:  States of ego agents before the step.
        :return: None - call `npc_states` to retrieve predictions.
        """
        self._update_ego_states(current_ego_agent_states)
        response = drive(
            location=self.location, agent_sizes=self._agent_attributes,
            agent_states=self.agent_states, recurrent_states=self._recurrent_states,
            get_infractions=self._monitor_infractions
        )
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        if self._monitor_infractions:
            self._infractions = response.infraction  # TODO: package infractions in dataclass
        self._time_step += 1

    def _update_ego_states(self, ego_agent_states):
        new_states = []
        ego_idx = 0
        for (i, s) in enumerate(self.agent_states):
            if self.ego_agent_mask[i]:
                new_states.append(ego_agent_states[ego_idx])
                ego_idx += 1
            else:
                new_states.append(self.agent_states[i])
        self._agent_states = new_states
