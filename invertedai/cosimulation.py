from typing import List, Optional

from invertedai import drive, initialize
from invertedai.common import AgentState, InfractionIndicators, Image


class BasicCosimulation:
    """
    Stateful wrapper around the Inverted AI API to simplify co-simulation.
    All arguments to :func:`initialize` can be passed to the constructor here
    and a sufficient combination of them must be passed as required by :func:`initialize`.
    This wrapper caches static agent attributes and propagates the recurrent state,
    so that only states of ego agents and NPCs need to be exchanged with it to
    perform co-simulation. Typically, each time step requires a single call to
    :func:`self.npc_states` and a single call to :func:`self.step`.

    This wrapper only supports a minimal co-simulation functionality.
    For more advanced use cases, call :func:`initialize` and :func:`drive` directly.

    :param location: Location name as expected by :func:`initialize`.
    :param ego_agent_mask: List indicating which agent is ego, meaning that it is
        controlled by you externally. The order in this list should be the same as that
        used in arguments to :func:`initialize`.
    :param monitor_infraction: Whether to monitor driving infractions, at a small increase
        in latency and payload size.
    :param render_birdview: Whether to render the bird's eye view of the simulation state
        at each time step. It drastically increases the payload received from Inverted AI
        servers and therefore slows down the simulation - use only for debugging.
    """

    def __init__(
        self,
        location: str,
        ego_agent_mask: Optional[List[bool]] = None,
        monitor_infractions: bool = False,
        render_birdview: bool = False,
        # sufficient arguments to initialize must also be included
        **kwargs,
    ):
        self._location = location
        response = initialize(location=location, **kwargs)
        self._agent_count = len(
            response.agent_attributes
        )  # initialize may produce different agent count
        self._agent_attributes = response.agent_attributes
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        self._monitor_infractions = monitor_infractions
        self._infractions = None
        self._render_birdview = render_birdview
        self._birdview = None
        if ego_agent_mask is None:
            self._ego_agent_mask = [False] * self._agent_count
        else:
            self._ego_agent_mask = ego_agent_mask[: self._agent_count]
        # initialize might not return the exact number of agents requested,
        # in which case we need to adjust the ego agent mask
        if len(self._ego_agent_mask) > self._agent_count:
            self._ego_agent_mask = self._ego_agent_mask[:self._agent_count]
        if len(self._ego_agent_mask) < self._agent_count:
            self._ego_agent_mask += [False] * self._agent_count
        self._time_step = 0

    @property
    def location(self) -> str:
        """
        Location name as recognized by Inverted AI API.
        """
        return self._location

    @property
    def agent_count(self) -> int:
        """
        The total number of agents, both ego and NPCs.
        """
        return self._agent_count

    @property
    def agent_states(self) -> List[AgentState]:
        """
        The predicted states for all agents, including ego.
        """
        return self._agent_states

    @property
    def ego_agent_mask(self) -> List[bool]:
        """
        Lists which agents are ego, which means that you control them externally.
        It can be updated during the simulation, but see caveats in user guide
        regarding the quality of resulting predictions.
        """
        return self._ego_agent_mask

    @ego_agent_mask.setter
    def ego_agent_mask(self, value):
        self.ego_agent_mask = value

    @property
    def ego_states(self):
        return [d for d, s in zip(self._agent_states, self._ego_agent_mask) if s]

    @property
    def infractions(self) -> List[InfractionIndicators]:
        """
        If `monitor_infractions` was set in the constructor,
        lists infractions currently committed by each agent, including ego agents.
        """
        return self._infractions

    @property
    def birdview(self) -> Image:
        """
        If `render_birdview` was set in the constructor,
        this is the image showing the current state of the simulation.
        """
        return self._birdview

    @property
    def npc_states(self) -> List[AgentState]:
        """
        Returns the predicted states of NPCs (non-ego agents) in order.
        The predictions for ego agents are excluded.
        """
        npc_states = []
        for (i, s) in enumerate(self._agent_states):
            if not self._ego_agent_mask[i]:
                npc_states.append(s)
        return npc_states

    def step(self, current_ego_agent_states: List[AgentState]) -> None:
        """
        Calls :func:`drive` to advance the simulation by one time step.
        Current states of ego agents need to be provided to synchronize with
        your local simulator.

        :param current_ego_agent_states:  States of ego agents before the step.
        :return: None - call :func:`self.npc_states` to retrieve predictions.
        """
        self._update_ego_states(current_ego_agent_states)
        response = drive(
            location=self.location,
            agent_attributes=self._agent_attributes,
            agent_states=self.agent_states,
            recurrent_states=self._recurrent_states,
            get_infractions=self._monitor_infractions,
            get_birdview=self._render_birdview,
        )
        self._agent_states = response.agent_states
        self._recurrent_states = response.recurrent_states
        if self._monitor_infractions and (response.infractions is not None):
            # TODO: package infractions in dataclass
            self._infractions = (
                [inf.collisions for inf in response.infractions],
                [inf.offroad for inf in response.infractions],
                [inf.wrong_way for inf in response.infractions],
            )
        if self._render_birdview:
            self._birdview = response.birdview
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
