from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass

import lanelet2
import numpy as np

from invertedai.common import AgentState, AgentProperties, AgentData, RecurrentState, SimulationAgentDict
from invertedai.helpers.waypoints import _find_aligned_lanelets

_traffic_rules = lanelet2.traffic_rules.create(
    lanelet2.traffic_rules.Locations.Germany,
    lanelet2.traffic_rules.Participants.Vehicle
)


class EndOfRoadConfig(BaseModel):
    """
    Configuration for :class:`EndOfRoadHandler`.

    Parameters:
    lanelet_map : lanelet2.core.LaneletMapLayers
        Projected lanelet map used to check for following lanelets.
    end_of_lanelet_spacing : float
        Distance threshold in meters. An agent whose best-aligned lanelet has
        no successors and whose distance to that lanelet's endpoint is less than
        this value is considered to be at the end of the road.
    remove_agent : bool
        If True, end-of-road agents are dropped from the simulation entirely —
        they will not appear in subsequent drive calls, visualization, or logs.
        If False, agents remain in drive calls but waypoint generation stops for them.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    lanelet_map: lanelet2.core.LaneletMapLayers
    end_of_lanelet_spacing: float = 3.0
    remove_agent: bool = True


class EndOfRoadHandler:
    """
    Detects agents approaching the end of the road.

    Detection uses only the lanelet routing graph and agent states.

    SimulationManager calls :func:`update` before calling WaypointManager,
    and passes the resulting IDs as an ``agents_mask`` to WaypointManager
    to stop waypoint generation for end-of-road agents.
    """

    def __init__(self, cfg: EndOfRoadConfig):
        self.cfg = cfg
        self._routing_graph = lanelet2.routing.RoutingGraph(cfg.lanelet_map, _traffic_rules)
        self._end_of_road_ids: set = set()
        self._end_of_road_agents: SimulationAgentDict = {}

    def is_end_of_road(self, state: AgentState) -> bool:
        """
        Returns True if the agent is near the end of a lanelet with
        no following lanelets in the routing graph.

        inspired by 'func:generate_lane_ids_from_lanelet_map' in helpers/waypoints.py
        """
        x, y, yaw = state.center.x, state.center.y, state.orientation
        filtered_lanelets = _find_aligned_lanelets(self.cfg.lanelet_map, x, y, yaw)
        if not filtered_lanelets:
            return False

        best_lanelet, _ = min(filtered_lanelets, key=lambda x: x[1])
        if self._routing_graph.following(best_lanelet, withLaneChanges=False):
            return False

        end_point = best_lanelet.centerline[-1]
        dist_to_end = np.sqrt((x - end_point.x) ** 2 + (y - end_point.y) ** 2)
        return dist_to_end < self.cfg.end_of_lanelet_spacing

    def update(
        self,
        agent_ids: List[str],
        agent_states: List[AgentState],
        properties: List[AgentProperties],
        recurrent_states: List[RecurrentState],
        external_ids: set,
    ) -> None:
        """
        Detect new end-of-road agents and record their state snapshot.

        Must be called with pre waypoint update agent_properties so that
        the recorded snapshot retains the last valid waypoints.
        """
        for i, aid in enumerate(agent_ids):
            if aid in external_ids or aid in self._end_of_road_ids:
                continue
            if self.is_end_of_road(agent_states[i]):
                self._end_of_road_ids.add(aid)
                self._end_of_road_agents[aid] = AgentData(
                    state=agent_states[i],
                    properties=properties[i],
                    recurrent=recurrent_states[i],
                )

    def get_agents_mask(self, agent_ids: List[str]) -> List[bool]:
        """
        Returns a boolean mask for WaypointManager.update() to skip
        waypoint generation for all end-of-road agents.
        """
        return [aid not in self._end_of_road_ids for aid in agent_ids]

    def get_end_of_road_agents(self) -> SimulationAgentDict:
        """
        Returns a SimulationAgentDict of every agent that has reached end-of-road,
        keyed by agent ID, with state/properties/recurrent at time of detection.
        """
        return self._end_of_road_agents

    def remove_agents(self, agent_ids: List[str]) -> None:
        """Clean up handler state when agents are manually removed."""
        for aid in agent_ids:
            self._end_of_road_ids.discard(aid)
            self._end_of_road_agents.pop(aid, None)
