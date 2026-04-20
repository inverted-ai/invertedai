from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass

import lanelet2
import numpy as np

from invertedai.common import AgentState, AgentProperties, AgentData, RecurrentState, SimulationAgentDict
from invertedai.helpers.waypoints import _find_direction_and_nearest_points

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
    waypoint_spacing : float
        Distance threshold in meters. An agent whose best-aligned lanelet has
        no successors and whose distance to that lanelet's endpoint is less than
        this value is considered to be at the end of the road.
    remove_agent : bool
        If True, end-of-road agents are dropped from the simulation entirely 
        they will not appear in subsequent drive calls, visualization, or logs.
        If False, agents are frozen at their last known position and
        continue to appear in visualization and logs.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    lanelet_map: lanelet2.core.LaneletMapLayers
    waypoint_spacing: float = 3.0
    remove_agent: bool = True


class EndOfRoadHandler:
    """
    Detects agents approaching the end of the road

    Detection uses only the lanelet routing graph and agent states
    
    SimulationManager calls :func:`update` before
    calling WaypointManager so frozen agents retain their last valid waypoints,
    and passes the resulting IDs as an ``agents_mask`` to
    WaypointManager to stop waypoint generation for those agents.
    """

    def __init__(self, cfg: EndOfRoadConfig):
        self.cfg = cfg
        self._routing_graph = lanelet2.routing.RoutingGraph(cfg.lanelet_map, _traffic_rules)
        self._end_of_road_ids: set = set()
        self._frozen_agents: SimulationAgentDict = {}

    def is_end_of_road(self, state: AgentState) -> bool:
        """
        Returns True if the agent is near the end of a lanelet with
        no following lanelets in the routing graph

        inspired by 'func:generate_lane_ids_from_lanelet_map' in helpers/waypoints.py
        """
        x, y, yaw = state.center.x, state.center.y, state.orientation
        filtered_lanelets = []
        for radius in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
            starting_lanelets = lanelet2.geometry.findWithin2d(
                self.cfg.lanelet_map.laneletLayer,
                lanelet2.core.BasicPoint2d(x, y),
                radius
            )
            for _, ll in sorted(starting_lanelets, key=lambda l: l[1].id):
                a, b = _find_direction_and_nearest_points(
                    ll.centerline,
                    lanelet2.core.BasicPoint3d(x, y, 0)
                )
                lane_orientation = np.arctan2(b.y - a.y, b.x - a.x)
                angle = np.absolute((yaw - lane_orientation + np.pi) % (2 * np.pi) - np.pi)
                if angle < 75 * np.pi / 180:
                    filtered_lanelets.append((ll, angle))
            if filtered_lanelets:
                break

        if not filtered_lanelets:
            return False

        best_lanelet, _ = min(filtered_lanelets, key=lambda x: x[1])
        if self._routing_graph.following(best_lanelet, withLaneChanges=False):
            return False

        end_point = best_lanelet.centerline[-1]
        dist_to_end = np.sqrt((x - end_point.x) ** 2 + (y - end_point.y) ** 2)
        return dist_to_end < self.cfg.waypoint_spacing

    def update(
        self,
        agent_ids: List[str],
        agent_states: List[AgentState],
        properties: List[AgentProperties],
        recurrent_states: List[RecurrentState],
        external_ids: set,
    ) -> None:
        """
        Detect new end of road agents and update internal state

        Must be called with pre waypoint update agent_properties so frozen agents
        retain their last valid waypoints rather than the empty list that
        results from a failed generation attempt.
        """
        print("offroad agents", self._end_of_road_ids)
        for i, aid in enumerate(agent_ids):
            if aid in external_ids or aid in self._end_of_road_ids:
                print(f"Skipping end of road check for agent {aid} since it is already marked as end of road or external")
                continue
            if self.is_end_of_road(agent_states[i]):
                self._end_of_road_ids.add(aid)
                if not self.cfg.remove_agent:
                    self._frozen_agents[aid] = AgentData(
                        state=agent_states[i],
                        properties=properties[i],
                        recurrent=recurrent_states[i],
                    )

    def get_agents_mask(self, agent_ids: List[str]) -> List[bool]:
        """
        Returns a boolean mask for WaypointManager.update() to skips
        waypoint generation for all end of road agents.
        """
        return [aid not in self._end_of_road_ids for aid in agent_ids]

    def get_frozen_agents(self) -> SimulationAgentDict:
        """Returns the frozen agent dict for visualization frames and logs"""
        return self._frozen_agents

    def remove_agents(self, agent_ids: List[str]) -> None:
        """Clean up handler state when agents are manually removed"""
        for aid in agent_ids:
            self._end_of_road_ids.discard(aid)
            self._frozen_agents.pop(aid, None)

