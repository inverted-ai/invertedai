from invertedai.keyed_agent import AgentData, KeyedAgents, AgentDict
from invertedai.common import AgentType, AgentState, RecurrentState, Point, AgentProperties
from invertedai.api.location import LocationResponse
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from invertedai.api.drive import InfractionIndicators
from invertedai.utils import get_default_agent_properties
# from driving_models.annotations_manager.map_visualization.iai_map.router.maps import highlight_intersection, highlight_close_points
import invertedai as iai
import lanelet2
import numpy as np
from matplotlib import pyplot as plt


class OffroadManager(BaseModel):
    remove_offroad: bool = True
    respawn: bool = False # change these var names in the end !
    removed_agents: AgentDict = {}
    pending_respawns: int = 0
    next_agent_idx: int = 0
    location_info_response: Optional[LocationResponse] = None
    def detect_offroad_agents(
        self,
        agent_ids: List[str],
        infractions: List[InfractionIndicators],
    ) -> List[str]:
        """
        Detect agents that are off the road according to Drive API infractions
        """
        return [agent_ids[i] for i, inf in enumerate(infractions) if inf.offroad]
    
    def remove_offroad_agents(
        self,
        keyed_agents: KeyedAgents,
        infractions: List[InfractionIndicators]
    ) -> List[str]:
        """
        Remove off-raod agents
        Returns list of removed agent IDs
        """
        agent_ids = keyed_agents.get_agent_ids()
        offroad_ids = self.detect_offroad_agents(agent_ids, infractions) # do we really need to return IDs??
        for id in offroad_ids:
            removed = keyed_agents.remove_agent(id)
            print(f"Removed offroad agent: {id}")
            self.removed_agents[id] = removed
            if self.respawn:
                states, _ = spawn_agents_on_map_edges(self.location_info_response.get_lanelet_map(), 1)[0]
                agent_id = self._next_agent_id(keyed_agents)
                self.spawn_agent(agent_id, state=states, keyed_agents=keyed_agents)
                print(f"Spawned agent: {agent_id}")
                # self.pending_respawns -=1
        return offroad_ids
    
    def spawn_agent(
        self,
        agent_id: str,
        state: AgentState,
        keyed_agents: KeyedAgents,
        properties: Optional[AgentProperties] = None,
        overwrite: bool = False,
    ):
        """
        Spawn a brand new agent with a zeroed recurrent state.
        """
        if properties is None:
            properties = get_default_agent_properties({AgentType.car: 1})[0]
            properties.length = 4.5
            properties.width = 2.0
            properties.rear_axis_offset=1.75
        new_agent = AgentData(
            state=state,
            properties=properties,
            recurrent=self._zero_recurrent_state(keyed_agents=keyed_agents),
        )
        keyed_agents.add_agent(agent_id, new_agent, overwrite=overwrite)

    def _zero_recurrent_state(
            self, 
            keyed_agents:KeyedAgents
        ) -> RecurrentState:
        """
        Create a zero recurrent state placeholder
        """
        for data in keyed_agents.agents_dict.values():
            if data.recurrent is not None:
                size = len(data.recurrent.packed)
                return RecurrentState(packed=([0.0] * size))

    def _next_agent_id(self, keyed_agents: KeyedAgents) -> str:
        """Return a new agent id guaranteed to be unique for this manager."""
        if self.next_agent_idx == 0:
            max_idx = -1
            for aid in keyed_agents.agents_dict.keys():
                suffix = self.get_id(aid)
                if suffix is not None and suffix > max_idx:
                    max_idx = suffix
            for aid in self.removed_agents.keys():
                suffix = self.get_id(aid)
                if suffix is not None and suffix > max_idx:
                    max_idx = suffix
            self.next_agent_idx = max_idx + 1 if max_idx >= 0 else len(keyed_agents.agents_dict)
        agent_id = f"agent_{self.next_agent_idx}"
        self.next_agent_idx += 1
        return agent_id

    @staticmethod
    def get_id(agent_id: str) -> Optional[int]:
        try:
            return int(agent_id.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None

def find_edge_lanelets(
    lanelet_map: lanelet2.core.LaneletMapLayers,
    entry_edges: bool = True
) -> List[lanelet2.core.ConstLanelet]:
    """
    Find lanelets that are at the edge of the map.
    """
    from lanelet2.routing import RoutingGraph
    from lanelet2.traffic_rules import create as create_traffic_rules
    
    # Create routing graph
    traffic_rules = create_traffic_rules(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle
    )
    routing_graph = RoutingGraph(lanelet_map, traffic_rules)
    
    edge_lanelets = []
    
    for lanelet in lanelet_map.laneletLayer:
        if entry_edges:
            # entry edges - no previous lanelets
            previous = routing_graph.previous(lanelet, withLaneChanges=False)
            if len(previous) == 0 and not is_undesired_lanelet(lanelet):
                edge_lanelets.append(lanelet) 
    return edge_lanelets

def is_undesired_lanelet(lanelet) -> bool:
    """
    Check lanelet attributes
    """
    attrs = lanelet.attributes
    if "subtype" in attrs: ## check for parking! 
        subtype = str(attrs["subtype"]).lower()
        if any(word in subtype for word in ["intersection", "crosswalk", "crossing"]):
            return True
    if "location" in attrs:
        location = str(attrs["location"]).lower()
        if "intersection" in location:
            return True
    if "turn_direction" in attrs:
        return True
    if "parking" in attrs:
        return True
    if "participant" in attrs:
        participant = str(attrs["participant"]).lower()
        if "intersection" in participant:
            return True
    if "type" in attrs:
        type_val = str(attrs["type"]).lower()
        if any(word in type_val for word in ["intersection", "junction"]):
            return True
    
    return False
# def highlight_intersection(xy_point: BasicPoint2d, lanelet_map: LaneletMap, ax: Axes) -> Axes:
#     intersecting_lanelets = [l[1] for l in lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, xy_point)]
#     if len(intersecting_lanelets) < 1:
#         return ax
#     lanelet_ids = [l.id for l in intersecting_lanelets]
#     polys = []
#     for lanelet_id in lanelet_ids:
#         road_mesh = road_mesh_from_lanelet_map(lanelet_map, lanelets=[lanelet_id]) 
#         corridor_verts, corridor_faces = road_mesh.verts, road_mesh.faces
#         verts_np = corridor_verts.squeeze(0).cpu().numpy()
#         faces_np = corridor_faces.squeeze(0).cpu().numpy()
#         polygons_from_faces = []
#         for face in faces_np:
#             triangle = verts_np[face]
#             polygons_from_faces.append(Polygon(triangle))
#         polygon_from_mesh = unary_union(polygons_from_faces)
#         polys.append(polygon_from_mesh)
#         for idx1 in range(len(polys)): 
#             for idx2 in range(idx1 + 1, len(polys)):
#                 poly1 = polys[idx1]
#                 poly2 = polys[idx2]
#                 if not poly1.is_valid:
#                     poly1 = poly1.buffer(0)
#                 if not poly2.is_valid:
#                     poly2 = poly2.buffer(0)
#                 if not poly1.is_valid or not poly2.is_valid:
#                     continue
#                 intersection = poly1.intersection(poly2)
#                 if intersection.area > 0:
#                     ax.fill(*poly1.exterior.xy, color='cyan', alpha=0.4, label='poly1')
#                     ax.fill(*poly2.exterior.xy, color='yellow', alpha=0.4, label='poly2')
#     return ax

# def highlight_close_points(lanelet_map, open_ends: np.ndarray, open_starts: np.ndarray, ax: Axes, distance_threshold: float = 2.0) -> Axes:
#     distances = cdist(open_starts, open_ends)
#     if np.any(distances < distance_threshold): 
#         start_indices, end_indices = np.where(distances < distance_threshold)
#         for s_idx, e_idx in zip(start_indices, end_indices):
#             start_point =  lanelet2.core.BasicPoint2d(open_starts[s_idx][0], open_starts[s_idx][1])
#             end_point = lanelet2.core.BasicPoint2d(open_ends[e_idx][0], open_ends[e_idx][1])
#             final_lanelets = [l[1] for l in lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, end_point)]
#             initial_lanelets = [l[1] for l in lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, start_point)]
#             end_point_array = np.array([end_point.x, end_point.y])
#             start_point_array = np.array([start_point.x, start_point.y])

#             final_directions = []
#             for final_lanelet in final_lanelets:
#                 final_direction = np.array([final_lanelet.centerline[-1].x - final_lanelet.centerline[-2].x,
#                                         final_lanelet.centerline[-1].y - final_lanelet.centerline[-2].y])
#                 final_direction = final_direction / np.linalg.norm(final_direction)
#                 final_directions.append(final_direction)
            
#             initial_directions = []
#             for initial_lanelet in initial_lanelets:
#                 initial_direction = np.array([initial_lanelet.centerline[1].x - initial_lanelet.centerline[0].x,
#                                             initial_lanelet.centerline[1].y - initial_lanelet.centerline[0].y])
#                 initial_direction = initial_direction / np.linalg.norm(initial_direction)
#                 initial_directions.append(initial_direction)
            
#             for initial_direction in initial_directions:
#                 for final_direction in final_directions:
#                     dot_product = np.dot(final_direction, initial_direction)
#                     same_direction = dot_product > 0.5 
#                     if same_direction:
#                         ax.text(end_point_array[0], end_point_array[1], 'O', color='red', fontsize=8, ha='center', va='center', zorder=11)
#                         ax.text(start_point_array[0], start_point_array[1], 'O', color='red', fontsize=8, ha='center', va='center', zorder=11)
#     return ax

def lanelet_map_open_ends_and_starts(lanelet_map: lanelet2.core.ConstLanelet,float, ax: plt.Axes) -> Tuple[np.ndarray, np.ndarray, plt.Axes]:
    """
    Find open ends and starts of lanelets.
    """
    # lanelet_map = LaneletMapLoadingInfo(
    #         path=osm_path,
    #         origin_latitude=origin_latitude,
    #         origin_longitude=origin_longitude,
    #         ).load()
    traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                    lanelet2.traffic_rules.Participants.Vehicle)
    graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    
    open_ends = []
    open_starts = []
    for l in lanelet_map.laneletLayer:
        if 'parking' in l.attributes:
            continue
        if not graph.following(l):
            last_xy = [l.centerline[-1].x, l.centerline[-1].y]
            open_ends.append(np.array(last_xy))
            
            # Find and highlight intersecting lanelets
            end_point = lanelet2.core.BasicPoint2d(last_xy[0], last_xy[1])
            ax = highlight_intersection(end_point, lanelet_map, ax)
            # Check if the point is valid (not empty/uninitialized)
            if end_point.x != 0 or end_point.y != 0:  # Basic validity check
                ax = highlight_intersection(end_point, lanelet_map, ax)
        if not graph.previous(l):
            first_xy = [l.centerline[0].x, l.centerline[0].y]
            open_starts.append(np.array(first_xy))

            # Find and highlight intersecting lanelets
            start_point = lanelet2.core.BasicPoint2d(first_xy[0], first_xy[1])
            if start_point.x != 0 or start_point.y != 0:  # Basic validity check
                ax = highlight_intersection(start_point, lanelet_map, ax)
    open_ends = np.unique(np.array(open_ends), axis=0)
    open_starts = np.unique(np.array(open_starts), axis=0)
    if open_ends.shape[0] > 0 and open_starts.shape[0] > 0:
        ax = highlight_close_points(lanelet_map,open_ends, open_starts, ax)
    
    return open_ends, open_starts, ax

def spawn_on_edge_lanelet(
    lanelet: lanelet2.core.ConstLanelet,
    inward_offset: float = 10.0, 
    speed: float = 5.0,
    rng: Optional[np.random.Generator] = None
) -> AgentState:
    """
    Spawn an agent at the start of an edge lanelet pointing inward 
    """
    if rng is None:
        rng = np.random.default_rng()
    centerline = lanelet.centerline
    if len(centerline) < 2:
        raise ValueError("Lanelet centerline must have at least 2 points")
    idx = 0
    center_pt = centerline[idx]
    next_pt = centerline[idx + 1]
    tangent = np.array([next_pt.x - center_pt.x, next_pt.y - center_pt.y])
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0])
        else:
            tangent = tangent / tangent_norm
    else:
        tangent = tangent / tangent_norm
    yaw = np.arctan2(tangent[1], tangent[0])
    spawn_xy = np.array([center_pt.x, center_pt.y])
    spawn_xy += inward_offset * tangent  
    agent_state = AgentState(
        center=Point(x=float(spawn_xy[0]), y=float(spawn_xy[1])),
        orientation=float(yaw),
        speed=speed
    )
    return agent_state


def spawn_agents_on_map_edges(
    lanelet_map: lanelet2.core.LaneletMapLayers,
    num_agents: int,
    speed_range: Tuple[float, float] = (3.0, 8.0),
    seed: Optional[int] = None
) -> List[Tuple[AgentState, lanelet2.core.ConstLanelet]]:
    """
    Spawn multiple agents at map entry edges
    Agents spawn at the first point of entry lanelets, pointing into the map
    """
    rng = np.random.default_rng(seed)
    edge_lanelets = find_edge_lanelets(lanelet_map, entry_edges=True)
    if not edge_lanelets:
        raise ValueError("No entry edge lanelets found in the map")
    spawned_agents = []
    for _ in range(num_agents):
        # pick random entry edge lanelet
        lanelet = rng.choice(edge_lanelets)
        speed = rng.uniform(*speed_range)
        state = spawn_on_edge_lanelet(
            lanelet,
            speed=speed,
            rng=rng
        ) 
        spawned_agents.append((state, lanelet)) # can return lanelet for debug purposes
    return spawned_agents