from typing import (
    List, 
    Optional, 
    Tuple, 
    Union
)
from pydantic import BaseModel, field_validator, model_validator
from dataclasses import dataclass
from enum import Enum
from math import sqrt, atan2, pi, hypot

import lanelet2
import numpy as np
import logging
import time

from invertedai.common import AgentState, Point, AgentProperties
from invertedai.api.location import LocationResponse
from invertedai.api.initialize import InitializeResponse
from invertedai.api.drive import DriveResponse

logger = logging.getLogger(__name__)
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)

class WaypointManagerConfig(BaseModel, validate_assignment=True):
    """
    Configuration class for the :class:`iai.WaypointManager` class.
    """
    
    waypoint_threshold: float = 10.0 #Distance in meters away from the waypoint to be considered reached
    waypoint_spacing: float = 30.0 #Distance in meters between waypoints along a path to an end goal
    random_seed: int = int(time.time())
    log_level: Optional[int] = None

class WaypointUpdateFlags(Enum):
    UNINITIALIZE_WAYPOINTS = 0
    WAYPOINT_REACHED = 1
    WAYPOINTS_EMPTY = 2
    MISSED_WAYPOINT = 3

@dataclass 
class WaypointManagerLogState:
    agent_state: AgentState
    agent_properties: AgentProperties
    flags: Optional[List[WaypointUpdateFlags]] = None

class WaypointManager:
    def __init__(
        self,
        location_info_response: LocationResponse,
        cfg: Optional[WaypointManagerConfig] = None
    ):
        if cfg is None:
            self.cfg = WaypointManagerConfig()
        else:
            self.cfg = cfg
        
        self.waypoint_threshold = self.cfg.waypoint_threshold
        self.waypoint_spacing = self.cfg.waypoint_spacing
        self.lanelet_map = location_info_response.get_lanelet_map()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        if self.cfg.log_level is not None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(self.cfg.log_level)
            self.logger.propagate = False

        self.debug_data = List[List[WaypointManagerLogState]] if self.cfg.log_level is not None else None

    def update(
        self,
        response: Union[InitializeResponse,DriveResponse],
        agent_properties: List[AgentProperties],
        target_paths: Optional[List[Optional[List[Point]]]] = None,
        agents_mask: Optional[List[bool]] = None
    ) -> List[AgentProperties]:
        """
        Given the current agent states, output agent properties populated with waypoints.

        Args:
            response (Union[InitializeResponse,DriveResponse]): A response object containing agent states used to calculate waypoints.
            agent_properties (List[AgentProperties]): The list of agent properties in which to check for existing waypoints and add any newly generated 
                waypoints. If the waypoints field of an agents AgentProperties is None, it is assumed this agent needs to be initialized with new
                waypoints unless the agents_mask parameter specifies otherwise. The length must match the number of provided agents.
            target_path (Optional[List[Optional[List[Point]]]]): A set of key target points for agents to achieve. Inner list is the set of key points and
                the outer list is per agent. If None is given to an agent, waypoints will be generated automatically. Once the agent has successfully executed
                the target path, it will no longer attempt to follow that target path. If this parameter is used, its length must match the number of 
                provided agents.
            agents_mask (List[bool]): All indices set to True will have their waypoints updated while indices set to False will be ignored
                and unchanged. If this parameter is used, its length must match the number of provided agents.

        Returns:
            List[AgentProperties]: List of agent properties containing waypoints to execute (unless specified otherwise by the agent mask).
        """
        
        agent_states = response.agent_states
        num_agents = len(agent_states)
        assert num_agents == len(agent_properties), "Given number of agent properties does not match given number of agent states."

        if agents_mask is None:
            agents_mask = [True for _ in range(num_agents)]
        else:
            assert num_agents == len(agents_mask), "Given number of agents in agents_mask does not match given number of agent states."

        if target_paths is None:
            target_paths = [None for _ in range(num_agents)]
        else:
            assert num_agents == len(target_paths), "Given number of paths in agents_mask does not match given number of agent states."

        log_update = []
        _agent_properties = [AgentProperties.deserialize(props.serialize()) for props in agent_properties]
        for i, mask in enumerate(agents_mask):
            props = _agent_properties[i]
            waypoint_flags = None
            state = agent_states[i]

            if mask or target_paths[i] is not None:
                waypoint_flags = []
                if props.waypoints is None:
                    # The current agents waypoints need to be initialized
                    waypoint_flags.append(WaypointUpdateFlags.UNINITIALIZE_WAYPOINTS)
                    props.waypoints = self.generate_waypoints(
                        state=state,
                        target_path = target_paths[i],
                        agent_properties = props
                    )
                
                if len(props.waypoints) > 0:
                    # Most common case, check if current waypoint is achieved
                    if self.check_waypoint_achieved(
                        agent_state = state,
                        waypoint = props.waypoints[0]
                    ):
                        waypoint_flags.append(WaypointUpdateFlags.WAYPOINT_REACHED)
                        props.waypoints.pop(0)

                if len(props.waypoints) == 0:
                    #Agent is done its route, generate a new route
                    #Do not pass the original target path as it should be completed if the list is empty
                   waypoint_flags.append(WaypointUpdateFlags.WAYPOINTS_EMPTY)
                   props.waypoints = self.generate_waypoints(state=state)


                if self.is_missed_waypoint(
                    state = state,
                    agent_properties = props
                ):
                    #If the current waypoint has been missed, reroute
                    waypoint_flags.append(WaypointUpdateFlags.MISSED_WAYPOINT)
                    props.waypoints = self.generate_waypoints(
                        state=state,
                        target_path = target_paths[i],
                        agent_properties = props
                    )

            if self.debug_data is not None:
                log_update.append(
                    WaypointManagerLogState(
                        agent_state=state,
                        agent_properties=props,
                        flags=waypoint_flags
                    )
                )
                
            _agent_properties[i] = props

        if self.debug_data is not None:
            self.debug_data.append(log_update)
        
        return _agent_properties

    def is_missed_waypoint(
        self,
        state: AgentState,
        agent_properties: AgentProperties,
    ) -> bool:
        # Two-stage check:
        # 1. Check if agent is facing the waypoint
        # 2. If not, check if high resolution path to waypoint is greater than waypoint spacing
        
        wp = agent_properties.waypoints[0]
        ap = state.center

        if self._is_vehicle_pointing_away(
            pt1=ap,
            pt2=wp,
            psi=state.orientation,
            threshold=pi/2
        ):
            props = AgentProperties.deserialize(agent_properties.serialize())
            props.waypoints = None
            wps = self.generate_waypoints(
                state = state,
                target_path = [wp],
                agent_properties = props,
                waypoint_spacing = 1.0
            )
            
            dist_sum = 0.0
            for i in range(len(wps)-1):
                dist_sum += self._get_L2_distance(wps[i],wps[i+1])
                if dist_sum > self.waypoint_spacing:
                    return True
            
        return False

    def _get_L2_distance(
        self,
        pt1: Point,
        pt2: Point
    ) -> float:
        return sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2)
    
    def _is_vehicle_pointing_away(
        self,
        pt1: Point,
        pt2: Point,
        psi: float,
        threshold: float
    ) -> bool:
        return abs((psi - atan2(pt2.y-pt1.y,pt2.x-pt1.x) + pi)%(2*pi) - pi) > threshold
    
    def generate_waypoints(
        self,
        state: AgentState,
        waypoint_spacing: Optional[float] = None,
        target_path: Optional[List[Point]] = None,
        agent_properties: Optional[AgentProperties] = None
    ) -> List[Point]:        
        ROUNDING_ERROR = 0.001
        waypoint_list = []
        default_target_path = [None]
        
        if target_path is None:
            target_path = default_target_path
        else:
            wps = agent_properties.waypoints
            if wps is None:
                # This state occurs when an agent with a defined target path needs its waypoints initialized
                pass
            else:
                target_index = None
                for i, target in enumerate(target_path):
                    for wp in wps:
                        if abs(target.x-wp.x) < ROUNDING_ERROR and abs(target.y-wp.y) < ROUNDING_ERROR:
                            target_index = i
                            break
                    if target_index is not None: break
                
                if target_index is not None:
                    #Assume this indicates that the target path has already been achieved
                    target_path = target_path[target_index:]
                else:
                    target_path = default_target_path
                
        for destination_waypoint in target_path:
            waypoint_list += generate_waypoints_from_lane_ids(
                start_state=state,
                lanelet_map=self.lanelet_map, 
                destination_waypoint=destination_waypoint,
                waypoint_spacing=self.waypoint_spacing if waypoint_spacing is None else waypoint_spacing,
                lane_ids=generate_lane_ids_from_lanelet_map(
                    start_state=state, 
                    lanelet_map=self.lanelet_map,
                    destination_waypoint=destination_waypoint,
                    seed=self.rng.integers(low=1, high=1000000000)
                )
            )

        return waypoint_list
    
    def check_waypoint_achieved(
        self,
        agent_state: AgentState,
        waypoint: Point
    ) -> bool:
        return self._get_L2_distance(agent_state.center,waypoint) < self.waypoint_threshold

def generate_waypoints_from_lane_ids(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    lane_ids: List[int], 
    waypoint_spacing: float = 15.0,
    destination_waypoint: Optional[Point] = None,
    transition_distance: int = 3,
    logger: Optional[logging.Logger] = None
) -> List[Point]:
    """
    Generates a list of waypoints from a sequence of lane ids. The start state should be within the first lane.

    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        lane_ids (List[int]): Sequence of lane ids to follow.
        waypoint_spacing (float): Spacing between the waypoints in meters. Defaults to 15.
        destination_waypoint (Optional[Point], optional): Desired final waypoint. Defaults to None.
        transition_distance (int): Distance over which to perform lane change transitions. Defaults to 3.

    Returns:
        List[Point]: List of waypoints for the agent to follow.
    """
    assert len(lane_ids) >= 1, "Expected the lane_ids to be populated"
    def get_lanelet(id):
        for l in lanelet_map.laneletLayer:
            if l.id == id:
                return l
        return None

    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    lanelets = [[]]
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    current_lanelet = None
    for i, current_lane_id in enumerate(lane_ids):
        prev_lanelet = current_lanelet
        current_lanelet = get_lanelet(current_lane_id)
        lane_centerline_points = [Point(x=point.x, y=point.y) for point in current_lanelet.centerline]
        if i == 0:
            distances = [(p.x-x)**2 + (p.y-y)**2 for p in lane_centerline_points]
            idx = distances.index(min(distances))
            # check if waypoints[idx] is in front of the given position and orientation
            forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
            waypoint_vec = np.array([lane_centerline_points[idx].x, lane_centerline_points[idx].y]) - np.array([x, y])
            dot_product = np.dot(forward_vec, waypoint_vec)
            if dot_product < 0:
                if idx < len(lane_centerline_points) - 1:
                    idx += 1
                else:
                    idx = -1
            if idx == -1:
                continue
            else:
                lane_centerline_points = lane_centerline_points[idx:]
            if len(lane_centerline_points) > 1:
                # check if the second point is already behind the current position
                second_point = lane_centerline_points[1]
                forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
                waypoint_vec = np.array([second_point.x, second_point.y]) - np.array([x, y])
                dot_product = np.dot(forward_vec, waypoint_vec)
                if dot_product < 0:
                    if logger is not None: logger.log("The starting position is behind the first lane centerline point even after adjustment. This may lead to unexpected behavior.")
            lane_centerline_points.insert(0, Point(x=x, y=y))
        else:
            if prev_lanelet:
                assert current_lanelet in routing_graph.following(prev_lanelet, withLaneChanges=True)
                if current_lanelet == routing_graph.left(prev_lanelet) or current_lanelet == routing_graph.right(prev_lanelet):
                    lanelets.append([])
        
        lanelets[-1].append(_sample_linestring([np.array([pt.x, pt.y]) for pt in lane_centerline_points], 1)) # sample at 1m interval

    for i, (lanes1, lanes2) in enumerate(zip(lanelets[:-1], lanelets[1:])):
        if not lanes1 or not lanes2: # lane change happened but no points were added... we should skip
            if logger is not None: logger.log("Lane change detected but no centerline points found in one of the lanes. Skipping lane change...")
            continue
        lane1_centerline = lanes1[-1]
        lane2_centerline = lanes2[0]
        starting_point_on_line1, ending_point_on_line2, start_idx, end_idx, m0, m1 = _lane_change_points(
            linestring1=lane1_centerline, 
            linestring2=lane2_centerline, 
            start_state=lane1_centerline[0], 
            transition_distance=transition_distance,
        )
        t_sample = np.linspace(0, 1, 50)
        points = _hermite_spline(starting_point_on_line1, ending_point_on_line2, m0, m1, t_sample)
        del lane1_centerline[start_idx:]
        del lane2_centerline[:end_idx]
        lane1_centerline.extend([np.array([points[0][t_idx], points[1][t_idx]]) for t_idx in range(t_sample.shape[0])])
    if destination_waypoint:
        dist, idx = _find_min_distance_from_point_to_line(
            np.array([destination_waypoint.x, destination_waypoint.y]),
            [point for point in lanelets[-1][-1]]
        )
        if dist < 5.0:
            del lanelets[-1][-1][idx:]
            lanelets[-1][-1].append(np.array([destination_waypoint.x, destination_waypoint.y]))
        else:
            if logger is not None: logger.log("Could not find the given waypoint on the last lane within 5 meters, ignoring the given waypoint. Try adjusting the transition distance or waypoint position.")
    all_centerline_points = np.array([point for lanes in lanelets for lane in lanes for point in lane])
    deltas = np.diff(all_centerline_points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    total_length = np.sum(seg_lengths)
    cumdist = np.concatenate(([0], np.cumsum(seg_lengths)))
    num_points = int(np.ceil(total_length / waypoint_spacing)) + 1
    if num_points < 2:
        return []
    new_distances = np.linspace(0, total_length, num_points)

    new_x = np.interp(new_distances, cumdist, all_centerline_points[:, 0])[1:]
    new_y = np.interp(new_distances, cumdist, all_centerline_points[:, 1])[1:]
    waypoints = [Point(x=x, y=y) for x, y in zip(new_x, new_y)]

    return waypoints

def generate_lane_ids_from_lanelet_map(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    min_distance: float = 1000.0, 
    destination_waypoint: Optional[Point] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> List[int]:
    """
    Generates a sequence of lane ids. If given a waypoint, it will generate the shortest possible route between
    current starting state and the specified waypoint. Otherwise, a random route will be generated that is at 
    least `min_distance` long in meters unless there are no more lanes to follow.
    
    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        min_distance (float): Minimum distance in meters to generate. Ignored if destination_waypoint is specified. Defaults to 1000.
        destination_waypoint (Optional[Point], optional): Desired final waypoint. Defaults to None.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        List[int]: Sequence of lane ids to follow. Empty if no routes are possible.
    """
    rng = np.random.default_rng(seed)
    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    filtered_lanelets = []
    radius_to_check = [0.1, 0.5, 1.0, 2.0, 5.0]
    for radius in radius_to_check:
        starting_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(x, y), radius)
        for _, lanelet in sorted(starting_lanelets, key=lambda lanelet: lanelet[1].id): # laneletLayer is backed by an unordered_map, so we sort by id to have deterministic behavior
            a, b = _find_direction_and_nearest_points(lanelet.centerline, lanelet2.core.BasicPoint3d(x, y, 0))
            lane_orientation = np.arctan2(b.y - a.y, b.x - a.x)
            angle = np.absolute((yaw - lane_orientation + np.pi) % (2 * np.pi) - np.pi)
            if angle < 75 * np.pi / 180:
                filtered_lanelets.append((lanelet, angle))
        if len(filtered_lanelets) > 0:
            break
    if len(starting_lanelets) == 0:
        if logger is not None: logger.log("Could not find any lanes in the starting position.")
        return []
    if len(filtered_lanelets) == 0:
        if logger is not None: logger.log("Could not find any lanes in the starting position that are aligned with the agent's orientation.")
        return []
    if destination_waypoint is not None:
        ending_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(destination_waypoint.x, destination_waypoint.y), 0)
        possible_routes = []
        for _, ending_lanelet in sorted(ending_lanelets, key=lambda lanelet: lanelet[1].id):
            for starting_lanelet, _ in filtered_lanelets:
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=True)
                if possible_route:
                    possible_routes.append(possible_route)
        if not possible_routes:
            if logger is not None: logger.log("Could not find any possible routes between the starting position and the given destination waypoint.")
            return []
        return [lanelet.id for lanelet in rng.choice(possible_routes).shortestPath()]
    else:
        best_starting_lanelet, _ = min(filtered_lanelets, key=lambda lane: lane[1])
        route = [best_starting_lanelet.id]
        total_distance = _lanelet_length(best_starting_lanelet, start_state.center)
        current = best_starting_lanelet
        p_straight = 0.8
        while total_distance < (min_distance) if min_distance is not None else 0.0:
            next_lanelet = None
            straight_candidates = [
                ll for ll in routing_graph.following(current, withLaneChanges=False)
            ]
            lane_change_candidates = []
            for adj in [routing_graph.left(current), routing_graph.right(current)]:
                if adj is not None:
                    lane_change_candidates.append(adj)
            choose_straight = (
                rng.random() < p_straight and straight_candidates
            ) or not lane_change_candidates or len(route) == 1
            if choose_straight and straight_candidates:
                next_lanelet = rng.choice(straight_candidates)
                total_distance += _lanelet_length(next_lanelet)
            elif lane_change_candidates:
                next_lanelet = rng.choice(lane_change_candidates)
                total_distance = total_distance - _lanelet_length(current) + _lanelet_length(next_lanelet)
            else:
                break
            route.append(next_lanelet.id)
            current = next_lanelet
        return route

    
def _find_direction_and_nearest_points(
    linestring: lanelet2.core.ConstLineString3d, 
    location3d: lanelet2.core.BasicPoint3d
) -> Tuple[lanelet2.core.Point2d, lanelet2.core.Point2d]:
    projected_reference = lanelet2.geometry.project(linestring, location3d)
    first, second = float("inf"), float("inf")
    closest_point_idx, second_closest_point_idx = 0, 0

    for i, point in enumerate(linestring):
        point_dist = lanelet2.geometry.distance(projected_reference, point)
        if point_dist < first:
            second = first
            first = point_dist
            second_closest_point_idx = closest_point_idx
            closest_point_idx = i
        elif point_dist < second:
            second = point_dist
            second_closest_point_idx = i

    if not abs(closest_point_idx - second_closest_point_idx) == 1:
        raise ValueError('Failed to find direction of the linestring at a given point')

    if closest_point_idx > second_closest_point_idx:
        point_a, point_b = linestring[second_closest_point_idx], linestring[closest_point_idx]
    else:
        point_b, point_a = linestring[second_closest_point_idx], linestring[closest_point_idx]

    return point_a, point_b

def _hermite_spline(
    p0: np.ndarray, 
    p1: np.ndarray, 
    m0: np.ndarray, 
    m1: np.ndarray, 
    t: np.ndarray
) -> np.ndarray:
    t = t[np.newaxis, :]
    p0 = p0[:, np.newaxis]
    p1 = p1[:, np.newaxis]
    m0 = m0[:, np.newaxis]
    m1 = m1[:, np.newaxis]
    return (2*t**3 - 3*t**2 + 1) * p0 + (t**3 - 2*t**2 + t) * m0 + (-2*t**3 + 3*t**2) * p1 + (t**3 - t**2) * m1 + 1e-10

def _lanelet_length(ll: lanelet2.core.ConstLanelet, starting_pos: Optional[Point] = None) -> float:
    cl = ll.centerline
    if starting_pos is None:
        return sum(
            hypot(cl[i].x - cl[i - 1].x, cl[i].y - cl[i - 1].y)
            for i in range(1, len(cl))
        )
    else:
        min_dist = float('inf')
        closest_segment_idx = 0
        for i in range(1, len(cl)):
            p1 = cl[i - 1]
            p2 = cl[i]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            px = starting_pos.x - p1.x
            py = starting_pos.y - p1.y
        
            segment_length_sq = dx * dx + dy * dy
            if segment_length_sq == 0:
                proj_point = p1
            else:
                t = max(0, min(1, (px * dx + py * dy) / segment_length_sq))
                proj_point = Point(x=p1.x + t * dx, y=p1.y + t * dy)
            dist = hypot(proj_point.x - starting_pos.x, proj_point.y - starting_pos.y)
            
            if dist < min_dist:
                min_dist = dist
                closest_segment_idx = i
                closest_point_on_segment = proj_point

        total_length = 0.0
        total_length += hypot(
            cl[closest_segment_idx].x - closest_point_on_segment.x,
            cl[closest_segment_idx].y - closest_point_on_segment.y
        )
        for i in range(closest_segment_idx + 1, len(cl)):
            total_length += hypot(cl[i].x - cl[i - 1].x, cl[i].y - cl[i - 1].y)
        
        return total_length
    
def _sample_linestring(
    linestring: List[np.ndarray], 
    spacing: float = 1.0
) -> List[np.ndarray]:
    if len(linestring) < 2:
        pt = linestring[0]
        return [pt]
    
    distances = np.sqrt(np.sum(np.diff(linestring, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    if cumulative_distances[-1] < 1e-10:
        return [linestring[0]]
    
    sample_distances = np.arange(0, cumulative_distances[-1], spacing)

    xs = np.interp(sample_distances, cumulative_distances, np.array([pt[0] for pt in linestring]))
    ys = np.interp(sample_distances, cumulative_distances, np.array([pt[1] for pt in linestring]))

    sampled_points = [np.array([x, y]) for x, y in zip(xs, ys)]
    
    return sampled_points


def _find_closest_point_on_line(
    point: np.ndarray, 
    line: List[np.ndarray]
) -> Tuple[np.ndarray, int]:
    px, py = point[0], point[1]
    arr = np.array(line)
    dx = arr[:, 0] - px
    dy = arr[:, 1] - py
    dist = np.sqrt(dx*dx + dy*dy)
    idx = np.argmin(dist).item()
    return arr[idx], idx

def _find_min_distance_from_point_to_line(
    point: np.ndarray, 
    line: List[np.ndarray]
) -> Tuple[float, int]:
    distances = []
    for i, (p1, p2) in enumerate(zip(line[:-1], line[1:])):
        line_vec = p2 - p1
        point_vec = point - p1
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            continue
        t = np.dot(point_vec, line_vec) / line_len
        t = max(0, min(1, t))
        projection = p1 + t * line_vec
        dist = np.linalg.norm(point - projection).item()
        distances.append((dist, i))
    if distances:
        return min(distances, key=lambda x: x[0])
    return float('inf'), -1

def _lane_change_points(
    linestring1: List[np.ndarray], 
    linestring2: List[np.ndarray],
    start_state: np.ndarray, 
    transition_distance: int
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    starting_point_on_line1, starting_point_on_line1_idx = _find_closest_point_on_line(start_state, linestring1)
    _, starting_point_on_line2_idx = _find_closest_point_on_line(starting_point_on_line1, linestring2)
    ending_point_on_line2_idx = starting_point_on_line2_idx + transition_distance if starting_point_on_line2_idx + transition_distance < len(linestring2) else len(linestring2) - 1
    ending_point_on_line2 = linestring2[ending_point_on_line2_idx]

    if len(linestring1) <= starting_point_on_line1_idx + 1:
        m0 = starting_point_on_line1 - linestring1[starting_point_on_line1_idx - 1]
    else:
        m0 = linestring1[starting_point_on_line1_idx + 1] - starting_point_on_line1
    m1 = ending_point_on_line2 - linestring2[ending_point_on_line2_idx - 1]

    m0 = m0 / (np.linalg.norm(m0) + 1e-10)
    m1 = m1 / (np.linalg.norm(m1) + 1e-10)

    return starting_point_on_line1, ending_point_on_line2, starting_point_on_line1_idx, ending_point_on_line2_idx, m0, m1