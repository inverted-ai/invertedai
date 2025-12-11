from typing import List, Optional, Tuple, Callable
import lanelet2
import random
import numpy as np
import logging

from scipy.interpolate import interp1d
from invertedai.common import AgentState, Point

logger = logging.getLogger(__name__)

traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)

def hermite_spline(
    p0: np.ndarray, 
    p1: np.ndarray, 
    m0: np.ndarray, 
    m1: np.ndarray, 
    t: np.ndarray
) -> np.ndarray:
    """
    Computes the Hermite spline interpolation between two points.

    Args:
        p0 (np.ndarray): n-D coordinates of the starting point.
        p1 (np.ndarray): n-D coordinates of the ending point.
        m0 (np.ndarray): n-D Tangent vector at the starting point.
        m1 (np.ndarray): n-D Tangent vector at the ending point.
        t (np.ndarray): Parameter values for interpolation, between 0 and 1.

    Returns:
        np.ndarray: Interpolated points along the Hermite spline.
    """
    t = t[np.newaxis, :]
    p0 = p0[:, np.newaxis]
    p1 = p1[:, np.newaxis]
    m0 = m0[:, np.newaxis]
    m1 = m1[:, np.newaxis]
    return (2*t**3 - 3*t**2 + 1) * p0 + (t**3 - 2*t**2 + t) * m0 + (-2*t**3 + 3*t**2) * p1 + (t**3 - t**2) * m1 + 1e-10

def sample_linestring(
    linestring: List[np.ndarray], 
    spacing: float = 1.0
) -> List[np.ndarray]:
    """
    Sample a linestring at `spacing` intervals.

    Args:
        linestring (List[np.ndarray]): List of points representing the linestring.
        spacing (float, optional): Distance between sampled points. Defaults to 1.

    Returns:
        List[np.ndarray]: List of sampled points as numpy arrays.
    """
    if len(linestring) < 2:
        pt = linestring[0]
        return [pt]
    
    distances = np.sqrt(np.sum(np.diff(linestring, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    if cumulative_distances[-1] < 1e-10:
        return [linestring[0]]
    
    interp_x = interp1d(cumulative_distances, np.array([pt[0] for pt in linestring]), kind='linear')
    interp_y = interp1d(cumulative_distances, np.array([pt[1] for pt in linestring]), kind='linear')
    
    sample_distances = np.arange(0, cumulative_distances[-1], spacing)
    
    sampled_points = []
    for d in sample_distances:
        sampled_points.append(np.array([interp_x(d), interp_y(d)]))
    
    return sampled_points

def find_closest_point_on_line(
    point: np.ndarray, 
    line: List[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """
    Finds the closest point on a line to a given point.

    Args:
        point (np.ndarray): The 2D reference point.
        line (List[np.ndarray]): List of 2D points representing the line.

    Returns:
        Tuple[np.ndarray, int]: The closest point on the line and its index.
    """
    px, py = point[0], point[1]
    arr = np.array(line)
    dx = arr[:, 0] - px
    dy = arr[:, 1] - py
    dist = np.sqrt(dx*dx + dy*dy)
    idx = np.argmin(dist).item()
    return arr[idx], idx

def find_min_distance_from_point_to_line(
    point: np.ndarray, 
    line: List[np.ndarray]
) -> Tuple[float, int]:
    """
    Finds the minimum distance from a point to a line

    Args:
        point (np.ndarray): The 2D reference point.
        line (List[np.ndarray]): List of 2D points representing the line.

    Returns:
        Tuple[float, int]: The minimum distance and the index of the segment on the line.
    """
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

def lane_change_points(
    linestring1: List[np.ndarray], 
    linestring2: List[np.ndarray],
    start_state: np.ndarray, 
    transition_distance: int
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """
    Finds the start and end points to initiate and complete a lane change between two lanes.

    Args:
        linestring1 (List[np.ndarray]): The lane to initiate the lane change from.
        linestring2 (List[np.ndarray]): The lane to complete the lane change to.
        start_state (np.ndarray): The starting state of the agent.
        transition_distance (int): The distance over which to perform the lane change.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int,  np.ndarray, np.ndarray]: The start and end points for the lane change, their indices, and the direction vectors at these points.
    """
    starting_point_on_line1, starting_point_on_line1_idx = find_closest_point_on_line(start_state, linestring1)
    _, starting_point_on_line2_idx = find_closest_point_on_line(starting_point_on_line1, linestring2)
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

def generate_waypoints_from_lane_ids(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    lane_ids: List[int], 
    waypoint_spacing: float = 15.0,
    destination_waypoint: Optional[Point] = None,
    transition_distance: int = 3,
    lane_change_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = hermite_spline,
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
        lane_change_fn (Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], optional): Function to use for lane change interpolation. Defaults to hermite_spline.

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
                    logger.warning("The starting position is behind the first lane centerline point even after adjustment. This may lead to unexpected behavior.")
            lane_centerline_points.insert(0, Point(x=x, y=y))
        if prev_lanelet:
            assert current_lanelet in routing_graph.following(prev_lanelet, withLaneChanges=True)
            if current_lanelet == routing_graph.left(prev_lanelet) or current_lanelet == routing_graph.right(prev_lanelet):
                lanelets.append([])
        
        lanelets[-1].append(sample_linestring([np.array([pt.x, pt.y]) for pt in lane_centerline_points], 1)) # sample at 1m interval

    for i, (lanes1, lanes2) in enumerate(zip(lanelets[:-1], lanelets[1:])):
        lane1_centerline = lanes1[-1]
        lane2_centerline = lanes2[0]
        starting_point_on_line1, ending_point_on_line2, start_idx, end_idx, m0, m1 = lane_change_points(
            linestring1=lane1_centerline, 
            linestring2=lane2_centerline, 
            start_state=lane1_centerline[0], 
            transition_distance=transition_distance,
        )
        t_sample = np.linspace(0, 1, 50)
        points = lane_change_fn(starting_point_on_line1, ending_point_on_line2, m0, m1, t_sample)
        del lane1_centerline[start_idx:]
        del lane2_centerline[:end_idx]
        lane1_centerline.extend([np.array([points[0][t_idx], points[1][t_idx]]) for t_idx in range(t_sample.shape[0])])
    if destination_waypoint:
        dist, idx = find_min_distance_from_point_to_line(
            np.array([destination_waypoint.x, destination_waypoint.y]),
            [point for point in lanelets[-1][-1]]
        )
        if dist < 5.0:
            del lanelets[-1][-1][idx:]
            lanelets[-1][-1].append(np.array([destination_waypoint.x, destination_waypoint.y]))
        else:
            logger.warning("Could not find the given waypoint on the last lane within 5 meters, ignoring the given waypoint. Try adjusting the transition distance or waypoint position.")
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
    min_distance: float = 600.0, 
    destination_waypoint: Optional[Point] = None,
    lane_change: bool = False,
    seed: int = 0,
) -> List[int]:
    """
    Generates a sequence of lane ids. If given a waypoint, it will generate the shortest possible route between
    current starting state and the specified waypoint. Otherwise, a random route will be generated that is at 
    least `min_distance` long in meters unless there are no more lanes to follow.
    
    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        min_distance (float): Minimum distance in meters to generate. Ignored if destination_waypoint is specified. Defaults to 600.
        destination_waypoint (Optional[Point], optional): Desired final waypoint. Defaults to None.
        lane_change (bool): Whether lane changes are supported. Defaults to False.
        seed (int): Random seed for reproducibility. Defaults to 0.

    Returns:
        List[int]: Sequence of lane ids to follow. Empty if no routes are possible.
    """
    random.seed(seed)
    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    starting_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(x, y), 0)
    filtered_lanelets = []
    for _, lanelet in starting_lanelets:
        a, b = find_direction_and_nearest_points(lanelet.centerline, lanelet2.core.BasicPoint3d(x, y, 0))
        lane_orientation = np.arctan2(b.y - a.y, b.x - a.x)
        angle = np.absolute((yaw - lane_orientation + np.pi) % (2 * np.pi) - np.pi)
        if angle < 75 * np.pi / 180:
            filtered_lanelets.append(lanelet)
    if len(filtered_lanelets) == 0:
        return []
    if destination_waypoint is not None:
        ending_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(destination_waypoint.x, destination_waypoint.y), 0)
        possible_routes = []
        for _, ending_lanelet in ending_lanelets:
            for starting_lanelet in filtered_lanelets:
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=lane_change)
                if possible_route:
                    possible_routes.append(possible_route)
        if not possible_routes:
            return []
        return [lanelet.id for lanelet in random.choice(possible_routes).shortestPath()]
    else:
        ending_lanelets = random.sample([lanelet for lanelet in lanelet_map.laneletLayer], len(lanelet_map.laneletLayer))
        for ending_lanelet in ending_lanelets:
            possible_routes = []
            for starting_lanelet in filtered_lanelets:
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=lane_change)
                if possible_route:
                    possible_routes.append(possible_route)
            if not possible_routes:
                continue
            for route in possible_routes:
                if route.length2d() >= min_distance:
                    return [lanelet.id for lanelet in route.shortestPath()]

    return []


def find_direction_and_nearest_points(
    linestring: lanelet2.core.ConstLineString3d, 
    location3d: lanelet2.core.BasicPoint3d
) -> Tuple[lanelet2.core.Point2d, lanelet2.core.Point2d]:
    """
    For a given linestring and a point near it, finds the nearest 2 points in forward direction.

    Args:
        linestring (lanelet2.core.ConstLineString3d): Linestring to check.
        location3d (lanelet2.core.BasicPoint3d): Point to check.

    Raises:
        ValueError: Raised when the method fails, usually because the linestring has a weird shape.

    Returns:
        Tuple[lanelet2.core.Point2d, lanelet2.core.Point2d]: The nearest 2 points in forward direction.
    """
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
    