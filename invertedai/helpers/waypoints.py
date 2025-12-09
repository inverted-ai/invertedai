from typing import List, Optional, Tuple, Callable
import lanelet2
import random
import numpy as np

from scipy.interpolate import interp1d
from invertedai.common import AgentState, Point

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
    linestring: List[lanelet2.core.ConstPoint3d], 
    spacing: float = 1.0
) -> List[np.ndarray]:
    """
    Sample a linestring at `spacing` meter intervals.

    Args:
        linestring (List[lanelet2.core.ConstPoint3d]): List of points representing the linestring.
        spacing (float, optional): Distance between sampled points. Defaults to 1.

    Returns:
        List[np.ndarray]: List of sampled points as numpy arrays.
    """
    if len(linestring) < 2:
        pt = linestring[0]
        return [np.array([pt.x, pt.y, pt.z])]
    
    points = np.array([[pt.x, pt.y, pt.z] for pt in linestring])
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    if cumulative_distances[-1] < 1e-10:
        return [points[0]]
    
    interp_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
    interp_y = interp1d(cumulative_distances, points[:, 1], kind='linear')
    interp_z = interp1d(cumulative_distances, points[:, 2], kind='linear')
    
    sample_distances = np.arange(0, cumulative_distances[-1], spacing)
    
    sampled_points = []
    for d in sample_distances:
        sampled_points.append(np.array([interp_x(d), interp_y(d), interp_z(d)]))
    
    return sampled_points

def closest_point_on_line(
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

def find_lane_change_region(
    line1: List[np.ndarray], 
    line2: List[np.ndarray], 
    tolerance: float = 0.2
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Finds overlapping regions between two lines where the points are approximately perpendicular within a given tolerance.

    Args:
        line1 (List[np.ndarray]): List of 2D points representing the first line.
        line2 (List[np.ndarray]): List of 2D points representing the second line.
        tolerance (float): Tolerance for considering points as approximately perpendicular. Defaults to 0.2.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of tuples containing pairs of overlapping points from both lines.
    """
    closest_pair = []
    for p1_idx, p1 in enumerate(line1[:-1]):
        p2, _ = closest_point_on_line(p1, line2)
        diff = p2 - p1
        dir = line1[p1_idx+1] - p1
        angle = np.arccos(np.dot(dir, diff))
        if angle >= np.pi / 2 - tolerance and angle <= np.pi / 2 + tolerance:
            closest_pair.append((p1, p2))
    return closest_pair

def lane_change_points(
    linestring1: List[lanelet2.core.ConstPoint3d], 
    linestring2: List[lanelet2.core.ConstPoint3d], 
    start_state: lanelet2.core.ConstPoint3d, 
    transition_distance: int
):
    if start_state is None:
        start_state = linestring1[0]
    line1 = sample_linestring(linestring1, 1) # do not change spacing without also modifying the ending point indexing
    line2 = sample_linestring(linestring2, 1) # do not change spacing without also modifying the ending point indexing

    pairs = find_lane_change_region(line1, line2)
    starting_point, starting_idx = closest_point_on_line(np.array((start_state.x, start_state.y)), [pair[0] for pair in pairs])
    assert len(pairs) > starting_idx + transition_distance, "Expected transition distance to be greater than the number of samples remaining"
    ending_point = pairs[starting_idx + transition_distance][1]

    m0 = pairs[starting_idx + 1][0] - starting_point
    m1 = ending_point - pairs[starting_idx + transition_distance - 1][1]

    m0 = m0 / (np.linalg.norm(m0) + 1e-10)
    m1 = m1 / (np.linalg.norm(m1) + 1e-10)

    return starting_point, ending_point, m0, m1

def generate_waypoints_from_lane_ids(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    lane_ids: List[int], 
    waypoint_spacing: float = 15.0,
    waypoint: Optional[Point] = None,
    transition_distance: int = 3,
    lane_change_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = hermite_spline,
) -> List[Point]:
    """
    Generates a list of waypoints from a sequence of lane ids. Assume that the start_state is close to the first lane in lane_ids

    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        lane_ids (List[int]): Sequence of lane ids to follow.
        waypoint_spacing (float): Spacing between the waypoints in meters. Defaults to 15.

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
    all_centerline_points = [[]]
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    current_lanelet = None
    for i, current_lane_id in enumerate(lane_ids):
        prev_lanelet = current_lanelet
        current_lanelet = get_lanelet(current_lane_id)
        lane_centerline_points = [point for point in current_lanelet.centerline]
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
                lane_centerline_points = []
            else:
                lane_centerline_points = lane_centerline_points[idx:]
            if len(lane_centerline_points) > 1:
                # check if the second point is already behind the current position
                second_point = lane_centerline_points[1]
                forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
                waypoint_vec = np.array([second_point.x, second_point.y]) - np.array([x, y])
                dot_product = np.dot(forward_vec, waypoint_vec)
                if dot_product < 0:
                    return []
        if prev_lanelet:
            assert current_lanelet in routing_graph.following(prev_lanelet, withLaneChanges=True)
            if current_lanelet in routing_graph.besides(prev_lanelet):
                all_centerline_points.append([])
        all_centerline_points[-1].extend(lane_centerline_points)

    for i, (lane1, lane2) in enumerate(zip(all_centerline_points[:-1], all_centerline_points[1:])):
        start_point, end_point, m0, m1 = lane_change_points(lane1, lane2, lane1[0], transition_distance)
        t_sample = np.linspace(0, 1, 50)
        points = lane_change_fn(start_point, end_point, m0, m1, t_sample)

        _, idx1 = closest_point_on_line(start_point, np.array([[pt.x, pt.y, pt.z] for pt in lane1]))
        _, idx2 = closest_point_on_line(end_point, np.array([[pt.x, pt.y, pt.z] for pt in lane2]))
        del lane1[idx1:]
        del lane2[:idx2]
        lane1.extend([Point(x=points[0][t_idx], y=points[1][t_idx]) for t_idx in range(t_sample.shape[0])])
        
    all_centerline_points = np.array([[point.x, point.y] for lane in all_centerline_points for point in lane])
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
    target_distance: float = 600.0, 
    waypoint: Optional[Point] = None,
    lane_change: bool = False
) -> List[int]:
    """
    Generates a sequence of lane ids. If given a waypoint, it will generate the shortest possible route between
    current starting state and the specified waypoint. Otherwise, a random route will be generated that is at 
    least `target_distance` long in meters unless there are no more lanes to follow.
    
    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        target_distance (float): Target distance in meters to generate. Ignored if waypoint is specified. Defaults to 600.
        waypoint (Optional[Point], optional): Desired final waypoint. Defaults to None.
        lane_change (bool): Whether lane changes are supported. Defaults to False.

    Returns:
        List[int]: Sequence of lane ids to follow. Empty if no routes are possible.
    """
    
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
    if len(filtered_lanelets) > 0:
        current_lanelet = random.choice(filtered_lanelets)
    else:
        return []
    if waypoint is not None:
        ending_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(waypoint.x, waypoint.y), 0)
        possible_routes = []
        for _, ending_lanelet in ending_lanelets:
            for starting_lanelet in filtered_lanelets:
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=lane_change)
                if possible_route:
                    possible_routes.append(possible_route)
        if not possible_routes:
            return []
        return [lanelet.id for lanelet in random.choice(possible_routes).shortestPath()]
    
    total_lane_distance = 0
    path = []
    while total_lane_distance < target_distance:
        lane_centerline_points = [point for point in current_lanelet.centerline]
        if len(lane_centerline_points) < 2:
            continue
        lane_length = lanelet2.geometry.length2d(current_lanelet)
        path.append(current_lanelet.id)
        total_lane_distance += lane_length
        reachable_lanelets = routing_graph.following(current_lanelet, withLaneChanges=lane_change)
        if reachable_lanelets:
            current_lanelet = random.choice(reachable_lanelets)
        else:
            break
    return path

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