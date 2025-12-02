from typing import List, Optional, Tuple
import lanelet2
import random
import numpy as np

from invertedai.common import AgentState, Point

def generate_waypoints_from_lane_ids(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    lane_ids: List[int], 
    waypoint_spacing: float
) -> List[Point]:
    """
    Generates a list of waypoints from a sequence of lane ids.

    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        lane_ids (List[int]): Sequence of lane ids to follow.
        waypoint_spacing (float): Spacing between the waypoints in meters.

    Returns:
        List[Point]: List of waypoints for the agent to follow.
    """
    assert len(lane_ids) >= 1, "Expected the lane_ids to be populated"
    def get_lanelet(id):
        for l in lanelet_map.laneletLayer:
            if l.id == id:
                return l
        return None

    all_centerline_points = []
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    for i, lane_id in enumerate(lane_ids):
        current_lanelet = get_lanelet(lane_id)
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
        all_centerline_points.extend(lane_centerline_points)
    all_centerline_points = np.array([[point.x, point.y] for point in all_centerline_points])
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
    waypoint: Optional[Point] = None
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

    Returns:
        List[int]: Sequence of lane ids to follow.
    """
    
    traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                    lanelet2.traffic_rules.Participants.Vehicle)
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
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=False)
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
        reachable_lanelets = routing_graph.following(current_lanelet, withLaneChanges=False)
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