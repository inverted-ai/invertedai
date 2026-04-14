from typing import (
    List, 
    Optional, 
    Tuple, 
    Union
)
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
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

class WaypointManagerConfig(BaseModel):
    """
    Configuration class for the :class:`iai.WaypointManager` class.'
    Used to initialize WaypointManager with lanelet map from LocationResponse.get_lanelet_map()
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, # allow for non-pydantic types such as lanelet maps to be used in the config
        validate_assignment=True,
    )
    lanelet_map: lanelet2.core.LaneletMapLayers
    waypoint_threshold: float = 10.0 #Distance in meters away from the waypoint to be considered reached
    waypoint_spacing: float = 30.0 #Distance in meters between waypoints along a path to an end goal
    random_seed: int = int(time.time()) #Pseudo-random seed for repeatability
    log_level: Optional[int] = logging.DEBUG #Configure the level of the logger for convenience 
    fail_soft: Optional[bool] = False #If an error is experienced, the manager will continue in a fail soft state instead of raising an Exception

class EndOfMapException(Exception):
    """
    Raised when an agent has reached the end of the map and no further waypoints
    can be generated. The routing graph has no successor lanes from the agent's current position.
    """
    pass

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
    """
    Helper class to manage waypoints statelessly for a simulation.
    """
    
    def __init__(
        self,
        cfg: WaypointManagerConfig
    ):
        if cfg is None:
            self.cfg = WaypointManagerConfig()
        else:
            self.cfg = cfg
        
        self.waypoint_threshold = self.cfg.waypoint_threshold
        self.waypoint_spacing = self.cfg.waypoint_spacing

        self.lanelet_map = self.cfg.lanelet_map
        self.rng = np.random.default_rng(self.cfg.random_seed)

        if self.cfg.log_level is not None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(self.cfg.log_level)
            self.logger.propagate = False

        self._debug_data: Optional[List[List[WaypointManagerLogState]]] = [] if self.cfg.log_level == logging.DEBUG else None

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
        if not num_agents == len(agent_properties): 
            raise ValueError(f"Given number of agent properties does not match given number of agent states.")

        if agents_mask is None:
            agents_mask = [True for _ in range(num_agents)]
        else:
            if not num_agents == len(agents_mask): 
                raise ValueError(f"Given number of agents in agents_mask does not match given number of agent states.")

        if target_paths is None:
            target_paths = [None for _ in range(num_agents)]
        else:
            if not num_agents == len(target_paths): 
                raise ValueError("Given number of paths in target_paths does not match given number of agent states.")

        log_update = []
        _agent_properties = [AgentProperties.deserialize(props.serialize()) for props in agent_properties]
        for i, mask in enumerate(agents_mask):
            props = _agent_properties[i]
            waypoint_flags = None
            state = agent_states[i]

            if mask or target_paths[i] is not None:
                try:
                    waypoint_flags = []
                    if props.waypoints is None:
                        # The current agents waypoints need to be initialized
                        waypoint_flags.append(WaypointUpdateFlags.UNINITIALIZE_WAYPOINTS)
                        props.waypoints = self.generate_waypoints(
                            state=state,
                            target_path = target_paths[i],
                            agent_properties = props
                        )
                        if len(props.waypoints) == 0:
                            raise EndOfMapException(f"No waypoints generated for agent at {state}")

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
                        if len(props.waypoints) == 0:
                            raise EndOfMapException(f"No waypoints generated for agent at {state}")

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
                        if len(props.waypoints) == 0:
                            raise EndOfMapException(f"No waypoints generated for agent at {state}")

                except (EndOfMapException, IndexError) as e:
                    if self.logger is not None:
                        self.logger.debug(msg=str(e))
                    props.waypoints = []

                except ValueError as e:
                    err_msg = str(e)
                    if self.cfg.fail_soft:
                        if self.logger is not None:
                            fail_soft_msg = "(Fail Soft): " + err_msg
                            self.logger.warning(msg=fail_soft_msg)
                        props.waypoints = []
                    else:
                        self.logger.error(msg=err_msg)
                        raise ValueError(err_msg)
                

            if self._debug_data is not None:
                log_update.append(
                    WaypointManagerLogState(
                        agent_state=state,
                        agent_properties=props,
                        flags=waypoint_flags
                    )
                )
                
            _agent_properties[i] = props

        if self._debug_data is not None:
            self._debug_data.append(log_update)
        
        return _agent_properties

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
            wps = generate_waypoints_from_lane_ids(
                start_state=state,
                lanelet_map=self.lanelet_map, 
                destination_waypoint=destination_waypoint,
                waypoint_spacing=self.waypoint_spacing if waypoint_spacing is None else waypoint_spacing,
                logger=self.logger,
                lane_ids=generate_lane_ids_from_lanelet_map(
                    start_state=state, 
                    lanelet_map=self.lanelet_map,
                    destination_waypoint=destination_waypoint,
                    seed=self.rng.integers(low=1, high=1000000000),
                    logger=self.logger,
                    waypoint_spacing=self.waypoint_spacing
                )
            )

            waypoint_list += wps

        return waypoint_list
    
    def is_missed_waypoint(
        self,
        state: AgentState,
        agent_properties: AgentProperties,
    ) -> bool:
        # Two-stage check:
        # 1. Check if agent is facing the waypoint
        # 2. If not, check if high resolution path to waypoint is greater than waypoint spacing
        
        if not agent_properties.waypoints:
            return False
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
            try:
                wps = self.generate_waypoints(
                    state = state,
                    target_path = [wp],
                    agent_properties = props,
                    waypoint_spacing = 1.0
                )
            except (EndOfMapException, ValueError, IndexError) as e:
                if logger is not None: 
                    msg = f"Error encountered when generating waypoints for missed waypoint check. This is likely due to the agent being close to the end of the map. Error details: {str(e)}"
                    logger.log(
                        level=logger.getEffectiveLevel(),
                        msg=msg
                    )
                return False

            dist_sum = 0.0
            for i in range(len(wps)-1):
                dist_sum += self._get_L2_distance(wps[i],wps[i+1])
                if dist_sum > self.waypoint_spacing:
                    return True
            
        return False
    
    def get_debug_data(self):
        return self._debug_data
    
    def check_waypoint_achieved(
        self,
        agent_state: AgentState,
        waypoint: Point
    ) -> bool:
        return self._get_L2_distance(agent_state.center,waypoint) < self.waypoint_threshold

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

def generate_waypoints_from_lane_ids(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    lane_ids: List[int], 
    waypoint_spacing: float = 15.0,
    destination_waypoint: Optional[Point] = None,
    transition_distance: int = 6,
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
        transition_distance (int): Distance over which to perform lane change transitions. Defaults to 6.

    Returns:
        List[Point]: List of waypoints for the agent to follow.
    """

    cannot_find_path_msg = f"Cannot find path for agent with state: {start_state}"
    
    if len(lane_ids) < 1:
        raise ValueError(cannot_find_path_msg)
    
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
                    if logger is not None: 
                        msg=f"The starting position is behind the first lane centerline point even after adjustment. This may lead to unexpected behavior."
                        logger.log(
                            level=logger.getEffectiveLevel(),
                            msg=msg
                        )
            lane_centerline_points.insert(0, Point(x=x, y=y))
        else:
            if prev_lanelet:
                if not current_lanelet in routing_graph.following(prev_lanelet, withLaneChanges=True):
                    msg = f"Current lanelet not in routing graph."
                    raise ValueError(msg)
                if current_lanelet == routing_graph.left(prev_lanelet) or current_lanelet == routing_graph.right(prev_lanelet):
                    lanelets.append([])
        
        lanelets[-1].append(_sample_linestring([np.array([pt.x, pt.y]) for pt in lane_centerline_points], 1)) # sample at 1m interval

    for i, (lanes1, lanes2) in enumerate(zip(lanelets[:-1], lanelets[1:])):
        if not lanes1 or not lanes2: # lane change happened but no points were added... we should skip
            if logger is not None: 
                msg = f"Lane change detected but no centerline points found in one of the lanes. Skipping lane change..."
                logger.log(
                    level=logger.getEffectiveLevel(),
                    msg=msg
                )
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
        if len(lanelets[-1]) == 0:
            del lanelets[-1]
        else:
            dist, idx = _find_min_distance_from_point_to_line(
                np.array([destination_waypoint.x, destination_waypoint.y]),
                [point for point in lanelets[-1][-1]]
            )
            if dist < 5.0:
                del lanelets[-1][-1][idx:]
                lanelets[-1][-1].append(np.array([destination_waypoint.x, destination_waypoint.y]))
            else:
                if logger is not None: 
                    msg = f"Could not find the given waypoint on the last lane within 5 meters, ignoring the given waypoint. Try adjusting the transition distance or waypoint position."
                    logger.log(
                        level=logger.getEffectiveLevel(),
                        msg=msg
                    )
    all_centerline_points = np.array([point for lanes in lanelets for lane in lanes for point in lane])
    if all_centerline_points.shape[0] < 2:
        if logger is not None: 
                msg = f"Could not calculate a path following the given lanes."
                logger.log(
                    level=logger.getEffectiveLevel(),
                    msg=msg
                )
        return []
    deltas = np.diff(all_centerline_points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    total_length = np.sum(seg_lengths)
    cumdist = np.concatenate(([0], np.cumsum(seg_lengths)))
    num_points = int(np.ceil(total_length / waypoint_spacing)) + 1
    if num_points < 2:
        if logger is not None: 
                msg = f"Could not calculate a path with the given waypoint spacing."
                logger.log(
                    level=logger.getEffectiveLevel(),
                    msg=msg
                )
        return []
    new_distances = np.linspace(0, total_length, num_points)

    new_x = np.interp(new_distances, cumdist, all_centerline_points[:, 0])[1:]
    new_y = np.interp(new_distances, cumdist, all_centerline_points[:, 1])[1:]
    waypoints = [Point(x=x, y=y) for x, y in zip(new_x, new_y)]

    return waypoints

def generate_lane_ids_from_lanelet_map(
    start_state: AgentState, 
    lanelet_map: lanelet2.core.LaneletMapLayers, 
    min_distance: Optional[float] = 1000.0, 
    destination_waypoint: Optional[Point] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    waypoint_spacing: Optional[float] = 20.0
) -> List[int]:
    """
    Generates a sequence of lane ids. If given a waypoint, it will generate the shortest possible route between
    current starting state and the specified waypoint. Otherwise, a random route will be generated that is at
    least `min_distance` long in meters unless there are no more lanes to follow.

    Args:
        start_state (AgentState): The starting state of the agent.
        lanelet_map (lanelet2.core.LaneletMapLayers): Projected lanelet map.
        min_distance (Optional[float]): Minimum distance in meters to generate. Ignored if destination_waypoint is specified. Defaults to 1000.
        destination_waypoint (Optional[Point], optional): Desired final waypoint. Defaults to None.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        waypoint_spacing (Optional[float]): Distance threshold in meters to detect when an agent is near the end of the map. Defaults to 30.

    Returns:
        List[int]: Sequence of lane ids to follow. Empty if no routes are possible.
    """
    rng = np.random.default_rng(seed)
    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)
    x, y, yaw = start_state.center.x, start_state.center.y, start_state.orientation
    filtered_lanelets = []
    radius_to_check = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    beta = 5.0 # parameter for lane change probability
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
        msg = f"Warning: Could not find any lanes in the starting position."
        if logger is not None: logger.log(
            level=logger.getEffectiveLevel(),
            msg=msg
        )
        return []
    if len(filtered_lanelets) == 0:
        if logger is not None: 
            msg = f"Warning: Could not find any lanes aligned with the agent's orientation."
            logger.log(
                level=logger.getEffectiveLevel(),
                msg=msg
            )
        return []
    # check if the best-aligned lanelet has no successors and the agent the end of road
    best_lanelet, _ = min(filtered_lanelets, key=lambda x: x[1])
    if not routing_graph.following(best_lanelet, withLaneChanges=True):
        end_point = best_lanelet.centerline[-1]
        dist_to_end = np.sqrt((x - end_point.x)**2 + (y - end_point.y)**2)
        if dist_to_end < (waypoint_spacing if waypoint_spacing is not None else 30.0):
            raise EndOfMapException(
                f"Agent is near the edge of the map with no following lanelets: {start_state}"
            )

    if destination_waypoint is not None:
        ending_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(destination_waypoint.x, destination_waypoint.y), 0)
        possible_routes = []
        for _, ending_lanelet in sorted(ending_lanelets, key=lambda lanelet: lanelet[1].id):
            for starting_lanelet, _ in filtered_lanelets:
                possible_route = routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=True)
                if possible_route:
                    possible_routes.append(possible_route)
        if not possible_routes:
            if logger is not None: 
                msg = f"Warning: Could not find any possible routes between the starting position and the given destination waypoint."
                logger.log(
                    level=logger.getEffectiveLevel(),
                    msg=msg
                )
            return []
        return [lanelet.id for lanelet in rng.choice(possible_routes).shortestPath()]
    else:
        maxRoutingCost = min_distance if min_distance is not None else 1000.0
        for starting_lanelet, _ in sorted(filtered_lanelets, key=lambda lane: lane[1]):
            ending_lanelets = sorted(list(routing_graph.reachableSet(starting_lanelet, maxRoutingCost=maxRoutingCost, allowLaneChanges=True)), key=lambda lanelet: lanelet.id)
            if ending_lanelets:
                break
        if not ending_lanelets:
            if logger is not None:
                logger.log(
                    level=logger.getEffectiveLevel(),
                    msg="Warning: Could not find any possible routes from the starting position."
                )
            return []
        route_lane_ids = [starting_lanelet.id]
        total_distance = 0
        while ending_lanelets and total_distance <= (min_distance if min_distance is not None else 0):
            routes = [(routing_graph.getRoute(starting_lanelet, ending_lanelet, withLaneChanges=True), ending_lanelet) for ending_lanelet in ending_lanelets]
            routes = [(route, ending_lanelet) for route, ending_lanelet in routes if route is not None]
            if not routes:
                if logger is not None:
                    logger.log(
                        level=logger.getEffectiveLevel(),
                        msg="Warning: Could not find any possible routes from the starting position."
                    )
                return []
            p = np.array([1 / np.e ** (beta * _find_max_num_lane_change(routing_graph, route.shortestPath())) for route, _ in routes])
            route, ending_lanelet = rng.choice(routes, p=p/p.sum())
            starting_lanelet = ending_lanelet
            ending_lanelets = sorted(list(routing_graph.reachableSet(starting_lanelet, maxRoutingCost=maxRoutingCost, allowLaneChanges=True)), key=lambda lanelet: lanelet.id)
            total_distance += route.length2d()
            route_lane_ids.extend([lanelet.id for lanelet in list(route.shortestPath())[1:]])
        if not ending_lanelets and total_distance < (waypoint_spacing if waypoint_spacing is not None else 20.0):
            raise EndOfMapException(
                f"Agent has less than one waypoint spacing of road remaining: {start_state}"
            )
        return route_lane_ids

def _find_max_num_lane_change(routing_graph: lanelet2.routing.RoutingGraph, route: lanelet2.routing.LaneletPath):
    lanelets = list(route)
    max_lane_change = 0
    num_lane_change = 0
    for lanelet1, lanelet2 in zip(lanelets[:-1], lanelets[1:]):
        if lanelet2 == routing_graph.left(lanelet1) or lanelet2 == routing_graph.right(lanelet1):
            num_lane_change += 1
        else:
            max_lane_change = max(max_lane_change, num_lane_change)
            num_lane_change = 0
    max_lane_change = max(max_lane_change, num_lane_change)
    return max_lane_change

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
    return (2*t**3 - 3*t**2 + 1) * p0 + (t**3 - 2*t**2 + t) * m0 + (-2*t**3 + 3*t**2) * p1 + (t**3 - t**2) * m1
    
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