import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import AgentType

import matplotlib.pyplot as plt
import os
import time
import json
import math
import argparse
from typing import List, Tuple, Dict
import random

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run waypoint example simulation with configurable location and coordinates'
    )
    parser.add_argument(
        '--town',
        type=str,
        required=True,
        help='Town name (e.g., Town10HD, Town03)'
    )
    parser.add_argument(
        '--x',
        type=float,
        default=None,
        help='Location x coordinate (defaults to map_center.x if not provided)'
    )
    parser.add_argument(
        '--y',
        type=float,
        default=None,
        help='Location y coordinate (defaults to map_center.y if not provided)'
    )
    parser.add_argument(
        '--simulation-length',
        type=int,
        default=80,
        help='Simulation length in steps (default: 60)'
    )
    parser.add_argument(
        '--num-agents',
        type=int,
        default=10,
        help='Number of agents (default: 10)'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=250.0,
        help='Field of view for rendering (default: 500.0)'
    )
    parser.add_argument(
        '--drive-model',
        type=str,
        default='nBu1',
        help='Drive model to use (default: nBu1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (defaults to current timestamp if not provided)'
    )
    
    return parser.parse_args()


def line_segment_intersects_rectangle(
    p1_x: float, p1_y: float,
    p2_x: float, p2_y: float,
    rect_center_x: float, rect_center_y: float,
    rect_length: float, rect_width: float,
    rect_orientation: float
) -> bool:
    """
    Check if a line segment intersects with a rotated rectangle (agent bounding box).
    
    Args:
        p1_x, p1_y: Start point of line segment
        p2_x, p2_y: End point of line segment
        rect_center_x, rect_center_y: Center of rectangle
        rect_length: Length of rectangle (along orientation)
        rect_width: Width of rectangle (perpendicular to orientation)
        rect_orientation: Orientation of rectangle in radians
    
    Returns:
        True if line segment intersects the rectangle, False otherwise
    """
    # Transform line segment to rectangle's local coordinate system
    cos_psi = math.cos(-rect_orientation)
    sin_psi = math.sin(-rect_orientation)
    
    # Translate to rectangle center
    p1_local_x = (p1_x - rect_center_x) * cos_psi - (p1_y - rect_center_y) * sin_psi
    p1_local_y = (p1_x - rect_center_x) * sin_psi + (p1_y - rect_center_y) * cos_psi
    p2_local_x = (p2_x - rect_center_x) * cos_psi - (p2_y - rect_center_y) * sin_psi
    p2_local_y = (p2_x - rect_center_x) * sin_psi + (p2_y - rect_center_y) * cos_psi
    
    # Rectangle bounds in local coordinates (axis-aligned)
    half_length = rect_length / 2.0
    half_width = rect_width / 2.0
    
    # Check if line segment intersects axis-aligned rectangle using Liang-Barsky algorithm
    dx = p2_local_x - p1_local_x
    dy = p2_local_y - p1_local_y
    
    # If line segment is a point
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return (-half_length <= p1_local_x <= half_length and 
                -half_width <= p1_local_y <= half_width)
    
    # Liang-Barsky line clipping algorithm
    t0, t1 = 0.0, 1.0
    
    for edge in range(4):
        if edge == 0:  # Left edge
            p, q = -dx, p1_local_x + half_length
        elif edge == 1:  # Right edge
            p, q = dx, half_length - p1_local_x
        elif edge == 2:  # Bottom edge
            p, q = -dy, p1_local_y + half_width
        else:  # Top edge
            p, q = dy, half_width - p1_local_y
        
        if abs(p) < 1e-9:  # Line is parallel to this edge
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                elif r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                elif r < t1:
                    t1 = r
    
    return t0 <= t1


def is_agent_in_fov(
    agent_a_state,
    agent_b_state,
    all_agent_states: List = None,
    all_agent_properties: List = None,
    agent_a_id: int = None,
    agent_b_id: int = None,
    fov_angle: float = 120.0,
    fov_range: float = 50.0
) -> bool:
    """
    Check if agent B is within agent A's field of view, considering line-of-sight blocking.
    
    Args:
        agent_a_state: AgentState of agent A (the observer)
        agent_b_state: AgentState of agent B (the target)
        all_agent_states: List of all agent states (for line-of-sight checking)
        all_agent_properties: List of all agent properties (for bounding box calculation)
        agent_a_id: ID of agent A (to exclude from blocking check)
        agent_b_id: ID of agent B (to exclude from blocking check)
        fov_angle: Field of view angle in degrees (default 120)
        fov_range: Field of view range in meters (default 50)
    
    Returns:
        True if agent B is within agent A's FoV and not blocked, False otherwise
    """
    # Calculate distance between agents
    dx = agent_b_state.center.x - agent_a_state.center.x
    dy = agent_b_state.center.y - agent_a_state.center.y
    distance = math.sqrt(dx**2 + dy**2)
    
    # Check if within range
    if distance > fov_range:
        return False
    
    # Calculate angle from agent A to agent B
    angle_to_b = math.atan2(dy, dx)
    
    # Normalize angles to [0, 2*pi]
    agent_a_orientation = agent_a_state.orientation % (2 * math.pi)
    angle_to_b = angle_to_b % (2 * math.pi)
    
    # Calculate angular difference
    angle_diff = abs(angle_to_b - agent_a_orientation)
    if angle_diff > math.pi:
        angle_diff = 2 * math.pi - angle_diff
    
    # Check if within FoV angle (half angle on each side)
    fov_half_angle_rad = (fov_angle / 2.0) * math.pi / 180.0
    
    if angle_diff > fov_half_angle_rad:
        return False
    
    # Check line-of-sight: see if any other agent blocks the view
    if all_agent_states is not None and all_agent_properties is not None:
        p1_x, p1_y = agent_a_state.center.x, agent_a_state.center.y
        p2_x, p2_y = agent_b_state.center.x, agent_b_state.center.y
        
        for agent_id, (other_state, other_props) in enumerate(zip(all_agent_states, all_agent_properties)):
            # Skip agent A and agent B
            if agent_id == agent_a_id or agent_id == agent_b_id:
                continue
            
            # Get agent dimensions
            length = other_props.length if other_props.length is not None else 5.0
            width = other_props.width if other_props.width is not None else 2.0
            if other_props.agent_type == "pedestrian":
                length = width = 1.5
            
            # Check if line segment from A to B intersects this agent's bounding box
            if line_segment_intersects_rectangle(
                p1_x, p1_y, p2_x, p2_y,
                other_state.center.x, other_state.center.y,
                length, width,
                other_state.orientation
            ):
                return False  # Blocked by this agent
    
    return True


def find_visibility_windows(
    drive_responses: List,
    agent_a_id: int,
    agent_b_id: int,
    all_agent_properties: List = None,
    fov_angle: float = 120.0,
    fov_range: float = 50.0,
    min_window_length: int = 40
) -> List[Tuple[int, int]]:
    """
    Find continuous time windows where agent A can see agent B.
    
    Returns:
        List of (start_timestep, end_timestep) tuples for valid windows
    """
    visibility = []
    
    for timestep, response in enumerate(drive_responses):
        # Check if both agents are present at this timestep
        if agent_a_id >= len(response.agent_states) or agent_b_id >= len(response.agent_states):
            visibility.append(False)
            continue
        
        agent_a_state = response.agent_states[agent_a_id]
        agent_b_state = response.agent_states[agent_b_id]
        
        # Use provided agent properties (they should be static across timesteps)
        agent_props = all_agent_properties
        
        is_visible = is_agent_in_fov(
            agent_a_state, 
            agent_b_state,
            all_agent_states=response.agent_states,
            all_agent_properties=agent_props,
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            fov_angle=fov_angle,
            fov_range=fov_range
        )
        visibility.append(is_visible)
    
    # Find continuous windows
    windows = []
    window_start = None
    
    for i, is_visible in enumerate(visibility):
        if is_visible and window_start is None:
            window_start = i
        elif not is_visible and window_start is not None:
            window_length = i - window_start
            if window_length >= min_window_length:
                windows.append((window_start, i - 1))
            window_start = None
    
    # Handle case where window extends to end
    if window_start is not None:
        window_length = len(visibility) - window_start
        if window_length >= min_window_length:
            windows.append((window_start, len(visibility) - 1))
    
    return windows


def find_nearby_agents(
    drive_responses: List,
    agent_a_id: int,
    agent_b_id: int,
    timestep: int,
    max_agents: int = 4,
    max_distance: float = None
) -> List[int]:
    """
    Find nearby agents to agent A at a given timestep.
    Agent B must be included. Returns up to max_agents agents.
    
    Args:
        max_distance: Maximum distance in meters. If None, no distance limit is applied.
    """
    if timestep >= len(drive_responses):
        return [agent_b_id]
    
    response = drive_responses[timestep]
    if agent_a_id >= len(response.agent_states) or agent_b_id >= len(response.agent_states):
        return [agent_b_id]
    
    agent_a_state = response.agent_states[agent_a_id]
    
    # Calculate distances to all agents
    distances = []
    for agent_id, agent_state in enumerate(response.agent_states):
        if agent_id == agent_a_id:
            continue
        
        dx = agent_state.center.x - agent_a_state.center.x
        dy = agent_state.center.y - agent_a_state.center.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Filter by max_distance if specified
        if max_distance is None or distance <= max_distance:
            distances.append((agent_id, distance))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    # Start with agent B (must be included, even if it exceeds max_distance)
    nearby = [agent_b_id]
    
    # Add other nearby agents (excluding A and B)
    for agent_id, _ in distances:
        if agent_id != agent_b_id and len(nearby) < max_agents:
            nearby.append(agent_id)
    
    return nearby


def has_stall_period(
    drive_responses: List,
    agent_id: int,
    windows: List[Tuple[int, int]],
    stall_threshold_speed: float = 0.1,
    min_stall_duration: int = 20
) -> bool:
    """
    Check if an agent has a stalling period (low/zero speed) of at least min_stall_duration timesteps
    within any of the given visibility windows.
    
    Args:
        drive_responses: List of drive responses for all timesteps
        agent_id: ID of the agent to check
        windows: List of (start, end) timestep tuples for visibility windows
        stall_threshold_speed: Speed threshold in m/s below which agent is considered stalling (default 0.1)
        min_stall_duration: Minimum consecutive timesteps of stalling to be considered a stall period (default 20)
    
    Returns:
        True if agent has a stall period of min_stall_duration or more timesteps, False otherwise
    """
    for window_start, window_end in windows:
        # Check for stalling within this window
        stall_start = None
        
        for timestep in range(window_start, window_end + 1):
            if timestep >= len(drive_responses):
                break
            
            response = drive_responses[timestep]
            if agent_id >= len(response.agent_states):
                continue
            
            agent_state = response.agent_states[agent_id]
            speed = abs(agent_state.speed)  # Use absolute speed
            
            if speed < stall_threshold_speed:
                # Agent is stalling
                if stall_start is None:
                    stall_start = timestep
            else:
                # Agent is moving
                if stall_start is not None:
                    stall_duration = timestep - stall_start
                    if stall_duration >= min_stall_duration:
                        return True  # Found a stall period of sufficient duration
                    stall_start = None
        
        # Check if stall period extends to end of window
        if stall_start is not None:
            stall_duration = (window_end + 1) - stall_start
            if stall_duration >= min_stall_duration:
                return True
    
    return False


def has_collisions_or_offroad(
    drive_responses: List,
    agent_ids: List[int],
    windows: List[Tuple[int, int]],
    check_collisions: bool = True,
    check_offroad: bool = True
) -> bool:
    """
    Check if any of the specified agents have collisions or go off-road within any of the given visibility windows.
    
    Args:
        drive_responses: List of drive responses for all timesteps
        agent_ids: List of agent IDs to check (e.g., [agent_a_id, agent_b_id] or replay_agent_ids)
        windows: List of (start, end) timestep tuples for visibility windows
        check_collisions: Whether to check for collisions
        check_offroad: Whether to check for off-road behavior
    
    Returns:
        True if any agent has collisions or goes off-road during any window (based on flags), False otherwise
    """
    for window_start, window_end in windows:
        for timestep in range(window_start, window_end + 1):
            if timestep >= len(drive_responses):
                break
            
            response = drive_responses[timestep]
            
            # Check if infractions are available
            if response.infractions is None:
                continue
            
            # Check each agent of interest
            for agent_id in agent_ids:
                if agent_id >= len(response.agent_states) or agent_id >= len(response.infractions):
                    continue
                
                infraction = response.infractions[agent_id]
                
                # Check for collisions or off-road behavior based on flags
                if (check_collisions and infraction.collisions) or (check_offroad and infraction.offroad):
                    return True  # Found collision or off-road behavior
    
    return False


def detect_t2f_scenarios(
    drive_responses: List,
    num_agents: int,
    all_agent_properties: List = None,
    fov_angle: float = 120.0,
    fov_range: float = 50.0,
    min_window_length: int = 40,
    filter_stalling: bool = True,
    min_stall_duration: int = 10,
    filter_collisions: bool = True,
    filter_offroad: bool = True
) -> List[Dict]:
    """
    Detect track-to-follow (t2f) scenarios from drive responses.
    
    Args:
        drive_responses: List of drive responses for all timesteps
        num_agents: Number of agents in the simulation
        all_agent_properties: List of agent properties (static across timesteps)
        fov_angle: Field of view angle in degrees
        fov_range: Field of view range in meters
        min_window_length: Minimum length of visibility window in timesteps
        filter_stalling: Whether to filter out scenarios where target agent is stalling
        min_stall_duration: Minimum consecutive timesteps of stalling to filter
        filter_collisions: Whether to filter out scenarios with collisions
        filter_offroad: Whether to filter out scenarios with off-road behavior
    
    Returns:
        List of scenario dictionaries with camera_mount_ids, replay_agent_ids, and valid_windows
    """
    scenarios = []
    
    # Check all pairs of agents
    for agent_a_id in range(num_agents):
        for agent_b_id in range(num_agents):
            if agent_a_id == agent_b_id:
                continue
            
            # Find visibility windows
            windows = find_visibility_windows(
                drive_responses,
                agent_a_id,
                agent_b_id,
                all_agent_properties=all_agent_properties,
                fov_angle=fov_angle,
                fov_range=fov_range,
                min_window_length=min_window_length
            )
            
            if windows:
                # Filter out scenarios where Agent B is stalling for 20+ timesteps
                if filter_stalling and has_stall_period(drive_responses, agent_b_id, windows, min_stall_duration=min_stall_duration):
                    continue  # Skip this scenario
                
                # Use the first window to determine nearby agents
                first_window_start = windows[0][0]
                max_distance = 2.0 * fov_range  # 2x FoV range
                replay_agent_ids = find_nearby_agents(
                    drive_responses,
                    agent_a_id,
                    agent_b_id,
                    first_window_start,
                    max_distance=max_distance
                )
                
                # Check for collisions or off-road behavior in relevant agents
                # Check both the observer (agent A), target (agent B), and replay agents
                agents_to_check = [agent_a_id, agent_b_id] + replay_agent_ids
                agents_to_check = list(set(agents_to_check))  # Remove duplicates
                
                if (filter_collisions or filter_offroad) and has_collisions_or_offroad(
                    drive_responses, agents_to_check, windows,
                    check_collisions=filter_collisions,
                    check_offroad=filter_offroad
                ):
                    continue  # Skip this scenario due to collisions or off-road behavior
                
                scenario = {
                    "camera_mount_ids": [agent_a_id, agent_b_id],
                    "observer_id": agent_a_id,  # Agent A is the camera being used
                    "instance_mask_id": agent_b_id,  # Agent B is the target being observed
                    "replay_agent_ids": replay_agent_ids+[agent_a_id],
                    "valid_windows": [[start, end] for start, end in windows]
                }
                scenarios.append(scenario)
    
    return scenarios


def main():
    args = parse_arguments()
    
    # Set up location string
    location = f"carla:{args.town}"
    
    # Set up other parameters
    simulation_length = args.simulation_length
    seed = args.seed if args.seed is not None else int(time.time())
    drive_model = args.drive_model
    num_agents = args.num_agents
    fov = args.fov
    
    api_key = os.environ.get("IAI_API_KEY", None)
    if api_key is None:
        iai.add_apikey('<INSERT_KEY_HERE>')  # specify your key here or through the IAI_API_KEY variable
    
    print("Begin initialization.")
    # get static information about a given location including map in osm
    # format and list traffic lights with their IDs and locations.
    location_info_response = iai.location_info(
        location=location, 
        include_map_source=True,
        rendering_fov=fov
    )
    
    # Set location coordinates - use provided values or default to map_center
    location_x = args.x if args.x is not None else location_info_response.map_center.x
    location_y = args.y if args.y is not None else location_info_response.map_center.y
    
    print(f"Using location: {location}")
    print(f"Using coordinates: x={location_x:.1f}, y={location_y:.1f}")
    
    # Create folder name based on town name and coordinates
    folder_name = f"iai_waypoints_{args.town}_{location_x:.1f}_{location_y:.1f}"
    print(f"Output folder: {folder_name}")
    
    # initialize the simulation by spawning NPCs
    response = iai.initialize(
        location=location,  # select one of available locations
        agent_properties=get_default_agent_properties({AgentType.car:num_agents}),  # number of NPCs to spawn
        location_of_interest=(location_x, location_y),
        random_seed=seed
    )
    initialize_response = response
    
    wp_manager = iai.WaypointManager(
        location_info_response = location_info_response,
        cfg = iai.WaypointManagerConfig(
            random_seed=seed,
            fail_soft=True
        )
    )
    agent_properties = wp_manager.update(
        response = response,
        agent_properties = response.agent_properties,
    )
    
    rendered_static_map = location_info_response.birdview_image.decode()
    scene_plotter = iai.utils.ScenePlotter(
        map_image = rendered_static_map,
        fov = fov,
        xy_offset = (location_info_response.map_center.x, location_info_response.map_center.y),
        static_actors = location_info_response.static_actors,
        resolution = (2048,2048),
        left_hand_coordinates = location.split(":")[0] == "carla"
    )
    scene_plotter.initialize_recording(
        agent_states=response.agent_states,
        agent_properties=agent_properties,
    )
    
    print("Begin stepping through simulation.")
    drive_responses = []
    for _ in range(simulation_length):  # how many simulation steps to execute (10 steps is 1 second)
        response = iai.drive(
            location=location,
            agent_properties=agent_properties,
            agent_states=response.agent_states,
            recurrent_states=response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states,
            random_seed=seed,
            api_model_version=drive_model,
            get_infractions=True  # Enable infraction tracking for collision and off-road detection
            # drive_model=drive_model
        )
        agent_properties = wp_manager.update(
            response = response,
            agent_properties = agent_properties,
        )
        drive_responses.append(response)
        # save the visualization
        scene_plotter.record_step(
            agent_states=response.agent_states,
            agent_properties=agent_properties, #This is important to capture the new waypoints every time step
            traffic_light_states=response.traffic_lights_states
        )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log_output_path = os.path.join(output_dir, f"waypoint_example_{seed}_{num_agents}_{args.town}_{location_x:.1f}_{location_y:.1f}.json")
    if log_output_path is not None:
        log_writer = iai.LogWriter()
        location_info_response = iai.location_info(
                location=location,
                rendering_fov=fov,
                rendering_center=(location_info_response.map_center.x, location_info_response.map_center.y),
        )
        log_writer.initialize(
                location=location,
                location_info_response=location_info_response,
                init_response=initialize_response,
                lights_random_seed=0,
                initialize_random_seed=0,
                drive_random_seed=0
            )
        if drive_model is not None:
            log_writer._scenario_log.drive_model_version = drive_model
        
        for response in drive_responses:
            log_writer.drive(drive_response=response)
        
        log_writer.export_to_file(log_path=log_output_path)
        
        # Detect t2f scenarios
        print("Detecting t2f scenarios...")
        # Get agent properties from the log writer (they should be consistent across timesteps)
        # Agent properties are static, so we can use them from any timestep
        agent_properties_for_detection = log_writer._scenario_log.agent_properties if hasattr(log_writer, '_scenario_log') else None
        if agent_properties_for_detection is None:
            # Fallback: use agent_properties from the last waypoint manager update
            # This should be available from the script's agent_properties variable
            agent_properties_for_detection = agent_properties
        # 2000 data use the following parameters to detect t2f scenarios    
        # t2f_scenarios = detect_t2f_scenarios(
        #     drive_responses=drive_responses,
        #     num_agents=num_agents,
        #     all_agent_properties=agent_properties_for_detection,
        #     fov_angle=120.0,  # Match the FoV angle used in visualization
        #     fov_range=30.0,  # Match the FoV range used in visualization
        #     min_window_length=40
        # )
        t2f_scenarios = detect_t2f_scenarios(
            drive_responses=drive_responses,
            num_agents=num_agents,
            all_agent_properties=agent_properties_for_detection,
            fov_angle=110.0,  # Match the FoV angle used in visualization
            fov_range=20.0,  # Match the FoV range used in visualization
            min_window_length=30,
            filter_stalling=True,
            filter_collisions=True,  # Filter out scenarios with collisions
            filter_offroad=True  # Filter out scenarios with off-road behavior
        )
        print(f"Found {len(t2f_scenarios)} t2f scenarios")
        
        # Add t2f_scenarios to output_dict
        log_writer.output_dict['t2f_scenarios'] = t2f_scenarios
        
        # log_writer.output_dict['vehicle_blueprints'] = [left_turning_vehicle_model, oncoming_vehicle_model]
        # log_writer.output_dict['intersection_id'] = intersection_id
        with open(log_output_path, "w") as outfile:
            json.dump(
                    log_writer.output_dict, 
                    outfile,
                    indent=4
            )
        print(f'Scenario log written to {os.path.abspath(log_output_path)}')
    
    print("Simulation finished, save visualization.")
    

    # save the visualization to disk
    # toss a coin to decide whether to save the visualization
    if random.random() < 0.1:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(50, 50))
        plt.axis('off')
        gif_name = os.path.join(output_dir, f'{args.town}_{location_x:.1f}_{location_y:.1f}_{seed}_waypoint_example.gif')
        scene_plotter.animate_scene(
            output_name=gif_name,
            ax=ax,
            direction_vec=False,
            velocity_vec=False,
            plot_frame_number=True,
            fov_vec=True,  # Enable FoV visualization
            fov_angle=110.0,  # 120 degree field of view
            fov_range=20.0,  # 50 meter range
            numbers=list(range(num_agents))
        )
        print(f"Visualization saved to {gif_name}")
        print("Done")


if __name__ == "__main__":
    main()
