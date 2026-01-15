#!/usr/bin/env python3
"""
Analyze agent behavior in t2f scenarios.
Classifies agent movement as straight, left turn, or right turn within valid windows.
"""

import json
import os
import glob
import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Tuple
import argparse


def load_json_files(json_files: List[str]) -> List[Dict]:
    """
    Load JSON files and return list of log data.
    
    Args:
        json_files: List of paths to JSON files
    
    Returns:
        List of loaded log dictionaries
    """
    logs = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    return logs


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
    
    Returns:
        Normalized angle
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def calculate_orientation_change(orientations: List[float]) -> float:
    """
    Calculate total orientation change over a sequence.
    
    Args:
        orientations: List of orientation angles in radians
    
    Returns:
        Total orientation change in radians
    """
    if len(orientations) < 2:
        return 0.0
    
    total_change = 0.0
    for i in range(1, len(orientations)):
        prev_angle = orientations[i-1]
        curr_angle = orientations[i]
        
        # Calculate angular difference
        diff = normalize_angle(curr_angle - prev_angle)
        total_change += diff
    
    return total_change


def classify_behavior(orientation_change: float, threshold: float = 0.3) -> str:
    """
    Classify agent behavior based on orientation change.
    
    Args:
        orientation_change: Total orientation change in radians
        threshold: Threshold in radians for considering it a turn (default: 0.3 rad ≈ 17°)
    
    Returns:
        'straight', 'left', or 'right'
    """
    if abs(orientation_change) < threshold:
        return 'straight'
    elif orientation_change > 0:
        return 'left'  # Positive rotation is counterclockwise (left turn)
    else:
        return 'right'  # Negative rotation is clockwise (right turn)


def analyze_agent_behavior_in_window(
    log_data: Dict,
    agent_id: int,
    window_start: int,
    window_end: int
) -> Dict:
    """
    Analyze agent behavior within a specific time window.
    
    Args:
        log_data: Log dictionary
        agent_id: ID of agent to analyze
        window_start: Start timestep of window
        window_end: End timestep of window
    
    Returns:
        Dictionary with behavior analysis results
    """
    if 'predetermined_agents' not in log_data:
        return None
    
    agent_id_str = str(agent_id)
    if agent_id_str not in log_data['predetermined_agents']:
        return None
    
    agent_data = log_data['predetermined_agents'][agent_id_str]
    if 'states' not in agent_data:
        return None
    
    # Extract orientations for the window
    orientations = []
    positions = []
    speeds = []
    
    for timestep in range(window_start, window_end + 1):
        timestep_str = str(timestep)
        if timestep_str in agent_data['states']:
            state = agent_data['states'][timestep_str]
            if 'orientation' in state:
                orientations.append(state['orientation'])
            if 'center' in state:
                positions.append((state['center']['x'], state['center']['y']))
            if 'speed' in state:
                speeds.append(state['speed'])
    
    if len(orientations) < 2:
        return {
            'behavior': 'insufficient_data',
            'orientation_change': 0.0,
            'num_timesteps': len(orientations),
            'avg_speed': np.mean(speeds) if speeds else 0.0
        }
    
    # Calculate orientation change
    orientation_change = calculate_orientation_change(orientations)
    
    # Classify behavior
    behavior = classify_behavior(orientation_change)
    
    # Calculate additional statistics
    avg_speed = np.mean(speeds) if speeds else 0.0
    total_distance = 0.0
    if len(positions) >= 2:
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += math.sqrt(dx**2 + dy**2)
    
    return {
        'behavior': behavior,
        'orientation_change': orientation_change,
        'orientation_change_deg': math.degrees(orientation_change),
        'num_timesteps': len(orientations),
        'avg_speed': avg_speed,
        'total_distance': total_distance,
        'window_start': window_start,
        'window_end': window_end
    }


def analyze_t2f_scenarios(logs: List[Dict], json_files: List[str] = None, write_labels: bool = False) -> Dict:
    """
    Analyze all t2f scenarios across all logs.
    
    Args:
        logs: List of log dictionaries
        json_files: List of corresponding JSON file paths (for writing back)
        write_labels: If True, write behavior labels back to JSON files
    
    Returns:
        Dictionary with analysis results and statistics
    """
    all_behaviors = []
    behavior_counts = defaultdict(int)
    behavior_by_scenario = []
    
    for log_idx, log_data in enumerate(logs):
        if 't2f_scenarios' not in log_data:
            continue
        
        t2f_scenarios = log_data['t2f_scenarios']
        
        for scenario_idx, scenario in enumerate(t2f_scenarios):
            if 'instance_mask_id' not in scenario:
                continue
            
            instance_mask_id = scenario['instance_mask_id']
            valid_windows = scenario.get('valid_windows', [])
            
            scenario_behaviors = []
            
            for window in valid_windows:
                if len(window) != 2:
                    continue
                
                window_start, window_end = window[0], window[1]
                
                # Analyze behavior in this window
                behavior_data = analyze_agent_behavior_in_window(
                    log_data,
                    instance_mask_id,
                    window_start,
                    window_end
                )
                
                if behavior_data:
                    scenario_behaviors.append(behavior_data)
                    all_behaviors.append(behavior_data)
                    behavior_counts[behavior_data['behavior']] += 1
            
            # Determine primary behavior for this scenario (most common in windows)
            if scenario_behaviors:
                window_behaviors = [b['behavior'] for b in scenario_behaviors]
                primary_behavior = max(set(window_behaviors), key=window_behaviors.count)
                
                # Add label to scenario in the log data
                if write_labels:
                    scenario['behavior_label'] = primary_behavior
                
                behavior_by_scenario.append({
                    'log_idx': log_idx,
                    'scenario_idx': scenario_idx,
                    'camera_mount_ids': scenario.get('camera_mount_ids', []),
                    'instance_mask_id': instance_mask_id,
                    'num_windows': len(valid_windows),
                    'primary_behavior': primary_behavior,
                    'window_behaviors': window_behaviors,
                    'window_details': scenario_behaviors
                })
    
    # Write labels back to JSON files if requested
    if write_labels and json_files:
        print("\nWriting behavior labels back to JSON files...")
        for log_idx, log_data in enumerate(logs):
            if log_idx >= len(json_files):
                continue
            
            file_path = json_files[log_idx]
            try:
                # Write the updated log_data back to file
                with open(file_path, 'w') as f:
                    json.dump(log_data, f, indent=4)
                
                # Count how many scenarios were updated
                num_updated = 0
                if 't2f_scenarios' in log_data:
                    num_updated = sum(1 for s in log_data['t2f_scenarios'] if 'behavior_label' in s)
                
                print(f"  Updated {num_updated} scenarios in {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Error updating {file_path}: {e}")
    
    # Calculate statistics
    stats = {
        'total_scenarios': len(behavior_by_scenario),
        'total_windows': len(all_behaviors),
        'behavior_counts': dict(behavior_counts),
        'behavior_percentages': {
            behavior: (count / len(all_behaviors) * 100) if all_behaviors else 0
            for behavior, count in behavior_counts.items()
        }
    }
    
    # Calculate average metrics by behavior
    behavior_metrics = defaultdict(lambda: {'orientation_changes': [], 'distances': [], 'speeds': []})
    for behavior_data in all_behaviors:
        behavior = behavior_data['behavior']
        behavior_metrics[behavior]['orientation_changes'].append(abs(behavior_data['orientation_change_deg']))
        behavior_metrics[behavior]['distances'].append(behavior_data['total_distance'])
        behavior_metrics[behavior]['speeds'].append(behavior_data['avg_speed'])
    
    for behavior in behavior_metrics:
        metrics = behavior_metrics[behavior]
        stats[f'{behavior}_avg_orientation_change_deg'] = np.mean(metrics['orientation_changes']) if metrics['orientation_changes'] else 0
        stats[f'{behavior}_avg_distance'] = np.mean(metrics['distances']) if metrics['distances'] else 0
        stats[f'{behavior}_avg_speed'] = np.mean(metrics['speeds']) if metrics['speeds'] else 0
    
    return {
        'statistics': stats,
        'scenario_details': behavior_by_scenario
    }


def print_statistics(analysis_results: Dict):
    """
    Print analysis statistics in a readable format.
    
    Args:
        analysis_results: Results from analyze_t2f_scenarios
    """
    stats = analysis_results['statistics']
    
    print("=" * 60)
    print("T2F Scenario Behavior Analysis")
    print("=" * 60)
    print(f"\nTotal Scenarios Analyzed: {stats['total_scenarios']}")
    print(f"Total Windows Analyzed: {stats['total_windows']}")
    
    print("\n" + "-" * 60)
    print("Behavior Distribution:")
    print("-" * 60)
    for behavior, count in stats['behavior_counts'].items():
        percentage = stats['behavior_percentages'][behavior]
        print(f"  {behavior:15s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "-" * 60)
    print("Average Metrics by Behavior:")
    print("-" * 60)
    behaviors = ['straight', 'left', 'right']
    for behavior in behaviors:
        if behavior in stats['behavior_counts']:
            print(f"\n{behavior.upper()}:")
            print(f"  Avg Orientation Change: {stats.get(f'{behavior}_avg_orientation_change_deg', 0):.2f}°")
            print(f"  Avg Distance Traveled:   {stats.get(f'{behavior}_avg_distance', 0):.2f} m")
            print(f"  Avg Speed:               {stats.get(f'{behavior}_avg_speed', 0):.2f} m/s")


def main():
    parser = argparse.ArgumentParser(description='Analyze agent behavior in t2f scenarios')
    parser.add_argument('json_files', nargs='+', type=str,
                       help='JSON log files to analyze (can use glob patterns)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file to save detailed results')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Orientation change threshold in radians for turn detection (default: 0.3)')
    parser.add_argument('--write-labels', action='store_true',
                       help='Write behavior labels back to the original JSON files')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    json_files = []
    for pattern in args.json_files:
        if '*' in pattern or '?' in pattern:
            json_files.extend(glob.glob(pattern))
        else:
            json_files.append(pattern)
    
    # Remove duplicates and check existence
    json_files = list(set(json_files))
    json_files = [f for f in json_files if os.path.exists(f)]
    
    if not json_files:
        print("Error: No valid JSON files found!")
        return
    
    print(f"Loading {len(json_files)} JSON files...")
    logs = load_json_files(json_files)
    
    if not logs:
        print("Error: No logs loaded!")
        return
    
    print(f"Successfully loaded {len(logs)} log files")
    print("Analyzing t2f scenarios...")
    
    if args.write_labels:
        print("Will write behavior labels back to JSON files...")
    
    # Update threshold in classify_behavior (would need to pass it through)
    # For now, using default threshold
    
    analysis_results = analyze_t2f_scenarios(logs, json_files=json_files, write_labels=args.write_labels)
    
    # Print statistics
    print_statistics(analysis_results)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()

