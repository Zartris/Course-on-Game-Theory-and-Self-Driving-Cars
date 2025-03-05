#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance metrics utilities for the Roundabout Traffic Management simulation.

This module provides functions for calculating and analyzing various performance
metrics related to traffic flow, safety, and efficiency.
"""

import numpy as np
from collections import defaultdict

def calculate_traffic_metrics(vehicles):
    """
    Calculate various traffic flow and efficiency metrics.
    
    Args:
        vehicles (list): List of Vehicle objects in the simulation
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Skip if no vehicles
    if not vehicles:
        return {
            'throughput': 0.0,
            'avg_speed': 0.0,
            'flow_efficiency': 0.0,
            'safety_score': 1.0,
            'fairness_index': 1.0
        }
    
    # Calculate average speed
    speeds = [v.velocity for v in vehicles if v.in_roundabout]
    metrics['avg_speed'] = np.mean(speeds) if speeds else 0.0
    
    # Calculate throughput (vehicles completed per time step)
    completed_vehicles = sum(1 for v in vehicles if v.completed)
    max_time = max(v.travel_time for v in vehicles) if vehicles else 1
    metrics['throughput'] = completed_vehicles / max_time if max_time > 0 else 0
    
    # Calculate flow efficiency (how well vehicles maintain desired speed)
    if speeds:
        target_speed = np.mean([v.max_speed for v in vehicles])
        speed_efficiency = np.mean([
            min(s / target_speed, 1.0) for s in speeds
        ])
        metrics['flow_efficiency'] = speed_efficiency
    else:
        metrics['flow_efficiency'] = 0.0
    
    # Calculate safety score
    safety_margins = []
    for v in vehicles:
        if v.in_roundabout:
            margin = v.calculate_safety_margin()
            if margin < float('inf'):
                safety_margins.append(margin)
    
    metrics['safety_score'] = np.mean(safety_margins) if safety_margins else 1.0
    
    # Calculate fairness index (Jain's fairness index for wait times)
    wait_times = [v.wait_time for v in vehicles]
    if wait_times:
        n = len(wait_times)
        sum_times = sum(wait_times)
        sum_square_times = sum(t * t for t in wait_times)
        metrics['fairness_index'] = (sum_times * sum_times) / (n * sum_square_times) if sum_square_times > 0 else 1.0
    else:
        metrics['fairness_index'] = 1.0
    
    return metrics

def analyze_vehicle_interactions(vehicles):
    """
    Analyze interactions between vehicles to identify patterns and potential issues.
    
    Args:
        vehicles (list): List of Vehicle objects in the simulation
        
    Returns:
        dict: Dictionary containing interaction analysis results
    """
    analysis = {
        'interaction_counts': defaultdict(int),
        'conflict_points': [],
        'cooperation_scores': {},
        'type_interactions': defaultdict(lambda: defaultdict(int))
    }
    
    # Analyze interactions between vehicle pairs
    for i, v1 in enumerate(vehicles):
        for v2 in vehicles[i+1:]:
            # Calculate distance between vehicles
            if hasattr(v1, 'position') and hasattr(v2, 'position'):
                distance = np.linalg.norm(
                    np.array(v1.position) - np.array(v2.position)
                )
                
                # Record close interactions
                if distance < (v1.size + v2.size) * 3:
                    interaction_type = _classify_interaction(v1, v2)
                    analysis['interaction_counts'][interaction_type] += 1
                    
                    # Record interaction between vehicle types
                    type_pair = tuple(sorted([v1.vehicle_type, v2.vehicle_type]))
                    analysis['type_interactions'][type_pair][interaction_type] += 1
                    
                    # Record potential conflict points
                    if interaction_type in ['conflict', 'near_miss']:
                        analysis['conflict_points'].append({
                            'position': tuple(np.array(v1.position + v2.position) / 2),
                            'vehicles': (v1.id, v2.id),
                            'type': interaction_type
                        })
    
    # Calculate cooperation scores for each vehicle
    for vehicle in vehicles:
        if hasattr(vehicle, 'cooperation_factor'):
            analysis['cooperation_scores'][vehicle.id] = _calculate_cooperation_score(vehicle)
    
    return analysis

def _classify_interaction(v1, v2):
    """
    Classify the type of interaction between two vehicles.
    
    Args:
        v1, v2 (Vehicle): The two vehicles involved in the interaction
        
    Returns:
        str: Type of interaction ('safe', 'conflict', 'near_miss', or 'collision')
    """
    if v1.collided or v2.collided:
        return 'collision'
    
    # Calculate relative velocity
    if hasattr(v1, 'velocity') and hasattr(v2, 'velocity'):
        rel_velocity = abs(v1.velocity - v2.velocity)
    else:
        rel_velocity = 0
    
    # Calculate distance
    distance = np.linalg.norm(
        np.array(v1.position) - np.array(v2.position)
    )
    
    # Classify based on distance and relative velocity
    safe_distance = (v1.size + v2.size) * 2
    
    if distance < (v1.size + v2.size) * 1.1:
        return 'conflict'
    elif distance < safe_distance and rel_velocity > 1.0:
        return 'near_miss'
    else:
        return 'safe'

def _calculate_cooperation_score(vehicle):
    """
    Calculate a cooperation score for a vehicle based on its behavior.
    
    Args:
        vehicle (Vehicle): The vehicle to analyze
        
    Returns:
        float: Cooperation score between 0 and 1
    """
    if not hasattr(vehicle, 'interaction_history'):
        return 0.5  # Default score if no history available
    
    # Initialize base score based on vehicle type
    base_scores = {
        'aggressive': 0.3,
        'conservative': 0.7,
        'cooperative': 0.8,
        'autonomous': 0.6
    }
    
    score = base_scores.get(vehicle.vehicle_type, 0.5)
    
    # Adjust score based on behavior history
    if hasattr(vehicle, 'cooperation_factor'):
        score = 0.7 * score + 0.3 * vehicle.cooperation_factor
    
    # Ensure score is between 0 and 1
    return np.clip(score, 0, 1)

def calculate_nash_equilibrium_stability(vehicle_states, n_steps=10):
    """
    Analyze the stability of the current traffic state with respect to Nash equilibrium.
    
    Args:
        vehicle_states (list): List of current vehicle states
        n_steps (int): Number of steps to look ahead for stability analysis
        
    Returns:
        dict: Stability analysis results
    """
    stability = {
        'is_stable': True,
        'deviation_incentives': {},
        'equilibrium_distance': 0.0
    }
    
    # Simple implementation focusing on immediate neighbors
    for v_state in vehicle_states:
        if not hasattr(v_state, 'utility'):
            continue
        
        # Check if any vehicle has incentive to deviate
        current_utility = v_state.utility
        max_alternative_utility = current_utility
        
        # Consider alternative strategies
        for action in range(3):  # 0: Enter/Accelerate, 1: Yield, 2: Change Lane
            if action != v_state.last_action:
                # Estimate utility of alternative action
                alternative_utility = _estimate_alternative_utility(v_state, action)
                if alternative_utility > max_alternative_utility:
                    max_alternative_utility = alternative_utility
                    stability['is_stable'] = False
                    stability['deviation_incentives'][v_state.id] = {
                        'current_action': v_state.last_action,
                        'better_action': action,
                        'utility_gain': alternative_utility - current_utility
                    }
    
    # Calculate "distance" from equilibrium
    if stability['deviation_incentives']:
        max_incentive = max(
            d['utility_gain'] for d in stability['deviation_incentives'].values()
        )
        stability['equilibrium_distance'] = max_incentive
    
    return stability

def _estimate_alternative_utility(vehicle_state, action):
    """
    Estimate the utility of an alternative action for a vehicle.
    
    Args:
        vehicle_state: Current state of the vehicle
        action: Alternative action to evaluate
        
    Returns:
        float: Estimated utility of the alternative action
    """
    # This is a simplified utility estimation
    base_utility = 0.0
    
    if hasattr(vehicle_state, 'velocity'):
        if action == 0:  # Enter/Accelerate
            # Reward for making progress
            base_utility += 2.0
            # Penalty for potential safety risks
            if hasattr(vehicle_state, 'calculate_safety_margin'):
                safety_margin = vehicle_state.calculate_safety_margin()
                base_utility -= max(0, 1.0 - safety_margin)
        elif action == 1:  # Yield
            # Small penalty for delay
            base_utility -= 0.5
            # Reward for safety
            base_utility += 1.0
        elif action == 2:  # Change Lane
            # Moderate reward/penalty based on current lane position
            base_utility += 0.0  # Neutral by default
    
    return base_utility