#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics utilities for evaluating fleet coordination performance.

This module provides functions for calculating and analyzing the performance
of fleet coordination strategies in terms of efficiency, safety, and information use.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

def calculate_fleet_metrics(vehicles, time_step, communication_graph=None):
    """
    Calculate comprehensive metrics for fleet performance.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        time_step (int): Current simulation time step
        communication_graph: NetworkX graph representing communications
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Calculate distance-based metrics
    metrics.update(calculate_distance_metrics(vehicles))
    
    # Calculate safety metrics
    metrics.update(calculate_safety_metrics(vehicles))
    
    # Calculate coordination metrics
    metrics.update(calculate_coordination_metrics(vehicles, communication_graph))
    
    # Calculate information-related metrics
    metrics.update(calculate_information_metrics(vehicles))
    
    # Calculate emergency response metrics
    metrics.update(calculate_emergency_metrics(vehicles))
    
    return metrics

def calculate_distance_metrics(vehicles):
    """
    Calculate metrics related to distances traveled and goal progress.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        
    Returns:
        dict: Distance-related metrics
    """
    metrics = {}
    
    # Total distance traveled by all vehicles
    total_distance = 0
    for vehicle in vehicles:
        if hasattr(vehicle, 'last_position') and vehicle.last_position is not None:
            distance = np.linalg.norm(np.array(vehicle.position) - np.array(vehicle.last_position))
            total_distance += distance
    
    metrics['total_distance'] = total_distance
    
    # Average progress towards goals
    progress_values = []
    for vehicle in vehicles:
        if hasattr(vehicle, 'goal'):
            current_distance = np.linalg.norm(np.array(vehicle.position) - np.array(vehicle.goal))
            
            if hasattr(vehicle, 'initial_distance_to_goal') and vehicle.initial_distance_to_goal > 0:
                progress = 1.0 - (current_distance / vehicle.initial_distance_to_goal)
                progress_values.append(progress)
    
    if progress_values:
        metrics['average_goal_progress'] = np.mean(progress_values)
    else:
        metrics['average_goal_progress'] = 0.0
    
    return metrics

def calculate_safety_metrics(vehicles):
    """
    Calculate safety-related metrics for the fleet.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        
    Returns:
        dict: Safety-related metrics
    """
    metrics = {}
    
    # Count near-misses (vehicles that came very close but didn't collide)
    near_misses = 0
    minimum_distances = []
    
    # Check all vehicle pairs
    for i, v1 in enumerate(vehicles):
        for j, v2 in enumerate(vehicles):
            if i < j:  # Avoid duplicate checks
                distance = np.linalg.norm(np.array(v1.position) - np.array(v2.position))
                minimum_distances.append(distance)
                
                # Define near-miss threshold
                if 0.5 < distance < 1.5:  # Close but not colliding
                    near_misses += 1
    
    metrics['near_misses'] = near_misses
    
    # Minimum distance between any two vehicles
    if minimum_distances:
        metrics['minimum_distance'] = min(minimum_distances)
    else:
        metrics['minimum_distance'] = float('inf')
    
    # Average minimum distance between vehicles
    if minimum_distances:
        metrics['average_minimum_distance'] = np.mean(minimum_distances)
    else:
        metrics['average_minimum_distance'] = float('inf')
    
    return metrics

def calculate_coordination_metrics(vehicles, communication_graph=None):
    """
    Calculate metrics related to fleet coordination.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        communication_graph: NetworkX graph representing communications
        
    Returns:
        dict: Coordination-related metrics
    """
    metrics = {}
    
    # If communication graph is provided, calculate metrics from it
    if communication_graph is not None:
        # Connectivity metrics
        if len(communication_graph.nodes()) > 0:
            # Convert directed graph to undirected for algebraic_connectivity calculation
            undirected_graph = communication_graph.to_undirected()
            try:
                metrics['connectivity'] = nx.algebraic_connectivity(undirected_graph, weight='weight')
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                # Handle any NetworkX errors by setting a default value
                metrics['connectivity'] = 0.0
        else:
            metrics['connectivity'] = 0.0
            
        # Communication efficiency - also requires undirected graph
        if len(communication_graph.edges()) > 0:
            undirected_graph = communication_graph.to_undirected()
            try:
                metrics['communication_efficiency'] = nx.global_efficiency(undirected_graph)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                metrics['communication_efficiency'] = 0.0
        else:
            metrics['communication_efficiency'] = 0.0
    
    # Calculate vehicle velocity alignment
    velocity_vectors = []
    for vehicle in vehicles:
        vel_magnitude = np.linalg.norm(vehicle.velocity)
        if vel_magnitude > 0.01:  # Only include non-stationary vehicles
            normalized_vel = np.array(vehicle.velocity) / vel_magnitude
            velocity_vectors.append(normalized_vel)
    
    if len(velocity_vectors) >= 2:
        # Calculate average pairwise dot product of velocity vectors
        alignment_sum = 0
        count = 0
        for i, v1 in enumerate(velocity_vectors):
            for j, v2 in enumerate(velocity_vectors):
                if i < j:  # Avoid duplicate checks
                    alignment_sum += np.dot(v1, v2)
                    count += 1
        
        if count > 0:
            metrics['velocity_alignment'] = alignment_sum / count
        else:
            metrics['velocity_alignment'] = 0.0
    else:
        metrics['velocity_alignment'] = 0.0
    
    return metrics

def calculate_information_metrics(vehicles):
    """
    Calculate metrics related to information usage and belief accuracy.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        
    Returns:
        dict: Information-related metrics
    """
    metrics = {}
    
    # Calculate belief accuracy - how well vehicles' beliefs match reality
    belief_kl_div = []
    belief_entropy = []
    
    for vehicle in vehicles:
        if hasattr(vehicle, 'beliefs'):
            for other_id, type_beliefs in vehicle.beliefs.items():
                # Find the actual type of the other vehicle
                other_vehicle = next((v for v in vehicles if v.id == other_id), None)
                if other_vehicle is None:
                    continue
                    
                actual_type = other_vehicle.vehicle_type
                
                # Create one-hot distribution for actual type
                actual_dist = {t: 1.0 if t == actual_type else 0.0 
                              for t in type_beliefs.keys()}
                
                # Calculate KL divergence between belief and reality
                kl_div = 0.0
                entropy = 0.0
                
                for type_name in type_beliefs.keys():
                    # Add small epsilon to avoid divide by zero or log(0)
                    p = type_beliefs[type_name] + 1e-10
                    q = actual_dist[type_name] + 1e-10
                    
                    # KL divergence
                    kl_div += q * np.log(q / p)
                    
                    # Entropy
                    entropy -= p * np.log(p)
                
                belief_kl_div.append(kl_div)
                belief_entropy.append(entropy)
    
    if belief_kl_div:
        metrics['belief_kl_divergence'] = np.mean(belief_kl_div)
    else:
        metrics['belief_kl_divergence'] = 0.0
        
    if belief_entropy:
        metrics['belief_entropy'] = np.mean(belief_entropy)
    else:
        metrics['belief_entropy'] = 0.0
    
    return metrics

def calculate_emergency_metrics(vehicles):
    """
    Calculate metrics related to emergency scenarios.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        
    Returns:
        dict: Emergency-related metrics
    """
    metrics = {}
    
    # Count active emergency vehicles
    active_emergencies = sum(1 for v in vehicles 
                           if v.vehicle_type == 'emergency' and getattr(v, 'emergency_active', False))
    
    metrics['active_emergencies'] = active_emergencies
    
    # Calculate average speed of emergency vehicles
    emergency_speeds = []
    for vehicle in vehicles:
        if vehicle.vehicle_type == 'emergency':
            speed = np.linalg.norm(vehicle.velocity)
            emergency_speeds.append(speed)
    
    if emergency_speeds:
        metrics['average_emergency_speed'] = np.mean(emergency_speeds)
    else:
        metrics['average_emergency_speed'] = 0.0
    
    return metrics

def analyze_belief_convergence(belief_history):
    """
    Analyze how beliefs converge over time.
    
    Args:
        belief_history (dict): History of belief updates for each vehicle
        
    Returns:
        dict: Convergence analysis metrics
    """
    results = {}
    
    # For each vehicle, analyze belief convergence
    for vehicle_id, history in belief_history.items():
        vehicle_results = {}
        
        # For each target vehicle, analyze convergence of beliefs
        target_ids = set()
        for beliefs in history:
            target_ids.update(beliefs.keys())
        
        for target_id in target_ids:
            # Extract belief history for this target
            target_history = []
            for t, beliefs in enumerate(history):
                if target_id in beliefs:
                    target_beliefs = beliefs[target_id]
                    target_history.append(target_beliefs)
            
            if len(target_history) < 2:
                continue
            
            # Calculate differences between consecutive beliefs
            diffs = []
            for t in range(1, len(target_history)):
                # Sum of absolute differences in probabilities
                diff = sum(abs(target_history[t].get(type_name, 0) - target_history[t-1].get(type_name, 0))
                          for type_name in set(target_history[t].keys()) | set(target_history[t-1].keys()))
                diffs.append(diff)
            
            # Calculate convergence metrics
            vehicle_results[target_id] = {
                'avg_change': np.mean(diffs),
                'max_change': max(diffs),
                'final_entropy': calc_entropy(target_history[-1])
            }
        
        results[vehicle_id] = vehicle_results
    
    return results

def calc_entropy(distribution):
    """
    Calculate entropy of a probability distribution.
    
    Args:
        distribution (dict): Probability distribution
        
    Returns:
        float: Entropy value
    """
    entropy = 0.0
    for p in distribution.values():
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def analyze_information_value(vehicle_performance, with_info, without_info):
    """
    Analyze the value of information by comparing performance with and without it.
    
    Args:
        vehicle_performance (list): List of performance metrics for vehicles
        with_info: Performance metrics with information
        without_info: Performance metrics without information
        
    Returns:
        dict: Analysis of information value
    """
    value = {}
    
    # Compare performance metrics
    for metric in set(with_info.keys()) & set(without_info.keys()):
        # Calculate absolute and relative improvement
        absolute_improvement = with_info[metric] - without_info[metric]
        
        if without_info[metric] != 0:
            relative_improvement = absolute_improvement / abs(without_info[metric])
        else:
            relative_improvement = float('inf') if absolute_improvement > 0 else 0.0
            
        value[metric] = {
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement
        }
    
    return value

def plot_fleet_performance(metrics_history):
    """
    Create plots of fleet performance metrics over time.
    
    Args:
        metrics_history (list): List of metrics dictionaries for each time step
        
    Returns:
        matplotlib.figure.Figure: Figure with performance plots
    """
    # Extract time steps and metrics
    time_steps = list(range(len(metrics_history)))
    
    # Select key metrics to plot
    key_metrics = [
        'average_goal_progress',
        'minimum_distance',
        'velocity_alignment',
        'belief_kl_divergence',
        'communication_efficiency'
    ]
    
    # Create figure
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(10, 3*len(key_metrics)))
    
    # Plot each metric
    for i, metric in enumerate(key_metrics):
        values = [metrics.get(metric, 0.0) for metrics in metrics_history]
        
        axes[i].plot(time_steps, values, 'b-')
        axes[i].set_title(f"{metric.replace('_', ' ').title()}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
    
    fig.tight_layout()
    return fig

def calculate_price_of_anarchy(optimal_performance, nash_performance):
    """
    Calculate the Price of Anarchy (PoA) comparing optimal vs. Nash performance.
    
    Args:
        optimal_performance (dict): Performance metrics under optimal coordination
        nash_performance (dict): Performance metrics under Nash equilibrium
        
    Returns:
        dict: Price of Anarchy for various metrics
    """
    poa = {}
    
    # Calculate PoA for relevant metrics
    for metric in set(optimal_performance.keys()) & set(nash_performance.keys()):
        opt_value = optimal_performance[metric]
        nash_value = nash_performance[metric]
        
        # Check if higher values are better for this metric
        higher_is_better = metric in [
            'average_goal_progress', 
            'velocity_alignment',
            'communication_efficiency'
        ]
        
        # Calculate PoA based on whether higher or lower values are better
        if higher_is_better:
            if nash_value > 0:
                poa[metric] = opt_value / nash_value
            else:
                poa[metric] = float('inf') if opt_value > 0 else 1.0
        else:
            if opt_value > 0:
                poa[metric] = nash_value / opt_value
            else:
                poa[metric] = float('inf') if nash_value > 0 else 1.0
    
    return poa