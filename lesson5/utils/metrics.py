#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performance metrics for the Task Allocation and Auction Mechanisms lesson.

This module provides functions for calculating and evaluating the performance
of different auction mechanisms and task allocation approaches.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_allocation_efficiency(allocated_tasks: List, available_tasks: List) -> float:
    """
    Calculate the allocation efficiency (percentage of tasks allocated).
    
    Args:
        allocated_tasks: List of tasks that were successfully allocated
        available_tasks: List of tasks that were available for allocation
        
    Returns:
        float: Allocation efficiency as a percentage
    """
    if not available_tasks:
        return 0.0
    
    return len(allocated_tasks) / len(available_tasks) * 100

def calculate_social_welfare(allocated_tasks: List, robot_valuations: Dict) -> float:
    """
    Calculate the social welfare (sum of valuations of allocated tasks).
    
    Args:
        allocated_tasks: List of allocated tasks
        robot_valuations: Dictionary mapping task IDs to robot valuations
        
    Returns:
        float: Total social welfare
    """
    welfare = 0.0
    
    for task in allocated_tasks:
        task_id = task.id
        if task_id in robot_valuations:
            welfare += robot_valuations[task_id]
    
    return welfare

def calculate_execution_time(allocation_times: List) -> Dict:
    """
    Calculate statistics about execution time for allocations.
    
    Args:
        allocation_times: List of execution times in seconds
        
    Returns:
        Dict: Statistics about execution times
    """
    if not allocation_times:
        return {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0
        }
    
    return {
        'mean': np.mean(allocation_times),
        'median': np.median(allocation_times),
        'min': np.min(allocation_times),
        'max': np.max(allocation_times),
        'std': np.std(allocation_times)
    }

def calculate_auction_revenue(payments: Dict) -> float:
    """
    Calculate the total revenue from all payments.
    
    Args:
        payments: Dictionary mapping task IDs to payment amounts
        
    Returns:
        float: Total revenue
    """
    return sum(payments.values())

def calculate_revenue_to_welfare_ratio(revenue: float, welfare: float) -> float:
    """
    Calculate the ratio of revenue to social welfare.
    
    Args:
        revenue: Total auction revenue
        welfare: Total social welfare
        
    Returns:
        float: Revenue to welfare ratio
    """
    if welfare == 0:
        return 0.0
    
    return revenue / welfare

def calculate_task_completion_metrics(tasks: List) -> Dict:
    """
    Calculate metrics about task completion.
    
    Args:
        tasks: List of task objects
        
    Returns:
        Dict: Metrics about task completion
    """
    total_tasks = len(tasks)
    if total_tasks == 0:
        return {
            'assigned_rate': 0.0,
            'completed_rate': 0.0,
            'failed_rate': 0.0,
            'pending_rate': 0.0
        }
    
    assigned_count = sum(1 for task in tasks if task.assigned and not task.completed and not task.failed)
    completed_count = sum(1 for task in tasks if task.completed)
    failed_count = sum(1 for task in tasks if task.failed)
    pending_count = total_tasks - assigned_count - completed_count - failed_count
    
    return {
        'assigned_rate': assigned_count / total_tasks * 100,
        'completed_rate': completed_count / total_tasks * 100,
        'failed_rate': failed_count / total_tasks * 100,
        'pending_rate': pending_count / total_tasks * 100
    }

def calculate_robot_performance(robots: List, tasks: List, allocations: Dict) -> Dict:
    """
    Calculate performance metrics for each robot.
    
    Args:
        robots: List of robot objects
        tasks: List of task objects
        allocations: Dictionary mapping task IDs to robot IDs
        
    Returns:
        Dict: Performance metrics for each robot
    """
    # Create inverse allocations mapping
    robot_tasks = defaultdict(list)
    for task_id, robot_id in allocations.items():
        robot_tasks[robot_id].append(task_id)
    
    # Create task lookup map
    task_map = {task.id: task for task in tasks}
    
    # Calculate metrics for each robot
    robot_metrics = {}
    for robot in robots:
        # Count assigned tasks
        assigned_task_ids = robot_tasks[robot.id]
        assigned_tasks = [task_map[task_id] for task_id in assigned_task_ids if task_id in task_map]
        
        # Calculate metrics
        task_count = len(assigned_tasks)
        completed_count = sum(1 for task in assigned_tasks if task.completed)
        failed_count = sum(1 for task in assigned_tasks if task.failed)
        
        # Calculate average task attributes
        if assigned_tasks:
            avg_priority = sum(task.priority for task in assigned_tasks) / len(assigned_tasks)
            avg_completion_time = sum(task.completion_time for task in assigned_tasks) / len(assigned_tasks)
        else:
            avg_priority = 0.0
            avg_completion_time = 0.0
        
        # Store metrics
        robot_metrics[robot.id] = {
            'task_count': task_count,
            'completed_count': completed_count,
            'failed_count': failed_count,
            'completion_rate': completed_count / task_count * 100 if task_count > 0 else 0.0,
            'avg_priority': avg_priority,
            'avg_completion_time': avg_completion_time,
            'strategy_type': robot.strategy_type
        }
    
    return robot_metrics

def calculate_bundle_metrics(tasks: List, allocations: Dict) -> Dict:
    """
    Calculate metrics specific to task bundles.
    
    Args:
        tasks: List of task objects
        allocations: Dictionary mapping task IDs to robot IDs
        
    Returns:
        Dict: Metrics about bundle allocations
    """
    # Group tasks by bundle
    bundle_tasks = defaultdict(list)
    bundle_ids = set()
    
    for task in tasks:
        if hasattr(task, 'bundle_id') and task.bundle_id:
            bundle_ids.add(task.bundle_id)
            bundle_tasks[task.bundle_id].append(task.id)
    
    if not bundle_ids:
        return {
            'bundle_count': 0,
            'complete_bundle_allocation_rate': 0.0,
            'partial_bundle_allocation_rate': 0.0,
            'unallocated_bundle_rate': 0.0
        }
    
    # Check allocation status for each bundle
    complete_bundle_allocations = 0
    partial_bundle_allocations = 0
    unallocated_bundles = 0
    
    for bundle_id, task_ids in bundle_tasks.items():
        # Check how many tasks in this bundle are allocated
        allocated_task_count = sum(1 for task_id in task_ids if task_id in allocations)
        
        if allocated_task_count == len(task_ids):
            # All tasks in bundle allocated (ideally to the same robot)
            # Check if allocated to the same robot
            robot_ids = set(allocations[task_id] for task_id in task_ids if task_id in allocations)
            if len(robot_ids) == 1:
                complete_bundle_allocations += 1
            else:
                # Split bundle (allocated but to different robots)
                partial_bundle_allocations += 1
        elif allocated_task_count > 0:
            # Some tasks in bundle allocated
            partial_bundle_allocations += 1
        else:
            # No tasks in bundle allocated
            unallocated_bundles += 1
    
    total_bundles = len(bundle_ids)
    
    return {
        'bundle_count': total_bundles,
        'complete_bundle_allocation_rate': complete_bundle_allocations / total_bundles * 100,
        'partial_bundle_allocation_rate': partial_bundle_allocations / total_bundles * 100,
        'unallocated_bundle_rate': unallocated_bundles / total_bundles * 100
    }

def calculate_coalition_task_metrics(tasks: List, allocations: Dict, robots: List) -> Dict:
    """
    Calculate metrics for coalition tasks.
    
    Args:
        tasks: List of task objects
        allocations: Dictionary mapping task IDs to robot IDs
        robots: List of robot objects
        
    Returns:
        Dict: Metrics about coalition task allocations
    """
    # Identify coalition tasks
    coalition_tasks = [task for task in tasks if hasattr(task, 'requires_coalition') and task.requires_coalition]
    
    if not coalition_tasks:
        return {
            'coalition_task_count': 0,
            'allocated_coalition_task_rate': 0.0,
            'coalition_size_avg': 0.0
        }
    
    # Check allocation status for coalition tasks
    allocated_coalition_tasks = 0
    coalition_sizes = []
    
    for task in coalition_tasks:
        # For coalition tasks, check if it's allocated to multiple robots
        # Note: This requires a modified allocation structure where task IDs can map to multiple robot IDs
        if hasattr(allocations, 'get_robot_coalition') and allocations.get_robot_coalition:
            coalition = allocations.get_robot_coalition(task.id)
            if coalition:
                allocated_coalition_tasks += 1
                coalition_sizes.append(len(coalition))
        elif task.id in allocations:
            # Simple case: just count allocated coalition tasks
            allocated_coalition_tasks += 1
    
    total_coalition_tasks = len(coalition_tasks)
    
    return {
        'coalition_task_count': total_coalition_tasks,
        'allocated_coalition_task_rate': allocated_coalition_tasks / total_coalition_tasks * 100 if total_coalition_tasks > 0 else 0.0,
        'coalition_size_avg': np.mean(coalition_sizes) if coalition_sizes else 0.0
    }

def compare_auction_mechanisms(results: Dict) -> Dict:
    """
    Compare performance metrics across different auction mechanisms.
    
    Args:
        results: Dictionary mapping auction types to result metrics
        
    Returns:
        Dict: Comparison metrics
    """
    if not results:
        return {}
    
    # Extract key metrics for each mechanism
    comparison = {}
    
    # Welfare comparison
    welfare_values = {
        mechanism: data.get('welfare', 0.0) 
        for mechanism, data in results.items()
    }
    best_welfare_mechanism = max(welfare_values.items(), key=lambda x: x[1])[0]
    welfare_improvement = {}
    
    for mechanism, welfare in welfare_values.items():
        if mechanism != best_welfare_mechanism and welfare_values[best_welfare_mechanism] > 0:
            improvement = (welfare_values[best_welfare_mechanism] - welfare) / welfare * 100 if welfare > 0 else float('inf')
            welfare_improvement[mechanism] = improvement
    
    comparison['welfare'] = {
        'values': welfare_values,
        'best_mechanism': best_welfare_mechanism,
        'improvement': welfare_improvement
    }
    
    # Revenue comparison
    revenue_values = {
        mechanism: data.get('revenue', 0.0) 
        for mechanism, data in results.items()
    }
    best_revenue_mechanism = max(revenue_values.items(), key=lambda x: x[1])[0]
    revenue_improvement = {}
    
    for mechanism, revenue in revenue_values.items():
        if mechanism != best_revenue_mechanism and revenue_values[best_revenue_mechanism] > 0:
            improvement = (revenue_values[best_revenue_mechanism] - revenue) / revenue * 100 if revenue > 0 else float('inf')
            revenue_improvement[mechanism] = improvement
    
    comparison['revenue'] = {
        'values': revenue_values,
        'best_mechanism': best_revenue_mechanism,
        'improvement': revenue_improvement
    }
    
    # Allocation efficiency comparison
    efficiency_values = {}
    for mechanism, data in results.items():
        allocated = len(data.get('allocated_tasks', []))
        total = allocated + len(data.get('unallocated_tasks', []))
        efficiency_values[mechanism] = allocated / total * 100 if total > 0 else 0.0
    
    best_efficiency_mechanism = max(efficiency_values.items(), key=lambda x: x[1])[0]
    efficiency_improvement = {}
    
    for mechanism, efficiency in efficiency_values.items():
        if mechanism != best_efficiency_mechanism and efficiency_values[best_efficiency_mechanism] > 0:
            improvement = (efficiency_values[best_efficiency_mechanism] - efficiency) / efficiency * 100 if efficiency > 0 else float('inf')
            efficiency_improvement[mechanism] = improvement
    
    comparison['allocation_efficiency'] = {
        'values': efficiency_values,
        'best_mechanism': best_efficiency_mechanism,
        'improvement': efficiency_improvement
    }
    
    # Execution time comparison
    time_values = {
        mechanism: data.get('execution_time', 0.0) 
        for mechanism, data in results.items()
    }
    fastest_mechanism = min(time_values.items(), key=lambda x: x[1])[0]
    time_improvement = {}
    
    for mechanism, execution_time in time_values.items():
        if mechanism != fastest_mechanism and execution_time > 0:
            improvement = (execution_time - time_values[fastest_mechanism]) / execution_time * 100
            time_improvement[mechanism] = improvement
    
    comparison['execution_time'] = {
        'values': time_values,
        'best_mechanism': fastest_mechanism,
        'improvement': time_improvement
    }
    
    return comparison

def calculate_incentive_compatibility_metrics(robot_bids: Dict, robot_values: Dict) -> Dict:
    """
    Calculate metrics related to incentive compatibility (truthfulness in bidding).
    
    Args:
        robot_bids: Dictionary mapping (robot_id, task_id) to bid values
        robot_values: Dictionary mapping (robot_id, task_id) to true valuations
        
    Returns:
        Dict: Metrics about bidding truthfulness
    """
    if not robot_bids or not robot_values:
        return {
            'truthful_bidding_rate': 0.0,
            'avg_bid_to_value_ratio': 0.0,
            'strategic_bidding_stats': {
                'underbidding_rate': 0.0,
                'overbidding_rate': 0.0,
                'avg_underbidding_percentage': 0.0,
                'avg_overbidding_percentage': 0.0
            }
        }
    
    # Count truthful vs. strategic bidding
    truthful_bids = 0
    underbidding_count = 0
    overbidding_count = 0
    underbidding_percentages = []
    overbidding_percentages = []
    bid_to_value_ratios = []
    
    for key, bid in robot_bids.items():
        if key in robot_values:
            true_value = robot_values[key]
            bid_to_value_ratio = bid / true_value if true_value > 0 else 0
            bid_to_value_ratios.append(bid_to_value_ratio)
            
            # Check if bid is truthful (allowing for small numerical differences)
            if abs(bid - true_value) / max(1e-6, true_value) < 0.01:  # Within 1% is considered truthful
                truthful_bids += 1
            elif bid < true_value:
                underbidding_count += 1
                underbidding_percentages.append((true_value - bid) / true_value * 100)
            else:  # bid > true_value
                overbidding_count += 1
                overbidding_percentages.append((bid - true_value) / true_value * 100)
    
    total_bids = len(robot_bids)
    
    return {
        'truthful_bidding_rate': truthful_bids / total_bids * 100 if total_bids > 0 else 0.0,
        'avg_bid_to_value_ratio': np.mean(bid_to_value_ratios) if bid_to_value_ratios else 0.0,
        'strategic_bidding_stats': {
            'underbidding_rate': underbidding_count / total_bids * 100 if total_bids > 0 else 0.0,
            'overbidding_rate': overbidding_count / total_bids * 100 if total_bids > 0 else 0.0,
            'avg_underbidding_percentage': np.mean(underbidding_percentages) if underbidding_percentages else 0.0,
            'avg_overbidding_percentage': np.mean(overbidding_percentages) if overbidding_percentages else 0.0
        }
    }

def calculate_spatial_distribution_metrics(tasks: List, allocations: Dict) -> Dict:
    """
    Calculate metrics related to spatial distribution of tasks and allocations.
    
    Args:
        tasks: List of task objects with position attributes
        allocations: Dictionary mapping task IDs to robot IDs
        
    Returns:
        Dict: Metrics about spatial distribution
    """
    if not tasks:
        return {
            'allocated_task_density': {},
            'unallocated_task_density': {},
            'spatial_allocation_bias': 0.0
        }
    
    # Create grid of task locations (10x10 grid)
    grid_size = 10
    allocated_grid = np.zeros((grid_size, grid_size))
    unallocated_grid = np.zeros((grid_size, grid_size))
    
    # Normalize task positions to grid
    max_x = max(task.position[0] for task in tasks)
    max_y = max(task.position[1] for task in tasks)
    
    for task in tasks:
        # Normalize position to grid coordinates
        grid_x = min(int(task.position[0] / (max_x + 1) * grid_size), grid_size - 1)
        grid_y = min(int(task.position[1] / (max_y + 1) * grid_size), grid_size - 1)
        
        if task.id in allocations:
            allocated_grid[grid_y, grid_x] += 1
        else:
            unallocated_grid[grid_y, grid_x] += 1
    
    # Calculate allocation density by grid cell
    allocated_density = {}
    unallocated_density = {}
    
    for y in range(grid_size):
        for x in range(grid_size):
            cell_key = f"({x},{y})"
            allocated_density[cell_key] = allocated_grid[y, x]
            unallocated_density[cell_key] = unallocated_grid[y, x]
    
    # Calculate spatial allocation bias (measures if allocations favor certain areas)
    # Bias is measured as coefficient of variation across grid cells
    total_allocations = np.sum(allocated_grid)
    total_tasks = np.sum(allocated_grid + unallocated_grid)
    
    if total_tasks > 0:
        # Calculate expected allocations per cell if uniform
        expected_allocations = total_allocations / (grid_size * grid_size)
        
        # Calculate variance of allocation density
        allocation_variance = 0
        for y in range(grid_size):
            for x in range(grid_size):
                total_cell_tasks = allocated_grid[y, x] + unallocated_grid[y, x]
                if total_cell_tasks > 0:
                    expected_cell_allocations = total_cell_tasks * (total_allocations / total_tasks)
                    allocation_variance += (allocated_grid[y, x] - expected_cell_allocations) ** 2
        
        # Normalize variance
        spatial_bias = np.sqrt(allocation_variance / (grid_size * grid_size)) / (expected_allocations + 1e-10)
    else:
        spatial_bias = 0.0
    
    return {
        'allocated_task_density': allocated_density,
        'unallocated_task_density': unallocated_density,
        'spatial_allocation_bias': spatial_bias
    }

def generate_comprehensive_metrics_report(tasks: List, robots: List, allocations: Dict, 
                                         auction_results: Dict, execution_times: List,
                                         all_auction_results: Dict = None) -> Dict:
    """
    Generate a comprehensive report of all metrics.
    
    Args:
        tasks: List of task objects
        robots: List of robot objects
        allocations: Dictionary mapping task IDs to robot IDs
        auction_results: Dictionary containing auction metrics
        execution_times: List of execution times
        all_auction_results: Dictionary mapping auction types to results
        
    Returns:
        Dict: Comprehensive metrics report
    """
    # List of allocated and available tasks
    allocated_tasks = [task for task in tasks if task.id in allocations]
    available_tasks = [task for task in tasks if not task.completed and not task.failed]
    
    # Calculate all metrics
    report = {
        'basic_metrics': {
            'allocation_efficiency': calculate_allocation_efficiency(allocated_tasks, available_tasks),
            'social_welfare': auction_results.get('welfare', 0.0),
            'revenue': auction_results.get('revenue', 0.0),
            'revenue_to_welfare_ratio': calculate_revenue_to_welfare_ratio(
                auction_results.get('revenue', 0.0), 
                auction_results.get('welfare', 0.0)
            ),
            'execution_time': calculate_execution_time(execution_times)
        },
        'task_metrics': calculate_task_completion_metrics(tasks),
        'robot_metrics': calculate_robot_performance(robots, tasks, allocations),
        'bundle_metrics': calculate_bundle_metrics(tasks, allocations),
        'coalition_metrics': calculate_coalition_task_metrics(tasks, allocations, robots),
        'spatial_metrics': calculate_spatial_distribution_metrics(tasks, allocations)
    }
    
    # Add auction comparison if available
    if all_auction_results:
        report['auction_comparison'] = compare_auction_mechanisms(all_auction_results)
    
    return report

class MetricsTracker:
    """
    Tracks and analyzes performance metrics for task allocation simulations.
    
    This class maintains time series data for various metrics and provides
    methods for computing summary statistics and visualizations.
    """
    
    def __init__(self):
        """Initialize an empty metrics tracker."""
        # Time series metrics
        self.metrics_history = defaultdict(list)
        
        # Robot-specific metrics
        self.robot_metrics = defaultdict(lambda: defaultdict(list))
        
        # Snapshot of current metrics
        self.current_metrics = {}
        
    def reset(self):
        """Clear all metrics."""
        self.metrics_history.clear()
        self.robot_metrics.clear()
        self.current_metrics.clear()
        
    def update_metric(self, metric_name: str, value: float):
        """
        Update a global metric with a new value.
        
        Args:
            metric_name: The name of the metric to update
            value: The new value
        """
        self.metrics_history[metric_name].append(value)
        self.current_metrics[metric_name] = value
        
    def update_robot_metric(self, robot_id: int, metric_name: str, value: float):
        """
        Update a robot-specific metric with a new value.
        
        Args:
            robot_id: The ID of the robot
            metric_name: The name of the metric to update
            value: The new value
        """
        self.robot_metrics[robot_id][metric_name].append(value)
        
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get the history of values for a specific metric.
        
        Args:
            metric_name: The name of the metric
            
        Returns:
            List[float]: The history of values
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_robot_metric_history(self, robot_id: int, metric_name: str) -> List[float]:
        """
        Get the history of values for a robot-specific metric.
        
        Args:
            robot_id: The ID of the robot
            metric_name: The name of the metric
            
        Returns:
            List[float]: The history of values
        """
        return self.robot_metrics.get(robot_id, {}).get(metric_name, [])
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get the current values of all metrics.
        
        Returns:
            Dict[str, float]: The current metrics
        """
        return self.current_metrics.copy()
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for all metrics.
        
        Returns:
            Dict: Summary statistics (mean, min, max, etc.) for all metrics
        """
        summary = {}
        
        # Process global metrics
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[metric_name] = values[-1]  # Last value
                summary[f"avg_{metric_name}"] = sum(values) / len(values)
                summary[f"min_{metric_name}"] = min(values)
                summary[f"max_{metric_name}"] = max(values)
        
        # Process robot-specific metrics
        robot_summary = {}
        for robot_id, metrics in self.robot_metrics.items():
            robot_summary[robot_id] = {}
            for metric_name, values in metrics.items():
                if values:
                    robot_summary[robot_id][metric_name] = values[-1]  # Last value
        
        summary["robot_metrics"] = robot_summary
        
        return summary
    
    def plot_metrics(self, metric_names: List[str] = None) -> plt.Figure:
        """
        Create a plot of the specified metrics over time.
        
        Args:
            metric_names: Names of metrics to plot (if None, plot all)
            
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        if metric_names is None:
            metric_names = list(self.metrics_history.keys())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name in metric_names:
            values = self.metrics_history.get(metric_name, [])
            if values:
                ax.plot(range(len(values)), values, label=metric_name)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Metrics Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_robot_comparison(self, metric_name: str) -> plt.Figure:
        """
        Create a comparison plot of a specific metric across all robots.
        
        Args:
            metric_name: Name of the metric to compare
            
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        # Get all robot IDs
        robot_ids = sorted(self.robot_metrics.keys())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot final value for each robot
        values = []
        for robot_id in robot_ids:
            robot_values = self.robot_metrics.get(robot_id, {}).get(metric_name, [])
            if robot_values:
                values.append(robot_values[-1])
            else:
                values.append(0)
        
        ax.bar([str(r) for r in robot_ids], values)
        ax.set_xlabel('Robot ID')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} by Robot')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def calculate_pareto_efficiency(self, 
                                   welfare_metric: str = 'total_welfare', 
                                   completion_metric: str = 'task_completion_rate') -> float:
        """
        Calculate the Pareto efficiency of the current allocation.
        
        Args:
            welfare_metric: Metric representing social welfare
            completion_metric: Metric representing task completion rate
            
        Returns:
            float: Pareto efficiency score (0-1)
        """
        welfare = self.current_metrics.get(welfare_metric, 0)
        completion = self.current_metrics.get(completion_metric, 0)
        
        # Simplified Pareto efficiency: product of normalized welfare and completion
        return (welfare * completion) ** 0.5  # Geometric mean
    
    def create_radar_chart(self, config_name: str) -> plt.Figure:
        """
        Create a radar chart showing multiple metrics for a configuration.
        
        Args:
            config_name: Name of the configuration for the chart title
            
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        # Metrics to include in the radar chart
        metrics = [
            'allocation_efficiency',
            'task_completion_rate', 
            'avg_utility_per_task',
            'total_welfare'
        ]
        
        # Get values
        values = []
        for metric in metrics:
            metric_values = self.metrics_history.get(metric, [])
            if metric_values:
                values.append(metric_values[-1])
            else:
                values.append(0)
        
        # Normalize values to [0, 1] range
        max_values = {
            'allocation_efficiency': 1.0,
            'task_completion_rate': 1.0,
            'avg_utility_per_task': max(5.0, max(self.metrics_history.get('avg_utility_per_task', [1.0]))),
            'total_welfare': max(50.0, max(self.metrics_history.get('total_welfare', [1.0])))
        }
        
        normalized_values = [values[i] / max_values.get(metrics[i], 1.0) for i in range(len(metrics))]
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set number of axes
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add the values (also close the loop)
        normalized_values += normalized_values[:1]
        
        # Plot
        ax.plot(angles, normalized_values, 'o-', linewidth=2)
        ax.fill(angles, normalized_values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Add title
        ax.set_title(f"Performance Metrics for {config_name}")
        
        return fig


def plot_metrics_report(report: Dict) -> Dict[str, plt.Figure]:
    """
    Generate visualization plots from a metrics report.
    
    Args:
        report: Comprehensive metrics report dictionary
        
    Returns:
        Dict[str, plt.Figure]: Dictionary mapping plot names to figures
    """
    plots = {}
    
    # 1. Basic metrics bar chart
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    
    basic_metrics = report['basic_metrics']
    metrics = ['allocation_efficiency', 'revenue_to_welfare_ratio']
    values = [basic_metrics['allocation_efficiency'], basic_metrics['revenue_to_welfare_ratio']]
    
    ax1.bar(metrics, values)
    ax1.set_ylabel('Percentage / Ratio')
    ax1.set_title('Basic Performance Metrics')
    ax1.grid(True, alpha=0.3)
    
    plots['basic_metrics'] = fig1
    
    # 2. Task completion status pie chart
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111)
    
    task_metrics = report['task_metrics']
    labels = ['Assigned', 'Completed', 'Failed', 'Pending']
    sizes = [
        task_metrics['assigned_rate'],
        task_metrics['completed_rate'],
        task_metrics['failed_rate'],
        task_metrics['pending_rate']
    ]
    
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Task Completion Status')
    
    plots['task_status'] = fig2
    
    # 3. Robot performance comparison
    fig3 = plt.figure(figsize=(12, 6))
    ax3 = fig3.add_subplot(111)
    
    robot_metrics = report['robot_metrics']
    robot_ids = list(robot_metrics.keys())
    
    if robot_ids:
        # Group robots by strategy type
        strategy_types = set(metrics['strategy_type'] for metrics in robot_metrics.values())
        
        # Create grouped bar chart
        x = np.arange(len(strategy_types))
        width = 0.35
        
        # Count robots and tasks by strategy
        robots_by_strategy = {strategy: 0 for strategy in strategy_types}
        tasks_by_strategy = {strategy: 0 for strategy in strategy_types}
        completion_by_strategy = {strategy: 0 for strategy in strategy_types}
        
        for robot_id, metrics in robot_metrics.items():
            strategy = metrics['strategy_type']
            robots_by_strategy[strategy] += 1
            tasks_by_strategy[strategy] += metrics['task_count']
            completion_by_strategy[strategy] += metrics['completion_rate'] * metrics['task_count'] / 100 if metrics['task_count'] > 0 else 0
        
        # Calculate average completion rate
        avg_completion_by_strategy = {}
        for strategy, tasks in tasks_by_strategy.items():
            avg_completion_by_strategy[strategy] = completion_by_strategy[strategy] / tasks if tasks > 0 else 0
            
        # Convert to lists for plotting
        strategies = list(strategy_types)
        robot_counts = [robots_by_strategy[s] for s in strategies]
        task_counts = [tasks_by_strategy[s] for s in strategies]
        completion_rates = [avg_completion_by_strategy[s] * 100 for s in strategies]
        
        # First set of bars: robot counts
        ax3.bar(x - width/3, robot_counts, width/3, label='Robot Count')
        # Second set of bars: task counts
        ax3.bar(x, task_counts, width/3, label='Task Count')
        # Third set of bars: completion rates
        ax3.bar(x + width/3, completion_rates, width/3, label='Avg Completion Rate (%)')
        
        ax3.set_xlabel('Robot Strategy Type')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Robot Performance by Strategy Type')
    
    plots['robot_performance'] = fig3
    
    # 4. Bundle allocation performance
    if report['bundle_metrics']['bundle_count'] > 0:
        fig4 = plt.figure(figsize=(8, 8))
        ax4 = fig4.add_subplot(111)
        
        bundle_metrics = report['bundle_metrics']
        labels = ['Complete Bundle Allocation', 'Partial Bundle Allocation', 'Unallocated Bundle']
        sizes = [
            bundle_metrics['complete_bundle_allocation_rate'],
            bundle_metrics['partial_bundle_allocation_rate'],
            bundle_metrics['unallocated_bundle_rate']
        ]
        
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        ax4.set_title(f'Bundle Allocation Performance (Total: {bundle_metrics["bundle_count"]} bundles)')
        
        plots['bundle_performance'] = fig4
    
    # 5. Spatial allocation heatmap
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111)
    
    spatial_metrics = report['spatial_metrics']
    allocated_density = spatial_metrics['allocated_task_density']
    unallocated_density = spatial_metrics['unallocated_task_density']
    
    # Convert dictionary to 2D grid
    grid_size = int(np.sqrt(len(allocated_density)))
    if grid_size > 0:
        allocation_ratio_grid = np.zeros((grid_size, grid_size))
        
        for y in range(grid_size):
            for x in range(grid_size):
                cell_key = f"({x},{y})"
                allocated = allocated_density.get(cell_key, 0)
                unallocated = unallocated_density.get(cell_key, 0)
                total = allocated + unallocated
                allocation_ratio_grid[y, x] = allocated / total if total > 0 else 0
        
        # Plot heatmap
        im = ax5.imshow(allocation_ratio_grid, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = fig5.colorbar(im, ax=ax5)
        cbar.set_label('Allocation Ratio')
        
        # Add grid lines
        ax5.grid(which='major', color='white', linestyle='-', linewidth=1, alpha=0.5)
        
        # Set ticks
        ax5.set_xticks(np.arange(grid_size))
        ax5.set_yticks(np.arange(grid_size))
        
        ax5.set_title(f'Spatial Allocation Distribution (Bias: {spatial_metrics["spatial_allocation_bias"]:.2f})')
    
    plots['spatial_distribution'] = fig5
    
    # 6. Auction comparison (if available)
    if 'auction_comparison' in report:
        fig6 = plt.figure(figsize=(12, 8))
        
        # Create 2x2 grid of subplots
        ax1 = fig6.add_subplot(2, 2, 1)
        ax2 = fig6.add_subplot(2, 2, 2)
        ax3 = fig6.add_subplot(2, 2, 3)
        ax4 = fig6.add_subplot(2, 2, 4)
        
        # Extract data
        auction_comparison = report['auction_comparison']
        mechanisms = list(auction_comparison['welfare']['values'].keys())
        
        # Plot welfare comparison
        welfare_values = [auction_comparison['welfare']['values'][m] for m in mechanisms]
        ax1.bar(mechanisms, welfare_values)
        ax1.set_title('Social Welfare')
        ax1.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot revenue comparison
        revenue_values = [auction_comparison['revenue']['values'][m] for m in mechanisms]
        ax2.bar(mechanisms, revenue_values)
        ax2.set_title('Revenue')
        ax2.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot allocation efficiency comparison
        efficiency_values = [auction_comparison['allocation_efficiency']['values'][m] for m in mechanisms]
        ax3.bar(mechanisms, efficiency_values)
        ax3.set_title('Allocation Efficiency (%)')
        ax3.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot execution time comparison
        time_values = [auction_comparison['execution_time']['values'][m] for m in mechanisms]
        ax4.bar(mechanisms, time_values)
        ax4.set_title('Execution Time (s)')
        ax4.set_xticklabels(mechanisms, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        fig6.tight_layout()
        plots['auction_comparison'] = fig6
    
    return plots


# Example usage if run directly
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import classes
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import classes for demonstration
    from task_manager import Task, TaskManager
    from bidding_robot import BiddingRobot
    
    # Create synthetic data for demonstration
    # Create task manager
    task_manager = TaskManager(
        environment_size=(20, 20),
        task_complexity_range=(1.0, 3.0)
    )
    
    # Generate tasks
    tasks = task_manager.generate_task_batch(10, include_complex=True)
    
    # Create robots
    robots = [
        BiddingRobot(0, {"speed": 1.5, "lift_capacity": 2.0}, (5, 5), "truthful"),
        BiddingRobot(1, {"speed": 1.0, "precision": 1.8}, (10, 10), "strategic"),
        BiddingRobot(2, {"lift_capacity": 1.2, "sensor_range": 2.0}, (15, 15), "cooperative")
    ]
    
    # Create example allocations
    allocations = {
        tasks[0].id: robots[0].id,
        tasks[1].id: robots[0].id,
        tasks[2].id: robots[1].id,
        tasks[3].id: robots[2].id
    }
    
    # Mark some tasks as completed or failed
    tasks[0].assigned = True
    tasks[1].assigned = True
    tasks[2].assigned = True
    tasks[3].assigned = True
    tasks[4].completed = True
    tasks[5].failed = True
    
    # Create example auction results
    auction_results = {
        'welfare': 25.5,
        'revenue': 22.0,
        'execution_time': 0.05
    }
    
    # Create example execution times
    execution_times = [0.05, 0.04, 0.06]
    
    # Create example results for different auction types
    all_auction_results = {
        'sequential': {
            'welfare': 25.5,
            'revenue': 22.0,
            'execution_time': 0.05,
            'allocated_tasks': [tasks[0], tasks[1], tasks[2], tasks[3]],
            'unallocated_tasks': [task for task in tasks if task.id not in allocations and not task.completed and not task.failed]
        },
        'combinatorial': {
            'welfare': 28.0,
            'revenue': 25.0,
            'execution_time': 0.12,
            'allocated_tasks': [tasks[0], tasks[1], tasks[2], tasks[3], tasks[6]],
            'unallocated_tasks': [task for task in tasks if task.id not in [0, 1, 2, 3, 6] and not task.completed and not task.failed]
        }
    }
    
    # Generate metrics report
    report = generate_comprehensive_metrics_report(
        tasks, robots, allocations, auction_results, execution_times, all_auction_results
    )
    
    # Plot metrics report
    plots = plot_metrics_report(report)
    
    # Save plots
    for name, fig in plots.items():
        fig.savefig(f"{name}_example.png")
        plt.close(fig)
    
    print("Metrics examples created successfully.")
    
    # Demonstrate the MetricsTracker class
    tracker = MetricsTracker()
    
    # Add some sample metrics
    for i in range(10):
        tracker.update_metric('allocation_efficiency', 0.5 + i * 0.05)
        tracker.update_metric('task_completion_rate', 0.4 + i * 0.06)
        tracker.update_metric('total_welfare', 10 + i * 2)
        
        # Add robot metrics
        for robot in robots:
            tracker.update_robot_metric(robot.id, 'energy', 100 - i * 5)
            tracker.update_robot_metric(robot.id, 'tasks_completed', i // 2)
    
    # Get summary metrics
    summary = tracker.get_summary_metrics()
    print("\nMetricsTracker Summary:")
    for key, value in summary.items():
        if key != "robot_metrics":
            print(f"  {key}: {value}")
    
    # Create some example plots
    fig1 = tracker.plot_metrics(['allocation_efficiency', 'task_completion_rate'])
    fig1.savefig("metrics_time_series.png")
    
    fig2 = tracker.plot_robot_comparison('energy')
    fig2.savefig("robot_energy_comparison.png")
    
    fig3 = tracker.create_radar_chart("Sequential Auction")
    fig3.savefig("radar_chart.png")
    
    print("MetricsTracker examples created successfully.")