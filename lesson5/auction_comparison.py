#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auction Comparison script for multi-robot task allocation.

This script runs different auction mechanisms and payment rules in parallel
and compares their performance for task allocation in multi-robot systems.
"""

import matplotlib.pyplot as plt
import argparse
import time
import random
import numpy as np
from typing import Dict, List, Tuple

from task_allocation import TaskAllocationEnv
from utils.metrics import MetricsTracker, plot_metrics_report, generate_comprehensive_metrics_report
from utils.visualization import visualize_auction_comparison, visualize_allocation_results

def run_comparison(args):
    """
    Run and compare different auction mechanisms and payment rules.
    
    Args:
        args: Command-line arguments
    """
    # Auction types to compare
    auction_types = ["sequential", "parallel", "combinatorial"]
    
    # Payment rules to compare
    payment_rules = ["first_price", "second_price", "vcg"]
    
    # Skip combinations that don't work well together
    skip_combinations = [
        ("combinatorial", "second_price")  # Problematic combination
    ]
    
    # Common parameters
    common_params = {
        "n_robots": args.num_robots,
        "n_tasks": args.num_tasks,
        "grid_size": args.grid_size,
        "dynamic_tasks": args.dynamic_tasks,
        "render_mode": None,  # Disable rendering for comparison
        "max_steps": args.max_steps,
        "verbose": False  # Disable verbose for comparison runs
    }
    
    # Store results
    results = {}
    all_metrics = {}
    task_completions = {}
    allocation_efficiencies = {}
    
    print(f"\n=== Comparing Auction Mechanisms ===")
    print(f"Robots: {args.num_robots}, Tasks: {args.num_tasks}, Steps: {args.max_steps}")
    
    # Run each auction type with each payment rule
    for auction_type in auction_types:
        for payment_rule in payment_rules:
            # Skip invalid combinations
            if (auction_type, payment_rule) in skip_combinations:
                print(f"Skipping {auction_type} with {payment_rule} (incompatible)")
                continue
            
            config_name = f"{auction_type}_{payment_rule}"
            print(f"\nRunning {config_name}...")
            
            # Create environment with this configuration
            env = TaskAllocationEnv(
                auction_type=auction_type,
                payment_rule=payment_rule,
                **common_params
            )
            
            # Run simulation
            observation = env.reset()
            done = False
            step_metrics = []
            
            start_time = time.time()
            while not done:
                observation, rewards, done, info = env.step()
                step_metrics.append(info['metrics'])
            
            run_time = time.time() - start_time
            
            # Get final results
            final_metrics = env.metrics.get_summary_metrics()
            allocated_tasks = env.task_manager.get_assigned_tasks()
            completed_tasks = env.task_manager.get_completed_tasks()
            failed_tasks = env.task_manager.get_failed_tasks()
            
            # Store results 
            results[config_name] = {
                "metrics": final_metrics,
                "run_time": run_time,
                "allocated_tasks": len(allocated_tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "step_metrics": step_metrics
            }
            
            # Track metrics for plotting
            all_metrics[config_name] = final_metrics
            task_completions[config_name] = len(completed_tasks)
            allocation_efficiencies[config_name] = final_metrics.get('allocation_efficiency', 0)
            
            # Print summary
            print(f"  Execution time: {run_time:.2f} seconds")
            print(f"  Completed tasks: {len(completed_tasks)}/{args.num_tasks}")
            print(f"  Allocation efficiency: {final_metrics.get('allocation_efficiency', 0):.2f}")
            print(f"  Task completion rate: {final_metrics.get('task_completion_rate', 0):.2f}")
            
            # Close the environment
            env.close()
    
    # Create comparison visualizations
    create_comparison_plots(results, args.output_dir)
    
    return results

def create_comparison_plots(results: Dict, output_dir: str = "."):
    """
    Create comparison plots for different auction mechanisms.
    
    Args:
        results: Results from different auction mechanisms
        output_dir: Directory to save plots
    """
    # Extract key metrics
    configs = list(results.keys())
    
    # Performance metrics
    metrics_to_plot = [
        "allocation_efficiency", 
        "task_completion_rate", 
        "avg_utility_per_task",
        "robot_metrics"
    ]
    
    # Create bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot allocation efficiency
    efficiencies = [results[config]["metrics"].get("allocation_efficiency", 0) for config in configs]
    axes[0].bar(configs, efficiencies)
    axes[0].set_title("Allocation Efficiency")
    axes[0].set_xticklabels(configs, rotation=45, ha="right")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot task completion rates
    completion_rates = [results[config]["metrics"].get("task_completion_rate", 0) for config in configs]
    axes[1].bar(configs, completion_rates)
    axes[1].set_title("Task Completion Rate")
    axes[1].set_xticklabels(configs, rotation=45, ha="right")
    axes[1].set_ylabel("Completion Rate (%)")
    axes[1].grid(True, alpha=0.3)
    
    # Plot utility per task
    utilities = [results[config]["metrics"].get("avg_utility_per_task", 0) for config in configs]
    axes[2].bar(configs, utilities)
    axes[2].set_title("Average Utility per Task")
    axes[2].set_xticklabels(configs, rotation=45, ha="right")
    axes[2].set_ylabel("Utility")
    axes[2].grid(True, alpha=0.3)
    
    # Plot execution time
    exec_times = [results[config]["run_time"] for config in configs]
    axes[3].bar(configs, exec_times)
    axes[3].set_title("Execution Time")
    axes[3].set_xticklabels(configs, rotation=45, ha="right")
    axes[3].set_ylabel("Time (seconds)")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/auction_comparison_metrics.png")
    
    # Create line chart for metrics over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    # Group step metrics by auction type
    for config in configs:
        step_metrics = results[config]["step_metrics"]
        steps = range(len(step_metrics))
        
        # Allocation efficiency over time
        if step_metrics and "allocation_efficiency" in step_metrics[0]:
            efficiency_values = [metrics.get("allocation_efficiency", 0) for metrics in step_metrics]
            axes[0].plot(steps, efficiency_values, label=config, marker='o', markersize=4, alpha=0.7)
        
        # Task completion over time
        if step_metrics and "task_completion_rate" in step_metrics[0]:
            completion_values = [metrics.get("task_completion_rate", 0) for metrics in step_metrics]
            axes[1].plot(steps, completion_values, label=config, marker='o', markersize=4, alpha=0.7)
        
        # Welfare over time
        if step_metrics and "total_welfare" in step_metrics[0]:
            welfare_values = [metrics.get("total_welfare", 0) for metrics in step_metrics]
            axes[2].plot(steps, welfare_values, label=config, marker='o', markersize=4, alpha=0.7)
    
    axes[0].set_title("Allocation Efficiency Over Time")
    axes[0].set_ylabel("Efficiency (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_title("Task Completion Rate Over Time")
    axes[1].set_ylabel("Completion Rate (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].set_title("Social Welfare Over Time")
    axes[2].set_ylabel("Welfare")
    axes[2].set_xlabel("Simulation Step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/auction_comparison_over_time.png")
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Define metrics for radar chart
    radar_metrics = ["allocation_efficiency", "task_completion_rate", "avg_utility_per_task", "run_time"]
    
    # Get max values for normalization
    max_values = {
        "allocation_efficiency": 100,
        "task_completion_rate": 100,
        "avg_utility_per_task": max(results[config]["metrics"].get("avg_utility_per_task", 0) for config in configs) or 1,
        "run_time": max(results[config]["run_time"] for config in configs) or 1
    }
    
    # Set number of metrics
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    
    # Draw the data for each auction type
    for config in configs:
        values = []
        for metric in radar_metrics:
            if metric == "run_time":
                # Invert run time (faster is better)
                value = 1 - (results[config]["run_time"] / max_values[metric])
            else:
                value = results[config]["metrics"].get(metric, 0) / max_values[metric]
            values.append(value)
        
        # Close the loop
        values += values[:1]
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=config)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_title("Auction Mechanism Comparison")
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(f"{output_dir}/auction_comparison_radar.png")
    plt.close()
    
    print(f"\nComparison plots saved to {output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare different auction mechanisms")
    
    parser.add_argument("--num_robots", type=int, default=5,
                        help="Number of robots in the simulation")
    parser.add_argument("--num_tasks", type=int, default=20,
                        help="Number of initial tasks to generate")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Size of the environment grid")
    parser.add_argument("--dynamic_tasks", type=bool, default=True,
                        help="Whether new tasks arrive during simulation")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum simulation steps")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output files")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    import os
    if args.output_dir == ".":
        args.output_dir = "results"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Run comparison
    results = run_comparison(args)
    
    # Print summary of best mechanism for different metrics
    print("\n=== Summary of Best Mechanisms ===")
    
    # Find best for allocation efficiency
    best_alloc = max(results.items(), key=lambda x: x[1]["metrics"].get("allocation_efficiency", 0))
    print(f"Best for allocation efficiency: {best_alloc[0]} ({best_alloc[1]['metrics'].get('allocation_efficiency', 0):.2f}%)")
    
    # Find best for task completion
    best_completion = max(results.items(), key=lambda x: x[1]["metrics"].get("task_completion_rate", 0))
    print(f"Best for task completion: {best_completion[0]} ({best_completion[1]['metrics'].get('task_completion_rate', 0):.2f}%)")
    
    # Find best for utility per task
    best_utility = max(results.items(), key=lambda x: x[1]["metrics"].get("avg_utility_per_task", 0))
    print(f"Best for utility per task: {best_utility[0]} ({best_utility[1]['metrics'].get('avg_utility_per_task', 0):.2f})")
    
    # Find fastest execution
    best_time = min(results.items(), key=lambda x: x[1]["run_time"])
    print(f"Fastest execution: {best_time[0]} ({best_time[1]['run_time']:.2f} seconds)")

if __name__ == "__main__":
    main()