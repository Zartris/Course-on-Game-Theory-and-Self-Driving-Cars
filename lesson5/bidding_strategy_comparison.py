#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bidding Strategy Comparison for multi-robot task allocation.

This script compares different bidding strategies in auction-based task allocation.
"""

import matplotlib.pyplot as plt
import argparse
import time
import random
import numpy as np
from typing import Dict, List, Tuple

from task_allocation import TaskAllocationEnv
from bidding_robot import BiddingRobot
from utils.metrics import MetricsTracker

def run_bidding_comparison(args):
    """
    Run and compare different robot bidding strategies.
    
    Args:
        args: Command-line arguments
    """
    # Bidding strategies to compare
    strategies = ["truthful", "strategic", "learning", "cooperative"]
    
    # Store results
    results = {}
    
    print(f"\n=== Comparing Bidding Strategies ===")
    print(f"Auction type: {args.auction_type}, Payment rule: {args.payment_rule}")
    print(f"Robots per strategy: {args.robots_per_strategy}, Tasks: {args.num_tasks}")
    
    # Configure environment
    n_robots = args.robots_per_strategy * len(strategies)
    env = TaskAllocationEnv(
        n_robots=n_robots,
        n_tasks=args.num_tasks,
        grid_size=args.grid_size,
        auction_type=args.auction_type,
        payment_rule=args.payment_rule,
        dynamic_tasks=args.dynamic_tasks,
        render_mode=None,  # Disable rendering for comparison
        max_steps=args.max_steps,
        verbose=False
    )
    
    # Ensure equal number of each strategy type
    for i, robot in enumerate(env.robots):
        strategy_index = i % len(strategies)
        robot.strategy_type = strategies[strategy_index]
    
    # Run simulation
    observation = env.reset()
    done = False
    
    # Track metrics by strategy
    robot_metrics = {strategy: [] for strategy in strategies}
    task_metrics = {strategy: [] for strategy in strategies}
    
    start_time = time.time()
    while not done:
        observation, rewards, done, info = env.step()
        
        # Collect per-strategy metrics
        for robot in env.robots:
            strategy = robot.strategy_type
            
            # Track task count, completion rate, and utility
            tasks_completed = len(robot.task_history)
            utility = sum(robot.utility_history) if robot.utility_history else 0
            
            robot_metrics[strategy].append({
                'id': robot.id,
                'tasks_completed': tasks_completed,
                'utility': utility,
                'energy': robot.energy
            })
    
    run_time = time.time() - start_time
    
    # Aggregate results per strategy
    for strategy in strategies:
        # Get robots of this strategy
        strategy_robots = [r for r in env.robots if r.strategy_type == strategy]
        
        # Calculate average tasks completed
        avg_tasks = sum(len(r.task_history) for r in strategy_robots) / len(strategy_robots)
        
        # Calculate average utility earned
        avg_utility = sum(sum(r.utility_history) if r.utility_history else 0 for r in strategy_robots) / len(strategy_robots)
        
        # Calculate average remaining energy
        avg_energy = sum(r.energy for r in strategy_robots) / len(strategy_robots)
        
        # Calculate per-task utility (efficiency)
        task_count = sum(len(r.task_history) for r in strategy_robots)
        total_utility = sum(sum(r.utility_history) if r.utility_history else 0 for r in strategy_robots)
        per_task_utility = total_utility / task_count if task_count > 0 else 0
        
        # Store results
        results[strategy] = {
            'avg_tasks_completed': avg_tasks,
            'avg_utility': avg_utility,
            'avg_energy': avg_energy,
            'per_task_utility': per_task_utility,
            'robot_count': len(strategy_robots),
            'task_count': task_count
        }
        
        # Print summary
        print(f"\n{strategy.capitalize()} strategy:")
        print(f"  Average tasks completed: {avg_tasks:.2f}")
        print(f"  Average utility earned: {avg_utility:.2f}")
        print(f"  Average remaining energy: {avg_energy:.2f}%")
        print(f"  Utility per task: {per_task_utility:.2f}")
    
    # Create comparison visualizations
    create_strategy_comparison_plots(results, args.auction_type, args.payment_rule, args.output_dir)
    
    return results

def create_strategy_comparison_plots(results: Dict, auction_type: str, payment_rule: str, output_dir: str = "."):
    """
    Create comparison plots for different bidding strategies.
    
    Args:
        results: Results from different strategies
        auction_type: Type of auction used
        payment_rule: Payment rule used
        output_dir: Directory to save plots
    """
    # Extract strategies
    strategies = list(results.keys())
    
    # Create bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot average tasks completed
    tasks_values = [results[strategy]['avg_tasks_completed'] for strategy in strategies]
    axes[0].bar(strategies, tasks_values)
    axes[0].set_title("Average Tasks Completed")
    axes[0].set_ylabel("Tasks")
    axes[0].grid(True, alpha=0.3)
    
    # Plot average utility earned
    utility_values = [results[strategy]['avg_utility'] for strategy in strategies]
    axes[1].bar(strategies, utility_values)
    axes[1].set_title("Average Utility Earned")
    axes[1].set_ylabel("Utility")
    axes[1].grid(True, alpha=0.3)
    
    # Plot average remaining energy
    energy_values = [results[strategy]['avg_energy'] for strategy in strategies]
    axes[2].bar(strategies, energy_values)
    axes[2].set_title("Average Remaining Energy")
    axes[2].set_ylabel("Energy (%)")
    axes[2].grid(True, alpha=0.3)
    
    # Plot utility per task
    per_task_values = [results[strategy]['per_task_utility'] for strategy in strategies]
    axes[3].bar(strategies, per_task_values)
    axes[3].set_title("Utility per Task")
    axes[3].set_ylabel("Utility")
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f"Bidding Strategy Comparison ({auction_type.capitalize()} auction, {payment_rule} payment)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bidding_strategy_comparison_{auction_type}_{payment_rule}.png")
    
    # Create spider chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Define metrics for radar chart
    radar_metrics = ["avg_tasks_completed", "avg_utility", "avg_energy", "per_task_utility"]
    metric_labels = ["Tasks Completed", "Utility Earned", "Remaining Energy", "Utility per Task"]
    
    # Get max values for normalization
    max_values = {
        "avg_tasks_completed": max(results[strategy]['avg_tasks_completed'] for strategy in strategies) or 1,
        "avg_utility": max(results[strategy]['avg_utility'] for strategy in strategies) or 1,
        "avg_energy": 100,  # Energy is already a percentage
        "per_task_utility": max(results[strategy]['per_task_utility'] for strategy in strategies) or 1
    }
    
    # Set number of metrics
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    
    # Draw the data for each strategy
    for strategy in strategies:
        values = []
        for metric in radar_metrics:
            value = results[strategy][metric] / max_values[metric]
            values.append(value)
        
        # Close the loop
        values += values[:1]
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy.capitalize())
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_title(f"Bidding Strategy Comparison\n({auction_type.capitalize()} auction, {payment_rule} payment)")
    ax.legend(loc='upper right')
    
    plt.savefig(f"{output_dir}/bidding_strategy_radar_{auction_type}_{payment_rule}.png")
    plt.close()
    
    print(f"\nComparison plots saved to {output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare different bidding strategies")
    
    parser.add_argument("--robots_per_strategy", type=int, default=3,
                        help="Number of robots per strategy type")
    parser.add_argument("--num_tasks", type=int, default=20,
                        help="Number of initial tasks to generate")
    parser.add_argument("--grid_size", type=int, default=20,
                        help="Size of the environment grid")
    parser.add_argument("--auction_type", type=str, default="sequential",
                        choices=["sequential", "parallel", "combinatorial"],
                        help="Type of auction mechanism to use")
    parser.add_argument("--payment_rule", type=str, default="first_price",
                        choices=["first_price", "second_price", "vcg"],
                        help="Payment rule for auction")
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
    results = run_bidding_comparison(args)
    
    # Print summary
    print("\n=== Summary ===")
    
    # Find best strategy for different metrics
    task_ranking = sorted(results.items(), key=lambda x: x[1]['avg_tasks_completed'], reverse=True)
    utility_ranking = sorted(results.items(), key=lambda x: x[1]['avg_utility'], reverse=True)
    efficiency_ranking = sorted(results.items(), key=lambda x: x[1]['per_task_utility'], reverse=True)
    
    print("Ranking by tasks completed:")
    for i, (strategy, metrics) in enumerate(task_ranking, 1):
        print(f"  {i}. {strategy.capitalize()}: {metrics['avg_tasks_completed']:.2f} tasks")
        
    print("\nRanking by utility earned:")
    for i, (strategy, metrics) in enumerate(utility_ranking, 1):
        print(f"  {i}. {strategy.capitalize()}: {metrics['avg_utility']:.2f}")
        
    print("\nRanking by utility per task:")
    for i, (strategy, metrics) in enumerate(efficiency_ranking, 1):
        print(f"  {i}. {strategy.capitalize()}: {metrics['per_task_utility']:.2f}")

if __name__ == "__main__":
    main()