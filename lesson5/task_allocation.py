#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task Allocation Simulation for Multi-Robot Systems using Auction Mechanisms.

This module implements a simulation environment for exploring auction-based 
task allocation in multi-robot systems, demonstrating market-based approaches
to distributed decision making in robotics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygame
import argparse
import time
import random
import math
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import os

# Import local modules
from task_manager import Task, TaskManager
from bidding_robot import BiddingRobot
from auction_mechanism import AuctionMechanism
from utils.visualization import SimulationVisualizer
from utils.metrics import MetricsTracker


class TaskAllocationEnv:
    """
    Simulation environment for task allocation using auction mechanisms.
    
    This environment simulates robots bidding on and executing tasks in a grid-based
    world, using various auction mechanisms and bidding strategies.
    """
    
    def __init__(self, 
                 n_robots: int = 5, 
                 n_tasks: int = 10, 
                 grid_size: int = 20,
                 auction_type: str = "sequential",
                 payment_rule: str = "first_price",
                 dynamic_tasks: bool = True,
                 render_mode: str = "pygame",
                 max_steps: int = 500,
                 verbose: bool = False):
        """
        Initialize the task allocation environment.
        
        Args:
            n_robots: Number of robots in the simulation
            n_tasks: Initial number of tasks to generate
            grid_size: Size of the environment grid (width and height)
            auction_type: Type of auction mechanism to use
            payment_rule: Payment rule for the auction
            dynamic_tasks: Whether new tasks arrive during simulation
            render_mode: Visualization mode ("pygame", "matplotlib", or None)
            max_steps: Maximum simulation steps
            verbose: Whether to print detailed debug information
        """
        self.verbose = verbose
        self.n_robots = n_robots
        self.n_tasks = n_tasks
        self.grid_size = grid_size
        self.auction_type = auction_type
        self.payment_rule = payment_rule
        self.dynamic_tasks = dynamic_tasks
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Initialize time step
        self.current_step = 0
        
        # Create task manager
        self.task_manager = TaskManager(
            environment_size=(grid_size, grid_size),
            task_complexity_range=(1.0, 5.0),
            dynamic=dynamic_tasks,
            task_arrival_rate=0.2,
            verbose=verbose
        )
        
        # Create auction mechanism
        self.auction_mechanism = AuctionMechanism(
            auction_type=auction_type,
            payment_rule=payment_rule,
            max_bundle_size=3,
            reserve_price=0.1
        )
        
        # Create robots with different strategies
        self.robots = self._create_robots()
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        
        # For visualization
        self.visualizer = None
        if render_mode:
            self.visualizer = SimulationVisualizer(
                grid_size=grid_size,
                render_mode=render_mode,
                task_manager=self.task_manager,
                robots=self.robots
            )
        
        # Initialize simulation state
        self.tasks = []
        self.reset()
        
    def _create_robots(self) -> List[BiddingRobot]:
        """Create a set of robots with various strategies and capabilities."""
        robots = []
        
        # Define possible strategies
        strategies = ["truthful", "strategic", "learning", "cooperative"]
        
        # Define possible capabilities
        capabilities = {
            "speed": (0.8, 2.0),          # Movement speed range
            "lift_capacity": (0.5, 3.0),  # Weight capacity range
            "precision": (0.7, 1.8),      # Fine manipulation range
            "sensor_range": (1.0, 3.0)    # Sensing range
        }
        
        # Create robots with diverse characteristics
        for i in range(self.n_robots):
            # Select strategy, cycling through options
            strategy = strategies[i % len(strategies)]
            
            # Generate random capabilities
            robot_capabilities = {}
            for cap, (min_val, max_val) in capabilities.items():
                # Each robot has a 70% chance of having each capability
                if random.random() < 0.7:
                    robot_capabilities[cap] = random.uniform(min_val, max_val)
            
            # Ensure each robot has at least one capability
            if not robot_capabilities:
                # Give a random capability if none was assigned
                cap = random.choice(list(capabilities.keys()))
                min_val, max_val = capabilities[cap]
                robot_capabilities[cap] = random.uniform(min_val, max_val)
            
            # Generate random starting position
            position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            
            # Create robot
            robot = BiddingRobot(
                robot_id=i,
                capabilities=robot_capabilities,
                initial_position=position,
                strategy_type=strategy,
                max_tasks=3 if strategy == "cooperative" else 2,
                verbose=self.verbose  # Pass the verbose flag from env to robots
            )
            
            robots.append(robot)
        
        return robots
    
    def reset(self):
        """Reset the environment to initial state."""
        # Reset time step
        self.current_step = 0
        
        # Generate initial tasks
        self.tasks = self.task_manager.generate_task_batch(
            self.n_tasks, 
            include_complex=True
        )
        
        # Reset robot states
        for robot in self.robots:
            # Reset position to a random location
            robot.position = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            robot.path = [robot.position]
            robot.current_tasks = set()
            robot.energy = 100.0
        
        # Reset metrics
        self.metrics.reset()
        
        # Reset visualizer if it exists
        if self.visualizer:
            self.visualizer.reset()
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, actions=None):
        """
        Execute one time step of the environment.
        
        Args:
            actions: Optional actions to override default behavior
            
        Returns:
            Tuple: (observations, rewards, done, info)
        """
        # Increment time step
        self.current_step += 1
        
        # Update tasks (handle expired tasks and create new ones)
        task_updates = self.task_manager.update_tasks(self.current_step)
        new_tasks = task_updates['new']
        expired_tasks = task_updates['expired']
        
        # Add new tasks to our tracking
        self.tasks.extend(new_tasks)
        
        # Get available tasks
        available_tasks = self.task_manager.get_available_tasks()
        
        # DEBUG
        completed_tasks = self.task_manager.get_completed_tasks()
        if self.verbose:
            print(f"Available tasks: {[t.id for t in available_tasks]}")
            print(f"Completed tasks: {[t.id for t in completed_tasks]}")
        
        # Run auction if there are available tasks
        allocation_results = None
        if available_tasks:
            # DEBUG
            if self.verbose:
                print(f"\nStep {self.current_step}: {len(available_tasks)} available tasks")
            
            # Announce tasks
            announcement = self.auction_mechanism.announce_tasks(available_tasks)
            
            # Collect bids from robots
            bids = self.auction_mechanism.collect_bids(self.robots, available_tasks, announcement)
            
            # DEBUG
            if self.verbose:
                if 'single_bids' in bids:
                    print(f"Collected {sum(len(b) for b in bids['single_bids'].values())} single bids")
                    if bids['bundle_bids']:
                        print(f"Collected {len(bids['bundle_bids'])} bundle bids")
                else:
                    print(f"Collected {sum(len(b) for b in bids.values() if b)} bids")
            
            # Determine winners
            winners = self.auction_mechanism.determine_winners(bids, available_tasks)
            
            # DEBUG
            if self.verbose:
                if 'allocations' in winners:
                    print(f"Winners determined: {len(winners['allocations'])} tasks allocated")
                    print(f"Allocations: {winners['allocations']}")
            
            # Allocate tasks to winners
            allocation_results = self.auction_mechanism.allocate_tasks(winners, available_tasks, self.robots)
            
            # DEBUG
            if self.verbose:
                print(f"Tasks allocated: {len(allocation_results['allocated_tasks'])}")
                for robot_id, tasks in allocation_results['robot_allocations'].items():
                    print(f"Robot {robot_id} allocated {len(tasks)} tasks: {[t.id for t in tasks]}")
                
            # CRITICAL FIX: Properly register tasks with the task manager
            for task_id, robot_id in winners.get('allocations', {}).items():
                if self.verbose:
                    print(f"Registering task {task_id} as assigned to robot {robot_id} with task manager")
                self.task_manager.assign_task(task_id, robot_id)
            
            # Handle coalition tasks (tasks that require multiple robots)
            if self.verbose:
                print("\nChecking for coalition tasks...")
                
            coalition_tasks = [t for t in available_tasks if t.requires_coalition]
            if coalition_tasks:
                # Find robots that can form coalitions
                for task in coalition_tasks:
                    if self.verbose:
                        print(f"Processing coalition task {task.id}")
                    
                    # Skip if the task is already allocated
                    if task.id in winners.get('allocations', {}):
                        continue
                        
                    # Try each robot as a potential coalition leader
                    for robot in self.robots:
                        # Skip if robot already has max tasks
                        if len(robot.current_tasks) >= robot.max_tasks:
                            continue
                            
                        # Try to form a coalition for this task
                        other_robots = [r for r in self.robots if r.id != robot.id]
                        coalition_info = robot.form_coalition(other_robots, task)
                        
                        if coalition_info and coalition_info['can_complete']:
                            if self.verbose:
                                print(f"Coalition formed for task {task.id} with robots {coalition_info['members']}")
                                
                            # Add task to all coalition members
                            for member_id in coalition_info['members']:
                                # Register with task manager
                                self.task_manager.assign_task(task.id, member_id)
                                
                                # Make sure each robot's current_tasks set includes this task
                                for robot in self.robots:
                                    if robot.id == member_id and task.id not in robot.current_tasks:
                                        robot.current_tasks.add(task.id)
                                        if self.verbose:
                                            print(f"Added coalition task {task.id} to robot {robot.id}'s task list")
                                
                            # We only need one successful coalition per task
                            break
            
            # Update robot bidding strategies based on auction results
            for robot in self.robots:
                if hasattr(robot, 'update_strategy'):
                    robot.update_strategy(winners)
        
        # DEBUG
        if self.verbose:
            print("\nExecuting robot tasks:")
            for robot in self.robots:
                print(f"Robot {robot.id} has {len(robot.current_tasks)} tasks: {robot.current_tasks}")
        
        # Robots execute their assigned tasks
        completed_tasks = self._execute_robot_tasks()
        if self.verbose:
            print(f"Completed tasks in this step: {completed_tasks}")
        
        # Calculate rewards (task utility minus payments)
        rewards = self._calculate_rewards()
        
        # Update metrics
        self._update_metrics(allocation_results)
        
        # Check if simulation is done
        done = self._check_done()
        
        # Gather information for return
        info = {
            'new_tasks': len(new_tasks),
            'expired_tasks': len(expired_tasks),
            'completed_tasks': len(self.task_manager.completed_tasks),
            'allocation_results': allocation_results,
            'metrics': self.metrics.get_current_metrics()
        }
        
        # Return step results
        return self._get_observation(), rewards, done, info
    
    def _get_observation(self):
        """
        Get the current state observation of the environment.
        
        Returns:
            Dict: Observation dictionary
        """
        # Collect observations about tasks
        task_observations = []
        for task in self.tasks:
            # Only include tasks that are still relevant (not completed/failed)
            if not task.completed and not task.failed:
                task_obs = {
                    'id': task.id,
                    'position': task.position,
                    'priority': task.priority,
                    'type': task.type,
                    'requires_coalition': task.requires_coalition,
                    'assigned': task.assigned,
                    'bundle_id': task.bundle_id
                }
                task_observations.append(task_obs)
        
        # Collect observations about robots
        robot_observations = []
        for robot in self.robots:
            robot_obs = {
                'id': robot.id,
                'position': robot.position,
                'capabilities': robot.capabilities,
                'strategy': robot.strategy_type,
                'current_tasks': list(robot.current_tasks),
                'energy': robot.energy
            }
            robot_observations.append(robot_obs)
        
        # Build full observation
        observation = {
            'tasks': task_observations,
            'robots': robot_observations,
            'current_step': self.current_step,
            'auction_type': self.auction_type,
            'payment_rule': self.payment_rule
        }
        
        return observation
    
    def _execute_robot_tasks(self):
        """Have robots execute their assigned tasks - focusing on one task at a time."""
        # Track completed tasks for this step
        completed_task_ids = set()
        
        # Each robot executes only its highest priority task
        for robot in self.robots:
            # Get all assigned tasks for this robot
            assigned_tasks = []
            for task_id in list(robot.current_tasks):  # Use list to avoid modification during iteration
                # Only execute tasks that are in the task manager's assigned_tasks set
                if (task_id in self.task_manager.tasks and 
                    task_id in self.task_manager.assigned_tasks):
                    task = self.task_manager.tasks[task_id]
                    assigned_tasks.append(task)
                elif task_id in robot.current_tasks:
                    # Task is not assigned in task manager but still in robot's list
                    if self.verbose:
                        print(f"Removing invalid task {task_id} from robot {robot.id}'s tasks")
                    robot.current_tasks.remove(task_id)
            
            # Skip if no tasks assigned
            if not assigned_tasks:
                continue
                
            # Sort tasks by priority (higher priority first) to prioritize important tasks
            assigned_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            # First check if robot is part of a coalition
            if robot.current_coalition is not None:
                # Find the coalition task
                coalition_task_id = robot.current_coalition
                if coalition_task_id in self.task_manager.tasks:
                    coalition_task = self.task_manager.tasks[coalition_task_id]
                    
                    # Find coalition info
                    coalition_info = None
                    for coalition in robot.coalition_history:
                        if coalition['task_id'] == coalition_task_id:
                            coalition_info = coalition
                            break
                    
                    if coalition_info:
                        # Set coalition robots function to get all members
                        def get_coalition_robots():
                            members = []
                            for member_id in coalition_info['members']:
                                for r in self.robots:
                                    if r.id == member_id:
                                        members.append(r)
                                        break
                            return members
                        
                        # Set the function in the robot
                        robot._get_coalition_robots = get_coalition_robots
                        
                        # Execute coalition task
                        success = robot.execute_coalition_task(coalition_task, coalition_info)
                        
                        if success:
                            # Check if all coalition members completed the task
                            all_completed = True
                            for member_id in coalition_info['members']:
                                for r in self.robots:
                                    if r.id == member_id and coalition_task_id in r.current_tasks:
                                        all_completed = False
                                        break
                            
                            # If this robot completed the task, check if everyone else is done too
                            if coalition_task_id not in robot.current_tasks:
                                if self.verbose:
                                    print(f"Robot {robot.id} completed their part of coalition task {coalition_task_id}")
                                # Check if all coalition members completed the task
                                all_completed = True
                                for member_id in coalition_info['members']:
                                    if member_id != robot.id:  # Skip this robot, we know it's done
                                        for r in self.robots:
                                            if r.id == member_id and coalition_task_id in r.current_tasks:
                                                all_completed = False
                                                break
                                    
                                # If all completed, mark the task as completed
                                if all_completed:
                                    if self.verbose:
                                        print(f"Coalition task {coalition_task_id} completed by all members")
                                    self.task_manager.mark_task_completed(coalition_task_id, success=True)
                                    completed_task_ids.add(coalition_task_id)
                        
                        # Skip regular task execution if in a coalition
                        continue
            
            # Execute highest priority regular task if not in coalition
            if assigned_tasks:
                task = assigned_tasks[0]
                success = robot.execute_task(task)
                
                if success:
                    # Mark task as completed
                    if self.verbose:
                        print(f"SUCCESS! Robot {robot.id} completed task {task.id}. Task was at {task.position}, robot at {robot.position}")
                    self.task_manager.mark_task_completed(task.id, success=True)
                    completed_task_ids.add(task.id)
                else:
                    # Robot is still moving towards the task
                    if self.verbose:
                        print(f"Task {task.id} not completed yet, robot {robot.id} still moving toward it") 
                    pass  # Do nothing - only mark tasks as failed when needed
        
        return completed_task_ids
    
    def _calculate_rewards(self):
        """Calculate rewards for each robot based on task execution and payments."""
        rewards = {}
        
        for robot in self.robots:
            # Base reward is the accumulated utility from completed tasks
            utility = sum(robot.utility_history[-1:] if robot.utility_history else [0])
            
            # Subtract energy costs
            energy_cost = (100 - robot.energy) * 0.01
            
            # Calculate final reward
            rewards[robot.id] = utility - energy_cost
        
        return rewards
    
    def _update_metrics(self, allocation_results):
        """Update performance metrics based on latest step results."""
        # Update allocation efficiency
        if allocation_results:
            self.metrics.update_metric('allocation_efficiency', 
                                      allocation_results['metrics']['allocation_rate'])
            
            # Update utilities and payments
            self.metrics.update_metric('total_welfare', 
                                      allocation_results['metrics']['total_welfare'])
            self.metrics.update_metric('total_revenue', 
                                      allocation_results['metrics']['total_revenue'])
        
        # Update task completion rate
        metrics = self.task_manager.get_allocation_metrics()
        self.metrics.update_metric('task_completion_rate', metrics['completion_rate'])
        self.metrics.update_metric('avg_utility_per_task', metrics['avg_utility_per_task'])
        
        # Record robot states
        for robot in self.robots:
            self.metrics.update_robot_metric(robot.id, 'energy', robot.energy)
            self.metrics.update_robot_metric(robot.id, 'tasks_completed', len(robot.task_history))
            self.metrics.update_robot_metric(robot.id, 'utility', sum(robot.utility_history))
    
    def _check_done(self):
        """Check if the simulation should end."""
        # End if we've reached the maximum number of steps
        if self.current_step >= self.max_steps:
            return True
            
        # End if all robots are out of energy
        if all(robot.energy <= 0 for robot in self.robots):
            return True
            
        # End if there are no more tasks and no possibility of new ones
        if not self.dynamic_tasks and len(self.task_manager.available_tasks) + len(self.task_manager.assigned_tasks) == 0:
            return True
        
        # Otherwise continue
        return False
    
    def render(self):
        """Render the current state of the environment."""
        if not self.visualizer:
            return
            
        self.visualizer.render(self.current_step)
    
    def close(self):
        """Clean up resources."""
        if self.visualizer:
            self.visualizer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Robot Task Allocation Simulation")
    
    parser.add_argument("--num_robots", type=int, default=5,
                        help="Number of robots in the simulation")
    parser.add_argument("--num_tasks", type=int, default=10,
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
    parser.add_argument("--render_mode", type=str, default="pygame",
                        choices=["pygame", "matplotlib", "none"],
                        help="Visualization mode")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum simulation steps")
    parser.add_argument("--compare", action="store_true",
                        help="Run multiple auction types and compare results")
    
    return parser.parse_args()


def run_single_simulation(args):
    """Run a single simulation with the specified parameters."""
    # Create environment
    env = TaskAllocationEnv(
        n_robots=args.num_robots,
        n_tasks=args.num_tasks,
        grid_size=args.grid_size,
        auction_type=args.auction_type,
        payment_rule=args.payment_rule,
        dynamic_tasks=args.dynamic_tasks,
        render_mode=args.render_mode if args.render_mode != "none" else None,
        max_steps=args.max_steps,
        verbose=True  # Enable verbose debugging
    )
    
    # Run simulation
    observation = env.reset()
    done = False
    
    print(f"Starting simulation with {args.auction_type} auction and {args.payment_rule} payment rule")
    print(f"Grid size: {args.grid_size}x{args.grid_size}, Robots: {args.num_robots}, Initial tasks: {args.num_tasks}")
    
    while not done:
        # Render current state
        env.render()
        
        # Execute step
        observation, rewards, done, info = env.step()
        
        # Optional: add sleep to slow down visualization
        if args.render_mode != "none":
            time.sleep(0.1)
        
        # Print step information (optional)
        if env.current_step % 10 == 0:
            completed = len(env.task_manager.completed_tasks)
            total = completed + len(env.task_manager.failed_tasks) + len(env.task_manager.assigned_tasks) + len(env.task_manager.available_tasks)
            print(f"Step {env.current_step}: Completed {completed}/{total} tasks")
    
    # Final rendering
    env.render()
    
    # Display final metrics
    print("\nFinal Metrics:")
    final_metrics = env.metrics.get_summary_metrics()
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Close environment
    env.close()
    
    return final_metrics


def run_comparison(args):
    """Run simulations with different auction types and compare results."""
    # Auction types to compare
    auction_types = ["sequential", "parallel", "combinatorial"]
    payment_rules = ["first_price", "second_price"]
    
    # Disable rendering for comparison runs
    args.render_mode = "none"
    
    # Store results
    results = {}
    
    for auction_type in auction_types:
        for payment_rule in payment_rules:
            # Skip invalid combinations
            if auction_type == "combinatorial" and payment_rule == "second_price":
                continue  # This combination is problematic
                
            print(f"\nRunning simulation with {auction_type} auction and {payment_rule} payment rule...")
            
            # Set parameters for this run
            args.auction_type = auction_type
            args.payment_rule = payment_rule
            
            # Run simulation
            metrics = run_single_simulation(args)
            
            # Store results
            results[f"{auction_type}_{payment_rule}"] = metrics
    
    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    
    # Compare key metrics
    for metric in ['allocation_efficiency', 'task_completion_rate', 'total_welfare', 'total_revenue']:
        print(f"\n{metric.upper()}:")
        for config, metrics in results.items():
            value = metrics.get(metric, 0)
            print(f"  {config}: {value:.2f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot data
    configs = list(results.keys())
    metrics_to_plot = ['allocation_efficiency', 'task_completion_rate', 'total_welfare', 'total_revenue']
    
    # Normalize welfare and revenue for better visualization
    max_welfare = max(results[config].get('total_welfare', 0) for config in configs)
    max_revenue = max(results[config].get('total_revenue', 0) for config in configs)
    
    # Prepare data for plotting
    data = []
    for metric in metrics_to_plot:
        metric_data = []
        for config in configs:
            value = results[config].get(metric, 0)
            if metric == 'total_welfare' and max_welfare > 0:
                value = value / max_welfare
            elif metric == 'total_revenue' and max_revenue > 0:
                value = value / max_revenue
            metric_data.append(value)
        data.append(metric_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (metric, metric_data) in enumerate(zip(metrics_to_plot, data)):
        axes[i].bar(configs, metric_data)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_xticklabels(configs, rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("auction_comparison.png")
    plt.show()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Run simulation(s)
    if args.compare:
        run_comparison(args)
    else:
        run_single_simulation(args)


if __name__ == "__main__":
    main()