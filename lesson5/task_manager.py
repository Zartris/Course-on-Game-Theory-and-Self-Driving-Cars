#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task Manager implementation for the Task Allocation and Auction Mechanisms lesson.

This module implements the generation, management, and evaluation of tasks in
a multi-robot task allocation scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Union, Optional
import random
import math
import heapq
from dataclasses import dataclass, field

@dataclass
class Task:
    """
    Class representing a task in the multi-robot system.
    
    Tasks have various properties including position, priority, required capabilities,
    and other attributes needed for task allocation.
    """
    id: int
    position: Tuple[int, int]
    priority: float = 1.0
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    completion_time: float = 1.0
    deadline: Optional[int] = None
    energy_cost: float = 1.0
    type: str = "simple"
    team_utility_factor: float = 1.0
    
    # For complex tasks requiring multiple robots
    requires_coalition: bool = False
    role_requirements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    base_utility: float = 10.0
    
    # For bundle tasks (combinatorial auctions)
    bundle_id: Optional[str] = None
    all_tasks: List = field(default_factory=list)
    
    # Dynamic task properties
    arrival_time: int = 0
    expiration_time: Optional[int] = None
    
    # Task state
    assigned: bool = False
    completed: bool = False
    failed: bool = False
    
    def __post_init__(self):
        """Initialize additional attributes after creation."""
        # Calculated at runtime based on auction participants
        self.expected_participants = 3
        self.competition_factor = 0.5

    def is_expired(self, current_time: int) -> bool:
        """Check if task has expired."""
        if self.expiration_time is None:
            return False
        return current_time > self.expiration_time
    
    def is_available(self, current_time: int) -> bool:
        """Check if task is available for allocation."""
        return (not self.assigned and 
                not self.completed and 
                not self.failed and 
                current_time >= self.arrival_time and
                not self.is_expired(current_time))
    
    def __str__(self) -> str:
        """String representation of the task."""
        status = "ASSIGNED" if self.assigned else "AVAILABLE" 
        if self.completed:
            status = "COMPLETED"
        elif self.failed:
            status = "FAILED"
        
        type_desc = f"{self.type}"
        if self.requires_coalition:
            type_desc += " (coalition)"
        if self.bundle_id:
            type_desc += f" (bundle: {self.bundle_id})"
            
        return f"Task {self.id} [{status}]: {type_desc} at {self.position}, priority={self.priority:.1f}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the task."""
        return (f"Task(id={self.id}, type={self.type}, position={self.position}, "
                f"priority={self.priority:.1f}, assigned={self.assigned}, completed={self.completed})")


class TaskManager:
    """
    Manages task generation, tracking, and evaluation in a multi-robot system.
    
    This class is responsible for creating tasks, managing their lifecycle,
    and providing metrics on task completion and allocation efficiency.
    """
    
    def __init__(self, 
                 environment_size: Tuple[int, int],
                 task_complexity_range: Tuple[float, float] = (1.0, 5.0),
                 dynamic: bool = True,
                 task_arrival_rate: float = 0.2,
                 task_types: List[str] = None,
                 capability_types: List[str] = None,
                 verbose: bool = False):
        """
        Initialize the TaskManager with environment parameters.
        
        Args:
            environment_size: (width, height) of the environment grid
            task_complexity_range: Range of task complexity values
            dynamic: Whether tasks arrive dynamically over time
            task_arrival_rate: Probability of new task arrival per time step
            task_types: Available task types (search, rescue, transport, etc.)
            capability_types: Available robot capability types
            verbose: Whether to print detailed debug information
        """
        self.environment_size = environment_size
        self.task_complexity_range = task_complexity_range
        self.dynamic = dynamic
        self.task_arrival_rate = task_arrival_rate
        self.verbose = verbose
        
        # Default task and capability types if none provided
        self.task_types = task_types or ["search", "transport", "monitor", "rescue", "build"]
        self.capability_types = capability_types or ["speed", "lift_capacity", "precision", "sensor_range"]
        
        # Task tracking
        self.tasks = {}  # id -> Task
        self.available_tasks = set()  # ids of available tasks
        self.assigned_tasks = set()  # ids of assigned tasks
        self.completed_tasks = set()  # ids of completed tasks
        self.failed_tasks = set()  # ids of failed tasks
        
        # Bundle management
        self.bundles = {}  # bundle_id -> list of task ids
        
        # Task generation tracking
        self.next_task_id = 0
        self.current_time = 0
        self.max_active_tasks = 20  # Maximum number of active tasks at once
        
        # Statistics
        self.allocation_stats = {
            'time_to_allocation': [],  # Time between task arrival and allocation
            'completion_rate': [],     # Percentage of tasks completed successfully
            'utility_per_task': [],    # Utility gained from each completed task
            'allocation_efficiency': []  # Ratio of allocated to available tasks
        }
        
    def generate_task(self, 
                      task_type: str = None, 
                      position: Tuple[int, int] = None, 
                      priority: float = None,
                      requires_coalition: bool = False,
                      bundle_id: str = None) -> Task:
        """
        Generate a single task with specified or random parameters.
        
        Args:
            task_type: Type of task (or random if None)
            position: Position of task (or random if None)
            priority: Priority level (or random if None)
            requires_coalition: Whether task requires multiple robots
            bundle_id: ID of bundle this task belongs to
            
        Returns:
            Task: The generated task
        """
        # Generate random parameters if not specified
        if task_type is None:
            task_type = random.choice(self.task_types)
            
        if position is None:
            position = (
                random.randint(0, self.environment_size[0] - 1),
                random.randint(0, self.environment_size[1] - 1)
            )
        
        if priority is None:
            # Higher priority for rescue and emergency tasks
            if task_type in ["rescue", "emergency"]:
                priority = random.uniform(3.0, 5.0)
            else:
                priority = random.uniform(1.0, 3.0)
                
        # Generate task complexity and determine required capabilities
        complexity = random.uniform(*self.task_complexity_range)
        required_capabilities = self._generate_required_capabilities(task_type, complexity)
        
        # Determine completion time based on complexity
        completion_time = complexity * random.uniform(0.8, 1.2)
        
        # Determine energy cost based on complexity and type
        energy_cost = complexity * random.uniform(0.5, 1.5)
        
        # Set expiration time for time-sensitive tasks
        expiration_time = None
        if task_type in ["rescue", "emergency"] or random.random() < 0.3:
            # Time-sensitive tasks expire after a while
            expiration_time = self.current_time + int(completion_time * 5)
            
        # Create the task
        task = Task(
            id=self.next_task_id,
            position=position,
            priority=priority,
            required_capabilities=required_capabilities,
            completion_time=completion_time,
            deadline=expiration_time,
            energy_cost=energy_cost,
            type=task_type,
            team_utility_factor=random.uniform(0.8, 1.2),
            requires_coalition=requires_coalition,
            bundle_id=bundle_id,
            arrival_time=self.current_time
        )
        
        # Add coalition requirements if needed
        if requires_coalition:
            task.role_requirements = self._generate_role_requirements(task_type, complexity)
            task.base_utility = complexity * 10.0  # Higher base utility for complex tasks
            
        # Increment task ID counter
        self.next_task_id += 1
        
        return task
    
    def _generate_required_capabilities(self, task_type: str, complexity: float) -> Dict[str, float]:
        """Generate required capabilities based on task type and complexity."""
        required_capabilities = {}
        
        # Different task types require different capabilities
        if task_type == "transport":
            required_capabilities["lift_capacity"] = complexity * random.uniform(0.8, 1.2)
            if random.random() < 0.3:
                required_capabilities["speed"] = complexity * 0.5
                
        elif task_type == "search":
            required_capabilities["sensor_range"] = complexity * random.uniform(0.8, 1.2)
            required_capabilities["speed"] = complexity * 0.7
            
        elif task_type == "rescue":
            required_capabilities["lift_capacity"] = complexity * 0.7
            required_capabilities["precision"] = complexity * random.uniform(0.8, 1.2)
            
        elif task_type == "monitor":
            required_capabilities["sensor_range"] = complexity * random.uniform(0.8, 1.2)
            
        elif task_type == "build":
            required_capabilities["precision"] = complexity * random.uniform(0.8, 1.2)
            required_capabilities["lift_capacity"] = complexity * 0.6
        
        # Add random capabilities with small probability
        for cap in self.capability_types:
            if cap not in required_capabilities and random.random() < 0.2:
                required_capabilities[cap] = complexity * random.uniform(0.3, 0.7)
                
        return required_capabilities
    
    def _generate_role_requirements(self, task_type: str, complexity: float) -> Dict[str, Dict[str, float]]:
        """Generate role requirements for coalition tasks."""
        roles = {}
        
        if task_type == "transport":
            # Heavy transport might need lifters and navigators
            roles["lifter"] = {"lift_capacity": complexity * 1.2}
            roles["navigator"] = {"sensor_range": complexity * 0.8, "speed": complexity * 0.6}
            
        elif task_type == "rescue":
            # Rescue missions might need scouts and operators
            roles["scout"] = {"sensor_range": complexity * 1.1, "speed": complexity}
            roles["operator"] = {"precision": complexity, "lift_capacity": complexity * 0.7}
            
        elif task_type == "build":
            # Building tasks might need different specialists
            roles["assembler"] = {"precision": complexity * 1.1}
            roles["supplier"] = {"lift_capacity": complexity, "speed": complexity * 0.8}
            
        else:
            # Default coalition roles
            roles["operator"] = {self.capability_types[0]: complexity}
            roles["assistant"] = {self.capability_types[1]: complexity * 0.8}
            
        return roles
    
    def generate_task_batch(self, n_tasks: int, include_complex: bool = True) -> List[Task]:
        """
        Generate a batch of tasks for static allocation scenarios.
        
        Args:
            n_tasks: Number of tasks to generate
            include_complex: Whether to include complex (coalition) tasks
            
        Returns:
            List[Task]: Generated tasks
        """
        tasks = []
        
        # Generate some tasks in bundles (for combinatorial auctions)
        bundle_count = n_tasks // 5  # About 20% of tasks in bundles
        for i in range(bundle_count):
            bundle_id = f"bundle_{i}"
            bundle_size = random.randint(2, 3)  # 2-3 tasks per bundle
            bundle_position = (
                random.randint(0, self.environment_size[0] - 1),
                random.randint(0, self.environment_size[1] - 1)
            )
            bundle_tasks = []
            
            # Generate tasks for this bundle
            for j in range(bundle_size):
                # Tasks in a bundle are close to each other
                position = (
                    min(max(0, bundle_position[0] + random.randint(-3, 3)), self.environment_size[0] - 1),
                    min(max(0, bundle_position[1] + random.randint(-3, 3)), self.environment_size[1] - 1)
                )
                
                task = self.generate_task(
                    task_type=random.choice(self.task_types),
                    position=position,
                    bundle_id=bundle_id
                )
                bundle_tasks.append(task)
                tasks.append(task)
                
            # Link tasks to each other
            for task in bundle_tasks:
                task.all_tasks = bundle_tasks
                
            # Track bundle
            self.bundles[bundle_id] = [task.id for task in bundle_tasks]
            
        # Generate remaining tasks
        remaining_count = n_tasks - len(tasks)
        complex_count = remaining_count // 4 if include_complex else 0  # About 25% complex tasks
        
        # Add complex tasks (requiring coalitions)
        for _ in range(complex_count):
            task = self.generate_task(requires_coalition=True)
            tasks.append(task)
            
        # Add regular tasks
        for _ in range(remaining_count - complex_count):
            task = self.generate_task()
            tasks.append(task)
            
        # Add tasks to manager's tracking
        for task in tasks:
            self.tasks[task.id] = task
            self.available_tasks.add(task.id)
            
        return tasks
    
    def update_tasks(self, time_step: int) -> Dict[str, List[Task]]:
        """
        Update task states and generate new tasks for dynamic scenarios.
        
        Args:
            time_step: Current time step
            
        Returns:
            Dict: {'new': new_tasks, 'expired': expired_tasks}
        """
        self.current_time = time_step
        new_tasks = []
        expired_tasks = []
        
        # Check for task expiration
        for task_id in list(self.available_tasks):
            task = self.tasks[task_id]
            if task.is_expired(time_step):
                self.available_tasks.remove(task_id)
                self.failed_tasks.add(task_id)
                task.failed = True
                expired_tasks.append(task)
                
        # Generate new tasks if dynamic mode is enabled
        if self.dynamic and len(self.available_tasks) + len(self.assigned_tasks) < self.max_active_tasks:
            # Generate tasks with probability based on arrival rate
            if random.random() < self.task_arrival_rate:
                # Occasionally generate bundles
                if random.random() < 0.2:
                    bundle_id = f"bundle_{time_step}_{random.randint(0, 999)}"
                    bundle_size = random.randint(2, 3)
                    bundle_position = (
                        random.randint(0, self.environment_size[0] - 1),
                        random.randint(0, self.environment_size[1] - 1)
                    )
                    bundle_tasks = []
                    
                    for _ in range(bundle_size):
                        position = (
                            min(max(0, bundle_position[0] + random.randint(-3, 3)), self.environment_size[0] - 1),
                            min(max(0, bundle_position[1] + random.randint(-3, 3)), self.environment_size[1] - 1)
                        )
                        
                        task = self.generate_task(position=position, bundle_id=bundle_id)
                        bundle_tasks.append(task)
                        new_tasks.append(task)
                        
                    # Link tasks to each other
                    for task in bundle_tasks:
                        task.all_tasks = bundle_tasks
                        
                    # Track bundle
                    self.bundles[bundle_id] = [task.id for task in bundle_tasks]
                    
                # Occasionally generate complex tasks
                elif random.random() < 0.25:
                    task = self.generate_task(requires_coalition=True)
                    new_tasks.append(task)
                    
                # Generate regular task
                else:
                    task = self.generate_task()
                    new_tasks.append(task)
        
        # Add new tasks to tracking
        for task in new_tasks:
            self.tasks[task.id] = task
            self.available_tasks.add(task.id)
            
        return {'new': new_tasks, 'expired': expired_tasks}
    
    def assign_task(self, task_id: int, robot_id: int) -> bool:
        """
        Mark a task as assigned to a robot.
        
        Args:
            task_id: ID of the task to assign
            robot_id: ID of the robot receiving the assignment
            
        Returns:
            bool: Success of assignment
        """
        if task_id not in self.available_tasks:
            return False
            
        task = self.tasks[task_id]
        task.assigned = True
        task.assigned_to = robot_id
        task.assignment_time = self.current_time
        
        # Update tracking sets
        self.available_tasks.remove(task_id)
        self.assigned_tasks.add(task_id)
        
        # Update statistics
        self.allocation_stats['time_to_allocation'].append(
            self.current_time - task.arrival_time
        )
        
        return True
    
    def mark_task_completed(self, task_id: int, success: bool = True) -> None:
        """
        Mark a task as completed or failed.
        
        Args:
            task_id: ID of the task to update
            success: Whether the task was completed successfully
        """
        # DEBUG
        if self.verbose:
            print(f"Marking task {task_id} as {'completed' if success else 'failed'}")
        
        if task_id not in self.assigned_tasks:
            if self.verbose:
                print(f"Task {task_id} is not in assigned_tasks set! Current assigned tasks: {self.assigned_tasks}")
            return
            
        task = self.tasks[task_id]
        self.assigned_tasks.remove(task_id)
        
        if success:
            task.completed = True
            self.completed_tasks.add(task_id)
            if self.verbose:
                print(f"Task {task_id} marked as completed. Completed tasks: {self.completed_tasks}")
        else:
            task.failed = True
            self.failed_tasks.add(task_id)
            if self.verbose:
                print(f"Task {task_id} marked as failed. Failed tasks: {self.failed_tasks}")
            
        # Update statistics
        if success:
            self.allocation_stats['utility_per_task'].append(
                task.priority * task.completion_time  # Simple utility metric
            )
    
    def evaluate_completion(self, task: Task, robot_capabilities: Dict[str, float]) -> bool:
        """
        Evaluate whether a robot can complete a task based on its capabilities.
        
        Args:
            task: The task to evaluate
            robot_capabilities: Robot's capabilities
            
        Returns:
            bool: Whether robot can complete the task
        """
        # For coalition tasks, a single robot cannot complete
        if task.requires_coalition:
            return False
            
        # Check each required capability
        for cap, level in task.required_capabilities.items():
            robot_level = robot_capabilities.get(cap, 0.0)
            if robot_level < level:
                return False
                
        return True
    
    def calculate_global_utility(self, allocation: Dict[int, List[int]]) -> float:
        """
        Calculate the global utility of a task allocation.
        
        Args:
            allocation: Dictionary mapping robot IDs to lists of task IDs
            
        Returns:
            float: Total utility of the allocation
        """
        total_utility = 0.0
        
        # Sum up utilities for all allocated tasks
        for robot_id, task_ids in allocation.items():
            for task_id in task_ids:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    # Calculate task utility (could be more complex in practice)
                    task_utility = task.priority * task.completion_time
                    
                    # Consider bundle synergies
                    if task.bundle_id:
                        # Check if all tasks in bundle are allocated to this robot
                        bundle_ids = self.bundles.get(task.bundle_id, [])
                        if all(t_id in task_ids for t_id in bundle_ids):
                            task_utility *= 1.2  # Bundle synergy bonus
                            
                    total_utility += task_utility
        
        return total_utility
    
    def get_available_tasks(self) -> List[Task]:
        """Get list of currently available tasks."""
        return [self.tasks[task_id] for task_id in self.available_tasks]
    
    def get_assigned_tasks(self) -> List[Task]:
        """Get list of currently assigned tasks."""
        return [self.tasks[task_id] for task_id in self.assigned_tasks]
    
    def get_completed_tasks(self) -> List[Task]:
        """Get list of completed tasks."""
        return [self.tasks[task_id] for task_id in self.completed_tasks]
    
    def get_failed_tasks(self) -> List[Task]:
        """Get list of failed tasks."""
        return [self.tasks[task_id] for task_id in self.failed_tasks]
    
    def visualize_task_distribution(self) -> plt.Figure:
        """
        Create visualization of task distribution in the environment.
        
        Returns:
            matplotlib.figure.Figure: Figure with visualization
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot
        ax.set_xlim(0, self.environment_size[0])
        ax.set_ylim(0, self.environment_size[1])
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Task Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot available tasks
        available_x = [self.tasks[tid].position[0] for tid in self.available_tasks]
        available_y = [self.tasks[tid].position[1] for tid in self.available_tasks]
        ax.scatter(available_x, available_y, color='green', marker='o', s=100, 
                  label='Available', alpha=0.7)
        
        # Plot assigned tasks
        assigned_x = [self.tasks[tid].position[0] for tid in self.assigned_tasks]
        assigned_y = [self.tasks[tid].position[1] for tid in self.assigned_tasks]
        ax.scatter(assigned_x, assigned_y, color='blue', marker='o', s=100, 
                  label='Assigned', alpha=0.7)
        
        # Plot completed tasks
        completed_x = [self.tasks[tid].position[0] for tid in self.completed_tasks]
        completed_y = [self.tasks[tid].position[1] for tid in self.completed_tasks]
        ax.scatter(completed_x, completed_y, color='gray', marker='x', s=80, 
                  label='Completed', alpha=0.5)
        
        # Plot failed tasks
        failed_x = [self.tasks[tid].position[0] for tid in self.failed_tasks]
        failed_y = [self.tasks[tid].position[1] for tid in self.failed_tasks]
        ax.scatter(failed_x, failed_y, color='red', marker='x', s=80, 
                  label='Failed', alpha=0.5)
        
        # Mark task bundles with connecting lines
        for bundle_id, task_ids in self.bundles.items():
            points_x = []
            points_y = []
            for task_id in task_ids:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    points_x.append(task.position[0])
                    points_y.append(task.position[1])
            
            if points_x:
                ax.plot(points_x, points_y, 'k--', alpha=0.4)
                center_x = sum(points_x) / len(points_x)
                center_y = sum(points_y) / len(points_y)
                ax.annotate(f"Bundle {bundle_id}", (center_x, center_y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add legend
        ax.legend()
        
        return fig
    
    def get_allocation_metrics(self) -> Dict:
        """
        Calculate and return metrics about task allocation performance.
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        # Task completion rate
        total_finished = len(self.completed_tasks) + len(self.failed_tasks)
        if total_finished > 0:
            metrics['completion_rate'] = len(self.completed_tasks) / total_finished
        else:
            metrics['completion_rate'] = 0.0
            
        # Average time to allocation
        if self.allocation_stats['time_to_allocation']:
            metrics['avg_time_to_allocation'] = sum(self.allocation_stats['time_to_allocation']) / len(self.allocation_stats['time_to_allocation'])
        else:
            metrics['avg_time_to_allocation'] = 0.0
            
        # Average utility per task
        if self.allocation_stats['utility_per_task']:
            metrics['avg_utility_per_task'] = sum(self.allocation_stats['utility_per_task']) / len(self.allocation_stats['utility_per_task'])
        else:
            metrics['avg_utility_per_task'] = 0.0
            
        # Current allocation efficiency
        total_tasks = len(self.available_tasks) + len(self.assigned_tasks)
        if total_tasks > 0:
            metrics['current_allocation_efficiency'] = len(self.assigned_tasks) / total_tasks
        else:
            metrics['current_allocation_efficiency'] = 0.0
            
        return metrics
    
# Example usage
if __name__ == "__main__":
    # Initialize task manager
    manager = TaskManager(
        environment_size=(50, 50),
        task_complexity_range=(1.0, 5.0),
        dynamic=True,
        task_arrival_rate=0.3
    )
    
    # Generate some initial tasks
    initial_tasks = manager.generate_task_batch(10, include_complex=True)
    print(f"Generated {len(initial_tasks)} initial tasks")
    
    # Assign some tasks
    for i, task in enumerate(initial_tasks[:5]):
        manager.assign_task(task.id, i % 3)  # Assign to robots 0, 1, 2
    
    # Complete some tasks
    for task in initial_tasks[:3]:
        manager.mark_task_completed(task.id, success=True)
    
    # Fail one task
    manager.mark_task_completed(initial_tasks[3].id, success=False)
    
    # Simulate some time steps
    for t in range(1, 6):
        updates = manager.update_tasks(t)
        print(f"Time step {t}:")
        print(f"  New tasks: {len(updates['new'])}")
        print(f"  Expired tasks: {len(updates['expired'])}")
    
    # Display current task counts
    print("\nTask status:")
    print(f"Available: {len(manager.available_tasks)}")
    print(f"Assigned: {len(manager.assigned_tasks)}")
    print(f"Completed: {len(manager.completed_tasks)}")
    print(f"Failed: {len(manager.failed_tasks)}")
    
    # Display metrics
    metrics = manager.get_allocation_metrics()
    print("\nPerformance metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Visualize task distribution
    fig = manager.visualize_task_distribution()
    plt.show()