"""
Real-Time Mechanism Design for Multi-Robot Systems

This module implements a real-time task allocation mechanism that operates under
strict timing constraints and adapts to hardware heterogeneity.
"""

import time
import random
import math
import heapq
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class HardwareClass(Enum):
    """Classification of robot hardware capabilities."""
    LOW_END = 0    # Limited computation, memory, and communication
    MID_RANGE = 1  # Moderate capabilities
    HIGH_END = 2   # High-performance hardware


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Representation of a task to be allocated."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    difficulty: float  # 0.0 to 1.0
    deadline: float  # seconds from now
    value: float
    required_capabilities: Set[str]
    priority: TaskPriority = TaskPriority.MEDIUM
    computation_requirement: float = 0.5  # 0.0 to 1.0
    communication_requirement: float = 0.5  # 0.0 to 1.0
    memory_requirement: float = 0.5  # 0.0 to 1.0


@dataclass
class Robot:
    """Representation of a robot in the system."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    capabilities: Set[str]
    hardware_class: HardwareClass
    computation_capacity: float  # 0.0 to 1.0
    communication_capacity: float  # 0.0 to 1.0
    memory_capacity: float  # 0.0 to 1.0
    energy_level: float  # 0.0 to 1.0
    max_computation_time: float  # Maximum time in ms for mechanism computation


class RealTimeMechanism:
    """
    Implements a real-time task allocation mechanism that operates under
    strict timing constraints and adapts to hardware heterogeneity.
    """
    
    def __init__(self, 
                 global_deadline: float = 100.0,  # ms
                 precomputation_enabled: bool = True,
                 adaptive_precision: bool = True):
        """
        Initialize the real-time mechanism.
        
        Args:
            global_deadline: Maximum time in ms for allocation decision
            precomputation_enabled: Whether to use precomputation
            adaptive_precision: Whether to adapt precision based on time constraints
        """
        self.global_deadline = global_deadline
        self.precomputation_enabled = precomputation_enabled
        self.adaptive_precision = adaptive_precision
        
        # Precomputed data
        self.distance_cache = {}  # (robot_id, task_id) -> distance
        self.capability_match_cache = {}  # (robot_id, task_id) -> match score
        self.value_estimates = {}  # (robot_id, task_id) -> estimated value
        
        # Allocation data
        self.allocation = {}  # task_id -> robot_id
        self.payments = {}  # robot_id -> payment
        
        # Performance metrics
        self.computation_times = []  # List of computation times in ms
        self.precision_levels = []  # List of precision levels used
        self.deadline_violations = 0  # Count of deadline violations
    
    def precompute_distances(self, robots: List[Robot], tasks: List[Task]) -> None:
        """
        Precompute distances between robots and tasks.
        
        Args:
            robots: List of robots
            tasks: List of tasks
        """
        start_time = time.time()
        
        for robot in robots:
            for task in tasks:
                key = (robot.id, task.id)
                distance = self.calculate_distance(robot.position, task.position)
                self.distance_cache[key] = distance
        
        end_time = time.time()
        print(f"Distance precomputation completed in {(end_time - start_time) * 1000:.2f} ms")
    
    def precompute_capability_matches(self, robots: List[Robot], tasks: List[Task]) -> None:
        """
        Precompute capability matches between robots and tasks.
        
        Args:
            robots: List of robots
            tasks: List of tasks
        """
        start_time = time.time()
        
        for robot in robots:
            for task in tasks:
                key = (robot.id, task.id)
                match = self.calculate_capability_match(robot, task)
                self.capability_match_cache[key] = match
        
        end_time = time.time()
        print(f"Capability match precomputation completed in {(end_time - start_time) * 1000:.2f} ms")
    
    def precompute_value_estimates(self, robots: List[Robot], tasks: List[Task]) -> None:
        """
        Precompute estimated values for robot-task pairs.
        
        Args:
            robots: List of robots
            tasks: List of tasks
        """
        start_time = time.time()
        
        for robot in robots:
            for task in tasks:
                key = (robot.id, task.id)
                
                # Use cached values if available
                distance = self.distance_cache.get(key, 
                                                 self.calculate_distance(robot.position, task.position))
                match = self.capability_match_cache.get(key,
                                                      self.calculate_capability_match(robot, task))
                
                # Estimate value
                value = self.estimate_value(robot, task, distance, match)
                self.value_estimates[key] = value
        
        end_time = time.time()
        print(f"Value estimation precomputation completed in {(end_time - start_time) * 1000:.2f} ms")
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_capability_match(self, robot: Robot, task: Task) -> float:
        """
        Calculate capability match between robot and task.
        
        Args:
            robot: The robot
            task: The task
            
        Returns:
            Match score (0.0 to 1.0)
        """
        # Check if robot has all required capabilities
        if not task.required_capabilities.issubset(robot.capabilities):
            return 0.0
        
        # Calculate match quality based on specialization
        specialization = len(task.required_capabilities) / max(1, len(robot.capabilities))
        
        # Check hardware requirements
        hardware_match = min(
            robot.computation_capacity / max(0.1, task.computation_requirement),
            robot.communication_capacity / max(0.1, task.communication_requirement),
            robot.memory_capacity / max(0.1, task.memory_requirement)
        )
        
        # Cap at 1.0
        hardware_match = min(1.0, hardware_match)
        
        return specialization * hardware_match
    
    def estimate_value(self, robot: Robot, task: Task, distance: float, capability_match: float) -> float:
        """
        Estimate value of a task for a robot.
        
        Args:
            robot: The robot
            task: The task
            distance: Distance between robot and task
            capability_match: Capability match score
            
        Returns:
            Estimated value
        """
        # If robot doesn't have required capabilities, value is 0
        if capability_match == 0.0:
            return 0.0
        
        # Base value depends on task value and difficulty
        base_value = task.value * (1.0 - task.difficulty)
        
        # Adjust for distance (closer is better)
        distance_factor = max(0.1, 1.0 - (distance / 100.0))
        
        # Adjust for capability match
        match_factor = capability_match
        
        # Adjust for hardware class
        if robot.hardware_class == HardwareClass.HIGH_END:
            hardware_factor = 1.0
        elif robot.hardware_class == HardwareClass.MID_RANGE:
            hardware_factor = 0.8
        else:  # LOW_END
            hardware_factor = 0.6
        
        # Adjust for task priority
        if task.priority == TaskPriority.CRITICAL:
            priority_factor = 2.0
        elif task.priority == TaskPriority.HIGH:
            priority_factor = 1.5
        elif task.priority == TaskPriority.MEDIUM:
            priority_factor = 1.0
        else:  # LOW
            priority_factor = 0.5
        
        # Calculate final value
        value = base_value * distance_factor * match_factor * hardware_factor * priority_factor
        
        return value
    
    def select_precision_level(self, available_time: float, robots: List[Robot]) -> Tuple[str, int, bool]:
        """
        Select appropriate precision level based on available time.
        
        Args:
            available_time: Available time in ms
            robots: List of robots to consider their hardware class
            
        Returns:
            Tuple of (mechanism type, approximation level, use_precomputation)
        """
        # Determine the lowest hardware class among robots
        min_hardware_class = min((robot.hardware_class for robot in robots), key=lambda x: x.value)
        
        # Adjust available time based on hardware class
        if min_hardware_class == HardwareClass.LOW_END:
            effective_time = available_time * 0.5  # Low-end hardware is slower
        elif min_hardware_class == HardwareClass.MID_RANGE:
            effective_time = available_time * 0.8  # Mid-range hardware is moderately fast
        else:  # HIGH_END
            effective_time = available_time  # High-end hardware is fast
        
        # Select mechanism and precision based on available time
        if effective_time > 50.0:
            # Plenty of time: use optimal mechanism
            return "vcg", 0, True
        elif effective_time > 20.0:
            # Moderate time: use approximate VCG
            return "vcg", 1, True
        elif effective_time > 10.0:
            # Limited time: use sequential auction
            return "sequential", 1, True
        elif effective_time > 5.0:
            # Very limited time: use greedy mechanism
            return "greedy", 1, True
        else:
            # Extremely limited time: use emergency allocation
            return "emergency", 2, False
    
    def execute_vcg_mechanism(self, robots: List[Robot], tasks: List[Task], 
                             approximation_level: int, use_precomputation: bool,
                             time_limit: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute VCG mechanism with time constraints.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            approximation_level: Level of approximation (0=exact, 1=approximate, 2=very approximate)
            use_precomputation: Whether to use precomputed values
            time_limit: Time limit in ms
            
        Returns:
            Tuple of (allocation, payments)
        """
        start_time = time.time()
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Determine which tasks to consider based on approximation level
        if approximation_level == 0:
            # Consider all tasks
            considered_tasks = tasks
        elif approximation_level == 1:
            # Consider high priority tasks first, then others if time permits
            high_priority_tasks = [t for t in tasks if t.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL)]
            other_tasks = [t for t in tasks if t.priority in (TaskPriority.MEDIUM, TaskPriority.LOW)]
            considered_tasks = high_priority_tasks + other_tasks
        else:  # approximation_level == 2
            # Consider only critical tasks
            considered_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
        
        # Calculate values for all robot-task pairs
        values = {}
        for robot in robots:
            for task in considered_tasks:
                key = (robot.id, task.id)
                
                if use_precomputation and key in self.value_estimates:
                    # Use precomputed value
                    value = self.value_estimates[key]
                else:
                    # Calculate value on the fly
                    distance = self.calculate_distance(robot.position, task.position)
                    match = self.calculate_capability_match(robot, task)
                    value = self.estimate_value(robot, task, distance, match)
                
                values[key] = value
        
        # Check if we're running out of time
        current_time = time.time()
        elapsed_ms = (current_time - start_time) * 1000
        if elapsed_ms > time_limit * 0.5:
            # Switch to faster mechanism if we've used half the time budget
            print(f"VCG taking too long ({elapsed_ms:.2f} ms), switching to greedy allocation")
            return self.execute_greedy_mechanism(robots, considered_tasks, 2, False, time_limit - elapsed_ms)
        
        # Solve winner determination problem
        allocation = self.solve_winner_determination(robots, considered_tasks, values, approximation_level)
        
        # Check if we're running out of time
        current_time = time.time()
        elapsed_ms = (current_time - start_time) * 1000
        if elapsed_ms > time_limit * 0.8:
            # Skip payment calculation if we've used 80% of the time budget
            print(f"Skipping payment calculation due to time constraints ({elapsed_ms:.2f} ms)")
            # Use simple payments based on task value
            for task_id, robot_id in allocation.items():
                task = next(t for t in considered_tasks if t.id == task_id)
                payments[robot_id] = payments.get(robot_id, 0.0) + task.value * 0.5
            return allocation, payments
        
        # Calculate VCG payments
        payments = self.calculate_vcg_payments(robots, considered_tasks, values, allocation, approximation_level)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"VCG mechanism completed in {elapsed_ms:.2f} ms (limit: {time_limit:.2f} ms)")
        
        return allocation, payments
    
    def solve_winner_determination(self, robots: List[Robot], tasks: List[Task], 
                                  values: Dict[Tuple[str, str], float], 
                                  approximation_level: int) -> Dict[str, str]:
        """
        Solve the winner determination problem.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            values: Dictionary mapping (robot_id, task_id) to value
            approximation_level: Level of approximation
            
        Returns:
            Task allocation (task_id -> robot_id)
        """
        # Start with empty allocation
        allocation = {}
        
        if approximation_level <= 1:
            # Use a greedy algorithm for approximate solution
            # Sort tasks by priority and value
            sorted_tasks = sorted(tasks, 
                                 key=lambda t: (t.priority.value, t.value), 
                                 reverse=True)
            
            # Keep track of allocated robots
            allocated_robots = set()
            
            # Allocate tasks one by one
            for task in sorted_tasks:
                # Find best robot for this task
                best_robot_id = None
                best_value = 0.0
                
                for robot in robots:
                    # Skip if robot is already allocated (in approximation level 1)
                    if approximation_level == 1 and robot.id in allocated_robots:
                        continue
                    
                    # Get value
                    key = (robot.id, task.id)
                    value = values.get(key, 0.0)
                    
                    # Update best robot if value is higher
                    if value > best_value:
                        best_value = value
                        best_robot_id = robot.id
                
                # Allocate task if any robot values it positively
                if best_robot_id is not None and best_value > 0:
                    allocation[task.id] = best_robot_id
                    allocated_robots.add(best_robot_id)
        else:
            # For highest approximation level, use an even simpler approach
            # Just allocate each task to the robot with highest value
            for task in tasks:
                best_robot_id = None
                best_value = 0.0
                
                for robot in robots:
                    key = (robot.id, task.id)
                    value = values.get(key, 0.0)
                    
                    if value > best_value:
                        best_value = value
                        best_robot_id = robot.id
                
                if best_robot_id is not None and best_value > 0:
                    allocation[task.id] = best_robot_id
        
        return allocation
    
    def calculate_vcg_payments(self, robots: List[Robot], tasks: List[Task], 
                              values: Dict[Tuple[str, str], float], 
                              allocation: Dict[str, str],
                              approximation_level: int) -> Dict[str, float]:
        """
        Calculate VCG payments.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            values: Dictionary mapping (robot_id, task_id) to value
            allocation: Task allocation (task_id -> robot_id)
            approximation_level: Level of approximation
            
        Returns:
            Robot payments (robot_id -> payment)
        """
        payments = {}
        
        if approximation_level == 0:
            # Full VCG payments
            # Calculate social welfare with all robots
            total_welfare = sum(values.get((allocation[task.id], task.id), 0.0) for task in tasks if task.id in allocation)
            
            # For each robot with allocated tasks
            for robot_id in set(allocation.values()):
                # Calculate welfare without this robot
                allocation_without_robot = {
                    task_id: r_id for task_id, r_id in allocation.items() if r_id != robot_id
                }
                
                # Re-solve winner determination without this robot
                welfare_without_robot = 0.0
                for task in tasks:
                    if task.id not in allocation_without_robot:
                        # Task was allocated to this robot, find next best robot
                        best_value = 0.0
                        for other_robot in robots:
                            if other_robot.id != robot_id:
                                key = (other_robot.id, task.id)
                                value = values.get(key, 0.0)
                                best_value = max(best_value, value)
                        welfare_without_robot += best_value
                    else:
                        # Task was allocated to another robot
                        key = (allocation_without_robot[task.id], task.id)
                        welfare_without_robot += values.get(key, 0.0)
                
                # Calculate robot's contribution to welfare
                robot_tasks = [task_id for task_id, r_id in allocation.items() if r_id == robot_id]
                robot_welfare = sum(values.get((robot_id, task_id), 0.0) for task_id in robot_tasks)
                
                # VCG payment: welfare without robot - (total welfare - robot's welfare)
                payment = welfare_without_robot - (total_welfare - robot_welfare)
                
                # Ensure payment is non-negative
                payments[robot_id] = max(0.0, payment)
        else:
            # Approximate VCG payments
            # Use second-price auction-like payments for each task
            for task_id, robot_id in allocation.items():
                # Find second-highest value
                values_for_task = [(r.id, values.get((r.id, task_id), 0.0)) for r in robots if r.id != robot_id]
                second_best = max(values_for_task, key=lambda x: x[1], default=(None, 0.0))
                
                # Payment is second-highest value
                payment = second_best[1]
                
                # Add to robot's total payment
                payments[robot_id] = payments.get(robot_id, 0.0) + payment
        
        return payments
    
    def execute_sequential_mechanism(self, robots: List[Robot], tasks: List[Task], 
                                    approximation_level: int, use_precomputation: bool,
                                    time_limit: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute sequential auction mechanism with time constraints.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            approximation_level: Level of approximation
            use_precomputation: Whether to use precomputed values
            time_limit: Time limit in ms
            
        Returns:
            Tuple of (allocation, payments)
        """
        start_time = time.time()
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Sort tasks by priority and value
        sorted_tasks = sorted(tasks, 
                             key=lambda t: (t.priority.value, t.value), 
                             reverse=True)
        
        # Keep track of allocated robots (for approximation level 1)
        allocated_robots = set()
        
        # For each task, run a separate second-price auction
        for task in sorted_tasks:
            # Check if we're running out of time
            current_time = time.time()
            elapsed_ms = (current_time - start_time) * 1000
            if elapsed_ms > time_limit:
                print(f"Sequential auction exceeded time limit ({elapsed_ms:.2f} ms), stopping early")
                break
            
            bids = {}  # Robot ID -> Bid
            
            # Collect bids from all robots
            for robot in robots:
                # Skip if robot is already allocated (in approximation level 1)
                if approximation_level == 1 and robot.id in allocated_robots:
                    continue
                
                # Get value
                key = (robot.id, task.id)
                
                if use_precomputation and key in self.value_estimates:
                    # Use precomputed value
                    value = self.value_estimates[key]
                else:
                    # Calculate value on the fly
                    distance = self.calculate_distance(robot.position, task.position)
                    match = self.calculate_capability_match(robot, task)
                    value = self.estimate_value(robot, task, distance, match)
                
                # Record bid if positive
                if value > 0:
                    bids[robot.id] = value
            
            # Determine winner (highest bid)
            if bids:
                winner_id = max(bids.items(), key=lambda x: x[1])[0]
                winning_bid = bids[winner_id]
                
                # Determine payment (second-highest bid)
                other_bids = [b for r, b in bids.items() if r != winner_id]
                if other_bids:
                    payment = max(other_bids)
                else:
                    payment = winning_bid * 0.5  # If no other bids, pay half the bid
                
                # Record allocation and payment
                allocation[task.id] = winner_id
                payments[winner_id] = payments.get(winner_id, 0.0) + payment
                
                # Mark robot as allocated (for approximation level 1)
                if approximation_level == 1:
                    allocated_robots.add(winner_id)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Sequential mechanism completed in {elapsed_ms:.2f} ms (limit: {time_limit:.2f} ms)")
        
        return allocation, payments
    
    def execute_greedy_mechanism(self, robots: List[Robot], tasks: List[Task], 
                                approximation_level: int, use_precomputation: bool,
                                time_limit: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute greedy mechanism with time constraints.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            approximation_level: Level of approximation
            use_precomputation: Whether to use precomputed values
            time_limit: Time limit in ms
            
        Returns:
            Tuple of (allocation, payments)
        """
        start_time = time.time()
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Sort tasks by priority
        if approximation_level <= 1:
            # Consider priority and value
            sorted_tasks = sorted(tasks, 
                                 key=lambda t: (t.priority.value, t.value), 
                                 reverse=True)
        else:
            # Consider only priority
            sorted_tasks = sorted(tasks, 
                                 key=lambda t: t.priority.value, 
                                 reverse=True)
        
        # For each task, find the best robot
        for task in sorted_tasks:
            # Check if we're running out of time
            current_time = time.time()
            elapsed_ms = (current_time - start_time) * 1000
            if elapsed_ms > time_limit:
                print(f"Greedy mechanism exceeded time limit ({elapsed_ms:.2f} ms), stopping early")
                break
            
            best_robot_id = None
            best_value = 0.0
            
            for robot in robots:
                # Get value
                key = (robot.id, task.id)
                
                if use_precomputation and key in self.value_estimates:
                    # Use precomputed value
                    value = self.value_estimates[key]
                else:
                    # Calculate value on the fly
                    if approximation_level <= 1:
                        # Full calculation
                        distance = self.calculate_distance(robot.position, task.position)
                        match = self.calculate_capability_match(robot, task)
                        value = self.estimate_value(robot, task, distance, match)
                    else:
                        # Simplified calculation
                        if task.required_capabilities.issubset(robot.capabilities):
                            value = task.value * (1.0 - task.difficulty)
                        else:
                            value = 0.0
                
                # Update best robot if value is higher
                if value > best_value:
                    best_value = value
                    best_robot_id = robot.id
            
            # Allocate task if any robot values it positively
            if best_robot_id is not None and best_value > 0:
                allocation[task.id] = best_robot_id
                
                # Simple payment: proportional to task value
                payment = task.value * 0.5  # 50% of task value
                payments[best_robot_id] = payments.get(best_robot_id, 0.0) + payment
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Greedy mechanism completed in {elapsed_ms:.2f} ms (limit: {time_limit:.2f} ms)")
        
        return allocation, payments
    
    def execute_emergency_mechanism(self, robots: List[Robot], tasks: List[Task], 
                                   time_limit: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute emergency mechanism for extremely limited time.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            time_limit: Time limit in ms
            
        Returns:
            Tuple of (allocation, payments)
        """
        start_time = time.time()
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Only consider critical tasks
        critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
        
        # Simple capability-based allocation
        for task in critical_tasks:
            # Check if we're running out of time
            current_time = time.time()
            elapsed_ms = (current_time - start_time) * 1000
            if elapsed_ms > time_limit:
                print(f"Emergency mechanism exceeded time limit ({elapsed_ms:.2f} ms), stopping early")
                break
            
            # Find any robot with required capabilities
            capable_robots = [r for r in robots if task.required_capabilities.issubset(r.capabilities)]
            
            if capable_robots:
                # Allocate to first capable robot
                robot = capable_robots[0]
                allocation[task.id] = robot.id
                
                # Fixed payment
                payment = task.value * 0.5
                payments[robot.id] = payments.get(robot.id, 0.0) + payment
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Emergency mechanism completed in {elapsed_ms:.2f} ms (limit: {time_limit:.2f} ms)")
        
        return allocation, payments
    
    def execute_real_time_allocation(self, robots: List[Robot], tasks: List[Task]) -> Tuple[Dict[str, str], Dict[str, float], str]:
        """
        Execute real-time task allocation with time constraints.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Tuple of (allocation, payments, mechanism type)
        """
        # Start timing
        start_time = time.time()
        
        # Determine time constraints
        # Use the most restrictive constraint among robots
        robot_time_limits = [robot.max_computation_time for robot in robots]
        time_limit = min(robot_time_limits) if robot_time_limits else self.global_deadline
        
        print(f"Executing real-time allocation with time limit: {time_limit:.2f} ms")
        
        # Precomputation phase
        if self.precomputation_enabled:
            # Estimate time for precomputation
            precomputation_time_limit = time_limit * 0.3  # Use up to 30% of time for precomputation
            
            precomputation_start = time.time()
            
            # Precompute distances
            self.precompute_distances(robots, tasks)
            
            current_time = time.time()
            elapsed_ms = (current_time - precomputation_start) * 1000
            
            if elapsed_ms < precomputation_time_limit:
                # Precompute capability matches
                self.precompute_capability_matches(robots, tasks)
            
            current_time = time.time()
            elapsed_ms = (current_time - precomputation_start) * 1000
            
            if elapsed_ms < precomputation_time_limit:
                # Precompute value estimates
                self.precompute_value_estimates(robots, tasks)
            
            precomputation_end = time.time()
            precomputation_elapsed = (precomputation_end - precomputation_start) * 1000
            print(f"Precomputation phase completed in {precomputation_elapsed:.2f} ms")
            
            # Update time limit
            time_limit -= precomputation_elapsed
        
        # Select mechanism and precision level
        mechanism_type, approximation_level, use_precomputation = self.select_precision_level(time_limit, robots)
        
        print(f"Selected mechanism: {mechanism_type}, approximation level: {approximation_level}")
        self.precision_levels.append(approximation_level)
        
        # Execute selected mechanism
        if mechanism_type == "vcg":
            allocation, payments = self.execute_vcg_mechanism(
                robots, tasks, approximation_level, use_precomputation, time_limit)
        elif mechanism_type == "sequential":
            allocation, payments = self.execute_sequential_mechanism(
                robots, tasks, approximation_level, use_precomputation, time_limit)
        elif mechanism_type == "greedy":
            allocation, payments = self.execute_greedy_mechanism(
                robots, tasks, approximation_level, use_precomputation, time_limit)
        else:  # emergency
            allocation, payments = self.execute_emergency_mechanism(
                robots, tasks, time_limit)
        
        # Record total computation time
        end_time = time.time()
        total_elapsed_ms = (end_time - start_time) * 1000
        self.computation_times.append(total_elapsed_ms)
        
        # Check for deadline violation
        if total_elapsed_ms > self.global_deadline:
            self.deadline_violations += 1
            print(f"WARNING: Deadline violation! Computation took {total_elapsed_ms:.2f} ms (limit: {self.global_deadline:.2f} ms)")
        
        print(f"Real-time allocation completed in {total_elapsed_ms:.2f} ms")
        
        return allocation, payments, mechanism_type


def simulate_real_time_allocation():
    """Run a simulation of the real-time task allocation mechanism."""
    # Create robots with different hardware classes
    robots = [
        Robot(id="R1", position=(0, 0), 
              capabilities={"sensing", "manipulation"}, 
              hardware_class=HardwareClass.HIGH_END,
              computation_capacity=0.9, communication_capacity=0.8,
              memory_capacity=0.7, energy_level=0.6,
              max_computation_time=100.0),
        Robot(id="R2", position=(10, 10), 
              capabilities={"sensing", "locomotion"}, 
              hardware_class=HardwareClass.MID_RANGE,
              computation_capacity=0.7, communication_capacity=0.9,
              memory_capacity=0.8, energy_level=0.5,
              max_computation_time=80.0),
        Robot(id="R3", position=(20, 20), 
              capabilities={"manipulation", "locomotion"}, 
              hardware_class=HardwareClass.MID_RANGE,
              computation_capacity=0.6, communication_capacity=0.7,
              memory_capacity=0.9, energy_level=0.8,
              max_computation_time=70.0),
        Robot(id="R4", position=(30, 30), 
              capabilities={"sensing", "manipulation", "locomotion"}, 
              hardware_class=HardwareClass.LOW_END,
              computation_capacity=0.5, communication_capacity=0.4,
              memory_capacity=0.3, energy_level=0.9,
              max_computation_time=50.0)
    ]
    
    # Create tasks with different priorities
    tasks = [
        Task(id="T1", position=(5, 5), difficulty=0.3, deadline=300, value=100, 
             required_capabilities={"sensing"}, priority=TaskPriority.MEDIUM,
             computation_requirement=0.3, communication_requirement=0.2, memory_requirement=0.4),
        Task(id="T2", position=(15, 15), difficulty=0.5, deadline=200, value=150, 
             required_capabilities={"manipulation"}, priority=TaskPriority.HIGH,
             computation_requirement=0.5, communication_requirement=0.4, memory_requirement=0.6),
        Task(id="T3", position=(25, 25), difficulty=0.7, deadline=100, value=200, 
             required_capabilities={"sensing", "manipulation"}, priority=TaskPriority.CRITICAL,
             computation_requirement=0.7, communication_requirement=0.6, memory_requirement=0.8),
        Task(id="T4", position=(35, 35), difficulty=0.9, deadline=50, value=250, 
             required_capabilities={"locomotion", "manipulation"}, priority=TaskPriority.LOW,
             computation_requirement=0.9, communication_requirement=0.8, memory_requirement=0.7)
    ]
    
    # Initialize mechanism
    mechanism = RealTimeMechanism(
        global_deadline=100.0,
        precomputation_enabled=True,
        adaptive_precision=True
    )
    
    # Run allocation
    allocation, payments, mechanism_type = mechanism.execute_real_time_allocation(robots, tasks)
    
    # Print results
    print(f"\nMechanism type used: {mechanism_type}")
    print("\nTask Allocation:")
    for task_id, robot_id in allocation.items():
        task = next(t for t in tasks if t.id == task_id)
        print(f"Task {task_id} (Priority: {task.priority.name}) -> Robot {robot_id} (Hardware: {next(r for r in robots if r.id == robot_id).hardware_class.name})")
    
    print("\nPayments:")
    for robot_id, payment in payments.items():
        print(f"Robot {robot_id}: {payment:.2f}")
    
    # Run with tighter time constraints
    print("\n\nRunning with tighter time constraints...")
    
    # Reduce computation time limits
    for robot in robots:
        robot.max_computation_time *= 0.3  # 30% of original
    
    # Initialize new mechanism with tighter deadline
    tight_mechanism = RealTimeMechanism(
        global_deadline=30.0,  # 30% of original
        precomputation_enabled=True,
        adaptive_precision=True
    )
    
    # Run allocation
    tight_allocation, tight_payments, tight_mechanism_type = tight_mechanism.execute_real_time_allocation(robots, tasks)
    
    # Print results
    print(f"\nMechanism type used with tight constraints: {tight_mechanism_type}")
    print("\nTask Allocation:")
    for task_id, robot_id in tight_allocation.items():
        task = next(t for t in tasks if t.id == task_id)
        print(f"Task {task_id} (Priority: {task.priority.name}) -> Robot {robot_id} (Hardware: {next(r for r in robots if r.id == robot_id).hardware_class.name})")
    
    print("\nPayments:")
    for robot_id, payment in tight_payments.items():
        print(f"Robot {robot_id}: {payment:.2f}")
    
    # Compare results
    print("\nComparison:")
    print(f"Normal constraints: {len(allocation)} tasks allocated using {mechanism_type}")
    print(f"Tight constraints: {len(tight_allocation)} tasks allocated using {tight_mechanism_type}")
    
    # Check if critical tasks were allocated in both cases
    critical_tasks = [t.id for t in tasks if t.priority == TaskPriority.CRITICAL]
    normal_critical_allocated = all(task_id in allocation for task_id in critical_tasks)
    tight_critical_allocated = all(task_id in tight_allocation for task_id in critical_tasks)
    
    print(f"Critical tasks allocated in normal constraints: {normal_critical_allocated}")
    print(f"Critical tasks allocated in tight constraints: {tight_critical_allocated}")


if __name__ == "__main__":
    simulate_real_time_allocation()
