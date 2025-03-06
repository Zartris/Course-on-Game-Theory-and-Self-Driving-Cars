"""
Resource-Aware Mechanism Design for Multi-Robot Systems

This module implements a resource-aware task allocation mechanism that explicitly
considers computational, communication, and energy resources in mechanism implementation.
"""

import random
import math
import time
from typing import Dict, List, Tuple, Set, Optional, Any, FrozenSet
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Types of resources to consider in resource-aware mechanism design."""
    COMPUTATION = 0
    COMMUNICATION = 1
    MEMORY = 2
    ENERGY = 3


@dataclass
class Task:
    """Representation of a task to be allocated."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    difficulty: float  # 0.0 to 1.0
    deadline: float  # seconds from now
    value: float
    required_capabilities: Set[str]


@dataclass
class Robot:
    """Representation of a robot in the system."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    capabilities: Set[str]
    computation_capacity: float  # 0.0 to 1.0
    communication_capacity: float  # 0.0 to 1.0
    memory_capacity: float  # 0.0 to 1.0
    energy_level: float  # 0.0 to 1.0


class ResourceAwareTaskAllocation:
    """
    Implements a resource-aware task allocation mechanism that adapts to
    hardware constraints and optimizes resource usage.
    """
    
    def __init__(self, resource_constraints: Dict[ResourceType, float]):
        """
        Initialize the resource-aware task allocation mechanism.
        
        Args:
            resource_constraints: Maximum available resources for each resource type
        """
        self.resource_constraints = resource_constraints
        self.current_resource_usage = {
            resource_type: 0.0 for resource_type in ResourceType
        }
        self.task_bundles = {}  # Task ID -> Bundle ID
        self.bundle_values = {}  # Bundle ID -> Value
        self.allocation = {}  # Task ID -> Robot ID
        self.payments = {}  # Robot ID -> Payment
    
    def select_mechanism_variant(self) -> str:
        """
        Select appropriate mechanism variant based on resource constraints.
        
        Returns:
            Name of the selected mechanism variant
        """
        # Calculate available resources as percentage of constraints
        available_resources = {
            resource_type: 1.0 - (self.current_resource_usage[resource_type] / 
                               self.resource_constraints.get(resource_type, float('inf')))
            for resource_type in ResourceType
        }
        
        # Select mechanism based on available resources
        if (available_resources[ResourceType.COMPUTATION] > 0.7 and 
            available_resources[ResourceType.COMMUNICATION] > 0.7 and
            available_resources[ResourceType.MEMORY] > 0.7 and
            available_resources[ResourceType.ENERGY] > 0.7):
            # High resources available: use sophisticated mechanism
            return "combinatorial_auction"
        elif (available_resources[ResourceType.COMPUTATION] > 0.3 and 
              available_resources[ResourceType.COMMUNICATION] > 0.3 and
              available_resources[ResourceType.MEMORY] > 0.3 and
              available_resources[ResourceType.ENERGY] > 0.3):
            # Medium resources available: use intermediate mechanism
            return "sequential_auction"
        else:
            # Low resources available: use lightweight mechanism
            return "greedy_allocation"
    
    def estimate_resource_usage(self, mechanism_variant: str, num_robots: int, 
                               num_tasks: int) -> Dict[ResourceType, float]:
        """
        Estimate resource usage for a mechanism variant.
        
        Args:
            mechanism_variant: Name of the mechanism variant
            num_robots: Number of robots in the system
            num_tasks: Number of tasks to allocate
            
        Returns:
            Dictionary mapping resource types to estimated usage
        """
        if mechanism_variant == "combinatorial_auction":
            # Combinatorial auction has exponential complexity in the number of tasks
            # and linear complexity in the number of robots
            computation = 0.1 * num_robots * min(2 ** num_tasks, 1000)  # Cap to avoid overflow
            communication = 0.05 * num_robots * num_tasks * 10  # Higher communication overhead
            memory = 0.2 * num_robots * num_tasks * 5  # Higher memory requirements
            energy = 0.1 * (computation + communication)  # Energy proportional to computation and communication
        elif mechanism_variant == "sequential_auction":
            # Sequential auction has quadratic complexity in the number of tasks
            # and linear complexity in the number of robots
            computation = 0.1 * num_robots * num_tasks * 2
            communication = 0.05 * num_robots * num_tasks * 3
            memory = 0.2 * num_robots * num_tasks
            energy = 0.1 * (computation + communication)
        else:  # greedy_allocation
            # Greedy allocation has linear complexity in both robots and tasks
            computation = 0.1 * num_robots * num_tasks
            communication = 0.05 * num_robots * num_tasks
            memory = 0.2 * num_tasks
            energy = 0.05 * (computation + communication)
        
        return {
            ResourceType.COMPUTATION: computation,
            ResourceType.COMMUNICATION: communication,
            ResourceType.MEMORY: memory,
            ResourceType.ENERGY: energy
        }
    
    def update_resource_usage(self, usage: Dict[ResourceType, float]) -> None:
        """
        Update current resource usage.
        
        Args:
            usage: Resource usage to add
        """
        for resource_type, amount in usage.items():
            self.current_resource_usage[resource_type] += amount
    
    def identify_key_bundles(self, tasks: List[Task], max_bundles: int) -> List[FrozenSet[str]]:
        """
        Identify key bundles for initial preference elicitation.
        
        Args:
            tasks: List of tasks
            max_bundles: Maximum number of bundles to identify
            
        Returns:
            List of task bundles (as frozensets of task IDs)
        """
        # Start with singleton bundles
        key_bundles = [frozenset([task.id]) for task in tasks]
        
        # If we have capacity for more bundles, add pairs of related tasks
        if max_bundles > len(tasks):
            # Group tasks by required capabilities
            capability_groups = {}
            for task in tasks:
                cap_key = frozenset(task.required_capabilities)
                if cap_key not in capability_groups:
                    capability_groups[cap_key] = []
                capability_groups[cap_key].append(task)
            
            # Create pairs within each capability group
            for group in capability_groups.values():
                if len(group) >= 2:
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            if len(key_bundles) < max_bundles:
                                key_bundles.append(frozenset([group[i].id, group[j].id]))
        
        return key_bundles[:max_bundles]
    
    def progressive_preference_elicitation(self, robots: List[Robot], tasks: List[Task], 
                                          max_bundles: Optional[int] = None) -> Tuple[Dict[str, Dict[FrozenSet[str], float]], Dict[str, str]]:
        """
        Elicit preferences progressively to reduce communication.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            max_bundles: Maximum number of bundles to query initially
            
        Returns:
            Tuple of (preferences, allocation)
        """
        # Start with minimal preference information
        preferences = {robot.id: {} for robot in robots}
        
        # Determine key bundles to query initially
        if max_bundles is None:
            # Adaptive based on resource constraints
            available_comm = (self.resource_constraints.get(ResourceType.COMMUNICATION, float('inf')) - 
                             self.current_resource_usage[ResourceType.COMMUNICATION])
            max_bundles = max(1, int(available_comm / (0.05 * len(robots))))
        
        # Initial preference elicitation for key bundles
        key_bundles = self.identify_key_bundles(tasks, max_bundles)
        
        print(f"Eliciting preferences for {len(key_bundles)} key bundles")
        
        for robot in robots:
            for bundle in key_bundles:
                # Evaluate bundle
                bundle_value = self.evaluate_bundle(robot, bundle, tasks)
                preferences[robot.id][bundle] = bundle_value
        
        # Update communication resource usage
        comm_usage = 0.05 * len(robots) * len(key_bundles)
        self.update_resource_usage({ResourceType.COMMUNICATION: comm_usage})
        
        # Initial allocation based on elicited preferences
        allocation = self.initial_allocation(preferences, tasks)
        
        # Progressive refinement if resources permit
        refinement_rounds = 0
        max_refinement_rounds = 3  # Limit refinement to control resource usage
        
        while (self.can_improve_allocation(allocation, preferences) and 
               self.has_sufficient_resources() and 
               refinement_rounds < max_refinement_rounds):
            
            refinement_rounds += 1
            print(f"Refinement round {refinement_rounds}")
            
            # Identify most valuable additional preference information
            next_queries = self.identify_next_queries(allocation, preferences, robots, tasks)
            
            # Elicit additional preferences
            for robot_id, bundles in next_queries.items():
                robot = next(r for r in robots if r.id == robot_id)
                for bundle in bundles:
                    bundle_value = self.evaluate_bundle(robot, bundle, tasks)
                    preferences[robot_id][bundle] = bundle_value
            
            # Update communication resource usage
            query_count = sum(len(bundles) for bundles in next_queries.values())
            comm_usage = 0.05 * query_count
            self.update_resource_usage({ResourceType.COMMUNICATION: comm_usage})
            
            # Refine allocation
            allocation = self.refine_allocation(allocation, preferences, tasks)
        
        return preferences, allocation
    
    def evaluate_bundle(self, robot: Robot, bundle: FrozenSet[str], tasks: List[Task]) -> float:
        """
        Evaluate a bundle of tasks for a robot.
        
        Args:
            robot: The robot
            bundle: Bundle of task IDs
            tasks: List of all tasks
            
        Returns:
            Value of the bundle for the robot
        """
        # Get tasks in the bundle
        bundle_tasks = [task for task in tasks if task.id in bundle]
        
        # Check if robot has required capabilities for all tasks
        for task in bundle_tasks:
            if not task.required_capabilities.issubset(robot.capabilities):
                return 0.0  # Cannot perform at least one task in the bundle
        
        # Base value is sum of task values
        base_value = sum(task.value for task in bundle_tasks)
        
        # Adjust for synergies between tasks
        # Tasks with similar required capabilities have positive synergy
        if len(bundle_tasks) > 1:
            capability_similarity = self.calculate_capability_similarity(bundle_tasks)
            synergy_factor = 1.0 + (0.2 * capability_similarity)
        else:
            synergy_factor = 1.0
        
        # Adjust for distance between tasks
        if len(bundle_tasks) > 1:
            distance_penalty = self.calculate_distance_penalty(robot, bundle_tasks)
        else:
            distance_penalty = 0.0
        
        # Adjust for robot capabilities
        capability_match = self.calculate_capability_match(robot, bundle_tasks)
        
        # Calculate final value
        bundle_value = base_value * synergy_factor * capability_match - distance_penalty
        
        return max(0.0, bundle_value)
    
    def calculate_capability_similarity(self, tasks: List[Task]) -> float:
        """
        Calculate similarity between task capabilities.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not tasks:
            return 0.0
        
        # Get all capability sets
        capability_sets = [task.required_capabilities for task in tasks]
        
        # Calculate average Jaccard similarity
        total_similarity = 0.0
        comparison_count = 0
        
        for i in range(len(capability_sets)):
            for j in range(i+1, len(capability_sets)):
                set_i = capability_sets[i]
                set_j = capability_sets[j]
                
                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                if union > 0:
                    similarity = intersection / union
                    total_similarity += similarity
                    comparison_count += 1
        
        # Return average similarity
        if comparison_count > 0:
            return total_similarity / comparison_count
        else:
            return 1.0  # Single task or all tasks have identical capabilities
    
    def calculate_distance_penalty(self, robot: Robot, tasks: List[Task]) -> float:
        """
        Calculate distance penalty for a set of tasks.
        
        Args:
            robot: The robot
            tasks: List of tasks
            
        Returns:
            Distance penalty value
        """
        if len(tasks) <= 1:
            return 0.0
        
        # Calculate total distance of the shortest path visiting all tasks
        # Starting from the robot's position
        # This is a simplified TSP calculation
        
        # Start with robot position
        current_position = robot.position
        unvisited = tasks.copy()
        total_distance = 0.0
        
        # Greedy nearest neighbor algorithm
        while unvisited:
            # Find nearest unvisited task
            nearest_task = min(unvisited, key=lambda t: self.distance(current_position, t.position))
            
            # Add distance
            total_distance += self.distance(current_position, nearest_task.position)
            
            # Update current position
            current_position = nearest_task.position
            
            # Mark as visited
            unvisited.remove(nearest_task)
        
        # Convert distance to penalty (higher distance = higher penalty)
        # Scale based on task values to make it comparable
        avg_task_value = sum(task.value for task in tasks) / len(tasks)
        distance_penalty = total_distance * 0.1 * avg_task_value
        
        return distance_penalty
    
    def calculate_capability_match(self, robot: Robot, tasks: List[Task]) -> float:
        """
        Calculate how well robot capabilities match task requirements.
        
        Args:
            robot: The robot
            tasks: List of tasks
            
        Returns:
            Capability match score (0.0 to 1.0)
        """
        if not tasks:
            return 1.0
        
        # Get all required capabilities across tasks
        all_required = set()
        for task in tasks:
            all_required.update(task.required_capabilities)
        
        # Check if robot has all required capabilities
        if not all_required.issubset(robot.capabilities):
            return 0.0  # Missing some required capabilities
        
        # Calculate match quality based on specialization
        # If robot has exactly the required capabilities, it's a perfect match
        # If robot has many more capabilities, it's less specialized
        specialization = len(all_required) / max(1, len(robot.capabilities))
        
        # Adjust match based on robot's resource levels
        resource_factor = min(
            robot.computation_capacity,
            robot.communication_capacity,
            robot.memory_capacity,
            robot.energy_level
        )
        
        return specialization * resource_factor
    
    def distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def initial_allocation(self, preferences: Dict[str, Dict[FrozenSet[str], float]], 
                          tasks: List[Task]) -> Dict[str, str]:
        """
        Create initial allocation based on elicited preferences.
        
        Args:
            preferences: Robot preferences (Robot ID -> {Bundle -> Value})
            tasks: List of tasks
            
        Returns:
            Task allocation (Task ID -> Robot ID)
        """
        # Start with empty allocation
        allocation = {}
        
        # Get all task IDs
        all_task_ids = {task.id for task in tasks}
        
        # Get all singleton bundles
        singleton_bundles = {frozenset([task_id]) for task_id in all_task_ids}
        
        # For each task, find the robot with highest valuation
        for task_id in all_task_ids:
            singleton = frozenset([task_id])
            
            # Find best robot
            best_robot_id = None
            best_value = float('-inf')
            
            for robot_id, robot_prefs in preferences.items():
                if singleton in robot_prefs:
                    value = robot_prefs[singleton]
                    if value > best_value:
                        best_value = value
                        best_robot_id = robot_id
            
            # Allocate task if any robot values it positively
            if best_robot_id is not None and best_value > 0:
                allocation[task_id] = best_robot_id
        
        return allocation
    
    def can_improve_allocation(self, allocation: Dict[str, str], 
                              preferences: Dict[str, Dict[FrozenSet[str], float]]) -> bool:
        """
        Check if allocation can potentially be improved with more information.
        
        Args:
            allocation: Current allocation (Task ID -> Robot ID)
            preferences: Current preferences (Robot ID -> {Bundle -> Value})
            
        Returns:
            True if improvement is possible, False otherwise
        """
        # If not all tasks are allocated, improvement is possible
        # This is a simplified check - in reality, we would need more sophisticated analysis
        return len(allocation) < sum(1 for prefs in preferences.values() for bundle in prefs if len(bundle) == 1)
    
    def has_sufficient_resources(self) -> bool:
        """
        Check if sufficient resources are available for further refinement.
        
        Returns:
            True if sufficient resources are available, False otherwise
        """
        # Check if any resource is near its limit
        for resource_type in ResourceType:
            if resource_type in self.resource_constraints:
                usage_ratio = self.current_resource_usage[resource_type] / self.resource_constraints[resource_type]
                if usage_ratio > 0.9:  # 90% usage is considered near limit
                    return False
        
        return True
    
    def identify_next_queries(self, allocation: Dict[str, str], 
                             preferences: Dict[str, Dict[FrozenSet[str], float]], 
                             robots: List[Robot], tasks: List[Task]) -> Dict[str, List[FrozenSet[str]]]:
        """
        Identify most valuable additional preference information to query.
        
        Args:
            allocation: Current allocation (Task ID -> Robot ID)
            preferences: Current preferences (Robot ID -> {Bundle -> Value})
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Dict mapping robot IDs to lists of bundles to query
        """
        # This is a simplified implementation that focuses on:
        # 1. Unallocated tasks
        # 2. Potential improvements to current allocation through bundles
        
        queries = {robot.id: [] for robot in robots}
        
        # Get all task IDs
        all_task_ids = {task.id for task in tasks}
        
        # Identify unallocated tasks
        unallocated_task_ids = all_task_ids - set(allocation.keys())
        
        # For each robot, identify potentially valuable bundles to query
        for robot in robots:
            # Skip if robot has no capabilities
            if not robot.capabilities:
                continue
            
            # Get tasks that match robot capabilities
            matching_tasks = [task for task in tasks if task.required_capabilities.issubset(robot.capabilities)]
            
            # Prioritize unallocated tasks
            unallocated_matching = [task for task in matching_tasks if task.id in unallocated_task_ids]
            
            # Query some pairs of tasks (one allocated + one unallocated)
            allocated_to_robot = [task for task in matching_tasks if allocation.get(task.id) == robot.id]
            
            # Create pairs of (allocated + unallocated) tasks
            for allocated_task in allocated_to_robot:
                for unallocated_task in unallocated_matching:
                    if allocated_task.id != unallocated_task.id:
                        bundle = frozenset([allocated_task.id, unallocated_task.id])
                        
                        # Check if we already know the value
                        if bundle not in preferences[robot.id]:
                            queries[robot.id].append(bundle)
            
            # Limit number of queries per robot to control communication
            queries[robot.id] = queries[robot.id][:3]  # Maximum 3 new queries per robot
        
        return queries
    
    def refine_allocation(self, allocation: Dict[str, str], 
                         preferences: Dict[str, Dict[FrozenSet[str], float]], 
                         tasks: List[Task]) -> Dict[str, str]:
        """
        Refine allocation based on updated preferences.
        
        Args:
            allocation: Current allocation (Task ID -> Robot ID)
            preferences: Updated preferences (Robot ID -> {Bundle -> Value})
            tasks: List of tasks
            
        Returns:
            Refined allocation (Task ID -> Robot ID)
        """
        # This is a simplified greedy refinement that:
        # 1. Keeps track of allocated robots and tasks
        # 2. Tries to find improvements by considering bundles
        
        # Start with current allocation
        new_allocation = allocation.copy()
        
        # Keep track of which robots and tasks are allocated
        allocated_robots = set(new_allocation.values())
        allocated_tasks = set(new_allocation.keys())
        
        # Get all task IDs
        all_task_ids = {task.id for task in tasks}
        
        # Identify unallocated tasks
        unallocated_task_ids = all_task_ids - allocated_tasks
        
        # Try to allocate unallocated tasks
        for task_id in unallocated_task_ids:
            singleton = frozenset([task_id])
            
            # Find best robot
            best_robot_id = None
            best_value = 0.0  # Only allocate if value is positive
            
            for robot_id, robot_prefs in preferences.items():
                if singleton in robot_prefs:
                    value = robot_prefs[singleton]
                    if value > best_value:
                        best_value = value
                        best_robot_id = robot_id
            
            # Allocate task if any robot values it positively
            if best_robot_id is not None and best_value > 0:
                new_allocation[task_id] = best_robot_id
        
        # Try to improve allocation by considering bundles
        # This is a simplified approach that doesn't guarantee optimality
        
        # Look for valuable bundles
        for robot_id, robot_prefs in preferences.items():
            for bundle, bundle_value in robot_prefs.items():
                # Skip singletons
                if len(bundle) <= 1:
                    continue
                
                # Check if bundle is better than current allocation
                current_value = 0.0
                
                # Calculate value of current allocation for these tasks
                for task_id in bundle:
                    current_robot_id = new_allocation.get(task_id)
                    if current_robot_id is not None:
                        singleton = frozenset([task_id])
                        if singleton in preferences[current_robot_id]:
                            current_value += preferences[current_robot_id][singleton]
                
                # If bundle value is higher, reallocate
                if bundle_value > current_value:
                    # Allocate all tasks in bundle to this robot
                    for task_id in bundle:
                        new_allocation[task_id] = robot_id
        
        return new_allocation
    
    def execute_greedy_allocation(self, robots: List[Robot], tasks: List[Task]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute greedy task allocation (lightweight mechanism).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Tuple of (allocation, payments)
        """
        print("Executing greedy allocation (lightweight mechanism)")
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Sort tasks by value (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.value, reverse=True)
        
        # For each task, find the best robot
        for task in sorted_tasks:
            best_robot = None
            best_score = float('-inf')
            
            for robot in robots:
                # Skip if robot doesn't have required capabilities
                if not task.required_capabilities.issubset(robot.capabilities):
                    continue
                
                # Calculate score based on distance and resource levels
                distance = self.distance(robot.position, task.position)
                resource_level = min(
                    robot.computation_capacity,
                    robot.communication_capacity,
                    robot.memory_capacity,
                    robot.energy_level
                )
                
                # Score: higher is better
                score = task.value * resource_level - distance
                
                if score > best_score:
                    best_score = score
                    best_robot = robot
            
            # Allocate task if any robot is suitable
            if best_robot is not None and best_score > 0:
                allocation[task.id] = best_robot.id
                
                # Simple payment: proportional to task value
                payment = task.value * 0.5  # 50% of task value
                payments[best_robot.id] = payments.get(best_robot.id, 0.0) + payment
        
        return allocation, payments
    
    def execute_sequential_auction(self, robots: List[Robot], tasks: List[Task]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute sequential auction (intermediate mechanism).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Tuple of (allocation, payments)
        """
        print("Executing sequential auction (intermediate mechanism)")
        
        # Start with empty allocation and payments
        allocation = {}
        payments = {}
        
        # Sort tasks by value (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.value, reverse=True)
        
        # For each task, run a separate second-price auction
        for task in sorted_tasks:
            bids = {}  # Robot ID -> Bid
            
            # Collect bids from all robots
            for robot in robots:
                # Skip if robot doesn't have required capabilities
                if not task.required_capabilities.issubset(robot.capabilities):
                    continue
                
                # Calculate bid based on distance and resource levels
                distance = self.distance(robot.position, task.position)
                resource_level = min(
                    robot.computation_capacity,
                    robot.communication_capacity,
                    robot.memory_capacity,
                    robot.energy_level
                )
                
                # Bid: higher is better
                bid = task.value * resource_level - distance
                
                # Record bid if positive
                if bid > 0:
                    bids[robot.id] = bid
            
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
        
        return allocation, payments
    
    def execute_combinatorial_auction(self, robots: List[Robot], tasks: List[Task]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute combinatorial auction (sophisticated mechanism).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Tuple of (allocation, payments)
        """
        print("Executing combinatorial auction (sophisticated mechanism)")
        
        # Use progressive preference elicitation to reduce communication
        preferences, allocation = self.progressive_preference_elicitation(robots, tasks)
        
        # Calculate VCG-like payments
        payments = self.compute_vcg_payments(preferences, allocation, robots, tasks)
        
        return allocation, payments
    
    def compute_vcg_payments(self, preferences: Dict[str, Dict[FrozenSet[str], float]], 
                            allocation: Dict[str, str], robots: List[Robot], 
                            tasks: List[Task]) -> Dict[str, float]:
        """
        Compute VCG payments based on preferences.
        
        Args:
            preferences: Robot preferences (Robot ID -> {Bundle -> Value})
            allocation: Task allocation (Task ID -> Robot ID)
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Robot payments (Robot ID -> Payment)
        """
        payments = {}
        
        # Group tasks by allocated robot
        robot_tasks = {}
        for task_id, robot_id in allocation.items():
            if robot_id not in robot_tasks:
                robot_tasks[robot_id] = []
            robot_tasks[robot_id].append(task_id)
        
        # Calculate welfare with all robots
        total_welfare = self.calculate_welfare(preferences, allocation)
        
        # For each robot with allocated tasks
        for robot_id, allocated_tasks in robot_tasks.items():
            # Calculate welfare without this robot
            welfare_without_robot = self.calculate_welfare_without_robot(
                preferences, allocation, robot_id)
            
            # Calculate this robot's contribution to welfare
            robot_bundle = frozenset(allocated_tasks)
            robot_value = preferences[robot_id].get(robot_bundle, 0.0)
            
            if len(allocated_tasks) == 1:
                # For singleton allocations, use the singleton value
                singleton = frozenset([allocated_tasks[0]])
                robot_value = preferences[robot_id].get(singleton, 0.0)
            
            # VCG payment: welfare without robot - (total welfare - robot's value)
            payment = welfare_without_robot - (total_welfare - robot_value)
            
            # Ensure payment is non-negative
            payments[robot_id] = max(0.0, payment)
        
        return payments
    
    def calculate_welfare(self, preferences: Dict[str, Dict[FrozenSet[str], float]], 
                         allocation: Dict[str, str]) -> float:
        """
        Calculate total welfare for an allocation.
        
        Args:
            preferences: Robot preferences (Robot ID -> {Bundle -> Value})
            allocation: Task allocation (Task ID -> Robot ID)
            
        Returns:
            Total welfare value
        """
        # Group tasks by allocated robot
        robot_tasks = {}
        for task_id, robot_id in allocation.items():
            if robot_id not in robot_tasks:
                robot_tasks[robot_id] = []
            robot_tasks[robot_id].append(task_id)
        
        # Calculate total welfare
        total_welfare = 0.0
        
        for robot_id, allocated_tasks in robot_tasks.items():
            # Get robot's value for its allocated tasks
            if len(allocated_tasks) == 1:
                # For singleton allocations, use the singleton value
                singleton = frozenset([allocated_tasks[0]])
                robot_value = preferences[robot_id].get(singleton, 0.0)
            else:
                # For bundles, use the bundle value if available
                bundle = frozenset(allocated_tasks)
                robot_value = preferences[robot_id].get(bundle, 0.0)
                
                # If bundle value not available, use sum of singleton values
                if robot_value == 0.0:
                    robot_value = sum(
                        preferences[robot_id].get(frozenset([task_id]), 0.0)
                        for task_id in allocated_tasks
                    )
            
            # Add to total welfare
            total_welfare += robot_value
        
        return total_welfare
    
    def calculate_welfare_without_robot(self, preferences: Dict[str, Dict[FrozenSet[str], float]], 
                                       allocation: Dict[str, str], excluded_robot_id: str) -> float:
        """
        Calculate welfare for an allocation without a specific robot.
        
        Args:
            preferences: Robot preferences (Robot ID -> {Bundle -> Value})
            allocation: Task allocation (Task ID -> Robot ID)
            excluded_robot_id: ID of the robot to exclude
            
        Returns:
            Welfare value without the excluded robot
        """
        # Create a new allocation without the excluded robot
        new_allocation = {
            task_id: robot_id
            for task_id, robot_id in allocation.items()
            if robot_id != excluded_robot_id
        }
        
        # Calculate welfare for the new allocation
        return self.calculate_welfare(preferences, new_allocation)
    
    def execute_allocation_mechanism(self, robots: List[Robot], tasks: List[Task]) -> Tuple[Dict[str, str], Dict[str, float], str]:
        """
        Execute task allocation with resource awareness.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Tuple of (allocation, payments, mechanism variant)
        """
        # Select mechanism variant based on resource constraints
        mechanism_variant = self.select_mechanism_variant()
        
        # Estimate resource usage
        estimated_usage = self.estimate_resource_usage(
            mechanism_variant, len(robots), len(tasks))
        
        # Check if we have sufficient resources
        for resource_type in ResourceType:
            if resource_type in self.resource_constraints:
                if self.current_resource_usage[resource_type] + estimated_usage[resource_type] > self.resource_constraints[resource_type]:
                    # Not enough resources for selected variant, downgrade
                    if mechanism_variant == "combinatorial_auction":
                        mechanism_variant = "sequential_auction"
                    elif mechanism_variant == "sequential_auction":
                        mechanism_variant = "greedy_allocation"
                    
                    # Re-estimate with downgraded variant
                    estimated_usage = self.estimate_resource_usage(
                        mechanism_variant, len(robots), len(tasks))
                    break
        
        # Execute selected mechanism
        if mechanism_variant == "combinatorial_auction":
            allocation, payments = self.execute_combinatorial_auction(robots, tasks)
        elif mechanism_variant == "sequential_auction":
            allocation, payments = self.execute_sequential_auction(robots, tasks)
        else:  # greedy_allocation
            allocation, payments = self.execute_greedy_allocation(robots, tasks)
        
        # Update resource usage
        self.update_resource_usage(estimated_usage)
        
        return allocation, payments, mechanism_variant


def simulate_resource_aware_allocation():
    """Run a simulation of the resource-aware task allocation mechanism."""
    # Define resource constraints
    resource_constraints = {
        ResourceType.COMPUTATION: 1000.0,
        ResourceType.COMMUNICATION: 500.0,
        ResourceType.MEMORY: 2000.0,
        ResourceType.ENERGY: 1500.0
    }
    
    # Create robots
    robots = [
        Robot(id="R1", position=(0, 0), 
              capabilities={"sensing", "manipulation"}, 
              computation_capacity=0.9, communication_capacity=0.8,
              memory_capacity=0.7, energy_level=0.6),
        Robot(id="R2", position=(10, 10), 
              capabilities={"sensing", "locomotion"}, 
              computation_capacity=0.7, communication_capacity=0.9,
              memory_capacity=0.8, energy_level=0.5),
        Robot(id="R3", position=(20, 20), 
              capabilities={"manipulation", "locomotion"}, 
              computation_capacity=0.6, communication_capacity=0.7,
              memory_capacity=0.9, energy_level=0.8),
        Robot(id="R4", position=(30, 30), 
              capabilities={"sensing", "manipulation", "locomotion"}, 
              computation_capacity=0.8, communication_capacity=0.6,
              memory_capacity=0.5, energy_level=0.9)
    ]
    
    # Create tasks
    tasks = [
        Task(id="T1", position=(5, 5), difficulty=0.3, deadline=300, value=100, 
             required_capabilities={"sensing"}),
        Task(id="T2", position=(15, 15), difficulty=0.5, deadline=200, value=150, 
             required_capabilities={"manipulation"}),
        Task(id="T3", position=(25, 25), difficulty=0.7, deadline=100, value=200, 
             required_capabilities={"sensing", "manipulation"}),
        Task(id="T4", position=(35, 35), difficulty=0.9, deadline=50, value=250, 
             required_capabilities={"locomotion", "manipulation"})
    ]
    
    # Initialize mechanism
    mechanism = ResourceAwareTaskAllocation(resource_constraints)
    
    # Run allocation
    allocation, payments, variant = mechanism.execute_allocation_mechanism(robots, tasks)
    
    # Print results
    print(f"\nMechanism variant selected: {variant}")
    print("\nTask Allocation:")
    for task_id, robot_id in allocation.items():
        print(f"Task {task_id} -> Robot {robot_id}")
    
    print("\nPayments:")
    for robot_id, payment in payments.items():
        print(f"Robot {robot_id}: {payment:.2f}")
    
    print("\nResource Usage:")
    for resource_type, usage in mechanism.current_resource_usage.items():
        constraint = resource_constraints.get(resource_type, float('inf'))
        percentage = (usage / constraint) * 100 if constraint > 0 else 0
        print(f"{resource_type.name}: {usage:.2f} / {constraint:.2f} ({percentage:.1f}%)")
    
    # Run another allocation with reduced resources
    print("\n\nSimulating with reduced resources...")
    
    # Reduce available resources
    reduced_constraints = {
        ResourceType.COMPUTATION: 200.0,  # 20% of original
        ResourceType.COMMUNICATION: 100.0,  # 20% of original
        ResourceType.MEMORY: 400.0,  # 20% of original
        ResourceType.ENERGY: 300.0  # 20% of original
    }
    
    # Initialize new mechanism with reduced resources
    reduced_mechanism = ResourceAwareTaskAllocation(reduced_constraints)
    
    # Run allocation
    reduced_allocation, reduced_payments, reduced_variant = reduced_mechanism.execute_allocation_mechanism(robots, tasks)
    
    # Print results
    print(f"\nMechanism variant selected with reduced resources: {reduced_variant}")
    print("\nTask Allocation:")
    for task_id, robot_id in reduced_allocation.items():
        print(f"Task {task_id} -> Robot {robot_id}")
    
    print("\nPayments:")
    for robot_id, payment in reduced_payments.items():
        print(f"Robot {robot_id}: {payment:.2f}")
    
    print("\nResource Usage:")
    for resource_type, usage in reduced_mechanism.current_resource_usage.items():
        constraint = reduced_constraints.get(resource_type, float('inf'))
        percentage = (usage / constraint) * 100 if constraint > 0 else 0
        print(f"{resource_type.name}: {usage:.2f} / {constraint:.2f} ({percentage:.1f}%)")


if __name__ == "__main__":
    simulate_resource_aware_allocation()
