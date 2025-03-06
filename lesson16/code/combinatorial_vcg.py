"""
Combinatorial VCG mechanism implementation for multi-robot task allocation.

This module demonstrates a combinatorial VCG mechanism for allocating bundles
of tasks to robots with complex valuation functions over task combinations.
"""

import numpy as np
import itertools
import time
from typing import Dict, List, Tuple, Set, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class Robot:
    """Represents a robot agent in a combinatorial VCG mechanism."""
    
    def __init__(self, robot_id: int, num_tasks: int, valuation_type: str = 'submodular'):
        """
        Initialize a robot agent with a valuation function over task bundles.
        
        Args:
            robot_id: Unique identifier for this robot
            num_tasks: Total number of tasks to be allocated
            valuation_type: Type of valuation function ('additive', 'submodular', 'superadditive')
        """
        self.id = robot_id
        self.num_tasks = num_tasks
        self.valuation_type = valuation_type
        
        # Generate base values for individual tasks
        self.base_values = {j: random.uniform(1, 10) for j in range(num_tasks)}
        
        # Generate synergy factors for task pairs (used in non-additive valuations)
        self.synergy_factors = {}
        for j1 in range(num_tasks):
            for j2 in range(j1+1, num_tasks):
                if valuation_type == 'submodular':
                    # Submodular: combinations worth less than the sum (diminishing returns)
                    self.synergy_factors[(j1, j2)] = random.uniform(0.7, 0.9)
                elif valuation_type == 'superadditive':
                    # Superadditive: combinations worth more than the sum (complementarities)
                    self.synergy_factors[(j1, j2)] = random.uniform(1.1, 1.3)
                else:
                    # Additive: no synergy factors needed
                    self.synergy_factors[(j1, j2)] = 1.0
        
        # Cache for computed valuations
        self.valuation_cache = {}
    
    def value(self, bundle: Set[int]) -> float:
        """
        Calculate the robot's valuation for a bundle of tasks.
        
        Args:
            bundle: Set of task IDs
            
        Returns:
            Valuation for the bundle
        """
        # Convert to frozenset for caching
        bundle_key = frozenset(bundle)
        
        # Return cached value if available
        if bundle_key in self.valuation_cache:
            return self.valuation_cache[bundle_key]
        
        if not bundle:
            return 0.0
        
        if self.valuation_type == 'additive':
            # Simple sum of individual values
            value = sum(self.base_values[j] for j in bundle)
        
        elif self.valuation_type == 'submodular' or self.valuation_type == 'superadditive':
            # Start with sum of individual values
            value = sum(self.base_values[j] for j in bundle)
            
            # Apply synergy factors for all pairs
            for j1 in bundle:
                for j2 in bundle:
                    if j1 < j2:
                        # Get the synergy factor for this pair
                        factor = self.synergy_factors.get((j1, j2), 1.0)
                        
                        # Adjust the value based on the synergy
                        # For submodular: reduces the contribution of the second task
                        # For superadditive: increases the contribution
                        synergy_value = self.base_values[j1] * self.base_values[j2] * (factor - 1.0) / 10.0
                        value += synergy_value
        
        # Cache and return the computed value
        self.valuation_cache[bundle_key] = value
        return value
    
    def report_valuations(self, max_bundle_size: Optional[int] = None) -> Dict[frozenset, float]:
        """
        Report valuations for all possible bundles up to a maximum size.
        
        Args:
            max_bundle_size: Maximum number of tasks in reported bundles (None for all)
            
        Returns:
            Dictionary mapping bundles (as frozensets) to valuations
        """
        all_tasks = set(range(self.num_tasks))
        valuations = {}
        
        # Determine the maximum bundle size
        if max_bundle_size is None:
            max_bundle_size = self.num_tasks
        
        # Generate all possible bundles up to the maximum size
        for size in range(1, max_bundle_size + 1):
            for bundle in itertools.combinations(all_tasks, size):
                bundle_set = frozenset(bundle)
                valuations[bundle_set] = self.value(bundle_set)
        
        return valuations


class CombinatorialVCG:
    """Implements a combinatorial VCG mechanism for task allocation."""
    
    def __init__(self, robots: List[Robot], num_tasks: int, max_bundle_size: Optional[int] = None):
        """
        Initialize the combinatorial VCG mechanism.
        
        Args:
            robots: List of Robot objects
            num_tasks: Total number of tasks to be allocated
            max_bundle_size: Maximum bundle size for each robot (None for unlimited)
        """
        self.robots = robots
        self.num_tasks = num_tasks
        self.max_bundle_size = max_bundle_size if max_bundle_size is not None else num_tasks
        
        # Collect reported valuations from all robots
        self.reported_valuations = {}
        for robot in robots:
            self.reported_valuations[robot.id] = robot.report_valuations(self.max_bundle_size)
        
        # Results of the mechanism
        self.allocation = None
        self.payments = None
        self.social_welfare = None
        
        # Performance metrics
        self.computation_time = 0
    
    def find_optimal_allocation(self) -> Tuple[Dict[int, Set[int]], float]:
        """
        Find the allocation that maximizes social welfare.
        
        Returns:
            Tuple of (allocation dictionary, social welfare value)
        """
        start_time = time.time()
        
        # For small instances, use brute force to find the optimal allocation
        if self.num_tasks <= 4:
            allocation, welfare = self._find_optimal_allocation_brute_force()
        else:
            # For larger instances, use a greedy approximation
            allocation, welfare = self._find_allocation_greedy()
        
        self.computation_time += time.time() - start_time
        return allocation, welfare
    
    def _find_optimal_allocation_brute_force(self) -> Tuple[Dict[int, Set[int]], float]:
        """
        Find the optimal allocation using brute force enumeration.
        
        Returns:
            Tuple of (allocation dictionary, social welfare value)
        """
        all_tasks = set(range(self.num_tasks))
        best_allocation = {}
        best_welfare = 0
        
        # Generate all possible partitions of tasks
        for partition in self._generate_partitions(all_tasks, len(self.robots)):
            # Skip partitions with more parts than robots
            if len(partition) > len(self.robots):
                continue
            
            # Try all possible assignments of partitions to robots
            for assignment in itertools.permutations(range(len(self.robots)), len(partition)):
                allocation = {}
                welfare = 0
                
                for i, bundle_idx in enumerate(assignment):
                    robot_id = self.robots[bundle_idx].id
                    bundle = frozenset(partition[i])
                    
                    if bundle:  # Skip empty bundles
                        allocation[robot_id] = bundle
                        welfare += self.reported_valuations[robot_id].get(bundle, 0)
                
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_allocation = allocation
        
        # Convert frozensets to regular sets in the allocation
        best_allocation = {k: set(v) for k, v in best_allocation.items()}
        return best_allocation, best_welfare
    
    def _generate_partitions(self, s: Set[int], max_parts: int) -> List[List[Set[int]]]:
        """
        Generate all possible partitions of a set into at most max_parts parts.
        
        Args:
            s: Set to partition
            max_parts: Maximum number of parts
            
        Returns:
            List of partitions, where each partition is a list of sets
        """
        if not s:
            return [[]]
        
        if max_parts == 1:
            return [[s]]
        
        result = []
        first = next(iter(s))
        rest = s - {first}
        
        # Put first in a new part
        for partition in self._generate_partitions(rest, max_parts - 1):
            result.append([{first}] + partition)
        
        # Add first to each existing part
        for partition in self._generate_partitions(rest, max_parts):
            for i in range(len(partition)):
                new_partition = partition.copy()
                new_partition[i] = new_partition[i].union({first})
                result.append(new_partition)
        
        return result
    
    def _find_allocation_greedy(self) -> Tuple[Dict[int, Set[int]], float]:
        """
        Find an allocation using a greedy approximation algorithm.
        
        Returns:
            Tuple of (allocation dictionary, social welfare value)
        """
        all_tasks = set(range(self.num_tasks))
        allocation = {robot.id: set() for robot in self.robots}
        remaining_tasks = all_tasks.copy()
        
        # Sort robots by their maximum valuation for any single task
        robot_order = sorted(
            self.robots, 
            key=lambda r: max(self.reported_valuations[r.id].get(frozenset({j}), 0) for j in all_tasks),
            reverse=True
        )
        
        # Greedy allocation: each robot picks its most valuable task in turn
        while remaining_tasks:
            for robot in robot_order:
                if not remaining_tasks:
                    break
                
                # Find the most valuable remaining task for this robot
                best_task = None
                best_marginal_value = 0
                
                for task in remaining_tasks:
                    current_bundle = allocation[robot.id]
                    new_bundle = current_bundle.union({task})
                    
                    # Calculate marginal value
                    current_value = self.reported_valuations[robot.id].get(frozenset(current_bundle), 0)
                    new_value = self.reported_valuations[robot.id].get(frozenset(new_bundle), 0)
                    marginal_value = new_value - current_value
                    
                    if marginal_value > best_marginal_value:
                        best_marginal_value = marginal_value
                        best_task = task
                
                # Allocate the best task if it has positive marginal value
                if best_task is not None and best_marginal_value > 0:
                    allocation[robot.id].add(best_task)
                    remaining_tasks.remove(best_task)
        
        # Calculate the social welfare
        welfare = sum(
            self.reported_valuations[robot.id].get(frozenset(bundle), 0)
            for robot in self.robots
            for robot_id, bundle in allocation.items()
            if robot.id == robot_id and bundle
        )
        
        return allocation, welfare
    
    def compute_payments(self, allocation: Dict[int, Set[int]], welfare: float) -> Dict[int, float]:
        """
        Compute VCG payments for each robot.
        
        Args:
            allocation: Allocation dictionary mapping robot IDs to task bundles
            welfare: Social welfare of the allocation
            
        Returns:
            Dictionary mapping robot IDs to payments
        """
        start_time = time.time()
        payments = {}
        
        # For each robot, compute the allocation without that robot
        for robot in self.robots:
            # Create a new VCG instance without this robot
            robots_without_i = [r for r in self.robots if r.id != robot.id]
            vcg_without_i = CombinatorialVCG(robots_without_i, self.num_tasks, self.max_bundle_size)
            
            # Find the optimal allocation without this robot
            allocation_without_i, welfare_without_i = vcg_without_i.find_optimal_allocation()
            
            # Calculate welfare of the current allocation excluding robot i's contribution
            welfare_others = welfare
            if robot.id in allocation:
                bundle = allocation[robot.id]
                welfare_others -= self.reported_valuations[robot.id].get(frozenset(bundle), 0)
            
            # VCG payment is the difference
            payments[robot.id] = welfare_without_i - welfare_others
        
        self.computation_time += time.time() - start_time
        return payments
    
    def run(self) -> Tuple[Dict[int, Set[int]], Dict[int, float], float]:
        """
        Run the combinatorial VCG mechanism.
        
        Returns:
            Tuple of (allocation, payments, social welfare)
        """
        # Find the optimal allocation
        self.allocation, self.social_welfare = self.find_optimal_allocation()
        
        # Compute VCG payments
        self.payments = self.compute_payments(self.allocation, self.social_welfare)
        
        return self.allocation, self.payments, self.social_welfare
    
    def calculate_utilities(self) -> Dict[int, float]:
        """
        Calculate the utility for each robot.
        
        Returns:
            Dictionary mapping robot IDs to utilities
        """
        if self.allocation is None or self.payments is None:
            raise ValueError("Mechanism must be run before calculating utilities")
        
        utilities = {}
        for robot in self.robots:
            # Value from allocated bundle
            value = 0
            if robot.id in self.allocation and self.allocation[robot.id]:
                bundle = self.allocation[robot.id]
                # Use the robot's true valuation, not the reported one
                value = robot.value(bundle)
            
            # Payment
            payment = self.payments.get(robot.id, 0)
            
            # Utility = value - payment
            utilities[robot.id] = value - payment
        
        return utilities


def visualize_results(mechanism: CombinatorialVCG):
    """
    Visualize the results of the combinatorial VCG mechanism.
    
    Args:
        mechanism: CombinatorialVCG instance after running
    """
    allocation = mechanism.allocation
    payments = mechanism.payments
    utilities = mechanism.calculate_utilities()
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot task allocation
    robot_ids = [robot.id for robot in mechanism.robots]
    
    # Create a mapping of tasks to colors
    task_colors = plt.cm.tab10(np.linspace(0, 1, mechanism.num_tasks))
    
    # Plot allocation as stacked bars
    bottom = np.zeros(len(robot_ids))
    for task in range(mechanism.num_tasks):
        task_allocation = [1 if robot.id in allocation and task in allocation[robot.id] else 0 
                          for robot in mechanism.robots]
        ax1.bar(robot_ids, task_allocation, bottom=bottom, color=task_colors[task], label=f'Task {task}')
        bottom += task_allocation
    
    ax1.set_xlabel('Robot ID')
    ax1.set_ylabel('Allocated Tasks')
    ax1.set_title('Task Bundle Allocation')
    ax1.set_xticks(robot_ids)
    ax1.legend(title='Tasks', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot payments and utilities
    width = 0.35
    payment_values = [payments.get(robot.id, 0) for robot in mechanism.robots]
    utility_values = [utilities.get(robot.id, 0) for robot in mechanism.robots]
    
    x = np.arange(len(robot_ids))
    ax2.bar(x - width/2, payment_values, width, label='Payment')
    ax2.bar(x + width/2, utility_values, width, label='Utility')
    
    ax2.set_xlabel('Robot ID')
    ax2.set_ylabel('Value')
    ax2.set_title('Payments and Utilities')
    ax2.set_xticks(x)
    ax2.set_xticklabels(robot_ids)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('combinatorial_vcg_results.png')
    plt.close()


def analyze_valuation_types():
    """
    Analyze the impact of different valuation types on the combinatorial VCG mechanism.
    """
    num_robots = 3
    num_tasks = 4
    
    # Create a fixed random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    valuation_types = ['additive', 'submodular', 'superadditive']
    results = {}
    
    for val_type in valuation_types:
        print(f"\nTesting with {val_type} valuations...")
        
        # Create robots with the specified valuation type
        robots = [Robot(i, num_tasks, val_type) for i in range(num_robots)]
        
        # Run the mechanism
        mechanism = CombinatorialVCG(robots, num_tasks)
        allocation, payments, welfare = mechanism.run()
        utilities = mechanism.calculate_utilities()
        
        # Store results
        results[val_type] = {
            'welfare': welfare,
            'payments': sum(payments.values()),
            'utilities': sum(utilities.values()),
            'computation_time': mechanism.computation_time
        }
        
        # Print results
        print(f"Allocation: {allocation}")
        print(f"Payments: {payments}")
        print(f"Social welfare: {welfare:.2f}")
        print(f"Total payments: {sum(payments.values()):.2f}")
        print(f"Total utilities: {sum(utilities.values()):.2f}")
        print(f"Computation time: {mechanism.computation_time:.4f} seconds")
        
        # Visualize
        visualize_results(mechanism)
        print(f"Results visualization saved to 'combinatorial_vcg_results.png'")
    
    # Compare results across valuation types
    print("\nComparison across valuation types:")
    for metric in ['welfare', 'payments', 'utilities', 'computation_time']:
        print(f"\n{metric.capitalize()}:")
        for val_type in valuation_types:
            print(f"  {val_type}: {results[val_type][metric]:.2f}")


def analyze_strategic_behavior():
    """
    Analyze the impact of strategic behavior in the combinatorial VCG mechanism.
    """
    num_robots = 3
    num_tasks = 4
    
    # Create a fixed random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create robots with true valuations
    true_robots = [Robot(i, num_tasks, 'submodular') for i in range(num_robots)]
    
    # Run the mechanism with truthful reporting
    print("\nRunning with truthful reporting...")
    true_mechanism = CombinatorialVCG(true_robots, num_tasks)
    true_allocation, true_payments, true_welfare = true_mechanism.run()
    true_utilities = true_mechanism.calculate_utilities()
    
    print(f"Allocation: {true_allocation}")
    print(f"Payments: {true_payments}")
    print(f"Utilities: {true_utilities}")
    
    # Create a strategic robot that misreports valuations
    print("\nRunning with strategic misreporting...")
    strategic_robots = true_robots.copy()
    
    # Modify the first robot to be strategic
    strategic_robot = strategic_robots[0]
    
    # Create a modified copy of the robot's reported valuations
    strategic_valuations = {}
    for bundle, value in true_mechanism.reported_valuations[strategic_robot.id].items():
        # Overreport values for bundles the robot likes
        if value > 5:
            strategic_valuations[bundle] = value * 1.5
        # Underreport values for bundles the robot doesn't like as much
        else:
            strategic_valuations[bundle] = value * 0.5
    
    # Create a new mechanism with the strategic robot
    strategic_mechanism = CombinatorialVCG(strategic_robots, num_tasks)
    
    # Replace the first robot's reported valuations with strategic ones
    strategic_mechanism.reported_valuations[strategic_robot.id] = strategic_valuations
    
    # Run the mechanism
    strategic_allocation, strategic_payments, strategic_welfare = strategic_mechanism.run()
    
    # Calculate utilities using true valuations
    strategic_utilities = {}
    for robot in strategic_robots:
        value = 0
        if robot.id in strategic_allocation and strategic_allocation[robot.id]:
            bundle = strategic_allocation[robot.id]
            # Use the robot's true valuation
            value = robot.value(bundle)
        
        payment = strategic_payments.get(robot.id, 0)
        strategic_utilities[robot.id] = value - payment
    
    print(f"Strategic allocation: {strategic_allocation}")
    print(f"Strategic payments: {strategic_payments}")
    print(f"Strategic utilities: {strategic_utilities}")
    
    # Compare utilities for the strategic robot
    print("\nComparison for the strategic robot (ID 0):")
    print(f"Utility when truthful: {true_utilities[0]:.2f}")
    print(f"Utility when strategic: {strategic_utilities[0]:.2f}")
    print(f"Difference: {strategic_utilities[0] - true_utilities[0]:.2f}")
    
    print("\nThis demonstrates that in a VCG mechanism, truthful reporting is a dominant strategy.")


if __name__ == "__main__":
    print("Combinatorial VCG Mechanism for Multi-Robot Task Allocation")
    print("=" * 60)
    
    # Run a basic simulation
    num_robots = 3
    num_tasks = 4
    
    print(f"Running simulation with {num_robots} robots and {num_tasks} tasks...")
    
    # Create robots with submodular valuations
    robots = [Robot(i, num_tasks, 'submodular') for i in range(num_robots)]
    
    # Run the mechanism
    mechanism = CombinatorialVCG(robots, num_tasks)
    allocation, payments, welfare = mechanism.run()
    utilities = mechanism.calculate_utilities()
    
    print("\nResults:")
    print(f"Allocation: {allocation}")
    print(f"Payments: {payments}")
    print(f"Social welfare: {welfare:.2f}")
    print(f"Utilities: {utilities}")
    print(f"Computation time: {mechanism.computation_time:.4f} seconds")
    
    # Visualize results
    visualize_results(mechanism)
    print("\nResults visualization saved to 'combinatorial_vcg_results.png'")
    
    print("\n" + "=" * 60)
    print("Analyzing different valuation types")
    analyze_valuation_types()
    
    print("\n" + "=" * 60)
    print("Analyzing strategic behavior in combinatorial VCG")
    analyze_strategic_behavior()
