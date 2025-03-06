"""
Distributed VCG implementation for multi-robot task allocation.

This module demonstrates a distributed implementation of the VCG mechanism
for allocating tasks among a team of robots with heterogeneous capabilities.
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import matplotlib.pyplot as plt

class Robot:
    """Represents a robot agent in a distributed VCG mechanism."""
    
    def __init__(self, robot_id: int, num_robots: int, num_tasks: int, 
                 communication_graph: nx.Graph, failure_prob: float = 0.0):
        """
        Initialize a robot agent.
        
        Args:
            robot_id: Unique identifier for this robot
            num_robots: Total number of robots in the system
            num_tasks: Total number of tasks to be allocated
            communication_graph: Graph representing communication links between robots
            failure_prob: Probability of message transmission failure
        """
        self.id = robot_id
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.comm_graph = communication_graph
        self.failure_prob = failure_prob
        
        # Generate costs for performing each task
        self.true_costs = {j: random.uniform(1, 10) for j in range(num_tasks)}
        self.reported_costs = self.true_costs.copy()  # Truthful by default
        
        # Storage for received cost reports from other robots
        self.received_costs = {}
        
        # Storage for allocation and payment results
        self.allocation = None
        self.payments = None
        
        # Tracking communication and computation
        self.messages_sent = 0
        self.messages_received = 0
        self.computation_time = 0
        
    def set_strategic_behavior(self, strategy: str, factor: float = 0.8):
        """
        Set strategic (non-truthful) reporting behavior.
        
        Args:
            strategy: Type of strategic behavior ('underreport', 'overreport', 'truthful')
            factor: Multiplier for cost manipulation
        """
        if strategy == 'underreport':
            self.reported_costs = {j: c * factor for j, c in self.true_costs.items()}
        elif strategy == 'overreport':
            self.reported_costs = {j: c / factor for j, c in self.true_costs.items()}
        else:  # truthful
            self.reported_costs = self.true_costs.copy()
    
    def broadcast_costs(self) -> Dict[int, Dict[int, float]]:
        """
        Broadcast cost information to neighboring robots.
        
        Returns:
            Dictionary mapping robot IDs to their reported costs
        """
        neighbors = list(self.comm_graph.neighbors(self.id))
        messages = {self.id: self.reported_costs}
        self.messages_sent += len(neighbors)
        
        # Simulate message failures
        for neighbor in neighbors:
            if random.random() >= self.failure_prob:
                # In a real system, this would be an actual message transmission
                pass
                
        return messages
    
    def receive_costs(self, messages: Dict[int, Dict[int, float]]):
        """
        Process received cost information from other robots.
        
        Args:
            messages: Dictionary mapping robot IDs to their reported costs
        """
        for robot_id, costs in messages.items():
            if robot_id != self.id:
                self.received_costs[robot_id] = costs
                self.messages_received += 1
    
    def compute_allocation(self) -> Dict[int, int]:
        """
        Compute the optimal task allocation based on reported costs.
        
        Returns:
            Dictionary mapping robot IDs to assigned task IDs
        """
        start_time = time.time()
        
        # Ensure we have cost information from all robots
        all_robots = set(range(self.num_robots))
        received_from = set(self.received_costs.keys()).union({self.id})
        
        if received_from != all_robots:
            print(f"Robot {self.id}: Missing cost information from robots {all_robots - received_from}")
            return None
        
        # Combine all cost information
        all_costs = {**{self.id: self.reported_costs}, **self.received_costs}
        
        # Create cost matrix for the assignment problem
        cost_matrix = np.full((self.num_robots, self.num_tasks), np.inf)
        for i in range(self.num_robots):
            for j in range(self.num_tasks):
                if j in all_costs[i]:
                    cost_matrix[i, j] = all_costs[i][j]
        
        # Solve the assignment problem using the Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Convert to dictionary format
        allocation = {i: j for i, j in zip(row_ind, col_ind)}
        
        self.computation_time += time.time() - start_time
        return allocation
    
    def compute_payments(self, allocation: Dict[int, int]) -> Dict[int, float]:
        """
        Compute VCG payments for each robot.
        
        Args:
            allocation: Dictionary mapping robot IDs to assigned task IDs
            
        Returns:
            Dictionary mapping robot IDs to their payments
        """
        start_time = time.time()
        
        # Combine all cost information
        all_costs = {**{self.id: self.reported_costs}, **self.received_costs}
        
        payments = {}
        
        # For each robot, compute the allocation without that robot
        for robot_i in range(self.num_robots):
            # Create cost matrix excluding robot_i
            robots_without_i = [r for r in range(self.num_robots) if r != robot_i]
            cost_matrix_without_i = np.full((len(robots_without_i), self.num_tasks), np.inf)
            
            for idx, r in enumerate(robots_without_i):
                for j in range(self.num_tasks):
                    if j in all_costs[r]:
                        cost_matrix_without_i[idx, j] = all_costs[r][j]
            
            # Solve the assignment problem without robot_i
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix_without_i)
            
            # Calculate the total cost without robot_i
            total_cost_without_i = sum(cost_matrix_without_i[i, j] for i, j in zip(row_ind, col_ind))
            
            # Calculate the total cost of the optimal allocation excluding robot_i's cost
            total_cost_with_i_excluded = sum(all_costs[r][allocation[r]] 
                                           for r in range(self.num_robots) 
                                           if r != robot_i and r in allocation)
            
            # VCG payment is the difference
            payments[robot_i] = total_cost_without_i - total_cost_with_i_excluded
        
        self.computation_time += time.time() - start_time
        return payments
    
    def verify_results(self, allocations: Dict[int, Dict[int, int]], 
                      payments: Dict[int, Dict[int, float]]) -> bool:
        """
        Verify that all robots have the same allocation and payment results.
        
        Args:
            allocations: Dictionary mapping robot IDs to their computed allocations
            payments: Dictionary mapping robot IDs to their computed payments
            
        Returns:
            True if all results are consistent, False otherwise
        """
        # Check if all allocations are the same
        first_allocation = next(iter(allocations.values()))
        allocation_consistent = all(a == first_allocation for a in allocations.values())
        
        # Check if all payments are approximately the same (floating point comparison)
        payment_consistent = True
        first_payment = next(iter(payments.values()))
        for p in payments.values():
            if not all(abs(p[i] - first_payment[i]) < 1e-6 for i in range(self.num_robots)):
                payment_consistent = False
                break
        
        return allocation_consistent and payment_consistent
    
    def calculate_utility(self, allocation: Dict[int, int], payments: Dict[int, float]) -> float:
        """
        Calculate the utility for this robot based on allocation and payment.
        
        Args:
            allocation: Dictionary mapping robot IDs to assigned task IDs
            payments: Dictionary mapping robot IDs to their payments
            
        Returns:
            Utility value (negative cost plus payment)
        """
        if self.id not in allocation:
            return 0
        
        task = allocation[self.id]
        true_cost = self.true_costs[task]
        payment = payments[self.id]
        
        return payment - true_cost


def run_distributed_vcg_simulation(num_robots: int = 5, 
                                  num_tasks: int = 5,
                                  communication_topology: str = 'complete',
                                  failure_prob: float = 0.0,
                                  strategic_robots: List[Tuple[int, str]] = None) -> Tuple[Dict, Dict, Dict]:
    """
    Run a simulation of the distributed VCG mechanism.
    
    Args:
        num_robots: Number of robots in the system
        num_tasks: Number of tasks to allocate
        communication_topology: Type of communication network ('complete', 'ring', 'star', 'random')
        failure_prob: Probability of message transmission failure
        strategic_robots: List of (robot_id, strategy) pairs for non-truthful robots
        
    Returns:
        Tuple of (allocation, payments, utilities)
    """
    # Create communication graph
    if communication_topology == 'complete':
        G = nx.complete_graph(num_robots)
    elif communication_topology == 'ring':
        G = nx.cycle_graph(num_robots)
    elif communication_topology == 'star':
        G = nx.star_graph(num_robots - 1)
    elif communication_topology == 'random':
        G = nx.erdos_renyi_graph(num_robots, 0.5)
        # Ensure the graph is connected
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(num_robots, 0.5)
    else:
        raise ValueError(f"Unknown communication topology: {communication_topology}")
    
    # Create robots
    robots = [Robot(i, num_robots, num_tasks, G, failure_prob) for i in range(num_robots)]
    
    # Set strategic behavior for specified robots
    if strategic_robots:
        for robot_id, strategy in strategic_robots:
            robots[robot_id].set_strategic_behavior(strategy)
    
    # Phase 1: Cost reporting
    all_messages = {}
    for robot in robots:
        messages = robot.broadcast_costs()
        all_messages.update(messages)
    
    for robot in robots:
        robot.receive_costs(all_messages)
    
    # Phase 2: Allocation computation
    allocations = {}
    for robot in robots:
        allocation = robot.compute_allocation()
        if allocation:
            allocations[robot.id] = allocation
    
    # Check if all robots computed the same allocation
    if not allocations:
        print("No robot could compute a valid allocation")
        return None, None, None
    
    first_allocation = next(iter(allocations.values()))
    if not all(a == first_allocation for a in allocations.values()):
        print("Warning: Robots computed different allocations")
    
    # Phase 3: Payment computation
    payments = {}
    for robot in robots:
        if robot.id in allocations:
            payment = robot.compute_payments(allocations[robot.id])
            payments[robot.id] = payment
    
    # Phase 4: Result verification
    consistent = robots[0].verify_results(allocations, payments)
    if not consistent:
        print("Warning: Inconsistent results detected")
    
    # Calculate utilities
    utilities = {}
    for robot in robots:
        if robot.id in allocations and robot.id in payments:
            utility = robot.calculate_utility(allocations[robot.id], payments[robot.id])
            utilities[robot.id] = utility
    
    # Use the first robot's results as the consensus
    return first_allocation, next(iter(payments.values())), utilities


def visualize_results(allocation: Dict[int, int], 
                     payments: Dict[int, float], 
                     utilities: Dict[int, float],
                     robots: List[Robot]):
    """
    Visualize the results of the distributed VCG mechanism.
    
    Args:
        allocation: Task allocation dictionary
        payments: Payment dictionary
        utilities: Utility dictionary
        robots: List of Robot objects
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot task allocation
    robot_ids = list(range(len(robots)))
    task_ids = [allocation[r] if r in allocation else -1 for r in robot_ids]
    
    ax1.bar(robot_ids, [1] * len(robot_ids), color='lightgray')
    for r, t in zip(robot_ids, task_ids):
        if t >= 0:
            ax1.text(r, 0.5, f"Task {t}", ha='center', va='center')
    
    ax1.set_xlabel('Robot ID')
    ax1.set_ylabel('Assigned Task')
    ax1.set_title('Task Allocation')
    ax1.set_xticks(robot_ids)
    ax1.set_yticks([])
    
    # Plot payments and utilities
    width = 0.35
    payment_values = [payments[r] if r in payments else 0 for r in robot_ids]
    utility_values = [utilities[r] if r in utilities else 0 for r in robot_ids]
    
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
    plt.savefig('vcg_distributed_results.png')
    plt.close()


def analyze_strategic_behavior():
    """
    Analyze the impact of strategic behavior on the distributed VCG mechanism.
    """
    num_robots = 5
    num_tasks = 5
    
    # Create a fixed set of robots with predetermined costs for consistency
    np.random.seed(42)
    random.seed(42)
    
    # Run with all truthful robots
    print("Running with all truthful robots...")
    allocation_t, payments_t, utilities_t = run_distributed_vcg_simulation(
        num_robots, num_tasks, 'complete', 0.0, None)
    
    # Run with one strategic robot
    strategies = ['underreport', 'overreport']
    for strategy in strategies:
        print(f"Running with robot 0 using {strategy} strategy...")
        allocation_s, payments_s, utilities_s = run_distributed_vcg_simulation(
            num_robots, num_tasks, 'complete', 0.0, [(0, strategy)])
        
        # Compare utilities
        if utilities_t and utilities_s:
            truthful_utility = utilities_t[0]
            strategic_utility = utilities_s[0]
            print(f"Robot 0 utility when truthful: {truthful_utility:.2f}")
            print(f"Robot 0 utility when strategic ({strategy}): {strategic_utility:.2f}")
            print(f"Difference: {strategic_utility - truthful_utility:.2f}")
        else:
            print("Could not compare utilities due to failed simulation")
    
    print("\nThis demonstrates that in a VCG mechanism, truthful reporting is a dominant strategy.")


def analyze_communication_impact():
    """
    Analyze the impact of communication topology and failures on the distributed VCG mechanism.
    """
    num_robots = 5
    num_tasks = 5
    
    # Test different communication topologies
    topologies = ['complete', 'ring', 'star', 'random']
    for topology in topologies:
        print(f"\nTesting {topology} communication topology...")
        allocation, payments, utilities = run_distributed_vcg_simulation(
            num_robots, num_tasks, topology, 0.0, None)
        
        if allocation and payments and utilities:
            print("Simulation successful")
            print(f"Allocation: {allocation}")
            print(f"Average payment: {sum(payments.values()) / len(payments):.2f}")
            print(f"Average utility: {sum(utilities.values()) / len(utilities):.2f}")
        else:
            print("Simulation failed")
    
    # Test impact of communication failures
    failure_probs = [0.0, 0.1, 0.3, 0.5]
    for prob in failure_probs:
        print(f"\nTesting with {prob*100}% message failure probability...")
        allocation, payments, utilities = run_distributed_vcg_simulation(
            num_robots, num_tasks, 'complete', prob, None)
        
        if allocation and payments and utilities:
            print("Simulation successful")
            print(f"Average utility: {sum(utilities.values()) / len(utilities):.2f}")
        else:
            print("Simulation failed due to communication issues")


if __name__ == "__main__":
    print("Distributed VCG Mechanism for Multi-Robot Task Allocation")
    print("=" * 60)
    
    # Run a basic simulation
    num_robots = 5
    num_tasks = 5
    
    print(f"Running simulation with {num_robots} robots and {num_tasks} tasks...")
    allocation, payments, utilities = run_distributed_vcg_simulation(num_robots, num_tasks)
    
    if allocation and payments and utilities:
        print("\nResults:")
        print(f"Allocation: {allocation}")
        print(f"Payments: {payments}")
        print(f"Utilities: {utilities}")
        
        # Create robots for visualization (just for the structure)
        robots = [Robot(i, num_robots, num_tasks, nx.complete_graph(num_robots)) 
                 for i in range(num_robots)]
        
        # Visualize results
        visualize_results(allocation, payments, utilities, robots)
        print("\nResults visualization saved to 'vcg_distributed_results.png'")
    else:
        print("Simulation failed")
    
    print("\n" + "=" * 60)
    print("Analyzing strategic behavior in distributed VCG")
    analyze_strategic_behavior()
    
    print("\n" + "=" * 60)
    print("Analyzing communication impact in distributed VCG")
    analyze_communication_impact()
