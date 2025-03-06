"""
Auction-based mechanisms for multi-robot task allocation.

This module implements various auction mechanisms for allocating tasks
among a team of robots, including single-item auctions, sequential auctions,
and combinatorial auctions with different bidding strategies.
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Callable
import matplotlib.pyplot as plt
from collections import defaultdict

class Task:
    """Represents a task to be allocated to robots."""
    
    def __init__(self, task_id: int, location: Tuple[float, float], difficulty: float = 1.0,
                 deadline: Optional[float] = None, prerequisites: Optional[List[int]] = None):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for this task
            location: (x, y) coordinates of the task
            difficulty: Relative difficulty factor affecting cost
            deadline: Optional deadline for task completion
            prerequisites: List of task IDs that must be completed before this task
        """
        self.id = task_id
        self.location = location
        self.difficulty = difficulty
        self.deadline = deadline
        self.prerequisites = prerequisites if prerequisites else []
        
        # Task state
        self.allocated = False
        self.completed = False
        self.allocated_to = None
        self.completion_time = None


class Robot:
    """Represents a robot agent participating in auction mechanisms."""
    
    def __init__(self, robot_id: int, location: Tuple[float, float], 
                 speed: float = 1.0, capabilities: Dict[str, float] = None):
        """
        Initialize a robot agent.
        
        Args:
            robot_id: Unique identifier for this robot
            location: (x, y) coordinates of the robot
            speed: Movement speed factor
            capabilities: Dictionary mapping capability types to proficiency levels
        """
        self.id = robot_id
        self.location = location
        self.speed = speed
        self.capabilities = capabilities if capabilities else {}
        
        # Robot state
        self.assigned_tasks = []
        self.completed_tasks = []
        self.current_location = location
        self.available_time = 0.0
        
        # Performance metrics
        self.total_distance = 0.0
        self.total_cost = 0.0
        self.total_utility = 0.0
        self.idle_time = 0.0
    
    def calculate_cost(self, task: Task, current_plan: List[int] = None) -> float:
        """
        Calculate the cost for this robot to perform a task.
        
        Args:
            task: The task to evaluate
            current_plan: List of task IDs in the robot's current plan
            
        Returns:
            Cost value for performing the task
        """
        # Base cost is distance to the task
        if current_plan and current_plan:
            # If we have a current plan, calculate cost from the last task in the plan
            last_task_id = current_plan[-1]
            last_task_location = self._get_task_location(last_task_id)
            distance = self._calculate_distance(last_task_location, task.location)
        else:
            # Otherwise, calculate from current location
            distance = self._calculate_distance(self.current_location, task.location)
        
        # Adjust for robot speed
        travel_cost = distance / self.speed
        
        # Adjust for task difficulty and robot capabilities
        execution_cost = task.difficulty
        for capability, level in self.capabilities.items():
            # Reduce cost if robot has relevant capabilities
            execution_cost *= (2.0 - min(level, 1.0))
        
        # Total cost is travel plus execution
        total_cost = travel_cost + execution_cost
        
        return total_cost
    
    def calculate_bundle_cost(self, tasks: List[Task]) -> float:
        """
        Calculate the cost for this robot to perform a bundle of tasks.
        
        Args:
            tasks: List of tasks in the bundle
            
        Returns:
            Total cost for performing all tasks in the bundle
        """
        if not tasks:
            return 0.0
        
        # Sort tasks to find a reasonable execution order
        # In a real system, this would use a more sophisticated TSP-like algorithm
        sorted_tasks = self._sort_tasks_by_distance(tasks)
        
        total_cost = 0.0
        current_loc = self.current_location
        
        for task in sorted_tasks:
            # Calculate travel cost to this task
            distance = self._calculate_distance(current_loc, task.location)
            travel_cost = distance / self.speed
            
            # Calculate execution cost
            execution_cost = task.difficulty
            for capability, level in self.capabilities.items():
                execution_cost *= (2.0 - min(level, 1.0))
            
            # Add to total
            total_cost += travel_cost + execution_cost
            
            # Update current location for next calculation
            current_loc = task.location
        
        return total_cost
    
    def _sort_tasks_by_distance(self, tasks: List[Task]) -> List[Task]:
        """
        Sort tasks by a greedy nearest-neighbor heuristic.
        
        Args:
            tasks: List of tasks to sort
            
        Returns:
            Sorted list of tasks
        """
        if not tasks:
            return []
        
        remaining = tasks.copy()
        sorted_tasks = []
        current_loc = self.current_location
        
        while remaining:
            # Find nearest task
            nearest = min(remaining, key=lambda t: self._calculate_distance(current_loc, t.location))
            sorted_tasks.append(nearest)
            remaining.remove(nearest)
            current_loc = nearest.location
        
        return sorted_tasks
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two locations.
        
        Args:
            loc1: First location (x, y)
            loc2: Second location (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def _get_task_location(self, task_id: int) -> Tuple[float, float]:
        """
        Get the location of a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            (x, y) location of the task
        """
        # In a real system, this would look up the task in a shared database
        # Here we'll just return a dummy location
        return (0.0, 0.0)
    
    def bid(self, task: Task, bidding_strategy: str = 'truthful') -> float:
        """
        Generate a bid for a task based on the specified strategy.
        
        Args:
            task: Task to bid on
            bidding_strategy: Strategy to use ('truthful', 'aggressive', 'conservative')
            
        Returns:
            Bid value
        """
        true_cost = self.calculate_cost(task, self.assigned_tasks)
        
        if bidding_strategy == 'truthful':
            return true_cost
        elif bidding_strategy == 'aggressive':
            # Bid lower than true cost to win more tasks
            return true_cost * random.uniform(0.7, 0.9)
        elif bidding_strategy == 'conservative':
            # Bid higher than true cost to ensure profit
            return true_cost * random.uniform(1.1, 1.3)
        else:
            return true_cost
    
    def bundle_bid(self, tasks: List[Task], bidding_strategy: str = 'truthful') -> float:
        """
        Generate a bid for a bundle of tasks.
        
        Args:
            tasks: List of tasks in the bundle
            bidding_strategy: Strategy to use
            
        Returns:
            Bid value for the entire bundle
        """
        true_cost = self.calculate_bundle_cost(tasks)
        
        if bidding_strategy == 'truthful':
            return true_cost
        elif bidding_strategy == 'aggressive':
            return true_cost * random.uniform(0.7, 0.9)
        elif bidding_strategy == 'conservative':
            return true_cost * random.uniform(1.1, 1.3)
        else:
            return true_cost


class AuctionMechanism:
    """Base class for auction-based task allocation mechanisms."""
    
    def __init__(self, robots: List[Robot], tasks: List[Task]):
        """
        Initialize the auction mechanism.
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
        """
        self.robots = robots
        self.tasks = tasks
        self.allocated_tasks = {}  # Maps task IDs to robot IDs
        self.payments = {}  # Maps robot IDs to payment amounts
        
        # Performance metrics
        self.total_cost = 0.0
        self.mechanism_runtime = 0.0
        self.communication_volume = 0
    
    def run_auction(self, bidding_strategy: str = 'truthful') -> Dict[int, List[int]]:
        """
        Run the auction to allocate tasks.
        
        Args:
            bidding_strategy: Bidding strategy for robots
            
        Returns:
            Dictionary mapping robot IDs to lists of allocated task IDs
        """
        raise NotImplementedError("Subclasses must implement run_auction")
    
    def calculate_social_welfare(self) -> float:
        """
        Calculate the social welfare of the allocation.
        
        Returns:
            Social welfare value (negative total cost)
        """
        total_cost = 0.0
        for robot in self.robots:
            # Get tasks allocated to this robot
            allocated_task_ids = [t.id for t in self.tasks if t.allocated_to == robot.id]
            allocated_tasks = [t for t in self.tasks if t.id in allocated_task_ids]
            
            if allocated_tasks:
                total_cost += robot.calculate_bundle_cost(allocated_tasks)
        
        # Social welfare is negative cost in this case
        return -total_cost
    
    def get_allocation_results(self) -> Dict:
        """
        Get the results of the allocation.
        
        Returns:
            Dictionary with allocation results
        """
        allocation = {}
        for robot in self.robots:
            allocation[robot.id] = [t.id for t in self.tasks if t.allocated_to == robot.id]
        
        return {
            'allocation': allocation,
            'payments': self.payments,
            'social_welfare': self.calculate_social_welfare(),
            'runtime': self.mechanism_runtime,
            'communication': self.communication_volume
        }


class SingleItemAuction(AuctionMechanism):
    """Implements a single-item auction mechanism for task allocation."""
    
    def run_auction(self, bidding_strategy: str = 'truthful') -> Dict[int, List[int]]:
        """
        Run a series of single-item auctions, one for each task.
        
        Args:
            bidding_strategy: Bidding strategy for robots
            
        Returns:
            Dictionary mapping robot IDs to lists of allocated task IDs
        """
        start_time = time.time()
        
        # Initialize payments
        self.payments = {robot.id: 0.0 for robot in self.robots}
        
        # Process each task in a separate auction
        for task in self.tasks:
            # Collect bids from all robots
            bids = {}
            for robot in self.robots:
                bid = robot.bid(task, bidding_strategy)
                bids[robot.id] = bid
                self.communication_volume += 1  # One message per bid
            
            # Find the winner (lowest bid)
            if bids:
                winner_id = min(bids, key=bids.get)
                winning_bid = bids[winner_id]
                
                # Allocate the task
                task.allocated = True
                task.allocated_to = winner_id
                
                # For a second-price auction, find the second-lowest bid
                sorted_bids = sorted(bids.values())
                second_price = sorted_bids[1] if len(sorted_bids) > 1 else winning_bid
                
                # Update payment (second price)
                self.payments[winner_id] += second_price
                
                # Update robot's assigned tasks
                for robot in self.robots:
                    if robot.id == winner_id:
                        robot.assigned_tasks.append(task.id)
        
        self.mechanism_runtime = time.time() - start_time
        return self.get_allocation_results()['allocation']


class SequentialAuction(AuctionMechanism):
    """Implements a sequential auction mechanism for task allocation."""
    
    def run_auction(self, bidding_strategy: str = 'truthful', 
                   task_ordering: str = 'random') -> Dict[int, List[int]]:
        """
        Run a sequential auction where tasks are auctioned in a specific order.
        
        Args:
            bidding_strategy: Bidding strategy for robots
            task_ordering: Method for ordering tasks ('random', 'deadline', 'difficulty')
            
        Returns:
            Dictionary mapping robot IDs to lists of allocated task IDs
        """
        start_time = time.time()
        
        # Initialize payments
        self.payments = {robot.id: 0.0 for robot in self.robots}
        
        # Order tasks according to the specified method
        ordered_tasks = self.tasks.copy()
        if task_ordering == 'deadline':
            # Sort by deadline (tasks without deadlines come last)
            ordered_tasks.sort(key=lambda t: t.deadline if t.deadline is not None else float('inf'))
        elif task_ordering == 'difficulty':
            # Sort by difficulty (hardest first)
            ordered_tasks.sort(key=lambda t: t.difficulty, reverse=True)
        else:
            # Random ordering
            random.shuffle(ordered_tasks)
        
        # Process tasks in the determined order
        for task in ordered_tasks:
            # Check if prerequisites are satisfied
            prerequisites_met = all(
                any(t.id == prereq and t.allocated for t in self.tasks)
                for prereq in task.prerequisites
            )
            
            if not prerequisites_met:
                continue
            
            # Collect bids from all robots
            bids = {}
            for robot in self.robots:
                # Robots consider their current allocation when bidding
                bid = robot.bid(task, bidding_strategy)
                bids[robot.id] = bid
                self.communication_volume += 1
            
            # Find the winner (lowest bid)
            if bids:
                winner_id = min(bids, key=bids.get)
                winning_bid = bids[winner_id]
                
                # Allocate the task
                task.allocated = True
                task.allocated_to = winner_id
                
                # For a second-price auction, find the second-lowest bid
                sorted_bids = sorted(bids.values())
                second_price = sorted_bids[1] if len(sorted_bids) > 1 else winning_bid
                
                # Update payment
                self.payments[winner_id] += second_price
                
                # Update robot's assigned tasks
                for robot in self.robots:
                    if robot.id == winner_id:
                        robot.assigned_tasks.append(task.id)
        
        self.mechanism_runtime = time.time() - start_time
        return self.get_allocation_results()['allocation']


class CombinatorialAuction(AuctionMechanism):
    """Implements a combinatorial auction mechanism for task allocation."""
    
    def __init__(self, robots: List[Robot], tasks: List[Task], max_bundle_size: int = 3):
        """
        Initialize the combinatorial auction mechanism.
        
        Args:
            robots: List of Robot objects
            tasks: List of Task objects
            max_bundle_size: Maximum number of tasks in a bundle
        """
        super().__init__(robots, tasks)
        self.max_bundle_size = max_bundle_size
    
    def run_auction(self, bidding_strategy: str = 'truthful') -> Dict[int, List[int]]:
        """
        Run a combinatorial auction where robots bid on bundles of tasks.
        
        Args:
            bidding_strategy: Bidding strategy for robots
            
        Returns:
            Dictionary mapping robot IDs to lists of allocated task IDs
        """
        import itertools
        start_time = time.time()
        
        # Initialize payments
        self.payments = {robot.id: 0.0 for robot in self.robots}
        
        # Generate all possible bundles up to max_bundle_size
        all_bundles = []
        for size in range(1, self.max_bundle_size + 1):
            for bundle in itertools.combinations(self.tasks, size):
                all_bundles.append(list(bundle))
        
        # Collect bids from all robots for all bundles
        bids = {}
        for robot in self.robots:
            bids[robot.id] = {}
            for bundle in all_bundles:
                bid = robot.bundle_bid(bundle, bidding_strategy)
                bundle_key = frozenset(t.id for t in bundle)
                bids[robot.id][bundle_key] = bid
                self.communication_volume += 1
        
        # Solve the winner determination problem
        # For simplicity, we'll use a greedy approach here
        # In a real system, this would use a more sophisticated algorithm
        
        # Sort all (robot, bundle) pairs by bid value
        all_bids = []
        for robot_id, robot_bids in bids.items():
            for bundle_key, bid_value in robot_bids.items():
                all_bids.append((robot_id, bundle_key, bid_value))
        
        all_bids.sort(key=lambda x: x[2])  # Sort by bid value (lowest first)
        
        # Greedy allocation
        allocated_tasks = set()
        allocation = {robot.id: [] for robot in self.robots}
        
        for robot_id, bundle_key, bid_value in all_bids:
            # Check if any task in this bundle is already allocated
            if any(task_id in allocated_tasks for task_id in bundle_key):
                continue
            
            # Allocate the bundle to this robot
            for task_id in bundle_key:
                allocated_tasks.add(task_id)
                allocation[robot_id].append(task_id)
                
                # Update task allocation status
                for task in self.tasks:
                    if task.id == task_id:
                        task.allocated = True
                        task.allocated_to = robot_id
            
            # Calculate VCG-like payment (simplified)
            # In a real system, this would be a proper VCG payment
            self.payments[robot_id] += bid_value
        
        self.mechanism_runtime = time.time() - start_time
        return allocation


def generate_random_scenario(num_robots: int = 5, num_tasks: int = 10, 
                            area_size: float = 100.0) -> Tuple[List[Robot], List[Task]]:
    """
    Generate a random scenario with robots and tasks.
    
    Args:
        num_robots: Number of robots to generate
        num_tasks: Number of tasks to generate
        area_size: Size of the square area
        
    Returns:
        Tuple of (robots list, tasks list)
    """
    # Generate robots
    robots = []
    for i in range(num_robots):
        location = (random.uniform(0, area_size), random.uniform(0, area_size))
        speed = random.uniform(0.8, 1.2)
        capabilities = {
            'sensing': random.uniform(0.5, 1.5),
            'manipulation': random.uniform(0.5, 1.5),
            'computation': random.uniform(0.5, 1.5)
        }
        robots.append(Robot(i, location, speed, capabilities))
    
    # Generate tasks
    tasks = []
    for i in range(num_tasks):
        location = (random.uniform(0, area_size), random.uniform(0, area_size))
        difficulty = random.uniform(0.5, 2.0)
        
        # Some tasks have deadlines
        deadline = random.uniform(10, 50) if random.random() < 0.3 else None
        
        # Some tasks have prerequisites
        prerequisites = []
        if i > 0 and random.random() < 0.2:
            num_prereqs = random.randint(1, min(3, i))
            prerequisites = random.sample(range(i), num_prereqs)
        
        tasks.append(Task(i, location, difficulty, deadline, prerequisites))
    
    return robots, tasks


def visualize_allocation(robots: List[Robot], tasks: List[Task], 
                        allocation: Dict[int, List[int]], title: str = "Task Allocation"):
    """
    Visualize the task allocation result.
    
    Args:
        robots: List of Robot objects
        tasks: List of Task objects
        allocation: Dictionary mapping robot IDs to lists of task IDs
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot area
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # Plot robots
    for robot in robots:
        plt.scatter(robot.location[0], robot.location[1], marker='o', s=100, 
                   label=f"Robot {robot.id}" if robot.id == 0 else "")
        plt.text(robot.location[0] + 2, robot.location[1] + 2, f"R{robot.id}")
    
    # Plot tasks
    for task in tasks:
        plt.scatter(task.location[0], task.location[1], marker='x', s=50, 
                   label=f"Task {task.id}" if task.id == 0 else "")
        plt.text(task.location[0] + 2, task.location[1] + 2, f"T{task.id}")
    
    # Plot allocation with lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for i, robot in enumerate(robots):
        if robot.id in allocation:
            for task_id in allocation[robot.id]:
                task = next(t for t in tasks if t.id == task_id)
                plt.plot([robot.location[0], task.location[0]], 
                        [robot.location[1], task.location[1]], 
                        '-', color=colors[i % len(colors)], alpha=0.5)
    
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def compare_auction_mechanisms(num_robots: int = 5, num_tasks: int = 10, 
                              bidding_strategy: str = 'truthful'):
    """
    Compare different auction mechanisms on the same scenario.
    
    Args:
        num_robots: Number of robots
        num_tasks: Number of tasks
        bidding_strategy: Bidding strategy for robots
    """
    # Generate a random scenario
    robots, tasks = generate_random_scenario(num_robots, num_tasks)
    
    # Create copies for each mechanism to ensure fair comparison
    robots1 = [Robot(r.id, r.location, r.speed, r.capabilities.copy()) for r in robots]
    tasks1 = [Task(t.id, t.location, t.difficulty, t.deadline, t.prerequisites.copy()) for t in tasks]
    
    robots2 = [Robot(r.id, r.location, r.speed, r.capabilities.copy()) for r in robots]
    tasks2 = [Task(t.id, t.location, t.difficulty, t.deadline, t.prerequisites.copy()) for t in tasks]
    
    robots3 = [Robot(r.id, r.location, r.speed, r.capabilities.copy()) for r in robots]
    tasks3 = [Task(t.id, t.location, t.difficulty, t.deadline, t.prerequisites.copy()) for t in tasks]
    
    # Run each mechanism
    print("Running Single-Item Auction...")
    single_auction = SingleItemAuction(robots1, tasks1)
    single_results = single_auction.run_auction(bidding_strategy)
    single_metrics = single_auction.get_allocation_results()
    
    print("Running Sequential Auction...")
    seq_auction = SequentialAuction(robots2, tasks2)
    seq_results = seq_auction.run_auction(bidding_strategy, 'difficulty')
    seq_metrics = seq_auction.get_allocation_results()
    
    print("Running Combinatorial Auction...")
    comb_auction = CombinatorialAuction(robots3, tasks3, max_bundle_size=3)
    comb_results = comb_auction.run_auction(bidding_strategy)
    comb_metrics = comb_auction.get_allocation_results()
    
    # Visualize allocations
    visualize_allocation(robots1, tasks1, single_results, "Single-Item Auction Allocation")
    visualize_allocation(robots2, tasks2, seq_results, "Sequential Auction Allocation")
    visualize_allocation(robots3, tasks3, comb_results, "Combinatorial Auction Allocation")
    
    # Compare metrics
    print("\nComparison of Auction Mechanisms:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Single-Item':<15} {'Sequential':<15} {'Combinatorial':<15}")
    print("-" * 50)
    
    metrics = ['social_welfare', 'runtime', 'communication']
    labels = ['Social Welfare', 'Runtime (s)', 'Messages']
    
    for metric, label in zip(metrics, labels):
        print(f"{label:<20} {single_metrics[metric]:<15.4f} {seq_metrics[metric]:<15.4f} {comb_metrics[metric]:<15.4f}")
    
    # Compare allocation distribution
    single_counts = [len(tasks) for robot_id, tasks in single_results.items()]
    seq_counts = [len(tasks) for robot_id, tasks in seq_results.items()]
    comb_counts = [len(tasks) for robot_id, tasks in comb_results.items()]
    
    print("\nTasks per Robot:")
    print(f"{'Robot ID':<10} {'Single-Item':<15} {'Sequential':<15} {'Combinatorial':<15}")
    for i in range(num_robots):
        print(f"{i:<10} {single_counts[i]:<15} {seq_counts[i]:<15} {comb_counts[i]:<15}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Social welfare comparison
    plt.subplot(1, 2, 1)
    mechanisms = ['Single-Item', 'Sequential', 'Combinatorial']
    welfare_values = [single_metrics['social_welfare'], seq_metrics['social_welfare'], comb_metrics['social_welfare']]
    plt.bar(mechanisms, welfare_values)
    plt.title('Social Welfare Comparison')
    plt.ylabel('Social Welfare (-Total Cost)')
    
    # Runtime and communication comparison
    plt.subplot(1, 2, 2)
    runtime_values = [single_metrics['runtime'], seq_metrics['runtime'], comb_metrics['runtime']]
    comm_values = [single_metrics['communication'] / 100, seq_metrics['communication'] / 100, comb_metrics['communication'] / 100]
    
    x = np.arange(len(mechanisms))
    width = 0.35
    
    plt.bar(x - width/2, runtime_values, width, label='Runtime (s)')
    plt.bar(x + width/2, comm_values, width, label='Messages (×100)')
    plt.xticks(x, mechanisms)
    plt.title('Performance Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('auction_mechanism_comparison.png')
    plt.close()
    
    print("\nComparison plots saved to 'auction_mechanism_comparison.png'")


def analyze_bidding_strategies():
    """
    Analyze the impact of different bidding strategies on auction outcomes.
    """
    num_robots = 5
    num_tasks = 10
    
    # Create a fixed random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate a random scenario
    robots, tasks = generate_random_scenario(num_robots, num_tasks)
    
    strategies = ['truthful', 'aggressive', 'conservative']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting with {strategy} bidding strategy...")
        
        # Create copies for this strategy
        robots_copy = [Robot(r.id, r.location, r.speed, r.capabilities.copy()) for r in robots]
        tasks_copy = [Task(t.id, t.location, t.difficulty, t.deadline, t.prerequisites.copy()) for t in tasks]
        
        # Run a sequential auction
        auction = SequentialAuction(robots_copy, tasks_copy)
        allocation = auction.run_auction(strategy, 'difficulty')
        metrics = auction.get_allocation_results()
        
        # Store results
        results[strategy] = {
            'allocation': allocation,
            'metrics': metrics
        }
        
        # Print results
        print(f"Social welfare: {metrics['social_welfare']:.4f}")
        print(f"Total payments: {sum(metrics['payments'].values()):.4f}")
        
        # Visualize
        visualize_allocation(robots_copy, tasks_copy, allocation, f"{strategy.capitalize()} Bidding Strategy")
    
    # Compare results across strategies
    print("\nComparison across bidding strategies:")
    print(f"{'Strategy':<15} {'Social Welfare':<15} {'Total Payments':<15}")
    for strategy, result in results.items():
        welfare = result['metrics']['social_welfare']
        payments = sum(result['metrics']['payments'].values())
        print(f"{strategy:<15} {welfare:<15.4f} {payments:<15.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Social welfare comparison
    plt.subplot(1, 2, 1)
    welfare_values = [results[s]['metrics']['social_welfare'] for s in strategies]
    plt.bar(strategies, welfare_values)
    plt.title('Social Welfare by Bidding Strategy')
    plt.ylabel('Social Welfare (-Total Cost)')
    
    # Payment comparison
    plt.subplot(1, 2, 2)
    payment_values = [sum(results[s]['metrics']['payments'].values()) for s in strategies]
    plt.bar(strategies, payment_values)
    plt.title('Total Payments by Bidding Strategy')
    plt.ylabel('Total Payments')
    
    plt.tight_layout()
    plt.savefig('bidding_strategy_comparison.png')
    plt.close()
    
    print("\nComparison plots saved to 'bidding_strategy_comparison.png'")


if __name__ == "__main__":
    print("Auction-Based Mechanisms for Multi-Robot Task Allocation")
    print("=" * 60)
    
    # Compare different auction mechanisms
    print("\nComparing auction mechanisms...")
    compare_auction_mechanisms(5, 10, 'truthful')
    
    # Analyze bidding strategies
    print("\n" + "=" * 60)
    print("Analyzing bidding strategies...")
    analyze_bidding_strategies()
