"""
Learning and Adaptation in Mechanism Participation

This module implements a learning-based mechanism for multi-robot task allocation,
demonstrating how robots can learn effective bidding strategies and how mechanisms
can adapt to learning participants.
"""

import random
import math
import time
import statistics
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


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
    learning_rate: float = 0.1
    exploration_rate: float = 0.2


class RLBiddingAgent:
    """
    Reinforcement learning agent for bidding in auctions.
    Uses Q-learning to adapt bidding strategy based on experience.
    """
    
    def __init__(self, robot: Robot, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, exploration_rate: float = 0.2):
        """
        Initialize the RL bidding agent.
        
        Args:
            robot: The robot this agent controls
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Probability of random exploration
        """
        self.robot = robot
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_values = {}  # State-action values
        self.history = []  # History of (state, action, reward, next_state)
        self.current_workload = 0.0
        self.last_market_context = {}
        self.bid_history = defaultdict(list)  # task_type -> list of (bid_ratio, outcome, utility)
    
    def get_state_features(self, task: Task, market_context: Dict[str, Any]) -> Tuple:
        """
        Extract relevant state features for learning.
        
        Args:
            task: The task to bid on
            market_context: Information about the current market
            
        Returns:
            Tuple of state features
        """
        # Features include:
        # - Task difficulty
        # - Task deadline (normalized)
        # - Number of competitors
        # - Average clearing price (normalized)
        # - Current workload
        # - Distance to task (normalized)
        
        # Calculate distance to task
        distance = math.sqrt((self.robot.position[0] - task.position[0])**2 + 
                            (self.robot.position[1] - task.position[1])**2)
        
        # Normalize distance to [0, 1] range assuming max distance of 100
        normalized_distance = min(1.0, distance / 100.0)
        
        # Normalize deadline to [0, 1] range assuming max deadline of 1000
        normalized_deadline = min(1.0, task.deadline / 1000.0)
        
        # Get market features with defaults
        num_competitors = market_context.get('num_competitors', 0)
        avg_clearing_price = market_context.get('avg_clearing_price', 0.0)
        
        # Normalize average clearing price to [0, 1] range assuming max price of 100
        normalized_avg_price = min(1.0, avg_clearing_price / 100.0)
        
        features = (
            task.difficulty,
            normalized_deadline,
            min(1.0, num_competitors / 10.0),  # Normalize to [0, 1]
            normalized_avg_price,
            min(1.0, self.current_workload / 5.0),  # Normalize to [0, 1]
            normalized_distance
        )
        
        return features
    
    def get_possible_actions(self, true_valuation: float) -> List[float]:
        """
        Generate possible bidding actions.
        
        Args:
            true_valuation: The true valuation for the task
            
        Returns:
            List of possible bid values
        """
        # Actions are bid values as percentages of true valuation
        bid_percentages = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        return [true_valuation * p for p in bid_percentages]
    
    def get_q_value(self, state: Tuple, action: float) -> float:
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: The state tuple
            action: The action (bid value)
            
        Returns:
            Q-value for the state-action pair
        """
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        if (state_key, action_key) not in self.q_values:
            # Initialize with optimistic values to encourage exploration
            self.q_values[(state_key, action_key)] = 1.0
        
        return self.q_values[(state_key, action_key)]
    
    def update_q_value(self, state: Tuple, action: float, reward: float, next_state: Optional[Tuple] = None) -> None:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state (None if terminal)
        """
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get maximum Q-value for next state
        if next_state is not None:
            true_valuation = self.get_true_valuation(next_state)
            next_actions = self.get_possible_actions(true_valuation)
            max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-value
        self.q_values[(state_key, action_key)] = new_q
    
    def select_action(self, task: Task, market_context: Dict[str, Any]) -> Tuple[float, float]:
        """
        Select bidding action using epsilon-greedy policy.
        
        Args:
            task: The task to bid on
            market_context: Information about the current market
            
        Returns:
            Tuple of (selected bid value, true valuation)
        """
        # Save market context for later use
        self.last_market_context = market_context
        
        # Get state features
        state = self.get_state_features(task, market_context)
        
        # Calculate true valuation
        true_valuation = self.get_true_valuation(task)
        
        # Get possible actions
        possible_actions = self.get_possible_actions(true_valuation)
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            selected_bid = random.choice(possible_actions)
            print(f"Robot {self.robot.id} exploring with bid {selected_bid:.2f} (true value: {true_valuation:.2f})")
            return selected_bid, true_valuation
        
        # Exploitation: best action according to Q-values
        selected_bid = max(possible_actions, key=lambda a: self.get_q_value(state, a))
        print(f"Robot {self.robot.id} exploiting with bid {selected_bid:.2f} (true value: {true_valuation:.2f})")
        return selected_bid, true_valuation
    
    def observe_outcome(self, task: Task, bid: float, true_value: float, 
                        won: bool, payment: float, utility: float) -> None:
        """
        Observe auction outcome and update strategy.
        
        Args:
            task: The task bid on
            bid: The bid value
            true_value: The true valuation
            won: Whether the bid won
            payment: The payment amount (if won)
            utility: The utility gained
        """
        # Get state features
        state = self.get_state_features(task, self.last_market_context)
        
        # Record action and outcome
        action = bid
        reward = utility
        
        # Store experience
        self.history.append((state, action, reward, None))  # Terminal state
        
        # Update Q-value
        self.update_q_value(state, action, reward)
        
        # Record bid history for this task type
        task_type = frozenset(task.required_capabilities)
        bid_ratio = bid / true_value if true_value > 0 else 1.0
        self.bid_history[task_type].append((bid_ratio, won, utility))
        
        # Gradually reduce exploration rate
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        # Update workload if task was won
        if won:
            self.current_workload += task.difficulty
        
        print(f"Robot {self.robot.id} observed outcome: won={won}, utility={utility:.2f}, " +
              f"new exploration rate={self.exploration_rate:.3f}")
    
    def get_state_key(self, state: Tuple) -> Tuple:
        """
        Convert state features to a hashable key.
        
        Args:
            state: The state tuple
            
        Returns:
            Discretized state tuple for use as dictionary key
        """
        # Discretize continuous features for better generalization
        discretized = tuple(round(f, 1) for f in state)
        return discretized
    
    def get_action_key(self, action: float) -> float:
        """
        Convert action to a hashable key.
        
        Args:
            action: The action value
            
        Returns:
            Discretized action value for use as dictionary key
        """
        return round(action, 2)
    
    def get_true_valuation(self, task: Task) -> float:
        """
        Calculate true valuation for a task.
        
        Args:
            task: The task to evaluate
            
        Returns:
            True valuation for the task
        """
        # Check if robot has required capabilities
        if not task.required_capabilities.issubset(self.robot.capabilities):
            return 0.0  # Cannot perform task
        
        # Calculate distance to task
        distance = math.sqrt((self.robot.position[0] - task.position[0])**2 + 
                            (self.robot.position[1] - task.position[1])**2)
        
        # Base value depends on task value and difficulty
        base_value = task.value * (1.0 - task.difficulty)
        
        # Adjust for distance (closer is better)
        distance_factor = max(0.1, 1.0 - (distance / 100.0))
        
        # Adjust for current workload (less workload is better)
        workload_factor = max(0.1, 1.0 - (self.current_workload / 5.0))
        
        # Calculate final valuation
        valuation = base_value * distance_factor * workload_factor
        
        return valuation
    
    def get_bidding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about bidding behavior.
        
        Returns:
            Dictionary of bidding statistics
        """
        stats = {}
        
        for task_type, history in self.bid_history.items():
            if not history:
                continue
                
            # Extract bid ratios and outcomes
            bid_ratios = [h[0] for h in history]
            win_rates = [1.0 if h[1] else 0.0 for h in history]
            utilities = [h[2] for h in history]
            
            # Calculate statistics
            avg_bid_ratio = sum(bid_ratios) / len(bid_ratios)
            win_rate = sum(win_rates) / len(win_rates)
            avg_utility = sum(utilities) / len(utilities) if utilities else 0.0
            
            # Calculate trend in bid ratios (are they increasing or decreasing?)
            if len(bid_ratios) >= 5:
                recent_ratios = bid_ratios[-5:]
                if len(recent_ratios) >= 2:
                    trend = (recent_ratios[-1] - recent_ratios[0]) / (len(recent_ratios) - 1)
                else:
                    trend = 0.0
            else:
                trend = 0.0
            
            # Store statistics
            task_type_str = ','.join(sorted(task_type))
            stats[task_type_str] = {
                'avg_bid_ratio': avg_bid_ratio,
                'win_rate': win_rate,
                'avg_utility': avg_utility,
                'bid_ratio_trend': trend,
                'num_bids': len(bid_ratios)
            }
        
        return stats


class LearningAwareMechanism:
    """
    A mechanism that adapts to learning participants, providing exploration
    incentives and adjusting parameters based on learning progress.
    """
    
    def __init__(self, learning_phase_length: int = 100, exploration_subsidy: float = 0.2):
        """
        Initialize the learning-aware mechanism.
        
        Args:
            learning_phase_length: Number of rounds considered the learning phase
            exploration_subsidy: Maximum subsidy for exploration (as fraction of payment)
        """
        self.round = 0
        self.learning_phase_length = learning_phase_length
        self.exploration_subsidy = exploration_subsidy
        self.robot_strategies = {}  # Robot ID -> Strategy statistics
        self.mechanism_performance = []  # History of performance metrics
        self.task_history = defaultdict(list)  # Task type -> list of allocation outcomes
    
    def update_learning_phase(self) -> Tuple[float, float]:
        """
        Update learning phase parameters.
        
        Returns:
            Tuple of (learning progress, current subsidy)
        """
        self.round += 1
        
        # Calculate learning progress (0.0 to 1.0)
        learning_progress = min(1.0, self.round / self.learning_phase_length)
        
        # Adjust exploration subsidy based on learning progress
        current_subsidy = self.exploration_subsidy * (1.0 - learning_progress)
        
        return learning_progress, current_subsidy
    
    def track_robot_strategy(self, robot_id: str, bid: float, true_value: float) -> None:
        """
        Track robot's strategic behavior.
        
        Args:
            robot_id: ID of the robot
            bid: The bid value
            true_value: The true valuation
        """
        if robot_id not in self.robot_strategies:
            self.robot_strategies[robot_id] = {
                'bids': [],
                'true_values': [],
                'bid_ratios': [],
                'strategy_consistency': 0.0
            }
        
        # Record bid and true value
        self.robot_strategies[robot_id]['bids'].append(bid)
        self.robot_strategies[robot_id]['true_values'].append(true_value)
        
        # Calculate bid ratio
        if true_value > 0:
            bid_ratio = bid / true_value
        else:
            bid_ratio = 1.0
        self.robot_strategies[robot_id]['bid_ratios'].append(bid_ratio)
        
        # Calculate strategy consistency (how consistent recent bid ratios are)
        if len(self.robot_strategies[robot_id]['bid_ratios']) >= 5:
            recent_ratios = self.robot_strategies[robot_id]['bid_ratios'][-5:]
            
            # Calculate consistency as inverse of standard deviation of bid ratios
            if len(recent_ratios) > 1:
                std_dev = statistics.stdev(recent_ratios)
                consistency = 1.0 / (1.0 + std_dev)
            else:
                consistency = 1.0
            
            self.robot_strategies[robot_id]['strategy_consistency'] = consistency
    
    def adjust_payment_for_learning(self, base_payment: float, robot_id: str, 
                                   learning_progress: float) -> float:
        """
        Adjust payment to account for learning phase.
        
        Args:
            base_payment: The base payment amount
            robot_id: ID of the robot
            learning_progress: Current learning progress (0.0 to 1.0)
            
        Returns:
            Adjusted payment amount
        """
        # Get strategy consistency
        if robot_id in self.robot_strategies:
            consistency = self.robot_strategies[robot_id]['strategy_consistency']
        else:
            consistency = 0.5  # Default for new robots
        
        # Calculate exploration subsidy
        current_subsidy = self.exploration_subsidy * (1.0 - learning_progress)
        
        # Higher subsidy for exploring (less consistent) robots
        exploration_factor = 1.0 - consistency
        subsidy_amount = base_payment * current_subsidy * exploration_factor
        
        # Adjust payment
        adjusted_payment = max(0, base_payment - subsidy_amount)
        
        return adjusted_payment
    
    def select_mechanism_variant(self, learning_progress: float) -> str:
        """
        Select appropriate mechanism variant based on learning progress.
        
        Args:
            learning_progress: Current learning progress (0.0 to 1.0)
            
        Returns:
            Name of the selected mechanism variant
        """
        if learning_progress < 0.3:
            # Early learning phase: use strategically simple variant
            return "posted_price"
        elif learning_progress < 0.7:
            # Middle learning phase: use intermediate variant
            return "sequential_auction"
        else:
            # Late learning phase: use sophisticated variant
            return "combinatorial_auction"
    
    def execute_posted_price(self, robots: List[Robot], tasks: List[Task], 
                            bidding_agents: Dict[str, RLBiddingAgent]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute posted-price mechanism (strategically simple).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            bidding_agents: Dict mapping robot IDs to bidding agents
            
        Returns:
            Tuple of (allocation, payments)
        """
        allocation = {}  # Task ID -> Robot ID
        payments = {}  # Robot ID -> Payment
        
        # Sort tasks by value (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.value, reverse=True)
        
        # For each task, offer it to robots in random order at a fixed price
        for task in sorted_tasks:
            # Calculate posted price (70% of estimated average value)
            avg_value_estimate = task.value * (1.0 - task.difficulty) * 0.7
            posted_price = avg_value_estimate
            
            # Shuffle robots to randomize order
            shuffled_robots = list(robots)
            random.shuffle(shuffled_robots)
            
            for robot in shuffled_robots:
                # Skip if task is already allocated
                if task.id in allocation:
                    continue
                
                # Skip if robot doesn't have required capabilities
                if not task.required_capabilities.issubset(robot.capabilities):
                    continue
                
                # Get bidding agent
                agent = bidding_agents[robot.id]
                
                # Get true valuation
                true_value = agent.get_true_valuation(task)
                
                # Robot accepts if true value >= posted price
                if true_value >= posted_price:
                    allocation[task.id] = robot.id
                    payments[robot.id] = payments.get(robot.id, 0.0) + posted_price
                    
                    # Update agent
                    utility = true_value - posted_price
                    agent.observe_outcome(task, posted_price, true_value, True, posted_price, utility)
                    break
                else:
                    # Robot rejected the task
                    agent.observe_outcome(task, posted_price, true_value, False, 0.0, 0.0)
        
        return allocation, payments
    
    def execute_sequential_auction(self, robots: List[Robot], tasks: List[Task], 
                                  bidding_agents: Dict[str, RLBiddingAgent]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute sequential auction mechanism (intermediate complexity).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            bidding_agents: Dict mapping robot IDs to bidding agents
            
        Returns:
            Tuple of (allocation, payments)
        """
        allocation = {}  # Task ID -> Robot ID
        payments = {}  # Robot ID -> Payment
        
        # Sort tasks by value (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.value, reverse=True)
        
        # For each task, run a separate second-price auction
        for task in sorted_tasks:
            bids = {}  # Robot ID -> Bid
            true_values = {}  # Robot ID -> True Value
            
            # Collect bids from all robots
            for robot in robots:
                # Skip if robot doesn't have required capabilities
                if not task.required_capabilities.issubset(robot.capabilities):
                    continue
                
                # Get bidding agent
                agent = bidding_agents[robot.id]
                
                # Get market context
                market_context = {
                    'num_competitors': len(robots) - 1,
                    'avg_clearing_price': sum(payments.values()) / max(1, len(payments)),
                    'mechanism': 'sequential_auction'
                }
                
                # Select bid
                bid, true_value = agent.select_action(task, market_context)
                
                # Record bid
                bids[robot.id] = bid
                true_values[robot.id] = true_value
                
                # Track strategy
                self.track_robot_strategy(robot.id, bid, true_value)
            
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
                
                # Update agents with outcome
                for robot_id, bid in bids.items():
                    agent = bidding_agents[robot_id]
                    true_value = true_values[robot_id]
                    
                    if robot_id == winner_id:
                        # Winner
                        utility = true_value - payment
                        agent.observe_outcome(task, bid, true_value, True, payment, utility)
                    else:
                        # Loser
                        agent.observe_outcome(task, bid, true_value, False, 0.0, 0.0)
        
        return allocation, payments
    
    def execute_combinatorial_auction(self, robots: List[Robot], tasks: List[Task], 
                                     bidding_agents: Dict[str, RLBiddingAgent]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Execute combinatorial auction mechanism (sophisticated).
        
        Args:
            robots: List of robots
            tasks: List of tasks
            bidding_agents: Dict mapping robot IDs to bidding agents
            
        Returns:
            Tuple of (allocation, payments)
        """
        # For simplicity, we'll implement a simplified version that considers only singleton bids
        # A full combinatorial auction would consider bids on bundles of tasks
        
        allocation = {}  # Task ID -> Robot ID
        payments = {}  # Robot ID -> Payment
        
        # Collect bids from all robots for all tasks
        bids = {}  # Robot ID -> {Task ID -> Bid}
        true_values = {}  # Robot ID -> {Task ID -> True Value}
        
        for robot in robots:
            bids[robot.id] = {}
            true_values[robot.id] = {}
            
            # Get bidding agent
            agent = bidding_agents[robot.id]
            
            for task in tasks:
                # Skip if robot doesn't have required capabilities
                if not task.required_capabilities.issubset(robot.capabilities):
                    continue
                
                # Get market context
                market_context = {
                    'num_competitors': len(robots) - 1,
                    'avg_clearing_price': sum(payments.values()) / max(1, len(payments)),
                    'mechanism': 'combinatorial_auction'
                }
                
                # Select bid
                bid, true_value = agent.select_action(task, market_context)
                
                # Record bid
                bids[robot.id][task.id] = bid
                true_values[robot.id][task.id] = true_value
                
                # Track strategy
                self.track_robot_strategy(robot.id, bid, true_value)
        
        # Solve winner determination problem (simplified greedy approach)
        # Sort tasks by highest bid
        task_bids = []  # List of (task_id, robot_id, bid)
        
        for robot_id, robot_bids in bids.items():
            for task_id, bid in robot_bids.items():
                task_bids.append((task_id, robot_id, bid))
        
        # Sort by bid (highest first)
        task_bids.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy allocation
        allocated_robots = set()
        
        for task_id, robot_id, bid in task_bids:
            # Skip if task is already allocated
            if task_id in allocation:
                continue
            
            # Skip if robot is already allocated (simplified constraint)
            if robot_id in allocated_robots:
                continue
            
            # Allocate task to robot
            allocation[task_id] = robot_id
            allocated_robots.add(robot_id)
        
        # Calculate VCG-like payments
        for task_id, robot_id in allocation.items():
            # Find the highest bid from another robot for this task
            other_bids = [b for t, r, b in task_bids if t == task_id and r != robot_id]
            if other_bids:
                payment = max(other_bids)
            else:
                payment = bids[robot_id][task_id] * 0.5  # If no other bids, pay half the bid
            
            # Record payment
            payments[robot_id] = payments.get(robot_id, 0.0) + payment
        
        # Update agents with outcome
        for robot_id, robot_bids in bids.items():
            agent = bidding_agents[robot_id]
            
            for task_id, bid in robot_bids.items():
                true_value = true_values[robot_id][task_id]
                
                if allocation.get(task_id) == robot_id:
                    # Winner
                    payment = payments[robot_id]  # Simplified: full payment for all tasks
                    utility = true_value - payment
                    agent.observe_outcome(tasks[tasks.index(next(t for t in tasks if t.id == task_id))], 
                                         bid, true_value, True, payment, utility)
                else:
                    # Loser
                    agent.observe_outcome(tasks[tasks.index(next(t for t in tasks if t.id == task_id))], 
                                         bid, true_value, False, 0.0, 0.0)
        
        return allocation, payments
    
    def execute_auction(self, robots: List[Robot], tasks: List[Task], 
                       bidding_agents: Dict[str, RLBiddingAgent]) -> Dict[str, Any]:
        """
        Execute auction with learning-aware adjustments.
        
        Args:
            robots: List of robots
            tasks: List of tasks
            bidding_agents: Dict mapping robot IDs to bidding agents
            
        Returns:
            Dict containing auction results
        """
        # Update learning phase parameters
        learning_progress, current_subsidy = self.update_learning_phase()
        
        # Select mechanism variant based on learning progress
        mechanism_variant = self.select_mechanism_variant(learning_progress)
        print(f"\nRound {self.round}: Using {mechanism_variant} (learning progress: {learning_progress:.2f})")
        
        # Execute selected mechanism
        if mechanism_variant == "posted_price":
            allocation, base_payments = self.execute_posted_price(robots, tasks, bidding_agents)
        elif mechanism_variant == "sequential_auction":
            allocation, base_payments = self.execute_sequential_auction(robots, tasks, bidding_agents)
        else:  # combinatorial_auction
            allocation, base_payments = self.execute_combinatorial_auction(robots, tasks, bidding_agents)
        
        # Adjust payments for learning phase
        payments = {}
        for robot_id, payment in base_payments.items():
            adjusted_payment = self.adjust_payment_for_learning(payment, robot_id, learning_progress)
            payments[robot_id] = adjusted_payment
            
            subsidy = payment - adjusted_payment
            if subsidy > 0:
                print(f"Robot {robot_id} received exploration subsidy: {subsidy:.2f} " +
                      f"({(subsidy/payment)*100:.1f}% of payment)")
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(allocation, payments, robots, tasks)
        self.mechanism_performance.append(performance_metrics)
        
        # Track task allocation history
        for task in tasks:
            task_type = frozenset(task.required_capabilities)
            allocated = task.id in allocation
            self.task_history[task_type].append(allocated)
        
        # Print summary
        print(f"Allocation rate: {performance_metrics['allocation_rate']:.2f}")
        print(f"Social welfare: {performance_metrics['social_welfare']:.2f}")
        print(f"Total payments: {performance_metrics['payment_total']:.2f}")
        
        return {
            'round': self.round,
            'mechanism_variant': mechanism_variant,
            'learning_progress': learning_progress,
            'allocation': allocation,
            'payments': payments,
            'performance': performance_metrics
        }
    
    def calculate_performance_metrics(self, allocation: Dict[str, str], payments: Dict[str, float], 
                                     robots: List[Robot], tasks: List[Task]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            allocation: Task allocation (Task ID -> Robot ID)
            payments: Robot payments (Robot ID -> Payment)
            robots: List of robots
            tasks: List of tasks
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate allocation rate
        allocation_rate = len(allocation) / len(tasks) if tasks else 0.0
        
        # Calculate social welfare (sum of true values minus payments)
        social_welfare = 0.0
        for task_id, robot_id in allocation.items():
            # Find task
            task = next(t for t in tasks if t.id == task_id)
            
            # Find robot
            robot = next(r for r in robots if r.id == robot_id)
            
            # Get bidding agent
            agent = next(a for a in [r for r in robots if r.id == robot_id] if a.id == robot_id)
            
            # Calculate true value
            true_value = agent.get_true_valuation(task) if hasattr(agent, 'get_true_valuation') else task.value
            
            # Add to social welfare
            social_welfare += true_value
        
        # Subtract payments
        for payment in payments.values():
            social_welfare -= payment
        
        # Calculate payment total
        payment_total = sum(payments.values())
        
        return {
            'allocation_rate': allocation_rate,
            'social_welfare': social_welfare,
            'payment_total': payment_total
        }


def simulate_learning_mechanism():
    """Run a simulation of the learning-aware mechanism."""
    # Create robots
    robots = [
        Robot(id="R1", position=(0, 0), capabilities={"sensing", "manipulation"}, 
              learning_rate=0.1, exploration_rate=0.3),
        Robot(id="R2", position=(10, 10), capabilities={"sensing", "locomotion"}, 
              learning_rate=0.2, exploration_rate=0.2),
        Robot(id="R3", position=(20, 20), capabilities={"manipulation", "locomotion"}, 
              learning_rate=0.1, exploration_rate=0.1),
        Robot(id="R4", position=(30, 30), capabilities={"sensing", "manipulation", "locomotion"}, 
              learning_rate=0.05, exploration_rate=0.4)
    ]
    
    # Create bidding agents
    bidding_agents = {
        robot.id: RLBiddingAgent(robot, learning_rate=robot.learning_rate, 
                                exploration_rate=robot.exploration_rate)
        for robot in robots
    }
    
    # Create tasks
    task_templates = [
        {"id_prefix": "T1", "position": (5, 5), "difficulty": 0.3, "deadline": 300, "value": 100, 
         "required_capabilities": {"sensing"}},
        {"id_prefix": "T2", "position": (15, 15), "difficulty": 0.5, "deadline": 200, "value": 150, 
         "required_capabilities": {"manipulation"}},
        {"id_prefix": "T3", "position": (25, 25), "difficulty": 0.7, "deadline": 100, "value": 200, 
         "required_capabilities": {"sensing", "manipulation"}},
        {"id_prefix": "T4", "position": (35, 35), "difficulty": 0.9, "deadline": 50, "value": 250, 
         "required_capabilities": {"locomotion", "manipulation"}}
    ]
    
    # Initialize mechanism
    mechanism = LearningAwareMechanism(
        learning_phase_length=20,
        exploration_subsidy=0.3
    )
    
    # Run simulation for multiple rounds
    num_rounds = 30
    
    for round_num in range(num_rounds):
        print(f"\n===== ROUND {round_num + 1} =====")
        
        # Generate new tasks for this round
        tasks = []
        for i, template in enumerate(task_templates):
            # Create task with unique ID
            task = Task(
                id=f"{template['id_prefix']}_{round_num}_{i}",
                position=template["position"],
                difficulty=template["difficulty"],
                deadline=template["deadline"],
                value=template["value"],
                required_capabilities=template["required_capabilities"]
            )
            tasks.append(task)
        
        # Execute auction
        results = mechanism.execute_auction(robots, tasks, bidding_agents)
        
        # Print allocation
        print("\nTask Allocation:")
        for task_id, robot_id in results['allocation'].items():
            print(f"Task {task_id} -> Robot {robot_id}")
        
        # Print payments
        print("\nPayments:")
        for robot_id, payment in results['payments'].items():
            print(f"Robot {robot_id}: {payment:.2f}")
        
        # Every 10 rounds, print bidding statistics
        if (round_num + 1) % 10 == 0:
            print("\nBidding Statistics:")
            for robot_id, agent in bidding_agents.items():
                stats = agent.get_bidding_statistics()
                print(f"\nRobot {robot_id}:")
                for task_type, task_stats in stats.items():
                    print(f"  Task Type {task_type}:")
                    print(f"    Avg Bid Ratio: {task_stats['avg_bid_ratio']:.2f}")
                    print(f"    Win Rate: {task_stats['win_rate']:.2f}")
                    print(f"    Avg Utility: {task_stats['avg_utility']:.2f}")
                    print(f"    Bid Ratio Trend: {task_stats['bid_ratio_trend']:.3f}")
                    print(f"    Number of Bids: {task_stats['num_bids']}")


if __name__ == "__main__":
    simulate_learning_mechanism()
