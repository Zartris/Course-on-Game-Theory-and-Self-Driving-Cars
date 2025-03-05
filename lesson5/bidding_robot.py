#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bidding Robot implementation for the Task Allocation and Auction Mechanisms lesson.

This module implements robots that can evaluate tasks, generate bids, and participate
in various auction mechanisms for task allocation.
"""

import numpy as np
import math
from typing import Dict, List, Set, Tuple, Union, Optional, Callable
from collections import defaultdict, deque

class BiddingRobot:
    """
    A robot agent capable of evaluating tasks, generating bids, and forming coalitions.
    
    This class models a robot with specific capabilities, position information, and bidding
    strategies for participating in market-based task allocation.
    """
    
    def __init__(self, 
                 robot_id: int, 
                 capabilities: Dict[str, float], 
                 initial_position: Tuple[int, int], 
                 strategy_type: str = 'truthful',
                 max_tasks: int = 1,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2,
                 verbose: bool = False):
        """
        Initialize a bidding robot with specified parameters.
        
        Args:
            robot_id: Unique identifier for the robot
            capabilities: Dictionary mapping capability types to proficiency levels
                          (e.g., {'speed': 0.8, 'lift_capacity': 5.0})
            initial_position: Starting position (x, y) on the grid
            strategy_type: Bidding strategy ('truthful', 'strategic', 'learning', 'cooperative')
            max_tasks: Maximum number of tasks the robot can handle simultaneously
            learning_rate: Rate at which the robot updates its strategy based on outcomes
            exploration_rate: Probability of trying a new bidding strategy during learning
            verbose: Whether to print detailed debug information
        """
        self.id = robot_id
        self.capabilities = capabilities
        self.position = initial_position
        self.strategy_type = strategy_type
        self.max_tasks = max_tasks
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.verbose = verbose
        
        # Robot state
        self.current_tasks = set()  # IDs of tasks currently assigned to this robot
        self.task_history = []      # History of completed tasks
        self.utility_history = []   # History of utilities earned
        self.bid_history = {}       # History of bids: {task_id: [(bid_amount, outcome)]}
        self.path = [initial_position]  # Movement history for visualization
        
        # For learning strategies
        self.q_values = defaultdict(lambda: defaultdict(float))  # {task_type: {bid_fraction: value}}
        
        # For reputation/trust
        self.reputation = 1.0  # Initial perfect reputation
        self.trust_in_others = {}  # {robot_id: trust_score}
        
        # Coalition information
        self.current_coalition = None  # Current coalition ID if part of a coalition
        self.coalition_role = None     # Role in the current coalition
        self.coalition_history = []    # History of coalitions participated in
        
        # Energy and resource model (optional)
        self.energy = 100.0
        self.energy_consumption_rate = 0.1  # Energy consumed per unit distance
        
    def evaluate_task(self, task) -> float:
        """
        Evaluate the utility of a task for this robot.
        
        Args:
            task: The task to evaluate
            
        Returns:
            float: The utility (positive) or cost (negative) of performing the task
        """
        if self.verbose:
            print(f"Robot {self.id} evaluating task {task.id}")
            print(f"  Task position: {task.position}")
            print(f"  Task type: {task.type}")
            print(f"  Task priority: {task.priority}")
            print(f"  Required capabilities: {task.required_capabilities}")
            print(f"  Robot capabilities: {self.capabilities}")
        
        # Basic utility calculation components
        distance_cost = self.calculate_distance_cost(task.position)
        capability_match = self.calculate_capability_match(task.required_capabilities)
        time_cost = self.calculate_time_cost(task)
        priority_value = task.priority * 2.0  # Scale priority to reasonable value
        
        if self.verbose:
            print(f"  Distance cost: {distance_cost}")
            print(f"  Capability match: {capability_match}")
            print(f"  Time cost: {time_cost}")
            print(f"  Priority value: {priority_value}")
        
        # Adjust for current workload
        workload_factor = 1.0 - (len(self.current_tasks) / self.max_tasks) if self.max_tasks > 0 else 0.0
        if self.verbose:
            print(f"  Workload factor: {workload_factor}")
        
        # Calculate final utility 
        utility = (
            priority_value +
            capability_match * 3.0 -  # Weight capability matching highly
            distance_cost -
            time_cost
        ) * workload_factor
        
        if self.verbose:
            print(f"  Base utility: {utility}")
        
        # Apply strategy-specific adjustments
        if self.strategy_type == 'cooperative':
            # Cooperative robots value tasks that align with team goals
            utility *= (1.0 + 0.2 * task.team_utility_factor)
            if self.verbose:
                print(f"  Applied cooperative strategy, new utility: {utility}")
            
        elif self.strategy_type == 'strategic':
            # Strategic robots might devalue tasks that others are likely to bid on
            if hasattr(task, 'competition_factor'):
                utility *= (1.0 - 0.1 * task.competition_factor)
                if self.verbose:
                    print(f"  Applied strategic adjustment, new utility: {utility}")
        
        if self.verbose:
            print(f"  Final utility: {utility}")
        return utility
    
    def calculate_distance_cost(self, task_position: Tuple[int, int]) -> float:
        """Calculate the cost associated with traveling to the task."""
        distance = math.sqrt(
            (task_position[0] - self.position[0])**2 +
            (task_position[1] - self.position[1])**2
        )
        
        # Scale by robot's speed capability (higher speed = lower cost)
        speed_factor = self.capabilities.get('speed', 1.0)
        return distance / speed_factor
    
    def calculate_capability_match(self, required_capabilities: Dict[str, float]) -> float:
        """Calculate how well the robot's capabilities match the task requirements."""
        if not required_capabilities:
            return 1.0  # Perfect match if no special capabilities required
        
        match_score = 0.0
        total_requirements = 0.0
        
        for cap, required_level in required_capabilities.items():
            total_requirements += required_level
            robot_level = self.capabilities.get(cap, 0.0)
            
            if robot_level >= required_level:
                # Full points if capability meets or exceeds requirement
                match_score += required_level
            else:
                # Partial points based on how close we are to requirement
                match_score += (robot_level / required_level) * required_level
        
        # Normalize to [0, 1] range
        return match_score / total_requirements if total_requirements > 0 else 1.0
    
    def calculate_time_cost(self, task) -> float:
        """Calculate the time cost of performing the task."""
        # Estimate based on distance and task complexity
        distance_time = self.calculate_distance_cost(task.position)
        
        # Use 'efficiency' capability if available, otherwise default to 1.0
        efficiency = self.capabilities.get('efficiency', 1.0)
        if self.verbose:
            print(f"  Robot efficiency: {efficiency}")
        
        execution_time = task.completion_time / efficiency
        if self.verbose:
            print(f"  Distance time: {distance_time}, Execution time: {execution_time}")
        
        return distance_time + execution_time
    
    def generate_bid(self, task, auction_type: str = 'sequential') -> Union[float, Dict, None]:
        """
        Generate a bid for a given task based on the robot's strategy.
        
        Args:
            task: The task to bid on
            auction_type: Type of auction ('sequential', 'combinatorial', etc.)
            
        Returns:
            A bid value, dictionary (for combinatorial auctions), or None if no bid
        """
        # First, evaluate the true utility
        true_utility = self.evaluate_task(task)
        
        # Check if we can even handle more tasks
        if len(self.current_tasks) >= self.max_tasks and task.id not in self.current_tasks:
            return None
        
        # Energy check - don't bid if we don't have enough energy
        distance = math.sqrt(
            (task.position[0] - self.position[0])**2 +
            (task.position[1] - self.position[1])**2
        )
        energy_needed = distance * self.energy_consumption_rate + task.energy_cost
        if energy_needed > self.energy:
            return None
            
        # Determine bid based on strategy
        bid_value = None
        
        # For bundle bids (combinatorial auctions)
        if auction_type == 'combinatorial' and hasattr(task, 'bundle_id'):
            return self._generate_combinatorial_bid(task)
        
        # For single-task bids
        if self.strategy_type == 'truthful':
            # Truthful bidding - bid true utility
            bid_value = true_utility
            
        elif self.strategy_type == 'strategic':
            # Strategic bidding - shade the bid to maximize profit
            competitors_estimate = getattr(task, 'expected_participants', 3)
            # Classic bid shading formula for first-price auctions
            bid_value = true_utility * ((competitors_estimate - 1) / competitors_estimate)
            
        elif self.strategy_type == 'learning':
            # Use learned Q-values to select bid, with exploration
            bid_value = self._generate_learning_bid(task, true_utility)
            
        elif self.strategy_type == 'cooperative':
            # In cooperative settings, might bid truthfully but consider team utility
            team_utility_factor = getattr(task, 'team_utility_factor', 1.0)
            bid_value = true_utility * team_utility_factor
        
        # Convert negative utility to a positive bid (to ensure robots bid on tasks even with negative utility)
        # In a real-world scenario, we might have more sophisticated logic here
        if bid_value is not None:
            if bid_value <= 0:
                # For negative utility, we'll create a small positive bid based on task priority
                # This ensures robots will still bid on tasks even if they're not optimal
                bid_value = task.priority * 0.5
            else:
                bid_value = max(0.1, bid_value)
        else:
            bid_value = task.priority * 0.1  # Default bid
        
        # Record the bid for learning purposes
        task_type = getattr(task, 'type', 'default')
        if task.id not in self.bid_history:
            self.bid_history[task.id] = []
        self.bid_history[task.id].append((bid_value, None))  # Outcome will be updated later
        
        return bid_value
    
    def _generate_learning_bid(self, task, true_utility: float) -> float:
        """Generate a bid using reinforcement learning approach."""
        task_type = getattr(task, 'type', 'default')
        
        # Exploration: try a random bid value with some probability
        if np.random.random() < self.exploration_rate:
            # Explore by bidding a random fraction of true utility
            bid_fraction = np.random.uniform(0.5, 1.2)
            return true_utility * bid_fraction
        
        # Exploitation: use the best known bidding fraction for this task type
        best_fraction = 1.0  # Default to truthful bidding
        best_value = -float('inf')
        
        for fraction in self.q_values[task_type]:
            if self.q_values[task_type][fraction] > best_value:
                best_value = self.q_values[task_type][fraction]
                best_fraction = fraction
        
        return true_utility * best_fraction
    
    def _generate_combinatorial_bid(self, task) -> Dict:
        """Generate a bid for a combinatorial auction."""
        # For simplicity, we'll bid on the task individually and as part of its bundle
        bundle_id = getattr(task, 'bundle_id', None)
        if not bundle_id:
            return {'single': self.evaluate_task(task)}
            
        # Check if we've already evaluated other tasks in this bundle
        bundle_tasks = [t for t in getattr(task, 'all_tasks', []) 
                       if getattr(t, 'bundle_id', None) == bundle_id]
        
        # Evaluate the entire bundle
        individual_values = [self.evaluate_task(t) for t in bundle_tasks]
        sum_individual = sum(individual_values)
        
        # Add synergy bonus for bundles
        bundle_synergy = 1.2  # Bundle is worth more than sum of parts
        bundle_value = sum_individual * bundle_synergy
        
        return {
            'single': self.evaluate_task(task),
            'bundle': {
                'id': bundle_id,
                'tasks': [t.id for t in bundle_tasks],
                'value': bundle_value
            }
        }
        
    def update_strategy(self, auction_results: Dict) -> None:
        """
        Update bidding strategy based on auction results.
        
        Args:
            auction_results: Dictionary containing auction outcomes
        """
        if self.strategy_type != 'learning':
            return  # Only learning agents update their strategy
            
        # Update Q-values based on auction outcomes
        for task_id, bid_info in self.bid_history.items():
            if task_id not in auction_results:
                continue
                
            # Get the most recent bid for this task
            bid_value, _ = bid_info[-1]
            
            # Get task details from results
            task_info = auction_results[task_id]
            task_type = task_info.get('type', 'default')
            true_utility = task_info.get('true_utility', 0)
            
            # Skip if we couldn't determine true utility or bid value
            if true_utility == 0 or bid_value == 0:
                continue
                
            # Calculate bid fraction (how much of true utility we bid)
            bid_fraction = round(bid_value / true_utility, 2)
            
            # Determine reward based on outcome
            reward = 0
            if task_info.get('winner') == self.id:
                # We won! Calculate profit (utility - payment)
                payment = task_info.get('payment', 0)
                profit = true_utility - payment
                reward = profit
            else:
                # We lost. If we bid too low, give a small negative reward
                winning_bid = task_info.get('winning_bid', float('inf'))
                if winning_bid > bid_value and winning_bid < true_utility:
                    # We should have bid higher
                    reward = -0.2 * (winning_bid - bid_value)
                else:
                    # We correctly avoided overpaying
                    reward = 0.1
            
            # Update Q-value for this bid fraction
            old_value = self.q_values[task_type][bid_fraction]
            self.q_values[task_type][bid_fraction] = old_value + self.learning_rate * (reward - old_value)
            
            # Update bid history with outcome
            self.bid_history[task_id][-1] = (bid_value, reward)
    
    def execute_task(self, assigned_task) -> bool:
        """
        Execute an assigned task.
        
        Args:
            assigned_task: The task to execute
            
        Returns:
            bool: True if task execution was successful, False otherwise
        """
        if self.verbose:
            print(f"Robot {self.id} executing task {assigned_task.id}")
        
        # Check if we have the task
        if assigned_task.id not in self.current_tasks:
            self.current_tasks.add(assigned_task.id)
            if self.verbose:
                print(f"Added task {assigned_task.id} to robot {self.id}'s current tasks")
        
        # Get target position for the task
        target_x, target_y = assigned_task.position
        current_x, current_y = self.position
        
        if self.verbose:
            print(f"Robot {self.id} at position {self.position}, target at {assigned_task.position}")
            print(f"Distance check: Robot {self.id} at {self.position}, task {assigned_task.id} at {assigned_task.position}")
            print(f"Distance check: dx={abs(current_x - target_x)}, dy={abs(current_y - target_y)}")
        
        # Consider task reached if robot is close to it - use a more generous threshold
        if abs(current_x - target_x) <= 1.5 and abs(current_y - target_y) <= 1.5:
            # We've reached the task, now complete it
            # Snap to the exact task position
            self.position = (target_x, target_y)
            self.path.append(self.position)
            if self.verbose:
                print(f"Robot {self.id} has reached task {assigned_task.id}")
            
            # Calculate task execution energy
            task_energy = assigned_task.energy_cost
            
            # Check if we have enough energy
            if task_energy > self.energy:
                # Task failed due to energy constraints
                if assigned_task.id in self.current_tasks:
                    self.current_tasks.remove(assigned_task.id)
                return False
            
            # Consume energy for the task itself
            self.energy -= task_energy
            
            # Mark task as completed
            self.task_history.append(assigned_task.id)
            if assigned_task.id in self.current_tasks:
                self.current_tasks.remove(assigned_task.id)
            
            # Record utility
            utility = self.evaluate_task(assigned_task)
            self.utility_history.append(utility)
            
            # Update reputation based on successful completion
            self.reputation = 0.95 * self.reputation + 0.05 * 1.0  # Slight boost for completion
            
            return True
            
        else:
            # Move toward the task (one step at a time)
            # Calculate direction vector
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if self.verbose:
                print(f"Moving robot {self.id} toward task {assigned_task.id}, distance={distance}")
            
            # Normalize the direction vector
            if distance > 0:
                # Move speed is just 1 grid cell per step
                step_size = min(1.0, distance)  # Don't overshoot
                if self.verbose:
                    print(f"Step size: {step_size}")
                
                # Calculate next position
                move_dx = dx / distance * step_size
                move_dy = dy / distance * step_size
                if self.verbose:
                    print(f"Movement vector: ({move_dx}, {move_dy})")
                
                # Update position (keep as floats for movement calculation)
                new_x = current_x + move_dx
                new_y = current_y + move_dy
                # Store the actual position (without rounding)
                old_pos = self.position
                self.position = (new_x, new_y)
                if self.verbose:
                    print(f"Robot {self.id} moved from {old_pos} to {self.position}")
                
                # For path visualization, we can use rounded positions
                path_pos = (round(new_x), round(new_y))
                self.path.append(path_pos)
                
                # Calculate energy consumption for this movement step
                step_distance = math.sqrt(
                    (new_x - current_x)**2 + (new_y - current_y)**2
                )
                movement_energy = step_distance * self.energy_consumption_rate
                if self.verbose:
                    print(f"Energy used: {movement_energy}, remaining: {self.energy}")
                
                # Check if we have enough energy
                if movement_energy > self.energy:
                    # Task failed due to energy constraints
                    if self.verbose:
                        print(f"Robot {self.id} has insufficient energy for task {assigned_task.id}")
                    if assigned_task.id in self.current_tasks:
                        self.current_tasks.remove(assigned_task.id)
                    return False
                
                # Consume energy for movement
                self.energy -= movement_energy
            else:
                if self.verbose:
                    print(f"Robot {self.id} is already at target position but not close enough?")
            
            # Haven't reached the task yet, keep moving next time
            return False
    
    def form_coalition(self, other_robots: List['BiddingRobot'], complex_task) -> Dict:
        """
        Form a coalition with other robots to handle a complex task.
        
        Args:
            other_robots: List of potential coalition partners
            complex_task: The task requiring multiple robots
            
        Returns:
            Dict: Coalition information including members, roles, and expected utility
        """
        if not complex_task.requires_coalition:
            return None
            
        required_capabilities = complex_task.required_capabilities
        coalition_members = [self]
        assigned_roles = {self.id: 'coordinator'}  # Initial role for proposing robot
        
        if self.verbose:
            print(f"Robot {self.id} attempting to form coalition for task {complex_task.id}")
            print(f"Task requires roles: {list(complex_task.role_requirements.keys())}")
        
        # Evaluate which robots should join the coalition
        for robot in other_robots:
            # Skip if robot is already in too many tasks
            if len(robot.current_tasks) >= robot.max_tasks:
                if self.verbose:
                    print(f"  Robot {robot.id} has too many tasks, skipping")
                continue
                
            # Skip if robot is already part of this coalition
            if robot.current_coalition == complex_task.id:
                if self.verbose:
                    print(f"  Robot {robot.id} is already part of this coalition")
                continue
                
            # Check if robot can meaningfully contribute
            contribution = 0
            potential_role = None
            
            # Find the best role for this robot
            for role, required_caps in complex_task.role_requirements.items():
                if role in assigned_roles.values():
                    continue  # Role already filled
                    
                match_score = robot.calculate_capability_match(required_caps)
                if match_score > contribution:
                    contribution = match_score
                    potential_role = role
            
            # Add robot to coalition if it can contribute meaningfully
            if contribution > 0.6 and potential_role:  # Threshold for meaningful contribution
                coalition_members.append(robot)
                assigned_roles[robot.id] = potential_role
                
                if self.verbose:
                    print(f"  Adding robot {robot.id} to coalition as {potential_role} (match={contribution:.2f})")
                
                # Check if we've filled all required roles
                if len(assigned_roles) == len(complex_task.role_requirements) + 1:  # +1 for coordinator
                    break
        
        # Evaluate if coalition can successfully complete the task
        can_complete = len(assigned_roles) >= len(complex_task.role_requirements) + 1
        
        if self.verbose:
            print(f"Coalition formation {'successful' if can_complete else 'failed'}")
            print(f"  Roles assigned: {assigned_roles}")
            print(f"  Members: {[r.id for r in coalition_members]}")
        
        if can_complete:
            # Calculate expected utility for the coalition
            coalition_utility = complex_task.base_utility
            
            # Distribute utility based on contribution (simplified Shapley value)
            utility_shares = {}
            total_capabilities = sum(robot.calculate_capability_match(required_capabilities) 
                                    for robot in coalition_members)
            
            for robot in coalition_members:
                capability_contribution = robot.calculate_capability_match(required_capabilities)
                utility_shares[robot.id] = (capability_contribution / total_capabilities) * coalition_utility
            
            # Create and return coalition information
            coalition_info = {
                'task_id': complex_task.id,
                'members': [r.id for r in coalition_members],
                'roles': assigned_roles,
                'utility_shares': utility_shares,
                'total_utility': coalition_utility,
                'can_complete': can_complete
            }
            
            # Record coalition for this robot
            self.current_coalition = complex_task.id
            self.coalition_role = assigned_roles[self.id]
            self.coalition_history.append(coalition_info)
            
            # Add the task to each coalition member's task list
            for robot in coalition_members:
                if robot.id != self.id:  # Already added to proposing robot
                    robot.current_tasks.add(complex_task.id)
                    robot.current_coalition = complex_task.id
                    robot.coalition_role = assigned_roles[robot.id]
                    robot.coalition_history.append(coalition_info)
            
            if self.verbose:
                print(f"Coalition formed successfully for task {complex_task.id}")
                print(f"  Utility shares: {utility_shares}")
            
            return coalition_info
        else:
            # Coalition formation failed
            if self.verbose:
                print(f"Coalition formation failed for task {complex_task.id}")
            return None
            
    def execute_coalition_task(self, task, coalition_info: Dict) -> bool:
        """
        Execute a task as part of a coalition.
        
        Args:
            task: The coalition task to execute
            coalition_info: Coalition information including roles and members
            
        Returns:
            bool: True if task execution was successful, False otherwise
        """
        if not task.requires_coalition or self.current_coalition != task.id:
            return False
            
        if self.verbose:
            print(f"Robot {self.id} executing coalition task {task.id} as {self.coalition_role}")
        
        # Get target position for the task
        target_x, target_y = task.position
        current_x, current_y = self.position
        
        # Check if we're already at or close enough to the task location
        if abs(current_x - target_x) <= 1.5 and abs(current_y - target_y) <= 1.5:
            if self.verbose:
                print(f"Robot {self.id} is at coalition task location")
                
            # We're at the task location, now wait for other coalition members
            # Check if all coalition members are at the task location
            all_members_present = True
            for member_id in coalition_info['members']:
                if member_id != self.id:
                    # Find the robot object
                    member_robot = None
                    for robot in self._get_coalition_robots():
                        if robot.id == member_id:
                            member_robot = robot
                            break
                    
                    if member_robot:
                        # Check if this member is at the task location
                        mx, my = member_robot.position
                        if abs(mx - target_x) > 1.5 or abs(my - target_y) > 1.5:
                            all_members_present = False
                            break
            
            # If all members are present, complete the task
            if all_members_present:
                if self.verbose:
                    print(f"All coalition members present for task {task.id}, completing task")
                    
                # Consume energy for task execution
                energy_cost = task.energy_cost / len(coalition_info['members'])  # Split energy cost
                
                # Check if we have enough energy
                if energy_cost > self.energy:
                    if self.verbose:
                        print(f"Robot {self.id} has insufficient energy for coalition task")
                    if task.id in self.current_tasks:
                        self.current_tasks.remove(task.id)
                    # Clear coalition status as we can't complete it
                    self.current_coalition = None
                    self.coalition_role = None
                    return False
                
                # Consume energy
                self.energy -= energy_cost
                
                # Record task completion
                self.task_history.append(task.id)
                if task.id in self.current_tasks:
                    self.current_tasks.remove(task.id)
                
                # Clear coalition status
                self.current_coalition = None
                self.coalition_role = None
                
                # Record utility
                utility = self.evaluate_task(task) * coalition_info['utility_shares'][self.id]
                self.utility_history.append(utility)
                
                return True
            else:
                # Not all members are present yet
                if self.verbose:
                    print(f"Waiting for other coalition members for task {task.id}")
                return False
        else:
            # Move toward the task location
            return self._move_toward_position(target_x, target_y)
            
    def _move_toward_position(self, target_x, target_y) -> bool:
        """Helper method to move toward a position."""
        current_x, current_y = self.position
        
        # Calculate direction vector
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx**2 + dy**2)
        
        if self.verbose:
            print(f"Robot {self.id} moving toward ({target_x}, {target_y}), distance={distance}")
        
        # If we're already there, return True
        if distance <= 0.1:
            return True
            
        # Normalize the direction vector
        step_size = min(1.0, distance)  # Don't overshoot
        move_dx = dx / distance * step_size
        move_dy = dy / distance * step_size
        
        if self.verbose:
            print(f"Step size: {step_size}, movement vector: ({move_dx}, {move_dy})")
            
        # Update position
        new_x = current_x + move_dx
        new_y = current_y + move_dy
        old_pos = self.position
        self.position = (new_x, new_y)
        
        if self.verbose:
            print(f"Robot {self.id} moved from {old_pos} to {self.position}")
            
        # For path visualization
        path_pos = (round(new_x), round(new_y))
        self.path.append(path_pos)
        
        # Calculate energy consumption
        step_distance = math.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        movement_energy = step_distance * self.energy_consumption_rate
        
        if self.verbose:
            print(f"Energy used: {movement_energy}, remaining: {self.energy}")
            
        # Check if we have enough energy
        if movement_energy > self.energy:
            if self.verbose:
                print(f"Robot {self.id} has insufficient energy for movement")
            # For coalition tasks, clean up
            if self.current_coalition is not None:
                coalition_task_id = self.current_coalition
                if coalition_task_id in self.current_tasks:
                    self.current_tasks.remove(coalition_task_id)
                self.current_coalition = None
                self.coalition_role = None
            return False
            
        # Consume energy
        self.energy -= movement_energy
        
        # Return False since we haven't reached the destination yet
        return False
        
    def _get_coalition_robots(self) -> List['BiddingRobot']:
        """Helper method to get all robots in the current coalition."""
        # This should be implemented by the task allocation environment
        # For now, return an empty list
        return []
    
    def compute_utility(self, assigned_tasks: List, costs: Dict) -> float:
        """
        Compute the total utility for a set of assigned tasks considering costs.
        
        Args:
            assigned_tasks: List of tasks assigned to this robot
            costs: Dictionary of costs (e.g., payments made for tasks)
            
        Returns:
            float: Total utility
        """
        total_utility = 0.0
        
        for task in assigned_tasks:
            # Calculate raw utility
            task_utility = self.evaluate_task(task)
            
            # Subtract costs
            task_cost = costs.get(task.id, 0)
            net_utility = task_utility - task_cost
            
            total_utility += net_utility
        
        return total_utility
        
    def update_position(self, new_position: Tuple[int, int]) -> None:
        """
        Update the robot's position on the grid.
        
        Args:
            new_position: New (x, y) position
        """
        old_position = self.position
        self.position = new_position
        self.path.append(new_position)
        
        # Calculate and consume energy for movement
        distance = math.sqrt(
            (new_position[0] - old_position[0])**2 +
            (new_position[1] - old_position[1])**2
        )
        energy_used = distance * self.energy_consumption_rate
        self.energy -= energy_used
        
    def __str__(self) -> str:
        """String representation of the robot."""
        return f"Robot {self.id} ({self.strategy_type}) at {self.position} with {len(self.current_tasks)} tasks"
    
    def __repr__(self) -> str:
        """Detailed string representation of the robot."""
        return (f"BiddingRobot(id={self.id}, strategy={self.strategy_type}, "
                f"position={self.position}, tasks={len(self.current_tasks)}, "
                f"energy={self.energy:.1f})")


# Example usage
if __name__ == "__main__":
    # Define a simple task class for testing
    class Task:
        def __init__(self, id, position, priority=1.0, required_capabilities=None, completion_time=1.0):
            self.id = id
            self.position = position
            self.priority = priority
            self.required_capabilities = required_capabilities or {}
            self.completion_time = completion_time
            self.energy_cost = 1.0
            self.type = "simple"
            self.team_utility_factor = 1.0
            
    # Create a robot
    robot = BiddingRobot(
        robot_id=1,
        capabilities={"speed": 1.5, "lift_capacity": 2.0, "precision": 0.8},
        initial_position=(0, 0),
        strategy_type="strategic"
    )
    
    # Create a task
    task = Task(
        id=1,
        position=(5, 5),
        priority=2.0,
        required_capabilities={"lift_capacity": 1.5},
        completion_time=2.0
    )
    
    # Evaluate and bid on the task
    utility = robot.evaluate_task(task)
    bid = robot.generate_bid(task)
    
    print(f"Robot evaluated task with utility: {utility}")
    print(f"Robot bid: {bid}")
    
    # Execute the task
    robot.current_tasks.add(task.id)
    success = robot.execute_task(task)
    
    print(f"Task execution {'succeeded' if success else 'failed'}")
    print(f"Robot is now at: {robot.position}")
    print(f"Robot energy: {robot.energy}")