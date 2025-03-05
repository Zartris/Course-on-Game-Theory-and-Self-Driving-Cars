#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BayesianVehicle class for autonomous vehicles with belief-based decision making.

This module implements vehicles that maintain beliefs about other vehicles' types
and use Bayesian reasoning for decision making under uncertainty.
"""

import numpy as np
import math
import random
from collections import defaultdict

class BayesianVehicle:
    """
    An autonomous vehicle with Bayesian reasoning capabilities for navigating
    under uncertainty and coordinating with other vehicles.
    """
    
    def __init__(self, vehicle_id, initial_position, goal, vehicle_type='standard'):
        """
        Initialize a Bayesian vehicle.
        
        Args:
            vehicle_id (int): Unique identifier for the vehicle
            initial_position (tuple): Initial (x, y) position
            goal (tuple): Target (x, y) position
            vehicle_type (str): Type of vehicle ('standard', 'premium', or 'emergency')
        """
        self.id = vehicle_id
        self.position = initial_position
        self.last_position = None
        self.goal = goal
        self.vehicle_type = vehicle_type
        
        # Motion state
        self.velocity = (0.0, 0.0)
        self.max_speed = self._get_max_speed_by_type()
        self.last_action = (0.0, 0.0)  # (acceleration, steering)
        self.reward = 0.0
        
        # Sensor and communication capabilities
        self.sensor_range = self._get_sensor_range_by_type()
        self.comm_reliability = self._get_comm_reliability_by_type()
        
        # Emergency handling
        self.emergency_active = False
        self.emergency_start_time = None
        
        # Beliefs about other vehicles' types
        self.beliefs = {}  # {vehicle_id: {type: probability}}
        self.observation_history = defaultdict(list)  # {vehicle_id: [observations]}
        
        # Bayesian game state
        self.bayesian_equilibria = None
        
        # Type-dependent utilities
        self.utility_weights = self._initialize_utility_weights()
    
    def _get_max_speed_by_type(self):
        """Get maximum speed based on vehicle type."""
        max_speeds = {
            'standard': 1.0,
            'premium': 1.2,
            'emergency': 1.5
        }
        return max_speeds.get(self.vehicle_type, 1.0)
    
    def _get_sensor_range_by_type(self):
        """Get sensor range based on vehicle type."""
        sensor_ranges = {
            'standard': 5.0,
            'premium': 7.0,
            'emergency': 8.0
        }
        return sensor_ranges.get(self.vehicle_type, 5.0)
    
    def _get_comm_reliability_by_type(self):
        """Get communication reliability based on vehicle type."""
        reliability = {
            'standard': 0.9,
            'premium': 0.95,
            'emergency': 0.99
        }
        return reliability.get(self.vehicle_type, 0.9)
    
    def _initialize_utility_weights(self):
        """Initialize utility weights based on vehicle type."""
        if self.vehicle_type == 'standard':
            return {
                'goal_progress': 0.5,
                'safety': 0.3,
                'efficiency': 0.2,
                'emergency_priority': 0.0
            }
        elif self.vehicle_type == 'premium':
            return {
                'goal_progress': 0.4,
                'safety': 0.3,
                'efficiency': 0.3,
                'emergency_priority': 0.0
            }
        elif self.vehicle_type == 'emergency':
            return {
                'goal_progress': 0.6,
                'safety': 0.2,
                'efficiency': 0.1,
                'emergency_priority': 0.1
            }
        else:
            return {
                'goal_progress': 0.5,
                'safety': 0.3,
                'efficiency': 0.2,
                'emergency_priority': 0.0
            }
    
    def update_state(self, action):
        """
        Update vehicle state based on action.
        
        Args:
            action (tuple/array): (acceleration, steering) action
        """
        # Store last position
        self.last_position = self.position
        
        # Extract action components (acceleration, steering)
        acceleration, steering = action[0], action[1]
        
        # Store last action
        self.last_action = (acceleration, steering)
        
        # Update velocity based on acceleration and steering
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed < 0.01:  # If almost stationary, use default direction
            direction = self._get_direction_to_goal()
            new_velocity = (
                acceleration * direction[0],
                acceleration * direction[1]
            )
        else:
            # Normalize current velocity to get direction
            direction = (self.velocity[0] / speed, self.velocity[1] / speed)
            
            # Apply steering to change direction
            c, s = math.cos(steering), math.sin(steering)
            new_direction = (
                c * direction[0] - s * direction[1],
                s * direction[0] + c * direction[1]
            )
            
            # Combine acceleration and direction
            new_speed = min(speed + acceleration, self.max_speed)
            new_speed = max(new_speed, 0)  # Prevent negative speed
            new_velocity = (
                new_speed * new_direction[0],
                new_speed * new_direction[1]
            )
        
        self.velocity = new_velocity
        
        # Update position
        new_position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )
        
        # Ensure the vehicle stays on the road (simplified for now)
        # In a real implementation, this would check against the road network
        self.position = new_position
    
    def _get_direction_to_goal(self):
        """Calculate normalized direction vector pointing to goal."""
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 0.001:
            return (0, 0)  # Already at goal
            
        return (dx / distance, dy / distance)
    
    def update_belief(self, other_vehicle_id, observation):
        """
        Update belief about another vehicle's type based on observation.
        
        Args:
            other_vehicle_id (int): ID of the observed vehicle
            observation (dict): Observation of the vehicle
        """
        # Skip if no prior belief exists
        if other_vehicle_id not in self.beliefs:
            return
            
        # Store observation in history
        self.observation_history[other_vehicle_id].append(observation)
        
        # Extract relevant features from observation
        position = observation.get('position')
        velocity = observation.get('velocity')
        acceleration = observation.get('acceleration', 0)
        steering = observation.get('steering', 0)
        
        # Calculate likelihood of observation given each type
        likelihoods = {}
        
        # Extract speed from velocity
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0
        
        # Likelihood models for different vehicle types
        for vehicle_type in self.beliefs[other_vehicle_id].keys():
            if vehicle_type == 'standard':
                # Standard vehicles typically have moderate speed and acceleration
                speed_likelihood = self._gaussian(speed, 1.0, 0.3)
                accel_likelihood = self._gaussian(abs(acceleration), 0.5, 0.3)
            elif vehicle_type == 'premium':
                # Premium vehicles might have higher speeds and smoother acceleration
                speed_likelihood = self._gaussian(speed, 1.2, 0.3)
                accel_likelihood = self._gaussian(abs(acceleration), 0.7, 0.2)
            elif vehicle_type == 'emergency':
                # Emergency vehicles typically have higher speeds and accelerations
                speed_likelihood = self._gaussian(speed, 1.5, 0.4)
                accel_likelihood = self._gaussian(abs(acceleration), 0.9, 0.3)
            else:
                # Default model
                speed_likelihood = 1.0
                accel_likelihood = 1.0
            
            # Combine likelihoods (assuming independence)
            likelihoods[vehicle_type] = speed_likelihood * accel_likelihood
        
        # Apply Bayes' rule to update beliefs
        prior = self.beliefs[other_vehicle_id]
        posterior = {}
        
        # Calculate normalization factor
        normalization = sum(likelihoods[t] * prior[t] for t in prior)
        
        if normalization > 0:
            # Update beliefs using Bayes' rule
            for vehicle_type in prior:
                posterior[vehicle_type] = (likelihoods[vehicle_type] * prior[vehicle_type]) / normalization
            
            # Update beliefs
            self.beliefs[other_vehicle_id] = posterior
    
    def _gaussian(self, x, mean, std_dev):
        """
        Calculate Gaussian probability density.
        
        Args:
            x (float): Value
            mean (float): Mean of distribution
            std_dev (float): Standard deviation
        
        Returns:
            float: Probability density
        """
        return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))
    
    def decide_action(self, state, belief_state):
        """
        Decide action based on state and belief state.
        
        Args:
            state (dict): Current state information
            belief_state (dict): Current belief state about other vehicles
            
        Returns:
            tuple: (acceleration, steering) action
        """
        # Use Bayesian equilibria if available
        if self.bayesian_equilibria is not None and len(self.bayesian_equilibria) > 0:
            # Choose the equilibrium with highest expected utility for this vehicle
            best_equilibrium = None
            best_utility = float('-inf')
            
            for eq in self.bayesian_equilibria:
                # Extract this vehicle's strategy in this equilibrium
                if self.id in eq['strategies']:
                    utility = self._evaluate_equilibrium_utility(eq)
                    if utility > best_utility:
                        best_utility = utility
                        best_equilibrium = eq
            
            if best_equilibrium:
                # Convert the strategy to an action
                return self._strategy_to_action(best_equilibrium['strategies'].get(self.id))
        
        # Fallback to heuristic decision making
        return self._heuristic_decision(state)
    
    def _evaluate_equilibrium_utility(self, equilibrium):
        """
        Evaluate expected utility of an equilibrium.
        
        Args:
            equilibrium (dict): Equilibrium strategies and expected utilities
            
        Returns:
            float: Expected utility for this vehicle
        """
        # If the equilibrium has precomputed utilities, use those
        if 'utilities' in equilibrium and self.id in equilibrium['utilities']:
            return equilibrium['utilities'][self.id]
        
        # Otherwise, compute utility based on the strategies
        return self.compute_expected_utility(equilibrium['strategies'].get(self.id, None), self.beliefs)
    
    def _strategy_to_action(self, strategy):
        """
        Convert a strategy to a concrete action.
        
        Args:
            strategy: Strategy representation (could be discrete action index or distribution)
            
        Returns:
            tuple: (acceleration, steering) action
        """
        if strategy is None:
            # Default action if no strategy provided
            return self._heuristic_decision(None)
        
        # If strategy is an action distribution, sample from it
        if isinstance(strategy, dict) or isinstance(strategy, list):
            # Sample action from distribution
            if isinstance(strategy, dict):
                actions = list(strategy.keys())
                probs = list(strategy.values())
            else:
                actions = list(range(len(strategy)))
                probs = strategy
                
            action_idx = np.random.choice(actions, p=probs)
            
            # Map action index to continuous action
            # This mapping depends on the action space definition
            action_map = {
                0: (0.0, 0.0),      # Maintain
                1: (0.2, 0.0),      # Accelerate
                2: (-0.2, 0.0),     # Decelerate
                3: (0.0, -0.2),     # Turn left
                4: (0.0, 0.2)       # Turn right
            }
            
            return action_map.get(action_idx, (0.0, 0.0))
        
        # If strategy is already an action, return it
        if isinstance(strategy, tuple) and len(strategy) == 2:
            return strategy
        
        # Fallback
        return self._heuristic_decision(None)
    
    def _heuristic_decision(self, state):
        """
        Make a heuristic decision when Bayesian equilibrium is not available.
        
        Args:
            state (dict): Current state information or None
            
        Returns:
            tuple: (acceleration, steering) action
        """
        # Simple goal-seeking behavior
        direction = self._get_direction_to_goal()
        
        # Calculate desired velocity
        desired_speed = self.max_speed
        desired_velocity = (desired_speed * direction[0], desired_speed * direction[1])
        
        # Calculate acceleration to reach desired velocity
        accel_x = desired_velocity[0] - self.velocity[0]
        accel_y = desired_velocity[1] - self.velocity[1]
        
        # Normalize and scale acceleration
        accel_magnitude = math.sqrt(accel_x**2 + accel_y**2)
        if accel_magnitude > 0.5:
            accel_x = (accel_x / accel_magnitude) * 0.5
            accel_y = (accel_y / accel_magnitude) * 0.5
        
        # Convert to scalar acceleration and steering
        current_speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        
        if current_speed < 0.01:
            # If almost stationary, just accelerate in goal direction
            acceleration = min(0.5, desired_speed)
            steering = 0.0
        else:
            # Project acceleration onto velocity direction to get scalar acceleration
            vel_direction = (self.velocity[0] / current_speed, self.velocity[1] / current_speed)
            acceleration = accel_x * vel_direction[0] + accel_y * vel_direction[1]
            
            # Calculate steering based on cross product
            # This is a simplified steering model
            steering = accel_x * vel_direction[1] - accel_y * vel_direction[0]
            steering = max(-0.5, min(0.5, steering))  # Limit steering angle
        
        return (acceleration, steering)
    
    def communicate(self, other_vehicles, communication_type='broadcast'):
        """
        Communicate with other vehicles to share information.
        
        Args:
            other_vehicles (list): List of vehicles to communicate with
            communication_type (str): Type of communication protocol
        """
        if communication_type == 'broadcast':
            # Broadcast own position, velocity, and type if emergency
            if self.vehicle_type == 'emergency' and self.emergency_active:
                for other in other_vehicles:
                    # Check if communication succeeds based on reliability
                    if random.random() < self.comm_reliability:
                        other.receive_communication(self.id, {
                            'position': self.position,
                            'velocity': self.velocity,
                            'type': self.vehicle_type,
                            'emergency': True,
                            'goal': self.goal
                        })
            else:
                # Regular broadcast
                for other in other_vehicles:
                    if random.random() < self.comm_reliability:
                        other.receive_communication(self.id, {
                            'position': self.position,
                            'velocity': self.velocity
                        })
        elif communication_type == 'targeted':
            # More selective information sharing based on relevance
            for other in other_vehicles:
                # Determine what information is relevant to share
                if self._is_interaction_likely(other):
                    if random.random() < self.comm_reliability:
                        other.receive_communication(self.id, {
                            'position': self.position,
                            'velocity': self.velocity,
                            'intention': self._get_intention()
                        })
    
    def _is_interaction_likely(self, other_vehicle):
        """
        Determine if interaction with another vehicle is likely.
        
        Args:
            other_vehicle: The other vehicle
            
        Returns:
            bool: True if interaction is likely
        """
        # Calculate distance
        distance = math.sqrt(
            (self.position[0] - other_vehicle.position[0])**2 +
            (self.position[1] - other_vehicle.position[1])**2
        )
        
        # Check if vehicles are moving toward each other
        if distance < 5.0:
            return True
            
        # Calculate relative velocity
        rel_vel_x = self.velocity[0] - other_vehicle.velocity[0]
        rel_vel_y = self.velocity[1] - other_vehicle.velocity[1]
        
        # Calculate time to closest approach
        pos_diff_x = other_vehicle.position[0] - self.position[0]
        pos_diff_y = other_vehicle.position[1] - self.position[1]
        
        # Dot product of position difference and relative velocity
        dot_product = pos_diff_x * rel_vel_x + pos_diff_y * rel_vel_y
        
        # Squared magnitude of relative velocity
        rel_vel_sq = rel_vel_x**2 + rel_vel_y**2
        
        if rel_vel_sq < 0.0001:
            return False  # Vehicles not moving relative to each other
            
        # Time to closest approach
        t_closest = -dot_product / rel_vel_sq
        
        if t_closest < 0 or t_closest > 10.0:
            return False  # Closest approach in past or too far in future
            
        # Position at closest approach
        closest_x = self.position[0] + self.velocity[0] * t_closest
        closest_y = self.position[1] + self.velocity[1] * t_closest
        other_closest_x = other_vehicle.position[0] + other_vehicle.velocity[0] * t_closest
        other_closest_y = other_vehicle.position[1] + other_vehicle.velocity[1] * t_closest
        
        # Distance at closest approach
        closest_distance = math.sqrt(
            (closest_x - other_closest_x)**2 +
            (closest_y - other_closest_y)**2
        )
        
        return closest_distance < 2.0
    
    def _get_intention(self):
        """
        Get current intention (e.g., turn at next intersection).
        
        Returns:
            dict: Intention information
        """
        # Calculate direction to goal
        goal_direction = self._get_direction_to_goal()
        
        # Simplified intention
        return {
            'heading': goal_direction,
            'target': self.goal
        }
    
    def receive_communication(self, sender_id, data):
        """
        Process received communication from another vehicle.
        
        Args:
            sender_id (int): ID of sending vehicle
            data (dict): Received data
        """
        # Update beliefs if type information is received
        if 'type' in data:
            # Direct observation of type (e.g., from emergency vehicle broadcast)
            self.beliefs[sender_id] = {t: 1.0 if t == data['type'] else 0.0 
                                      for t in self.beliefs.get(sender_id, {})}
        
        # If it's an emergency vehicle, adjust behavior
        if data.get('emergency', False):
            self._respond_to_emergency(sender_id, data)
    
    def receive_emergency_notification(self, emergency_vehicle_id, emergency_destination):
        """
        Process emergency notification and adjust behavior.
        
        Args:
            emergency_vehicle_id (int): ID of emergency vehicle
            emergency_destination (tuple): Destination of emergency vehicle
        """
        # Update belief to be certain this is an emergency vehicle
        if emergency_vehicle_id in self.beliefs:
            self.beliefs[emergency_vehicle_id] = {
                'standard': 0.0,
                'premium': 0.0,
                'emergency': 1.0
            }
        
        # Adjust utility weights to prioritize giving way to emergency vehicles
        if self.vehicle_type != 'emergency':
            self.utility_weights['emergency_priority'] = 0.3
            # Reduce other weights proportionally
            total = sum(self.utility_weights.values())
            scale = (1.0 - self.utility_weights['emergency_priority']) / (total - self.utility_weights['emergency_priority'])
            
            for key in self.utility_weights:
                if key != 'emergency_priority':
                    self.utility_weights[key] *= scale
    
    def _respond_to_emergency(self, emergency_vehicle_id, data):
        """
        Adjust behavior in response to nearby emergency vehicle.
        
        Args:
            emergency_vehicle_id (int): ID of emergency vehicle
            data (dict): Data about the emergency vehicle
        """
        # Similar to receive_emergency_notification but with more detail if available
        self.receive_emergency_notification(emergency_vehicle_id, data.get('goal', None))
    
    def compute_expected_utility(self, action, belief_state):
        """
        Compute expected utility of an action given belief state.
        
        Args:
            action: Action to evaluate
            belief_state (dict): Beliefs about other vehicles' types
            
        Returns:
            float: Expected utility
        """
        if action is None:
            return 0.0
            
        # Convert action to continuous form if needed
        if not isinstance(action, tuple):
            action = self._strategy_to_action(action)
            
        # Simulate action to evaluate outcome
        old_position = self.position
        old_velocity = self.velocity
        
        # Temporarily update state to evaluate action
        self.update_state(action)
        
        # Calculate utility components
        goal_progress = self._calculate_goal_progress()
        safety = self._calculate_safety()
        efficiency = self._calculate_efficiency()
        emergency_priority = self._calculate_emergency_priority()
        
        # Restore original state
        self.position = old_position
        self.velocity = old_velocity
        
        # Combine utilities with weights
        utility = (
            self.utility_weights['goal_progress'] * goal_progress +
            self.utility_weights['safety'] * safety +
            self.utility_weights['efficiency'] * efficiency +
            self.utility_weights['emergency_priority'] * emergency_priority
        )
        
        return utility
    
    def _calculate_goal_progress(self):
        """Calculate utility component for progress towards goal."""
        # Calculate distance to goal
        distance = math.sqrt(
            (self.position[0] - self.goal[0])**2 +
            (self.position[1] - self.goal[1])**2
        )
        
        # Normalize by initial distance (approximate)
        if hasattr(self, 'initial_distance_to_goal'):
            normalized = 1.0 - distance / self.initial_distance_to_goal
        else:
            # Estimate initial distance if not set
            self.initial_distance_to_goal = 20.0  # Rough estimate
            normalized = 1.0 - distance / self.initial_distance_to_goal
        
        return max(0.0, min(1.0, normalized))
    
    def _calculate_safety(self):
        """Calculate utility component for safety."""
        # Placeholder for safety calculation based on distance to other vehicles
        # In a full implementation, this would consider proximity to other known vehicles
        return 1.0  # Simplified placeholder
    
    def _calculate_efficiency(self):
        """Calculate utility component for efficiency."""
        # Calculate speed as a fraction of max speed
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        return min(1.0, speed / self.max_speed)
    
    def _calculate_emergency_priority(self):
        """Calculate utility component for emergency priority."""
        if self.vehicle_type == 'emergency' and self.emergency_active:
            # Emergency vehicle on active duty gets high priority
            return 1.0
        else:
            # Non-emergency vehicles should yield to emergency vehicles
            return 0.0
    
    def compute_bayesian_nash(self, game_state):
        """
        Compute Bayesian Nash equilibrium for a given game state.
        
        This is a placeholder - actual computation should typically
        be done by the BayesianGameAnalyzer class.
        
        Args:
            game_state (dict): Description of the current game state
            
        Returns:
            list: Possible Bayesian Nash equilibria
        """
        # This method would typically delegate to the game analyzer
        if hasattr(self, 'env') and hasattr(self.env, 'game_analyzer'):
            return self.env.game_analyzer.find_bayesian_nash_equilibria(game_state)
        else:
            return []  # Return empty list if no analyzer is available