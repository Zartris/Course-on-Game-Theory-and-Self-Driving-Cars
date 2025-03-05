#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vehicle class for the Roundabout Traffic Management simulation.

This module implements different driver behaviors and strategies for vehicles
navigating a roundabout, incorporating game theory concepts such as best response
strategies and repeated game learning.
"""

import numpy as np
import math
from collections import deque

class Vehicle:
    """
    A vehicle that can navigate a roundabout with different behavioral strategies.
    Implements game theory concepts for decision making and learning from interactions.
    """
    
    def __init__(self, id, entry_point, exit_point, vehicle_type='conservative', env=None):
        """
        Initialize a vehicle with specified parameters and behavior type.
        
        Args:
            id (int): Unique identifier for the vehicle
            entry_point (int): Index of roundabout entry point (0 to n-1)
            exit_point (int): Index of intended exit point
            vehicle_type (str): One of 'aggressive', 'conservative', 'cooperative', 'autonomous'
            env (RoundaboutEnv): Reference to the environment
        """
        self.id = id
        self.entry_point = entry_point
        self.exit_point = exit_point
        self.vehicle_type = vehicle_type
        self.env = env
        
        # Physical properties
        self.size = 10.0  # Vehicle radius in pixels
        self.max_speed = 5.0
        self.min_speed = 1.0
        self.max_acceleration = 0.5
        self.max_deceleration = -0.8
        
        # Current state
        self.position = self._calculate_entry_position()
        self.velocity = 0.0
        self.angle = self._calculate_entry_angle()
        self.in_roundabout = False
        self.completed = False
        self.collided = False
        self.wait_time = 0
        self.travel_time = 0
        self.last_action = None
        
        # Strategy parameters (adjusted based on vehicle type)
        self._init_strategy_parameters()
        
        # History tracking for repeated games
        self.interaction_history = deque(maxlen=100)
        self.strategy_history = deque(maxlen=100)
        
        # Nash equilibrium computation history
        self.nash_history = []
        
    def _init_strategy_parameters(self):
        """Initialize strategy parameters based on vehicle type."""
        # Base parameters
        self.time_value = 1.0  # Value of time (higher means more impatient)
        self.safety_value = 1.0  # Value of safety (higher means more cautious)
        self.cooperation_factor = 0.5  # Willingness to cooperate
        self.learning_rate = 0.1  # Rate of strategy adaptation
        
        # Adjust parameters based on vehicle type
        if self.vehicle_type == 'aggressive':
            self.time_value *= 2.0
            self.safety_value *= 0.5
            self.cooperation_factor *= 0.5
            self.min_gap_accept = 2.0  # Minimum gap to accept for entry
            
        elif self.vehicle_type == 'conservative':
            self.time_value *= 0.5
            self.safety_value *= 2.0
            self.cooperation_factor *= 0.8
            self.min_gap_accept = 4.0
            
        elif self.vehicle_type == 'cooperative':
            self.time_value *= 0.8
            self.safety_value *= 1.2
            self.cooperation_factor *= 2.0
            self.min_gap_accept = 3.0
            
        elif self.vehicle_type == 'autonomous':
            self.time_value *= 1.0
            self.safety_value *= 1.5
            self.cooperation_factor *= 1.5
            self.min_gap_accept = 2.5
            self.learning_rate *= 2.0  # Faster learning for autonomous vehicles
    
    def _calculate_entry_position(self):
        """Calculate the initial position at the entry point."""
        if not self.env:
            return (0, 0)
        
        angle = 2 * math.pi * self.entry_point / self.env.n_entry_points
        entry_radius = self.env.roundabout_radius + self.env.lane_width / 2
        
        x = self.env.window_size / 2 + entry_radius * math.cos(angle)
        y = self.env.window_size / 2 + entry_radius * math.sin(angle)
        
        return (x, y)
    
    def _calculate_entry_angle(self):
        """Calculate the initial angle at the entry point."""
        if not self.env:
            return 0
        
        # Calculate angle based on entry point position
        return 2 * math.pi * self.entry_point / self.env.n_entry_points
    
    def update_state(self, action):
        """
        Update the vehicle's state based on the chosen action.
        
        Args:
            action (int): 0 = Enter/Accelerate, 1 = Yield/Decelerate, 2 = Change Lane
            
        Returns:
            float: Reward for the action taken
        """
        if self.completed or self.collided:
            return 0.0
        
        self.last_action = action
        reward = 0.0
        
        # Update times
        if not self.in_roundabout:
            self.wait_time += 1
        self.travel_time += 1
        
        # Process action
        if action == 0:  # Enter/Accelerate
            if not self.in_roundabout:
                # Check if entry is safe
                if self._is_entry_safe():
                    self.in_roundabout = True
                    self.velocity = self.min_speed
                    reward += 2.0  # Reward for successful entry
                else:
                    reward -= 1.0  # Penalty for unsafe entry attempt
            else:
                # Accelerate if in roundabout
                self.velocity = min(self.velocity + self.max_acceleration, self.max_speed)
                reward += 0.1  # Small reward for making progress
                
        elif action == 1:  # Yield/Decelerate
            if self.in_roundabout:
                self.velocity = max(self.velocity + self.max_deceleration, self.min_speed)
            reward += 0.5 * self.calculate_safety_margin()  # Reward for safe behavior
            
        elif action == 2:  # Change Lane (if applicable)
            # Lane changing logic could be implemented here
            pass
        
        # Update position if in roundabout
        if self.in_roundabout:
            self._update_position()
            
            # Check if exit reached
            if self._is_at_exit():
                self.completed = True
                reward += 10.0  # Reward for completing journey
        
        # Add time penalty
        reward -= 0.01 * self.time_value  # Small time penalty
        
        # Safety bonus/penalty
        safety_margin = self.calculate_safety_margin()
        reward += 0.1 * self.safety_value * safety_margin
        
        return reward
    
    def _update_position(self):
        """Update position based on current velocity and angle."""
        if not self.env:
            return
        
        # Calculate movement along the circular path
        angular_velocity = self.velocity / self.env.roundabout_radius
        self.angle += angular_velocity
        
        # Keep angle in [0, 2π]
        self.angle = self.angle % (2 * math.pi)
        
        # Update position
        center_x = self.env.window_size / 2
        center_y = self.env.window_size / 2
        
        self.position = (
            center_x + self.env.roundabout_radius * math.cos(self.angle),
            center_y + self.env.roundabout_radius * math.sin(self.angle)
        )
    
    def _is_entry_safe(self):
        """Check if it's safe to enter the roundabout."""
        if not self.env:
            return False
        
        # Check distance to nearest vehicle in roundabout
        min_gap = float('inf')
        for other in self.env.vehicles:
            if other != self and other.in_roundabout:
                # Calculate angular distance
                angle_diff = abs(self.angle - other.angle)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                
                # Convert to arc length
                gap = angle_diff * self.env.roundabout_radius
                min_gap = min(min_gap, gap)
        
        return min_gap >= self.min_gap_accept * self.size
    
    def _is_at_exit(self):
        """Check if the vehicle has reached its exit."""
        if not self.env:
            return False
        
        exit_angle = 2 * math.pi * self.exit_point / self.env.n_entry_points
        angle_diff = abs(self.angle - exit_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        
        return angle_diff < 0.1  # Allow some tolerance
    
    def calculate_safety_margin(self):
        """Calculate the current safety margin to other vehicles."""
        if not self.env:
            return float('inf')
        
        min_distance = float('inf')
        for other in self.env.vehicles:
            if other != self and other.in_roundabout:
                distance = np.linalg.norm(
                    np.array(self.position) - np.array(other.position)
                )
                min_distance = min(min_distance, distance)
        
        # Normalize by vehicle size
        safety_margin = max(0, (min_distance - 2 * self.size) / self.size)
        return safety_margin
    
    def compute_best_response(self, other_vehicles):
        """
        Compute the best response strategy given other vehicles' states.
        This implements a simplified version of best response dynamics.
        
        Args:
            other_vehicles (list): List of other vehicles to consider
            
        Returns:
            int: Best action to take (0, 1, or 2)
        """
        best_action = 1  # Default to yielding
        best_utility = float('-inf')
        
        # For each possible action
        for action in [0, 1, 2]:
            utility = self._evaluate_action_utility(action, other_vehicles)
            
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action
    
    def _evaluate_action_utility(self, action, other_vehicles):
        """Evaluate the utility of an action considering other vehicles."""
        utility = 0.0
        
        # Time component
        if action == 0:  # Enter/Accelerate
            utility += self.time_value * 2.0
        elif action == 1:  # Yield
            utility -= self.time_value
            
        # Safety component
        min_distance = self._predict_min_distance(action, other_vehicles)
        safety_utility = self.safety_value * (min_distance / (2 * self.size) - 1)
        utility += safety_utility
        
        # Cooperation component (for repeated games)
        if self.env and self.env.repeated_game:
            cooperation_utility = self._evaluate_cooperation(action, other_vehicles)
            utility += self.cooperation_factor * cooperation_utility
        
        return utility
    
    def _predict_min_distance(self, action, other_vehicles):
        """Predict minimum distance to other vehicles if action is taken."""
        min_distance = float('inf')
        
        # Simple prediction one step ahead
        predicted_pos = self.position
        if self.in_roundabout:
            if action == 0:  # Accelerate
                angular_velocity = (self.velocity + self.max_acceleration) / self.env.roundabout_radius
            else:  # Decelerate or change lane
                angular_velocity = (self.velocity + self.max_deceleration) / self.env.roundabout_radius
            
            next_angle = (self.angle + angular_velocity) % (2 * math.pi)
            predicted_pos = (
                self.env.window_size / 2 + self.env.roundabout_radius * math.cos(next_angle),
                self.env.window_size / 2 + self.env.roundabout_radius * math.sin(next_angle)
            )
        
        # Check distance to each other vehicle
        for other in other_vehicles:
            distance = np.linalg.norm(
                np.array(predicted_pos) - np.array(other.position)
            )
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _evaluate_cooperation(self, action, other_vehicles):
        """Evaluate the cooperative aspect of an action based on interaction history."""
        if not self.env or not self.env.repeated_game:
            return 0.0
        
        cooperation_score = 0.0
        
        for other in other_vehicles:
            # Get history with this vehicle
            pair_key = tuple(sorted([self.id, other.id]))
            if pair_key in self.env.interaction_history:
                history = self.env.interaction_history[pair_key]
                
                # Analyze recent interactions
                recent_cooperation = 0
                total_interactions = 0
                
                for interaction in reversed(list(history)):
                    if total_interactions >= 5:  # Look at last 5 interactions
                        break
                    
                    # Check if interaction was cooperative
                    if interaction['type'] != 'collision':
                        if interaction['vehicle1']['id'] == self.id:
                            if interaction['vehicle1']['action'] == 1:  # Yielding is cooperative
                                recent_cooperation += 1
                        else:
                            if interaction['vehicle2']['action'] == 1:
                                recent_cooperation += 1
                    
                    total_interactions += 1
                
                if total_interactions > 0:
                    cooperation_score += recent_cooperation / total_interactions
        
        return cooperation_score if other_vehicles else 0.0
    
    def update_strategy(self, interaction_history):
        """
        Update strategy based on past interactions (for repeated games).
        Implements a form of learning from past experiences.
        
        Args:
            interaction_history (list): List of past interactions
        """
        if not self.env or not self.env.repeated_game:
            return
        
        # Analyze recent history to adjust strategy parameters
        recent_collisions = 0
        recent_successful_interactions = 0
        total_interactions = 0
        
        for interaction in reversed(list(interaction_history)):
            if total_interactions >= 10:  # Look at last 10 interactions
                break
            
            if interaction['type'] == 'collision':
                recent_collisions += 1
            else:
                recent_successful_interactions += 1
            
            total_interactions += 1
        
        if total_interactions > 0:
            collision_rate = recent_collisions / total_interactions
            success_rate = recent_successful_interactions / total_interactions
            
            # Adjust strategy parameters based on outcomes
            if collision_rate > 0.2:  # Too many collisions
                self.safety_value *= (1 + self.learning_rate)
                self.time_value *= (1 - self.learning_rate)
            elif success_rate > 0.8:  # Very successful
                self.time_value *= (1 + self.learning_rate * 0.5)  # Modest increase in aggression
            
            # Update cooperation factor based on others' behavior
            if success_rate > 0.7:
                self.cooperation_factor *= (1 + self.learning_rate * 0.2)
            else:
                self.cooperation_factor *= (1 - self.learning_rate * 0.1)
            
            # Keep parameters within reasonable bounds
            self.safety_value = np.clip(self.safety_value, 0.5, 3.0)
            self.time_value = np.clip(self.time_value, 0.5, 3.0)
            self.cooperation_factor = np.clip(self.cooperation_factor, 0.1, 2.0)
    
    def compute_nash_equilibria(self, game_state):
        """
        Compute Nash equilibria for the current game state using Nashpy.
        This is used for autonomous vehicles to make optimal decisions.
        
        Args:
            game_state (dict): Current state of the game including payoff matrices
            
        Returns:
            list: List of Nash equilibria
        """
        if self.vehicle_type != 'autonomous':
            return None
        
        try:
            import nashpy as nash
            
            # Extract payoff matrices from game state
            payoff_matrix_self = game_state.get('payoff_matrix_self', None)
            payoff_matrix_other = game_state.get('payoff_matrix_other', None)
            
            if payoff_matrix_self is None or payoff_matrix_other is None:
                return None
            
            # Create the game
            game = nash.Game(payoff_matrix_self, payoff_matrix_other)
            
            # Compute Nash equilibria
            equilibria = list(game.support_enumeration())
            
            # Store for later analysis
            self.nash_history.append({
                'time': self.env.step_count if self.env else 0,
                'equilibria': equilibria
            })
            
            return equilibria
            
        except ImportError:
            print("Nashpy not available. Nash equilibrium computation skipped.")
            return None
    
    def decide_action(self):
        """
        Decide the next action based on vehicle type and current state.
        
        Returns:
            int: Chosen action (0 = Enter/Accelerate, 1 = Yield/Decelerate, 2 = Change Lane)
        """
        if self.completed or self.collided:
            return 1  # Yield/Stop if completed or collided
        
        # Get nearby vehicles
        nearby_vehicles = []
        if self.env:
            for other in self.env.vehicles:
                if other != self and not other.completed and not other.collided:
                    distance = np.linalg.norm(
                        np.array(self.position) - np.array(other.position)
                    )
                    if distance < self.env.roundabout_radius * 2:
                        nearby_vehicles.append(other)
        
        # Decision logic based on vehicle type
        if self.vehicle_type == 'aggressive':
            # Aggressive vehicles prioritize speed over safety
            if not self.in_roundabout:
                return 0 if self._is_entry_safe() else 1
            return 0  # Usually accelerate when in roundabout
            
        elif self.vehicle_type == 'conservative':
            # Conservative vehicles prioritize safety
            if not self.in_roundabout:
                gap_multiple = 1.5  # Require larger gaps
                return 0 if self._is_entry_safe() and self.calculate_safety_margin() > gap_multiple else 1
            return 1 if self.calculate_safety_margin() < 2.0 else 0
            
        elif self.vehicle_type == 'cooperative':
            # Cooperative vehicles consider others' states
            return self.compute_best_response(nearby_vehicles)
            
        elif self.vehicle_type == 'autonomous':
            # Autonomous vehicles use more sophisticated decision making
            if self.env and self.env.repeated_game:
                # Use learning from repeated games
                game_state = self._create_game_state(nearby_vehicles)
                equilibria = self.compute_nash_equilibria(game_state)
                
                if equilibria and len(equilibria) > 0:
                    # Choose the equilibrium that maximizes social welfare
                    best_equilibrium = max(equilibria, key=lambda e: sum(e[0]) + sum(e[1]))
                    # Convert probability distribution to action
                    action_probs = best_equilibrium[0]
                    return np.argmax(action_probs)
            
            # Fallback to best response if Nash computation fails
            return self.compute_best_response(nearby_vehicles)
        
        # Default behavior
        return 1  # Yield by default
    
    def _create_game_state(self, nearby_vehicles):
        """Create a game state representation for Nash equilibrium computation."""
        if not nearby_vehicles:
            return {'payoff_matrix_self': None, 'payoff_matrix_other': None}
        
        # Simple 2x2 game matrix for demonstration
        # Actions: [Enter/Accelerate, Yield/Decelerate]
        payoff_matrix_self = np.zeros((2, 2))
        payoff_matrix_other = np.zeros((2, 2))
        
        # Fill payoff matrices based on current state
        for i in range(2):
            for j in range(2):
                # Evaluate outcomes of action combinations
                utility_self = self._evaluate_action_utility(i, nearby_vehicles)
                utility_other = sum(other._evaluate_action_utility(j, [self]) 
                                 for other in nearby_vehicles) / len(nearby_vehicles)
                
                payoff_matrix_self[i][j] = utility_self
                payoff_matrix_other[i][j] = utility_other
        
        return {
            'payoff_matrix_self': payoff_matrix_self,
            'payoff_matrix_other': payoff_matrix_other
        }