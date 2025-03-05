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
        self.final_goal = goal  # Store the final goal
        self.current_goal = self._get_intermediate_waypoint(initial_position, goal)  # Current waypoint goal
        self.vehicle_type = vehicle_type
        
        # Movement properties
        self.velocity = [0, 0]
        self.max_speed = self._get_max_speed()
        self.sensor_range = self._get_sensor_range()
        self.comm_reliability = self._get_comm_reliability()
        self.wall_collision_margin = 1.0  # Margin for wall collisions
        
        # State tracking
        self.beliefs = {}
        self.last_action = (0, 0)  # acceleration, steering
        self.path = []
        self.waiting_at_intersection = False
        self.intersection_priority = 0
        self.time_waiting = 0
        
        # Emergency properties
        self.emergency_active = False
        self.emergency_start_time = None
        self.give_way_to_emergency = False
    
    def _get_max_speed(self):
        """Get maximum speed based on vehicle type."""
        base_speed = 0.04  # 25x slower than original
        if self.vehicle_type == 'emergency':
            return base_speed * 1.5  # Faster for emergency vehicles
        elif self.vehicle_type == 'premium':
            return base_speed * 1.2  # Slightly faster for premium
        else:
            return base_speed  # Standard speed
    
    def _get_sensor_range(self):
        """Get sensor range based on vehicle type."""
        if self.vehicle_type == 'emergency':
            return 8.0  # Better sensors for emergency
        elif self.vehicle_type == 'premium':
            return 6.0  # Better sensors for premium
        else:
            return 4.0  # Standard range
    
    def _get_comm_reliability(self):
        """Get communication reliability based on vehicle type."""
        if self.vehicle_type == 'emergency':
            return 0.95  # Most reliable
        elif self.vehicle_type == 'premium':
            return 0.85  # More reliable
        else:
            return 0.75  # Standard reliability
    
    def _get_intermediate_waypoint(self, current_pos, goal):
        """Calculate intermediate waypoint to avoid center of intersection."""
        # Center of the grid (assuming 20x20 grid)
        center_x, center_y = 10, 10
        intersection_radius = 3  # Radius to go around intersection
        
        # Determine which quadrant the current position and goal are in
        curr_quad = self._get_quadrant(current_pos)
        goal_quad = self._get_quadrant(goal)
        
        # If crossing the intersection
        if curr_quad != goal_quad:
            # Calculate waypoint based on clockwise movement around intersection
            angle = self._get_waypoint_angle(curr_quad, goal_quad)
            waypoint_x = center_x + intersection_radius * np.cos(angle)
            waypoint_y = center_y + intersection_radius * np.sin(angle)
            return (waypoint_x, waypoint_y)
        else:
            # If in same quadrant, return final goal
            return goal
    
    def _get_quadrant(self, position):
        """Determine which quadrant a position is in (1-4, clockwise from top-right)."""
        x, y = position
        center_x, center_y = 10, 10
        
        if x >= center_x and y < center_y:
            return 1  # Top-right
        elif x >= center_x and y >= center_y:
            return 2  # Bottom-right
        elif x < center_x and y >= center_y:
            return 3  # Bottom-left
        else:
            return 4  # Top-left
    
    def _get_waypoint_angle(self, current_quad, goal_quad):
        """Calculate angle for waypoint based on current and goal quadrants."""
        # Base angles for each quadrant (in radians)
        quad_angles = {
            1: 0,           # Right
            2: np.pi/2,     # Bottom
            3: np.pi,       # Left
            4: -np.pi/2     # Top
        }
        
        # Get base angle for current quadrant
        angle = quad_angles[current_quad]
        
        # Adjust angle based on goal quadrant to ensure clockwise movement
        if goal_quad <= current_quad:
            goal_quad += 4
        steps = goal_quad - current_quad
        angle += (steps * np.pi/2) / 2  # Divide by 2 to get midpoint
        
        return angle
    
    def decide_action(self, state, belief_state):
        """
        Decide next action using Bayesian game theory.
        
        Args:
            state (dict): Current state information
            belief_state (dict): Current beliefs about other vehicles
        
        Returns:
            tuple: (acceleration, steering) action
        """
        # Check if we've reached the current waypoint
        curr_pos = np.array(self.position)
        curr_goal = np.array(self.current_goal)
        dist_to_waypoint = np.linalg.norm(curr_goal - curr_pos)
        
        if dist_to_waypoint < 0.5:  # If reached waypoint
            if self.current_goal != self.final_goal:
                # Update to next waypoint or final goal
                self.current_goal = self._get_intermediate_waypoint(self.position, self.final_goal)
                curr_goal = np.array(self.current_goal)
        
        # Calculate direction to current goal (waypoint or final)
        goal_vector = curr_goal - curr_pos
        distance_to_goal = np.linalg.norm(goal_vector)
        
        if distance_to_goal > 0:
            direction = goal_vector / distance_to_goal
        else:
            direction = np.array([0, 0])
        
        # Add wall avoidance
        direction = self._avoid_walls(direction)
        
        # Initialize acceleration and steering
        acceleration = 0
        steering = 0
        
        # Check for nearby vehicles
        nearby_vehicles = state.get('nearby_vehicles', [])
        
        # Emergency vehicle behavior
        if self.vehicle_type == 'emergency' and self.emergency_active:
            # Emergency vehicles get priority and move at max speed
            acceleration = self.max_speed
            steering = self._calculate_steering(direction, nearby_vehicles)
            self.waiting_at_intersection = False
            return acceleration, steering
        
        # Handle intersection behavior
        in_intersection = self._is_in_intersection(self.position)
        if in_intersection:
            if not self.waiting_at_intersection:
                # We've just entered the intersection
                self._evaluate_intersection_priority(nearby_vehicles, belief_state)
            
            # Check if we should proceed through intersection
            can_proceed = self._can_proceed_through_intersection(nearby_vehicles)
            
            if can_proceed:
                # Proceed through intersection
                acceleration = self.max_speed
                self.waiting_at_intersection = False
                self.time_waiting = 0
            else:
                # Wait at intersection
                acceleration = 0
                self.waiting_at_intersection = True
                self.time_waiting += 1
        else:
            # Normal road behavior
            self.waiting_at_intersection = False
            self.time_waiting = 0
            
            # Adjust speed based on nearby vehicles
            safe_distance = self._calculate_safe_distance(nearby_vehicles)
            if safe_distance:
                acceleration = min(self.max_speed, safe_distance)
            else:
                acceleration = self.max_speed
        
        # Calculate steering to avoid collisions
        steering = self._calculate_steering(direction, nearby_vehicles)
        
        # Store last action
        self.last_action = (acceleration, steering)
        
        return acceleration, steering
    
    def _evaluate_intersection_priority(self, nearby_vehicles, belief_state):
        """Evaluate priority for crossing the intersection."""
        self.intersection_priority = 0
        
        # Base priority on vehicle type
        if self.vehicle_type == 'emergency':
            self.intersection_priority += 100
        elif self.vehicle_type == 'premium':
            self.intersection_priority += 50
        else:
            self.intersection_priority += 25
        
        # Add priority based on waiting time
        self.intersection_priority += self.time_waiting * 5
        
        # Consider beliefs about nearby vehicles
        for vehicle in nearby_vehicles:
            if vehicle['id'] in belief_state:
                beliefs = belief_state[vehicle['id']]
                # Reduce priority if we believe others are emergency vehicles
                if beliefs.get('emergency', 0) > 0.5:
                    self.intersection_priority -= 30
    
    def _can_proceed_through_intersection(self, nearby_vehicles):
        """Determine if it's safe to proceed through intersection."""
        # Emergency vehicles always proceed
        if self.vehicle_type == 'emergency' and self.emergency_active:
            return True
        
        # Check each nearby vehicle
        for vehicle in nearby_vehicles:
            # Calculate relative position
            rel_pos = np.array([vehicle['position'][0] - self.position[0],
                              vehicle['position'][1] - self.position[1]])
            distance = np.linalg.norm(rel_pos)
            
            # If vehicle is too close, don't proceed
            if distance < 1.5:
                return False
            
            # If vehicle has higher priority and is close, don't proceed
            if (hasattr(vehicle, 'intersection_priority') and 
                vehicle.intersection_priority > self.intersection_priority and
                distance < 3):
                return False
        
        return True
    
    def _calculate_safe_distance(self, nearby_vehicles):
        """Calculate safe following distance based on nearby vehicles."""
        min_safe_distance = None
        
        for vehicle in nearby_vehicles:
            # Calculate relative position and velocity
            rel_pos = np.array([vehicle['position'][0] - self.position[0],
                              vehicle['position'][1] - self.position[1]])
            distance = np.linalg.norm(rel_pos)
            
            # If vehicle is in front of us
            if distance < self.sensor_range:
                safe_distance = distance - 1.0  # Maintain 1 unit minimum gap
                if min_safe_distance is None or safe_distance < min_safe_distance:
                    min_safe_distance = max(0, safe_distance)
        
        return min_safe_distance
    
    def _calculate_steering(self, goal_direction, nearby_vehicles):
        """Calculate steering to avoid collisions while moving towards goal."""
        # Start with direction to goal
        steering_vector = goal_direction.copy()
        
        # Add avoidance vectors from nearby vehicles
        for vehicle in nearby_vehicles:
            rel_pos = np.array([vehicle['position'][0] - self.position[0],
                              vehicle['position'][1] - self.position[1]])
            distance = np.linalg.norm(rel_pos)
            
            if distance < self.sensor_range and distance > 0.001:  # Avoid division by zero
                # Calculate repulsion vector (stronger at closer distances)
                repulsion = -rel_pos / (distance * distance)
                steering_vector += repulsion
        
        # Normalize steering vector
        steering_norm = np.linalg.norm(steering_vector)
        if steering_norm > 0.001:  # Avoid division by very small numbers
            steering_vector = steering_vector / steering_norm
        
        return steering_vector
    
    def _avoid_walls(self, direction):
        """Add wall avoidance behavior to direction vector."""
        pos_x, pos_y = self.position
        margin = self.wall_collision_margin
        
        # Get repulsion from walls
        wall_force = np.zeros(2)
        
        # Left wall
        if pos_x < margin:
            wall_force[0] += (margin - pos_x) / margin
        # Right wall
        elif pos_x > 19 - margin:
            wall_force[0] -= (pos_x - (19 - margin)) / margin
        # Top wall
        if pos_y < margin:
            wall_force[1] += (margin - pos_y) / margin
        # Bottom wall
        elif pos_y > 19 - margin:
            wall_force[1] -= (pos_y - (19 - margin)) / margin
        
        # Combine original direction with wall avoidance
        new_direction = direction + wall_force
        
        # Normalize if non-zero
        norm = np.linalg.norm(new_direction)
        if norm > 0:
            new_direction = new_direction / norm
        
        return new_direction
    
    def _is_in_intersection(self, position):
        """Check if position is in the intersection zone."""
        center_x, center_y = 10, 10
        x, y = position
        intersection_size = 2
        return (abs(x - center_x) < intersection_size and 
                abs(y - center_y) < intersection_size)
    
    def update_state(self, action):
        """Update vehicle state based on action."""
        acceleration, steering = action
        
        # Update velocity (with steering influence)
        self.velocity[0] = steering[0] * acceleration
        self.velocity[1] = steering[1] * acceleration
        
        # Update position
        new_position = (
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        )
        
        # Ensure we stay within bounds (20x20 grid)
        self.position = (
            max(0, min(19, new_position[0])),
            max(0, min(19, new_position[1]))
        )
    
    def update_belief(self, other_id, observation):
        """Update beliefs about another vehicle based on observation."""
        if other_id not in self.beliefs:
            self.beliefs[other_id] = {
                'standard': 1/3,
                'premium': 1/3,
                'emergency': 1/3
            }
        
        # Extract features from observation
        speed = np.linalg.norm(observation['velocity'])
        acceleration = observation['acceleration']
        
        # Update beliefs based on observed behavior
        if speed > 1.3:  # Fast movement suggests emergency
            self._update_type_probability(other_id, 'emergency', 0.6)
        elif speed > 1.1:  # Moderately fast suggests premium
            self._update_type_probability(other_id, 'premium', 0.6)
        else:  # Normal speed suggests standard
            self._update_type_probability(other_id, 'standard', 0.6)
    
    def _update_type_probability(self, other_id, vehicle_type, confidence):
        """Update probability of a specific vehicle type."""
        # Increase probability of observed type
        self.beliefs[other_id][vehicle_type] *= confidence
        
        # Normalize probabilities
        total = sum(self.beliefs[other_id].values())
        if total > 0:
            for type_name in self.beliefs[other_id]:
                self.beliefs[other_id][type_name] /= total
    
    def communicate(self, neighbors):
        """Communicate with neighboring vehicles."""
        # Send information to neighbors
        for neighbor in neighbors:
            if self.vehicle_type == 'emergency' and self.emergency_active:
                neighbor.receive_emergency_notification(self.id, self.final_goal)
            
            # Share observations and beliefs
            neighbor.receive_observation(self.id, {
                'position': self.position,
                'velocity': self.velocity,
                'type': self.vehicle_type,
                'acceleration': self.last_action[0]  # Add acceleration from last action
            })
            
            # Share beliefs if confident
            for other_id, beliefs in self.beliefs.items():
                if max(beliefs.values()) > 0.8:  # Only share if confident
                    neighbor.receive_belief(other_id, beliefs)
    
    def receive_emergency_notification(self, emergency_id, emergency_goal):
        """Receive notification of emergency vehicle status."""
        if emergency_id in self.beliefs:
            # Update beliefs to reflect emergency status
            self.beliefs[emergency_id] = {
                'standard': 0.0,
                'premium': 0.0,
                'emergency': 1.0
            }
        
        # Adjust own behavior if in path of emergency vehicle
        if self._is_in_emergency_path(emergency_goal):
            self.give_way_to_emergency = True
    
    def _is_in_emergency_path(self, emergency_goal):
        """
        Check if this vehicle is in the path of an emergency vehicle.
        
        Args:
            emergency_goal (tuple): Goal position of emergency vehicle
            
        Returns:
            bool: True if vehicle is in emergency path
        """
        # Get current position relative to center of intersection
        curr_x, curr_y = self.position
        center_x, center_y = 10, 10  # Center of 20x20 grid
        
        # Determine which road segments we're on (horizontal or vertical)
        on_horizontal_road = abs(curr_y - center_y) < 2
        on_vertical_road = abs(curr_x - center_x) < 2
        
        # Get emergency vehicle's approach direction
        emergency_x, emergency_y = emergency_goal
        emergency_horizontal = abs(emergency_y - center_y) < 2
        emergency_vertical = abs(emergency_x - center_x) < 2
        
        # If we're on the same road as the emergency vehicle's path
        if (on_horizontal_road and emergency_horizontal) or (on_vertical_road and emergency_vertical):
            # Check if we're between the emergency vehicle and its goal
            if on_horizontal_road:
                # Check if we're between emergency vehicle and its goal horizontally
                if emergency_x > center_x and curr_x > center_x:
                    return True
                if emergency_x < center_x and curr_x < center_x:
                    return True
            if on_vertical_road:
                # Check if we're between emergency vehicle and its goal vertically
                if emergency_y > center_y and curr_y > center_y:
                    return True
                if emergency_y < center_y and curr_y < center_y:
                    return True
        
        # If we're in the intersection and emergency vehicle needs to pass through
        if on_horizontal_road and on_vertical_road:  # We're in intersection
            if emergency_horizontal or emergency_vertical:
                return True
        
        return False
    
    def receive_belief(self, other_id, beliefs):
        """
        Receive and update beliefs about another vehicle from a neighbor.
        
        Args:
            other_id (int): ID of the vehicle the beliefs are about
            beliefs (dict): Dictionary of beliefs about vehicle types
        """
        if other_id not in self.beliefs:
            self.beliefs[other_id] = beliefs.copy()
        else:
            # Combine received beliefs with existing beliefs
            for vehicle_type, probability in beliefs.items():
                if probability > self.beliefs[other_id].get(vehicle_type, 0):
                    self.beliefs[other_id][vehicle_type] = probability
            
            # Normalize probabilities
            total = sum(self.beliefs[other_id].values())
            if total > 0:
                for type_name in self.beliefs[other_id]:
                    self.beliefs[other_id][type_name] /= total

    def receive_observation(self, other_id, observation):
        """
        Receive and process observation data from another vehicle.
        
        Args:
            other_id (int): ID of the observed vehicle
            observation (dict): Contains observed vehicle data (position, velocity, type)
        """
        # Update beliefs based on observation
        self.update_belief(other_id, observation)