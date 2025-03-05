#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lesson 3 Coding Exercise: Roundabout Game

This module implements a game theory scenario where multiple vehicles navigate a roundabout,
demonstrating concepts of Pareto efficiency, best response strategies, and repeated games.

The game is implemented as a Gymnasium environment, allowing for easy integration with
reinforcement learning algorithms and standardized interaction patterns.
Visualization is handled by Pygame for better interactivity.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import time
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import nashpy as nash
from scipy.spatial import ConvexHull

# Import the Vehicle class from vehicle.py
from vehicle import Vehicle

# Import game analysis tools from game_analyzer.py
from game_analyzer import GameAnalyzer

# Import utilities for visualization and metrics
from utils.visualization import draw_roundabout, draw_vehicles, draw_info_panel
from utils.metrics import calculate_traffic_metrics


class RoundaboutEnv(gym.Env):
    """
    A Gymnasium environment representing a roundabout with multiple entry/exit points where
    vehicles with different driver behaviors interact.
    
    This environment demonstrates Pareto efficiency, best response strategies, and repeated games
    in the context of traffic flow management.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, n_entry_points=4, max_vehicles=8, render_mode=None, 
                 repeated_game=False, history_length=100):
        """
        Initialize the roundabout environment with specified number of entry points and vehicles.
        
        Args:
            n_entry_points (int): Number of entry/exit points on the roundabout
            max_vehicles (int): Maximum number of vehicles in the simulation
            render_mode (str, optional): The rendering mode ('human', 'rgb_array', or None)
            repeated_game (bool): Whether to enable repeated game features
            history_length (int): How many past interactions to track for repeated games
        """
        super(RoundaboutEnv, self).__init__()
        
        # Roundabout configuration
        self.n_entry_points = n_entry_points
        self.max_vehicles = max_vehicles
        self.roundabout_radius = 100.0  # Pixels
        self.lane_width = 30.0  # Pixels
        self.entry_width = 20.0  # Pixels
        
        # Vehicle action space: 0 = Enter/Accelerate, 1 = Yield/Decelerate, 2 = Change Lane (if applicable)
        self.action_space = spaces.Discrete(3)
        
        # Observation space includes positions, velocities, and vehicle states for all vehicles
        # For each vehicle: [x, y, velocity, angle, entry_point, exit_point, in_roundabout, vehicle_type]
        # vehicle_type is encoded as 0 = aggressive, 1 = conservative, 2 = cooperative, 3 = autonomous
        obs_size = max_vehicles * 8 + 1  # +1 for time step
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Define the reward range
        self.reward_range = (-10.0, 10.0)
        
        # Set the render mode
        self.render_mode = render_mode
        
        # Repeated game features
        self.repeated_game = repeated_game
        self.history_length = history_length
        self.interaction_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Pygame setup
        self.window = None
        self.clock = None
        self.window_size = 800
        
        # Reset the environment
        self.reset()
        
        # Game analyzer for computation and visualization of game theory concepts
        self.game_analyzer = GameAnalyzer()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and returns the initial observation.
        
        Args:
            seed (int, optional): A seed for random number generation
            options (dict, optional): Additional options for environment configuration
            
        Returns:
            tuple: (observation, info)
        """
        # Reset the random seed
        super().reset(seed=seed)
        
        # Initialize vehicles
        self.vehicles = []
        initial_vehicle_count = min(4, self.max_vehicles)
        
        # Create initial vehicles at different entry points
        for i in range(initial_vehicle_count):
            entry_point = i % self.n_entry_points
            exit_point = (entry_point + np.random.randint(1, self.n_entry_points)) % self.n_entry_points
            
            # Randomly assign vehicle types
            vehicle_type = np.random.choice(['aggressive', 'conservative', 'cooperative', 'autonomous'])
            
            # Create a vehicle with specified parameters
            vehicle = Vehicle(
                id=i,
                entry_point=entry_point,
                exit_point=exit_point,
                vehicle_type=vehicle_type,
                env=self
            )
            
            self.vehicles.append(vehicle)
        
        # Time step counter
        self.step_count = 0
        self.max_steps = 1000
        
        # State tracking variables
        self.game_over = False
        self.collisions = []
        
        # Traffic flow metrics
        self.total_wait_time = 0
        self.total_travel_time = 0
        self.vehicles_completed = 0
        
        # Initialize rendering if needed
        if self.render_mode == "human" and self.window is None:
            self._init_rendering()
        
        # Return the initial observation and info
        return self._get_observation(), self._get_info()
    
    def _init_rendering(self):
        """Initialize pygame and create rendering components."""
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Roundabout Traffic Management")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            numpy.ndarray: The observation array
        """
        # Initialize observation with zeros
        obs = np.zeros(self.observation_space.shape)
        
        # Add time step
        obs[0] = self.step_count
        
        # Add vehicle states to observation
        for i, vehicle in enumerate(self.vehicles):
            if i < self.max_vehicles:
                start_idx = 1 + i * 8
                
                # Vehicle position
                obs[start_idx] = vehicle.position[0]
                obs[start_idx + 1] = vehicle.position[1]
                
                # Vehicle velocity
                obs[start_idx + 2] = vehicle.velocity
                
                # Vehicle angle
                obs[start_idx + 3] = vehicle.angle
                
                # Entry and exit points
                obs[start_idx + 4] = vehicle.entry_point
                obs[start_idx + 5] = vehicle.exit_point
                
                # In roundabout flag
                obs[start_idx + 6] = float(vehicle.in_roundabout)
                
                # Vehicle type (encoded)
                type_encoding = {'aggressive': 0, 'conservative': 1, 'cooperative': 2, 'autonomous': 3}
                obs[start_idx + 7] = type_encoding[vehicle.vehicle_type]
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            dict: Additional information about the current state
        """
        info = {
            'vehicle_count': len(self.vehicles),
            'collisions': len(self.collisions),
            'step_count': self.step_count,
            'vehicles_completed': self.vehicles_completed,
            'mean_wait_time': 0 if self.vehicles_completed == 0 else self.total_wait_time / self.vehicles_completed,
            'mean_travel_time': 0 if self.vehicles_completed == 0 else self.total_travel_time / self.vehicles_completed
        }
        
        # Add traffic metrics
        if len(self.vehicles) > 0:
            metrics = calculate_traffic_metrics(self.vehicles)
            info.update(metrics)
        
        return info
    
    def step(self, actions):
        """
        Take a step in the environment by applying actions to vehicles.
        
        Args:
            actions (list): List of actions for each vehicle
            
        Returns:
            tuple: (observation, rewards, terminated, truncated, info)
        """
        # Ensure actions is a list of the correct length
        if not isinstance(actions, list):
            if isinstance(actions, int):
                actions = [actions]
            else:
                actions = list(actions)
        
        # Pad actions if needed
        actions = actions[:len(self.vehicles)]
        while len(actions) < len(self.vehicles):
            actions.append(1)  # Default action: Yield/Decelerate
        
        rewards = []
        
        # Apply actions to vehicles and collect rewards
        for vehicle, action in zip(self.vehicles, actions):
            # Update vehicle state based on action
            reward = vehicle.update_state(action)
            rewards.append(reward)
        
        # Check for collisions
        self._check_collisions()
        
        # Update environment state
        self.step_count += 1
        
        # Check if max steps reached
        truncated = self.step_count >= self.max_steps
        
        # Check if all vehicles have completed their journeys or collided
        completed_or_collided = all(v.completed or v.collided for v in self.vehicles)
        terminated = completed_or_collided or not self.vehicles
        
        # Create new vehicles to replace completed ones (up to max_vehicles)
        self._spawn_new_vehicles()
        
        # Analyze game state if needed for visualization or analysis
        self._analyze_game_state()
        
        # Render if required
        if self.render_mode == "human":
            self._render_frame()
        
        # Return step information
        return (
            self._get_observation(),
            rewards,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _check_collisions(self):
        """Check for collisions between vehicles and update collision status."""
        # Check each pair of vehicles for collisions
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                v1 = self.vehicles[i]
                v2 = self.vehicles[j]
                
                # Skip if either vehicle has already collided or completed its journey
                if v1.collided or v2.collided or v1.completed or v2.completed:
                    continue
                
                # Calculate distance between vehicles
                distance = np.linalg.norm(np.array(v1.position) - np.array(v2.position))
                
                # Check if vehicles are too close (collision)
                if distance < v1.size + v2.size:
                    v1.collided = True
                    v2.collided = True
                    self.collisions.append((v1.id, v2.id))
                    
                    # Record interaction in history for repeated games
                    if self.repeated_game:
                        self._record_interaction(v1, v2, 'collision')
    
    def _spawn_new_vehicles(self):
        """Spawn new vehicles to replace completed or collided ones."""
        # Remove completed or collided vehicles and track statistics
        active_vehicles = []
        for vehicle in self.vehicles:
            if vehicle.completed:
                self.vehicles_completed += 1
                self.total_wait_time += vehicle.wait_time
                self.total_travel_time += vehicle.travel_time
            
            if not (vehicle.completed or vehicle.collided):
                active_vehicles.append(vehicle)
        
        # Update the list of active vehicles
        self.vehicles = active_vehicles
        
        # Add new vehicles if needed
        while len(self.vehicles) < self.max_vehicles:
            # Choose an entry point that's not too crowded
            entry_point_counts = [0] * self.n_entry_points
            for v in self.vehicles:
                if not v.in_roundabout:
                    entry_point_counts[v.entry_point] += 1
            
            # Find the entry point with the fewest waiting vehicles
            entry_point = np.argmin(entry_point_counts)
            
            # Choose an exit point different from the entry
            exit_point = (entry_point + np.random.randint(1, self.n_entry_points)) % self.n_entry_points
            
            # Assign a vehicle type
            vehicle_type = np.random.choice(['aggressive', 'conservative', 'cooperative', 'autonomous'])
            
            # Create a new vehicle
            new_id = max([v.id for v in self.vehicles] + [-1]) + 1
            new_vehicle = Vehicle(
                id=new_id,
                entry_point=entry_point,
                exit_point=exit_point,
                vehicle_type=vehicle_type,
                env=self
            )
            
            self.vehicles.append(new_vehicle)
    
    def _record_interaction(self, v1, v2, interaction_type):
        """Record an interaction between two vehicles for repeated game analysis."""
        if not self.repeated_game:
            return
        
        # Create a unique key for this pair of vehicles
        pair_key = tuple(sorted([v1.id, v2.id]))
        
        # Record interaction with type and timestamp
        self.interaction_history[pair_key].append({
            'type': interaction_type,
            'time': self.step_count,
            'vehicle1': {
                'id': v1.id,
                'type': v1.vehicle_type,
                'action': v1.last_action
            },
            'vehicle2': {
                'id': v2.id,
                'type': v2.vehicle_type,
                'action': v2.last_action
            }
        })
    
    def _analyze_game_state(self):
        """Analyze the current game state for game theory concepts."""
        if len(self.vehicles) < 2:
            return
        
        # This is a placeholder for game analysis
        # In a full implementation, we would:
        # 1. Compute payoff matrices for current vehicle interactions
        # 2. Find Nash equilibria
        # 3. Identify Pareto improvements
        # 4. Update best response strategies
        
        # For vehicles that can see each other, analyze their interactions
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                v1 = self.vehicles[i]
                v2 = self.vehicles[j]
                
                # Skip if vehicles are too far apart to interact
                distance = np.linalg.norm(np.array(v1.position) - np.array(v2.position))
                if distance > self.roundabout_radius * 2:
                    continue
                
                # Compute best response for each vehicle given the other's strategy
                if hasattr(v1, 'compute_best_response') and hasattr(v2, 'compute_best_response'):
                    v1.compute_best_response([v2])
                    v2.compute_best_response([v1])
                
                # For cooperative vehicles in repeated games, consider history
                if self.repeated_game and v1.vehicle_type == 'cooperative' and v2.vehicle_type == 'cooperative':
                    pair_key = tuple(sorted([v1.id, v2.id]))
                    if pair_key in self.interaction_history and len(self.interaction_history[pair_key]) > 0:
                        # Use history to inform current decisions
                        v1.update_strategy(self.interaction_history[pair_key])
                        v2.update_strategy(self.interaction_history[pair_key])
    
    def _render_frame(self):
        """Render a single frame of the environment."""
        if self.window is None and self.render_mode == "human":
            self._init_rendering()
        
        if self.window is not None:
            # Fill the background
            self.window.fill((240, 240, 240))
            
            # Draw the roundabout
            draw_roundabout(
                self.window, 
                self.window_size // 2, 
                self.window_size // 2, 
                self.roundabout_radius,
                self.lane_width,
                self.n_entry_points
            )
            
            # Draw vehicles
            draw_vehicles(
                self.window,
                self.vehicles,
                self.window_size // 2,
                self.window_size // 2
            )
            
            # Draw information panel with metrics
            info = self._get_info()
            draw_info_panel(
                self.window,
                self.font,
                info,
                self.vehicles
            )
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def render(self):
        """
        Render the environment according to the set render mode.
        
        Returns:
            numpy.ndarray or None: RGB array if mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame()
            return None
    
    def _render_frame_rgb(self):
        """
        Render a frame as an RGB array.
        
        Returns:
            numpy.ndarray: RGB array of the rendered frame
        """
        if self.window is None:
            self._init_rendering()
        
        self._render_frame()
        
        # Convert pygame surface to RGB array
        rgb_array = pygame.surfarray.array3d(self.window)
        return np.transpose(rgb_array, (1, 0, 2))
    
    def close(self):
        """Clean up resources used by the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def get_pareto_frontier(self):
        """
        Calculate the Pareto frontier for current vehicle strategies.
        
        Returns:
            tuple: (pareto_points, vehicle_ids) where pareto_points contains the
                   coordinates of points on the Pareto frontier, and vehicle_ids
                   contains the corresponding vehicle ids.
        """
        if len(self.vehicles) < 2:
            return [], []
        
        # For simplicity, we'll represent the utility space in 2D: 
        # (travel time, safety margin)
        points = []
        vehicle_ids = []
        
        for v in self.vehicles:
            # Skip vehicles that have collided
            if v.collided:
                continue
            
            # Calculate utilities based on travel time and safety
            travel_time_utility = -v.travel_time  # Negated because lower is better
            safety_margin = v.calculate_safety_margin()
            
            points.append([travel_time_utility, safety_margin])
            vehicle_ids.append(v.id)
        
        if len(points) < 2:
            return [], []
        
        # Convert to numpy array
        points = np.array(points)
        
        # Find Pareto optimal points (non-dominated points)
        pareto_points = self.game_analyzer.find_pareto_optimal_solutions(points)
        
        # Get the vehicle ids corresponding to Pareto optimal points
        pareto_vehicle_ids = [vehicle_ids[i] for i in range(len(points)) if i in pareto_points]
        
        return points[pareto_points], pareto_vehicle_ids
    
    def plot_pareto_frontier(self):
        """
        Plot the current Pareto frontier.
        
        Returns:
            matplotlib.figure.Figure: The figure object with the Pareto frontier plot
        """
        pareto_points, vehicle_ids = self.get_pareto_frontier()
        
        if len(pareto_points) < 2:
            return None
        
        return self.game_analyzer.plot_pareto_frontier(pareto_points, vehicle_ids)


def interact_with_roundabout(env, n_steps=1000):
    """
    Interactive mode for the roundabout environment.
    
    Args:
        env (RoundaboutEnv): The roundabout environment
        n_steps (int): Maximum number of steps to run
    """
    # Reset the environment
    obs, info = env.reset()
    
    # Main interaction loop
    for step in range(n_steps):
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
        
        # Get AI actions for all vehicles
        actions = []
        for vehicle in env.vehicles:
            # Let each vehicle decide its action based on its type and the environment
            action = vehicle.decide_action()
            actions.append(action)
        
        # Take a step in the environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Render the environment
        env.render()
        
        # Show game theory analysis periodically
        if step % 100 == 0 and step > 0:
            print(f"\nStep {step} - Game Theory Analysis:")
            print(f"Vehicles in simulation: {len(env.vehicles)}")
            print(f"Vehicles completed: {env.vehicles_completed}")
            print(f"Collisions: {len(env.collisions)}")
            
            # Calculate and show Pareto frontier
            pareto_points, pareto_vehicle_ids = env.get_pareto_frontier()
            if len(pareto_points) > 0:
                print("Vehicles on the Pareto frontier:")
                for i, v_id in enumerate(pareto_vehicle_ids):
                    # Find the vehicle object with this id
                    vehicle = next((v for v in env.vehicles if v.id == v_id), None)
                    if vehicle:
                        print(f"  Vehicle {v_id} ({vehicle.vehicle_type}): "
                              f"Travel time: {vehicle.travel_time:.2f}, "
                              f"Safety margin: {vehicle.calculate_safety_margin():.2f}")
            
            # Plot Pareto frontier
            fig = env.plot_pareto_frontier()
            if fig:
                plt.show(block=False)
                plt.pause(2)
                plt.close()
        
        # Delay to control simulation speed
        time.sleep(0.01)
        
        # Check if episode is done
        if terminated or truncated:
            break
    
    # Final analysis
    print("\nSimulation Complete!")
    print(f"Vehicles completed: {env.vehicles_completed}")
    print(f"Collisions: {len(env.collisions)}")
    print(f"Average wait time: {info['mean_wait_time']:.2f}")
    print(f"Average travel time: {info['mean_travel_time']:.2f}")
    
    # Clean up resources
    env.close()


def main():
    """Main function to run the roundabout simulation."""
    print("Roundabout Traffic Management Simulation")
    print("This simulation demonstrates how different driver behaviors and strategies")
    print("affect traffic flow in a roundabout, with a focus on game theory concepts.")
    print("Close the pygame window or press ESC to exit.")
    
    # Ask for user input on simulation parameters
    try:
        n_entry_points = int(input("\nEnter number of entry points (2-8, default 4): ") or 4)
        n_entry_points = max(2, min(8, n_entry_points))
        
        max_vehicles = int(input("Enter maximum number of vehicles (2-20, default 8): ") or 8)
        max_vehicles = max(2, min(20, max_vehicles))
        
        enable_repeated_games = input("Enable repeated game features? (y/n, default y): ").lower() != 'n'
        
        print("\nCreating roundabout environment...")
        env = RoundaboutEnv(
            n_entry_points=n_entry_points,
            max_vehicles=max_vehicles,
            render_mode="human",
            repeated_game=enable_repeated_games
        )
        
        print("Starting simulation...")
        interact_with_roundabout(env)
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        print("Using default values.")
        env = RoundaboutEnv(render_mode="human")
        interact_with_roundabout(env)
    except KeyboardInterrupt:
        print("\nExiting simulation.")
    finally:
        # Ensure pygame is properly shut down
        pygame.quit()


if __name__ == "__main__":
    # Set appropriate environment variables for pygame
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center the pygame window
    
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.quit()  # Ensure pygame is properly shut down in case of errors