#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson 2 Coding Exercise: Lane Changing Game

This module implements a game theory scenario where autonomous vehicles make
lane-changing decisions while considering the behaviors of other vehicles.
The game demonstrates concepts of dominant strategies and Nash equilibria
in the context of autonomous driving.

The environment is implemented using Gymnasium, visualization is handled by Pygame,
and Nash equilibrium calculations are performed using Nashpy.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import nashpy as nash
import time
import os

class LaneChangingGameEnv(gym.Env):
    """
    A Gymnasium environment representing a game-theoretic model of lane-changing decisions
    for autonomous vehicles. The environment simulates a three-lane highway where vehicles
    must make strategic decisions about lane changes while considering other vehicles' behaviors.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, render_mode=None):
        """
        Initialize the environment with payoff matrices for different driver behaviors
        and lane-changing scenarios.
        
        The payoff matrices represent the outcomes (utility values) for different
        combinations of actions between two vehicles, considering safety, time efficiency,
        and different driver behaviors (aggressive, defensive, cooperative).
        
        Args:
            render_mode (str, optional): The mode to render the environment ('human' or 'rgb_array')
        """
        super(LaneChangingGameEnv, self).__init__()
        
        # Define actions: 0 = Change Left, 1 = Stay, 2 = Change Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: positions and velocities of both vehicles
        # [car1_x, car1_y, car1_vel, car2_x, car2_y, car2_vel]
        self.observation_space = spaces.Box(
            low=np.array([-5, -1, 0, -5, -1, 0], dtype=np.float32),
            high=np.array([5, 1, 2, 5, 1, 2], dtype=np.float32)
        )
        
        # Define different driver behavior types and their payoff matrices
        self.driver_types = ['aggressive', 'defensive', 'cooperative']
        
        # Payoff matrices for different driver combinations
        # Format: (car1_payoff, car2_payoff) for each action combination
        # Actions: [Change Left, Stay, Change Right]
        
        # Aggressive vs Aggressive
        self.payoff_aggressive_vs_aggressive = np.array([
            [(-8,-8), (-2,1), (-5,-5)],    # Car1: Change Left
            [(1,-2), (0,0), (1,-2)],       # Car1: Stay
            [(-5,-5), (-2,1), (-8,-8)]     # Car1: Change Right
        ])
        
        # Defensive vs Defensive
        self.payoff_defensive_vs_defensive = np.array([
            [(-10,-10), (-1,2), (-7,-7)],  # Car1: Change Left
            [(2,-1), (1,1), (2,-1)],       # Car1: Stay
            [(-7,-7), (-1,2), (-10,-10)]   # Car1: Change Right
        ])
        
        # Cooperative vs Cooperative
        self.payoff_cooperative_vs_cooperative = np.array([
            [(-6,-6), (0,3), (-4,-4)],     # Car1: Change Left
            [(3,0), (2,2), (3,0)],         # Car1: Stay
            [(-4,-4), (0,3), (-6,-6)]      # Car1: Change Right
        ])
        
        # Action names for better readability
        self.action_names = ['Change Left', 'Stay', 'Change Right']
        
        # Set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Initialize pygame if rendering is required
        self.window = None
        self.clock = None
        self.window_size = 800
        self.car_sprites = None
        
        # Initialize game state
        self.reset()
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for resetting
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize car positions and velocities
        self.car1_state = {
            'x': -2.0,
            'y': 0.0,  # Lane position
            'vel': 1.0,
            'type': np.random.choice(self.driver_types)
        }
        
        self.car2_state = {
            'x': 2.0,
            'y': 0.0,  # Lane position
            'vel': 1.0,
            'type': np.random.choice(self.driver_types)
        }
        
        # Game state variables
        self.collision = False
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.max_steps = 100
        
        # Actions taken by cars (None until step is called)
        self.car1_action = None
        self.car2_action = None
        
        # Initialize rendering if needed
        if self.render_mode == "human" and self.window is None:
            self._init_rendering()
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Get the current observation from the environment.
        
        Returns:
            numpy.ndarray: The current state observation
        """
        return np.array([
            self.car1_state['x'],
            self.car1_state['y'],
            self.car1_state['vel'],
            self.car2_state['x'],
            self.car2_state['y'],
            self.car2_state['vel']
        ], dtype=np.float32)
    
    def _get_payoff_matrix(self, car1_type, car2_type):
        """
        Get the appropriate payoff matrix based on driver types.
        
        Args:
            car1_type (str): Type of first driver
            car2_type (str): Type of second driver
            
        Returns:
            numpy.ndarray: The payoff matrix for the given driver types
        """
        if car1_type == 'aggressive' and car2_type == 'aggressive':
            return self.payoff_aggressive_vs_aggressive
        elif car1_type == 'defensive' and car2_type == 'defensive':
            return self.payoff_defensive_vs_defensive
        elif car1_type == 'cooperative' and car2_type == 'cooperative':
            return self.payoff_cooperative_vs_cooperative
        else:
            # For mixed types, use an average of the matrices
            return (self.payoff_aggressive_vs_aggressive + 
                   self.payoff_defensive_vs_defensive + 
                   self.payoff_cooperative_vs_cooperative) / 3
    
    def step(self, action):
        """
        Take a step in the environment given the actions of both cars.
        
        Args:
            action (tuple): A tuple of (car1_action, car2_action) where:
                          0 = Change Left, 1 = Stay, 2 = Change Right
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        car1_action, car2_action = action
        self.car1_action = car1_action
        self.car2_action = car2_action
        
        # Get payoffs based on driver types
        payoff_matrix = self._get_payoff_matrix(self.car1_state['type'], 
                                              self.car2_state['type'])
        reward = tuple(payoff_matrix[car1_action, car2_action])
        
        # Update positions based on actions
        old_y1 = self.car1_state['y']
        old_y2 = self.car2_state['y']
        
        # Update lane positions
        if car1_action == 0:  # Change Left
            self.car1_state['y'] = max(-1.0, self.car1_state['y'] - 1.0)
        elif car1_action == 2:  # Change Right
            self.car1_state['y'] = min(1.0, self.car1_state['y'] + 1.0)
            
        if car2_action == 0:  # Change Left
            self.car2_state['y'] = max(-1.0, self.car2_state['y'] - 1.0)
        elif car2_action == 2:  # Change Right
            self.car2_state['y'] = min(1.0, self.car2_state['y'] + 1.0)
        
        # Update x positions (cars moving forward)
        self.car1_state['x'] += self.car1_state['vel'] * 0.1
        self.car2_state['x'] += self.car2_state['vel'] * 0.1
        
        # Check for collision
        if (abs(self.car1_state['x'] - self.car2_state['x']) < 0.5 and
            abs(self.car1_state['y'] - self.car2_state['y']) < 0.5):
            self.collision = True
            self.terminated = True
        
        # Increment step counter
        self.step_count += 1
        
        # Check if episode should end
        if not self.terminated:
            self.terminated = (abs(self.car1_state['x']) > 5 or 
                             abs(self.car2_state['x']) > 5)
        
        self.truncated = self.step_count >= self.max_steps
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            self.truncated,
            {
                'car1_action': self.action_names[car1_action],
                'car2_action': self.action_names[car2_action],
                'car1_type': self.car1_state['type'],
                'car2_type': self.car2_state['type'],
                'collision': self.collision
            }
        )
    
    def _init_rendering(self):
        """Initialize pygame and create sprites for rendering."""
        pygame.init()
        pygame.display.init()
        
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Lane Changing Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Create car sprites
        car_size = int(self.window_size / 15)
        self.car_sprites = {}
        
        # Create car sprites for different types
        colors = {
            'aggressive': (255, 50, 50),    # Red
            'defensive': (50, 50, 255),     # Blue
            'cooperative': (50, 255, 50)    # Green
        }
        
        for car_type, color in colors.items():
            car = pygame.Surface((car_size*2, car_size), pygame.SRCALPHA)
            pygame.draw.rect(car, color, (0, 0, car_size*2, car_size), 0, 3)
            pygame.draw.rect(car, (255, 255, 255), 
                           (car_size/2, car_size/4, car_size, car_size/2), 0, 2)
            self.car_sprites[car_type] = car
    
    def _render_frame(self):
        """Render a single frame of the environment."""
        if self.window is None:
            self._init_rendering()
        
        # Fill background (road color)
        self.window.fill((100, 100, 100))
        
        # Draw lane markings
        for y in range(1, 3):
            y_pos = int(self.window_size * y / 3)
            pygame.draw.line(self.window, (255, 255, 255), 
                           (0, y_pos), (self.window_size, y_pos), 2)
        
        # Convert world coordinates to screen coordinates
        def world_to_screen(x, y):
            screen_x = int((x + 5) / 10 * self.window_size)
            screen_y = int((y + 1) / 2 * self.window_size)
            return screen_x, screen_y
        
        # Draw cars
        for car_state, action in [(self.car1_state, self.car1_action),
                                (self.car2_state, self.car2_action)]:
            screen_x, screen_y = world_to_screen(car_state['x'], car_state['y'])
            car_sprite = self.car_sprites[car_state['type']]
            
            # Center the sprite on the car's position
            sprite_rect = car_sprite.get_rect(center=(screen_x, screen_y))
            self.window.blit(car_sprite, sprite_rect)
        
        # Draw game information
        if self.car1_action is not None and self.car2_action is not None:
            info_texts = [
                f"Car 1 ({self.car1_state['type']}): {self.action_names[self.car1_action]}",
                f"Car 2 ({self.car2_state['type']}): {self.action_names[self.car2_action]}"
            ]
            
            if self.collision:
                info_texts.append("COLLISION!")
            
            # Draw info box
            padding = 10
            line_height = 40
            box_height = len(info_texts) * line_height + 2 * padding
            pygame.draw.rect(self.window, (255, 255, 255),
                           pygame.Rect(10, 10, 400, box_height))
            
            for i, text in enumerate(info_texts):
                color = (255, 0, 0) if "COLLISION" in text else (0, 0, 0)
                text_surface = self.font.render(text, True, color)
                self.window.blit(text_surface, (20, 20 + i * line_height))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame()
            return None
    
    def _render_frame_rgb(self):
        """Render a frame as an RGB array."""
        self._render_frame()
        array = pygame.surfarray.array3d(self.window)
        return np.transpose(array, (1, 0, 2))
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
    
    def analyze_game(self):
        """
        Analyze the game to find dominant strategies and Nash equilibria using Nashpy.
        
        Returns:
            dict: Analysis results including Nash equilibria and dominant strategies
        """
        # Get current payoff matrix based on driver types
        payoff_matrix = self._get_payoff_matrix(self.car1_state['type'],
                                              self.car2_state['type'])
        
        # Create nashpy game
        game = nash.Game(payoff_matrix[:,:,0], payoff_matrix[:,:,1])
        
        # Find pure strategy Nash equilibria
        pure_equilibria = list(game.support_enumeration())
        
        # Convert equilibria to readable format
        nash_equilibria = []
        for eq in pure_equilibria:
            car1_strategy = np.argmax(eq[0])
            car2_strategy = np.argmax(eq[1])
            nash_equilibria.append((
                self.action_names[car1_strategy],
                self.action_names[car2_strategy]
            ))
        
        # Find dominant strategies
        car1_dominant = None
        car2_dominant = None
        
        # Check for car1 dominant strategy
        for action1 in range(3):
            is_dominant = True
            for action2 in range(3):
                for alt_action in range(3):
                    if (action1 != alt_action and
                        payoff_matrix[alt_action, action2, 0] >
                        payoff_matrix[action1, action2, 0]):
                        is_dominant = False
                        break
                if not is_dominant:
                    break
            if is_dominant:
                car1_dominant = self.action_names[action1]
                break
        
        # Check for car2 dominant strategy
        for action2 in range(3):
            is_dominant = True
            for action1 in range(3):
                for alt_action in range(3):
                    if (action2 != alt_action and
                        payoff_matrix[action1, alt_action, 1] >
                        payoff_matrix[action1, action2, 1]):
                        is_dominant = False
                        break
                if not is_dominant:
                    break
            if is_dominant:
                car2_dominant = self.action_names[action2]
                break
        
        return {
            'nash_equilibria': nash_equilibria,
            'car1_dominant': car1_dominant,
            'car2_dominant': car2_dominant,
            'car1_type': self.car1_state['type'],
            'car2_type': self.car2_state['type']
        }

def print_payoff_matrix(env):
    """Print the current payoff matrix in a readable format."""
    payoff_matrix = env._get_payoff_matrix(env.car1_state['type'],
                                         env.car2_state['type'])
    
    print(f"\nPayoff Matrix for {env.car1_state['type']} vs {env.car2_state['type']}:")
    print("-" * 70)
    print("                   Car 2")
    print("            Change Left    Stay    Change Right")
    print("-" * 70)
    
    for i, action1 in enumerate(env.action_names):
        row = f"Car 1 {action1:<10} | "
        for j, _ in enumerate(env.action_names):
            payoffs = payoff_matrix[i, j]
            row += f"({payoffs[0]}, {payoffs[1]})".center(12)
        print(row)
    print("-" * 70)

def main():
    """Main function to demonstrate the lane changing game environment."""
    print("Welcome to the Lane Changing Game!")
    print("This simulation models strategic lane-changing decisions between two vehicles.")
    print("Each car can either 'Change Left' (0), 'Stay' (1), or 'Change Right' (2).")
    
    # Create game environment
    env = LaneChangingGameEnv(render_mode="human")
    
    while True:
        # Reset environment
        observation, _ = env.reset()
        
        # Show current game analysis
        print("\nAnalyzing current game configuration...")
        analysis = env.analyze_game()
        
        print(f"\nCar Types:")
        print(f"Car 1: {analysis['car1_type']}")
        print(f"Car 2: {analysis['car2_type']}")
        
        print_payoff_matrix(env)
        
        print("\nGame Analysis:")
        if analysis['car1_dominant']:
            print(f"Car 1 has a dominant strategy: {analysis['car1_dominant']}")
        else:
            print("Car 1 does not have a dominant strategy")
            
        if analysis['car2_dominant']:
            print(f"Car 2 has a dominant strategy: {analysis['car2_dominant']}")
        else:
            print("Car 2 does not have a dominant strategy")
        
        if analysis['nash_equilibria']:
            print("\nNash Equilibria:")
            for i, ne in enumerate(analysis['nash_equilibria'], 1):
                print(f"{i}. Car 1: {ne[0]}, Car 2: {ne[1]}")
        else:
            print("\nNo pure strategy Nash Equilibria found.")
        
        # Get user input for actions
        print("\nChoose actions for both cars:")
        print("0. Change Left")
        print("1. Stay")
        print("2. Change Right")
        print("3. Quit")
        
        try:
            car1_action = int(input("\nEnter action for Car 1 (0-3): "))
            if car1_action == 3:
                break
                
            car2_action = int(input("Enter action for Car 2 (0-3): "))
            if car2_action == 3:
                break
            
            if not (0 <= car1_action <= 2 and 0 <= car2_action <= 2):
                print("Invalid action(s). Please try again.")
                continue
            
            # Step through the environment
            observation, reward, terminated, truncated, info = env.step((car1_action, car2_action))
            
            print(f"\nResults:")
            print(f"Car 1 ({info['car1_type']}) chose to {info['car1_action']} and received a payoff of {reward[0]}")
            print(f"Car 2 ({info['car2_type']}) chose to {info['car2_action']} and received a payoff of {reward[1]}")
            
            if info['collision']:
                print("A collision occurred!")
            
            # Wait for a moment to show the result
            time.sleep(2)
            
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            break
    
    env.close()

if __name__ == "__main__":
    # Set appropriate environment variables for pygame
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.quit()