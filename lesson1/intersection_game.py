#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Day 1 Coding Exercise: Intersection Game

This module implements a simple game theory scenario where two self-driving cars
approach an intersection and must decide whether to yield or proceed.

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


class IntersectionGameEnv(gym.Env):
    """
    A Gymnasium environment representing a game-theoretic model of two self-driving cars at an intersection.
    
    This class implements the game logic, payoff matrix, and visualization for a scenario
    where two autonomous vehicles must decide whether to yield or proceed at an intersection.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, render_mode=None):
        """
        Initialize the environment with the payoff matrix for the intersection scenario.
        
        The payoff matrix represents the outcomes (utility values) for each combination
        of actions chosen by the two cars.
        
        Payoff structure: [car1_payoff, car2_payoff]
        - Both proceed: [-10, -10] (collision, very negative for both)
        - Car1 proceeds, Car2 yields: [3, 1] (Car1 gains time, Car2 waits)
        - Car1 yields, Car2 proceeds: [1, 3] (Car1 waits, Car2 gains time)
        - Both yield: [0, 0] (Both wait, slight inefficiency but no collision)
        
        Args:
            render_mode (str, optional): The mode to render the environment ('human' or 'rgb_array')
        """
        super(IntersectionGameEnv, self).__init__()
        
        # Define actions: 0 = Proceed, 1 = Yield
        self.action_space = spaces.Discrete(2)
        
        # Observation space for each car: position (x, y)
        # We include positions for both cars in the observation
        self.observation_space = spaces.Box(
            low=np.array([-5, -1, -1, -5], dtype=np.float32),  # Min positions for car1_x, car1_y, car2_x, car2_y
            high=np.array([5, 1, 5, 1], dtype=np.float32),     # Max positions for car1_x, car1_y, car2_x, car2_y
        )
        
        # Payoff matrix for [car1, car2]
        self.payoff_matrix = np.array([
            [[-10, -10], [3, 1]],   # Car1: Proceed, Car2: [Proceed, Yield]
            [[1, 3], [0, 0]]        # Car1: Yield, Car2: [Proceed, Yield]
        ])
        
        # Action names for better readability
        self.action_names = ['Proceed', 'Yield']
        
        # Set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Initialize pygame if rendering is required
        self.window = None
        self.clock = None
        self.window_size = 800  # Size of the PyGame window
        self.car_sprites = None
        self.traffic_light_sprites = None
        
        # Reset the environment
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
        # Set random seed for reproducibility
        super().reset(seed=seed)
        
        # Car positions: [x, y]
        self.car1_position = [-4, 0]  # Car1 starts from the left
        self.car2_position = [0, -4]  # Car2 starts from the bottom
        
        # Car rotations (in degrees)
        self.car1_rotation = 0   # Facing right
        self.car2_rotation = 90  # Facing up
        
        # Car actions (None until step is called)
        self.car1_action = None
        self.car2_action = None
        
        # Simulation variables
        self.collision = False
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.max_steps = 80
        
        # Initialize rendering if needed
        if self.render_mode == "human" and self.window is None:
            self._init_rendering()
            
        # Return the initial observation and info
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _init_rendering(self):
        """Initialize pygame and load sprites for rendering."""
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Intersection Game Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Load or create car sprites
        self.car_sprites = {}
        
        # Create simple car sprites if no images are available
        car_size = int(self.window_size / 15)
        
        # Blue car (car1)
        blue_car = pygame.Surface((car_size*2, car_size), pygame.SRCALPHA)
        pygame.draw.rect(blue_car, (30, 30, 255), (0, 0, car_size*2, car_size), 0, 3)
        pygame.draw.rect(blue_car, (100, 100, 255), (car_size/2, car_size/4, car_size, car_size/2), 0, 2)
        pygame.draw.circle(blue_car, (0, 0, 0), (car_size/2, car_size/2), car_size/6)
        pygame.draw.circle(blue_car, (0, 0, 0), (car_size*3/2, car_size/2), car_size/6)
        self.car_sprites["blue"] = blue_car
        
        # Red car (car2)
        red_car = pygame.Surface((car_size*2, car_size), pygame.SRCALPHA)
        pygame.draw.rect(red_car, (255, 30, 30), (0, 0, car_size*2, car_size), 0, 3)
        pygame.draw.rect(red_car, (255, 100, 100), (car_size/2, car_size/4, car_size, car_size/2), 0, 2)
        pygame.draw.circle(red_car, (0, 0, 0), (car_size/2, car_size/2), car_size/6)
        pygame.draw.circle(red_car, (0, 0, 0), (car_size*3/2, car_size/2), car_size/6)
        self.car_sprites["red"] = red_car
        
        # Create traffic light sprites
        light_size = int(self.window_size / 25)
        
        # Traffic lights (horizontal and vertical)
        self.traffic_light_sprites = {}
        
        for state in ["red", "green", "yellow"]:
            # Horizontal traffic light
            h_light = pygame.Surface((light_size*3, light_size), pygame.SRCALPHA)
            h_light.fill((50, 50, 50))
            if state == "red":
                color = (255, 0, 0)
                pos = light_size/2
            elif state == "green":
                color = (0, 255, 0)
                pos = light_size*2.5
            else:  # yellow
                color = (255, 255, 0)
                pos = light_size*1.5
                
            pygame.draw.circle(h_light, color, (pos, light_size/2), light_size/3)
            self.traffic_light_sprites["h_" + state] = h_light
            
            # Vertical traffic light
            v_light = pygame.Surface((light_size, light_size*3), pygame.SRCALPHA)
            v_light.fill((50, 50, 50))
            if state == "red":
                color = (255, 0, 0)
                pos = light_size/2
            elif state == "green":
                color = (0, 255, 0)
                pos = light_size*2.5
            else:  # yellow
                color = (255, 255, 0)
                pos = light_size*1.5
                
            pygame.draw.circle(v_light, color, (light_size/2, pos), light_size/3)
            self.traffic_light_sprites["v_" + state] = v_light
            
    def step(self, action):
        """
        Take a step in the environment given the actions of both cars.
        
        Args:
            action (tuple): A tuple of (car1_action, car2_action) where:
                           0 = Proceed, 1 = Yield for both cars
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Parse actions
        car1_action, car2_action = action
        
        # Set car actions for simulation
        self.car1_action = car1_action
        self.car2_action = car2_action
        
        # Increment step counter
        self.step_count += 1
        
        # Determine movement patterns based on actions
        car1_moves = car1_action == 0  # Proceed = 0
        car2_moves = car2_action == 0  # Proceed = 0
        
        # Update car positions
        if car1_moves and self.car1_position[0] < 4:
            self.car1_position[0] += 0.5
        
        if car2_moves and self.car2_position[1] < 4:
            self.car2_position[1] += 0.5
        
        # Check for collision
        if (abs(self.car1_position[0]) < 0.5 and abs(self.car2_position[1]) < 0.5):
            self.collision = True
            self.terminated = True
        
        # Check if episode is terminated or truncated
        if not self.terminated:
            # Terminate if both cars have crossed the intersection
            if self.car1_position[0] >= 4 and self.car2_position[1] >= 4:
                self.terminated = True
        
        # Truncate if max steps reached
        self.truncated = self.step_count >= self.max_steps
        
        # Get payoffs for the current actions
        reward = self._get_payoffs(car1_action, car2_action)
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        # Return step information
        return (
            self._get_observation(),  # Observation
            reward,                   # Reward (as a tuple for both cars)
            self.terminated,          # Terminated flag
            self.truncated,           # Truncated flag
            {                         # Info dictionary
                'car1_action': self.action_names[car1_action],
                'car2_action': self.action_names[car2_action],
                'collision': self.collision,
                'car1_position': self.car1_position,
                'car2_position': self.car2_position,
                'step': self.step_count
            }
        )
    
    def _get_observation(self):
        """
        Get the current observation from the environment.
        
        Returns:
            numpy.ndarray: The current state observation
        """
        return np.array([
            self.car1_position[0],  # Car1 x position
            self.car1_position[1],  # Car1 y position
            self.car2_position[0],  # Car2 x position
            self.car2_position[1]   # Car2 y position
        ], dtype=np.float32)
    
    def _get_payoffs(self, car1_action, car2_action):
        """
        Get the payoffs for both cars given their actions.
        
        Args:
            car1_action (int): Action of car1 (0 = Proceed, 1 = Yield)
            car2_action (int): Action of car2 (0 = Proceed, 1 = Yield)
            
        Returns:
            tuple: (car1_payoff, car2_payoff)
        """
        return tuple(self.payoff_matrix[car1_action, car2_action])
    
    def _world_to_pixels(self, x, y):
        """
        Convert world coordinates to pixel coordinates for rendering.
        
        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            
        Returns:
            tuple: (pixel_x, pixel_y)
        """
        # Map from [-5, 5] to [0, window_size]
        pixel_x = int((x + 5) / 10 * self.window_size)
        pixel_y = int((5 - y) / 10 * self.window_size)  # Y is flipped in pygame
        return pixel_x, pixel_y
    
    def _render_frame(self):
        """Render a single frame of the environment using pygame."""
        if self.window is None:
            self._init_rendering()
        
        # Fill background with green (grass)
        self.window.fill((100, 180, 100))
        
        # Draw road (gray asphalt)
        road_width = self.window_size // 5
        half_road = road_width // 2
        
        # Horizontal road
        pygame.draw.rect(
            self.window,
            (80, 80, 80),  # Gray
            pygame.Rect(0, self.window_size // 2 - half_road, self.window_size, road_width)
        )
        
        # Vertical road
        pygame.draw.rect(
            self.window,
            (80, 80, 80),  # Gray
            pygame.Rect(self.window_size // 2 - half_road, 0, road_width, self.window_size)
        )
        
        # Draw road markings (white dashed lines)
        dash_length = 20
        dash_gap = 20
        line_width = 3
        
        # Horizontal road center line
        for x in range(0, self.window_size, dash_length + dash_gap):
            pygame.draw.line(
                self.window,
                (255, 255, 255),  # White
                (x, self.window_size // 2),
                (x + dash_length, self.window_size // 2),
                line_width
            )
        
        # Vertical road center line
        for y in range(0, self.window_size, dash_length + dash_gap):
            pygame.draw.line(
                self.window,
                (255, 255, 255),  # White
                (self.window_size // 2, y),
                (self.window_size // 2, y + dash_length),
                line_width
            )
        
        # Draw stop lines
        stop_line_width = 5
        stop_line_offset = 40  # distance from intersection
        
        # Horizontal stop lines
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # White
            (self.window_size // 2 - half_road - stop_line_offset, self.window_size // 2 - half_road),
            (self.window_size // 2 - half_road - stop_line_offset, self.window_size // 2 + half_road),
            stop_line_width
        )
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # White
            (self.window_size // 2 + half_road + stop_line_offset, self.window_size // 2 - half_road),
            (self.window_size // 2 + half_road + stop_line_offset, self.window_size // 2 + half_road),
            stop_line_width
        )
        
        # Vertical stop lines
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # White
            (self.window_size // 2 - half_road, self.window_size // 2 - half_road - stop_line_offset),
            (self.window_size // 2 + half_road, self.window_size // 2 - half_road - stop_line_offset),
            stop_line_width
        )
        pygame.draw.line(
            self.window,
            (255, 255, 255),  # White
            (self.window_size // 2 - half_road, self.window_size // 2 + half_road + stop_line_offset),
            (self.window_size // 2 + half_road, self.window_size // 2 + half_road + stop_line_offset),
            stop_line_width
        )
        
        # Draw traffic lights based on car actions
        if self.car1_action is not None and self.car2_action is not None:
            # Traffic light for Car 1 (horizontal road)
            h_light_state = "red" if self.car1_action == 1 else "green"
            h_traffic_light = self.traffic_light_sprites["h_" + h_light_state]
            h_light_pos = self._world_to_pixels(-2, 1)
            self.window.blit(h_traffic_light, (h_light_pos[0], h_light_pos[1]))
            
            # Traffic light for Car 2 (vertical road)
            v_light_state = "red" if self.car2_action == 1 else "green"
            v_traffic_light = self.traffic_light_sprites["v_" + v_light_state]
            v_light_pos = self._world_to_pixels(1, -2)
            self.window.blit(v_traffic_light, (v_light_pos[0], v_light_pos[1]))
        
        # Draw cars with proper rotation
        car_size = int(self.window_size / 15)
        
        # Car 1 (blue)
        car1_x, car1_y = self._world_to_pixels(self.car1_position[0], self.car1_position[1])
        blue_car = self.car_sprites["blue"]
        # Center the sprite on the car's position
        car1_pos = (car1_x - blue_car.get_width() // 2, car1_y - blue_car.get_height() // 2)
        self.window.blit(blue_car, car1_pos)
        
        # Car 2 (red)
        car2_x, car2_y = self._world_to_pixels(self.car2_position[0], self.car2_position[1])
        # Rotate the red car to face up
        red_car = pygame.transform.rotate(self.car_sprites["red"], -90)  # -90 degrees to face up
        # Center the sprite on the car's position
        car2_pos = (car2_x - red_car.get_width() // 2, car2_y - red_car.get_height() // 2)
        self.window.blit(red_car, car2_pos)
        
        # Draw information text
        if self.car1_action is not None and self.car2_action is not None:
            car1_action_name = self.action_names[self.car1_action]
            car2_action_name = self.action_names[self.car2_action]
            
            text_color = (0, 0, 0)  # Black text
            
            # Render car actions
            car1_text = self.font.render(f"Car 1 (Blue): {car1_action_name}", True, text_color)
            car2_text = self.font.render(f"Car 2 (Red): {car2_action_name}", True, text_color)
            
            # Render payoffs
            payoffs = self._get_payoffs(self.car1_action, self.car2_action)
            payoffs_text = self.font.render(f"Payoffs: Car1 = {payoffs[0]}, Car2 = {payoffs[1]}", True, text_color)
            
            # Position text
            bg_rect = pygame.Rect(10, 10, 400, 110)
            pygame.draw.rect(self.window, (255, 255, 255, 180), bg_rect)
            pygame.draw.rect(self.window, (0, 0, 0), bg_rect, 2)
            
            self.window.blit(car1_text, (20, 20))
            self.window.blit(car2_text, (20, 60))
            self.window.blit(payoffs_text, (20, 100))
            
            # Show collision alert if needed
            if self.collision:
                collision_text = self.font.render("COLLISION!", True, (255, 0, 0))
                text_width = collision_text.get_width()
                self.window.blit(collision_text, (self.window_size//2 - text_width//2, 20))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def render(self):
        """
        Render the environment.
        
        Returns:
            numpy.ndarray: RGB image data if mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        elif self.render_mode == "human":
            self._render_frame()
            return None
    
    def _render_frame_rgb(self):
        """Render a frame of the environment as an RGB array."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
        self._render_frame()
        
        # Convert pygame surface to RGB array
        rgb_array = pygame.surfarray.array3d(self.window)
        return np.transpose(rgb_array, (1, 0, 2))  # Transpose to match expected format
    
    def close(self):
        """Clean up resources and close rendering."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
    
    def analyze_game(self):
        """
        Analyze the game to find dominant strategies and Nash equilibria.
        
        Returns:
            dict: A dictionary containing game analysis information
        """
        # Calculate expected payoffs for all strategy combinations
        payoffs = {}
        
        for car1_action in range(2):  # 0=Proceed, 1=Yield
            for car2_action in range(2):
                payoffs[(car1_action, car2_action)] = self._get_payoffs(car1_action, car2_action)
        
        # Check for dominant strategies
        car1_dominant = None
        if (payoffs[(0, 0)][0] > payoffs[(1, 0)][0] and 
            payoffs[(0, 1)][0] > payoffs[(1, 1)][0]):
            car1_dominant = 0  # Proceed
        elif (payoffs[(1, 0)][0] > payoffs[(0, 0)][0] and 
              payoffs[(1, 1)][0] > payoffs[(0, 1)][0]):
            car1_dominant = 1  # Yield
        
        car2_dominant = None
        if (payoffs[(0, 0)][1] > payoffs[(0, 1)][1] and 
            payoffs[(1, 0)][1] > payoffs[(1, 1)][1]):
            car2_dominant = 0  # Proceed
        elif (payoffs[(0, 1)][1] > payoffs[(0, 0)][1] and 
              payoffs[(1, 1)][1] > payoffs[(1, 0)][1]):
            car2_dominant = 1  # Yield
        
        # Find Nash equilibria
        nash_equilibria = []
        for car1_action in range(2):
            for car2_action in range(2):
                # Check if this is a Nash equilibrium
                car1_payoff = payoffs[(car1_action, car2_action)][0]
                car2_payoff = payoffs[(car1_action, car2_action)][1]
                
                # Check if car1 can improve by changing strategy
                car1_can_improve = False
                for alt_action in range(2):
                    if alt_action != car1_action:
                        alt_payoff = payoffs[(alt_action, car2_action)][0]
                        if alt_payoff > car1_payoff:
                            car1_can_improve = True
                            break
                
                # Check if car2 can improve by changing strategy
                car2_can_improve = False
                for alt_action in range(2):
                    if alt_action != car2_action:
                        alt_payoff = payoffs[(car1_action, alt_action)][1]
                        if alt_payoff > car2_payoff:
                            car2_can_improve = True
                            break
                
                # If neither car can improve, this is a Nash equilibrium
                if not car1_can_improve and not car2_can_improve:
                    nash_equilibria.append((car1_action, car2_action))
        
        return {
            'payoffs': payoffs,
            'car1_dominant': self.action_names[car1_dominant] if car1_dominant is not None else None,
            'car2_dominant': self.action_names[car2_dominant] if car2_dominant is not None else None,
            'nash_equilibria': [(self.action_names[car1], self.action_names[car2]) for car1, car2 in nash_equilibria]
        }


def print_payoff_matrix(env):
    """
    Print the payoff matrix in a readable format.
    
    Args:
        env (IntersectionGameEnv): The game environment
    """
    print("\nPayoff Matrix for Intersection Game:")
    print("-" * 50)
    print("                   Car 2")
    print("                 Proceed     Yield")
    print("-" * 50)
    
    for i, action1 in enumerate(['Proceed', 'Yield']):
        row = f"Car 1 {action1:<7} | "
        for j, action2 in enumerate(['Proceed', 'Yield']):
            payoffs = env.payoff_matrix[i, j]
            row += f"({payoffs[0]}, {payoffs[1]})  "
        print(row)
    print("-" * 50)


def main():
    """
    Main function to demonstrate the intersection game environment.
    """
    print("Welcome to the Intersection Game!")
    print("This simulation models two self-driving cars at an intersection.")
    print("Each car can either 'Proceed' (0) or 'Yield' (1).")
    
    # Create game environment
    env = IntersectionGameEnv(render_mode="human")
    
    # Show payoff matrix
    print_payoff_matrix(env)
    
    # Analyze the game
    analysis = env.analyze_game()
    
    print("\nGame Analysis:")
    print("-" * 50)
    
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
        for ne in analysis['nash_equilibria']:
            ne_key = (0 if ne[0] == 'Proceed' else 1, 0 if ne[1] == 'Proceed' else 1)
            print(f"- Car 1: {ne[0]}, Car 2: {ne[1]} with payoffs: {analysis['payoffs'][ne_key]}")
    else:
        print("\nNo pure strategy Nash Equilibria found.")
    
    # Interactive simulation
    print("\nLet's run a simulation of the game using the Gymnasium environment!")
    print("Close the pygame window or press Ctrl+C to exit the current simulation.")
    
    # Ask for user input for car actions
    while True:
        print("\nChoose actions for both cars:")
        print("1. Car 1: Proceed, Car 2: Proceed")
        print("2. Car 1: Proceed, Car 2: Yield")
        print("3. Car 1: Yield, Car 2: Proceed")
        print("4. Car 1: Yield, Car 2: Yield")
        print("5. Quit")
        
        try:
            choice = int(input("Enter your choice (1-5): "))
            if choice == 5:
                env.close()
                break
                
            if choice == 1:
                actions = (0, 0)  # (Proceed, Proceed)
            elif choice == 2:
                actions = (0, 1)  # (Proceed, Yield)
            elif choice == 3:
                actions = (1, 0)  # (Yield, Proceed)
            elif choice == 4:
                actions = (1, 1)  # (Yield, Yield)
            else:
                print("Invalid choice. Please try again.")
                continue
            
            # Reset the environment
            observation, info = env.reset()
            
            # Render initial state
            env.render()
            
            # Step through until done
            terminated = False
            truncated = False
            
            # Event handling for pygame
            running = True
            while not (terminated or truncated) and running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                
                if not running:
                    break
                    
                # Take a step in the environment
                observation, reward, terminated, truncated, info = env.step(actions)
                env.render()
                
                # Control simulation speed
                pygame.time.delay(100)  # 100ms delay between steps
            
            print(f"\nSimulation complete!")
            print(f"Car 1 chose to {env.action_names[actions[0]]} and received a payoff of {reward[0]}")
            print(f"Car 2 chose to {env.action_names[actions[1]]} and received a payoff of {reward[1]}")
            
            if env.collision:
                print("A collision occurred at the intersection!")
            else:
                print("Both cars navigated the intersection safely.")
                
        except ValueError as e:
            print(f"Error: {e}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            env.close()
            break
    
    env.close()


if __name__ == "__main__":
    # Set appropriate environment variables for pygame
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center the pygame window
    
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        pygame.quit()  # Ensure pygame is properly shut down in case of errors