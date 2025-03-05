#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fleet coordination simulator for autonomous vehicles using Bayesian game theory.

This module implements a simulator for autonomous vehicle coordination using
Bayesian game-theoretic approaches to handle uncertain information about other vehicles.
"""

import numpy as np
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import random
import time
from collections import defaultdict

from bayesian_vehicle import BayesianVehicle
from bayesian_game_analyzer import BayesianGameAnalyzer
from utils.visualization import (draw_city_grid, draw_vehicles, 
                                draw_belief_states, draw_info_panel,
                                create_belief_heatmap, visualize_information_flow)
from utils.metrics import (calculate_fleet_metrics, analyze_belief_convergence,
                          analyze_information_value, plot_fleet_performance)

class FleetCoordinationEnvironment:
    """
    Environment for simulating fleet coordination with Bayesian vehicles.
    """
    
    def __init__(self, config=None):
        """
        Initialize the fleet coordination environment.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Default configuration
        self.config = {
            'grid_size': 20,
            'block_size': 4,
            'num_standard_vehicles': 3,
            'num_premium_vehicles': 1,
            'num_emergency_vehicles': 1,
            'max_steps': 500,
            'emergency_scenario': True,
            'emergency_start_step': 100,
            'visualization': True,
            'visualization_frequency': 1,
            'screen_width': 1200,  # Increased resolution
            'screen_height': 900   # Increased resolution
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Initialize pygame if visualization is enabled
        if self.config['visualization']:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.config['screen_width'],
                self.config['screen_height']
            ))
            pygame.display.set_caption("Fleet Coordination Simulation")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            
        # Initialize vehicles
        self.vehicles = []
        self.next_vehicle_id = 0
        
        # Initialize communication graph
        self.communication_graph = nx.DiGraph()
        
        # Initialize game analyzer
        self.game_analyzer = BayesianGameAnalyzer()
        
        # State tracking
        self.current_step = 0
        self.emergency_active = False
        self.metrics_history = []
        self.belief_history = defaultdict(list)
        
        # Initialize fleet
        self._initialize_fleet()
        
        # Store heatmap surface
        self.belief_heatmap = None
        self.update_heatmap_frequency = 20  # Update heatmap every 20 steps
        
        # Debug mode flags
        self.debug_mode = False
        self.step_requested = False
        
    def _initialize_fleet(self):
        """Initialize the fleet of vehicles with different types."""
        # Clear existing vehicles
        self.vehicles = []
        
        # Create standard vehicles
        for _ in range(self.config['num_standard_vehicles']):
            self._create_vehicle('standard')
        
        # Create premium vehicles
        for _ in range(self.config['num_premium_vehicles']):
            self._create_vehicle('premium')
        
        # Create emergency vehicles
        for _ in range(self.config['num_emergency_vehicles']):
            self._create_vehicle('emergency')
            
        # Initialize beliefs about other vehicles
        self._initialize_beliefs()
        
        # Add vehicles to communication graph
        for vehicle in self.vehicles:
            self.communication_graph.add_node(vehicle.id)
    
    def _create_vehicle(self, vehicle_type):
        """
        Create a new vehicle of the specified type.
        
        Args:
            vehicle_type (str): Type of vehicle to create
        
        Returns:
            BayesianVehicle: The newly created vehicle
        """
        # Generate random position
        # Ensure vehicles start on roads (at grid positions divisible by block_size)
        block_size = self.config['block_size']
        grid_size = self.config['grid_size']
        
        # Random road position
        if random.random() < 0.5:
            # Horizontal road
            x = random.uniform(0, grid_size)
            y = block_size * random.randint(0, grid_size // block_size)
        else:
            # Vertical road
            x = block_size * random.randint(0, grid_size // block_size)
            y = random.uniform(0, grid_size)
            
        position = (x, y)
        
        # Generate random goal on a different road
        while True:
            if random.random() < 0.5:
                # Horizontal road
                goal_x = random.uniform(0, grid_size)
                goal_y = block_size * random.randint(0, grid_size // block_size)
            else:
                # Vertical road
                goal_x = block_size * random.randint(0, grid_size // block_size)
                goal_y = random.uniform(0, grid_size)
                
            goal = (goal_x, goal_y)
            
            # Ensure goal is different from position
            if np.linalg.norm(np.array(goal) - np.array(position)) > 5.0:
                break
        
        # Create vehicle
        vehicle = BayesianVehicle(
            vehicle_id=self.next_vehicle_id,
            initial_position=position,
            goal=goal,
            vehicle_type=vehicle_type
        )
        
        # Calculate initial distance to goal for progress tracking
        vehicle.initial_distance_to_goal = np.linalg.norm(
            np.array(position) - np.array(goal)
        )
        
        # Link vehicle to environment
        vehicle.env = self
        
        # Increment ID counter
        self.next_vehicle_id += 1
        
        # Add to vehicle list
        self.vehicles.append(vehicle)
        
        return vehicle
    
    def _initialize_beliefs(self):
        """Initialize beliefs for each vehicle about other vehicles' types."""
        for vehicle in self.vehicles:
            # Initialize beliefs about other vehicles
            vehicle.beliefs = {}
            
            for other in self.vehicles:
                if other.id != vehicle.id:
                    # Initialize with uniform distribution
                    vehicle.beliefs[other.id] = {
                        'standard': 1/3,
                        'premium': 1/3,
                        'emergency': 1/3
                    }
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.emergency_active = False
        self.metrics_history = []
        self.belief_history = defaultdict(list)
        self._initialize_fleet()
        
        # Initial observation
        return self._get_observation()
    
    def step(self):
        """
        Run one step of the simulation.
        
        Returns:
            dict: Observation of the current state
            dict: Metrics for the current step
            bool: Done flag
            dict: Additional info
        """
        # Check if emergency scenario should activate
        if (self.config['emergency_scenario'] and 
            self.current_step == self.config['emergency_start_step']):
            self._activate_emergency()
        
        # Update communication graph based on positions and capabilities
        self._update_communication_graph()
        
        # Perform communication between vehicles
        self._perform_communication()
        
        # Record belief state before updates
        for vehicle in self.vehicles:
            if hasattr(vehicle, 'beliefs'):
                # Make a deep copy of beliefs
                beliefs_copy = {}
                for other_id, type_beliefs in vehicle.beliefs.items():
                    beliefs_copy[other_id] = type_beliefs.copy()
                
                self.belief_history[vehicle.id].append(beliefs_copy)
        
        # Compute Bayesian game state
        game_state = self._compute_game_state()
        
        # Find Bayesian Nash equilibria
        equilibria = self.game_analyzer.find_bayesian_nash_equilibria(game_state)
        
        # Update vehicles with equilibria
        for vehicle in self.vehicles:
            vehicle.bayesian_equilibria = equilibria
        
        # Let each vehicle decide its action
        for vehicle in self.vehicles:
            # Get state and belief state for this vehicle
            state = self._get_vehicle_state(vehicle)
            belief_state = vehicle.beliefs
            
            # Decide action using Bayesian reasoning
            action = vehicle.decide_action(state, belief_state)
            
            # Apply action
            vehicle.update_state(action)
            
            # Update beliefs based on observations
            self._update_vehicle_beliefs(vehicle)
        
        # Calculate metrics
        metrics = calculate_fleet_metrics(
            self.vehicles, 
            self.current_step, 
            self.communication_graph
        )
        
        # Store metrics for analysis
        self.metrics_history.append(metrics)
        
        # Increment step counter
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.config['max_steps']
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'belief_history': self.belief_history,
            'metrics_history': self.metrics_history
        }
        
        return observation, metrics, done, info
    
    def _activate_emergency(self):
        """Activate emergency scenario for emergency vehicles."""
        self.emergency_active = True
        
        # Find emergency vehicles
        for vehicle in self.vehicles:
            if vehicle.vehicle_type == 'emergency':
                vehicle.emergency_active = True
                vehicle.emergency_start_time = self.current_step
                
                # Set high priority destination
                # For simplicity, choose a random position
                x = random.uniform(0, self.config['grid_size'])
                y = random.uniform(0, self.config['grid_size'])
                
                # Update goal
                old_goal = vehicle.goal
                vehicle.goal = (x, y)
                
                # Update initial distance for new goal
                vehicle.initial_distance_to_goal = np.linalg.norm(
                    np.array(vehicle.position) - np.array(vehicle.goal)
                )
                
                # Broadcast emergency status to all vehicles
                for other in self.vehicles:
                    if other.id != vehicle.id:
                        other.receive_emergency_notification(vehicle.id, vehicle.goal)
    
    def _update_communication_graph(self):
        """Update the communication graph based on vehicle positions and capabilities."""
        # Reset graph edges (keep nodes)
        self.communication_graph.clear_edges()
        
        # For each pair of vehicles, check if communication is possible
        for v1 in self.vehicles:
            for v2 in self.vehicles:
                if v1.id != v2.id:
                    # Calculate distance between vehicles
                    distance = np.linalg.norm(
                        np.array(v1.position) - np.array(v2.position)
                    )
                    
                    # Check if v1 can communicate with v2
                    if distance <= v1.sensor_range:
                        # Add edge with weight based on distance and reliability
                        # Higher weight = less reliable communication
                        weight = distance * (1 - v1.comm_reliability)
                        self.communication_graph.add_edge(v1.id, v2.id, weight=weight)
    
    def _perform_communication(self):
        """Have vehicles communicate with each other based on the communication graph."""
        # Each vehicle communicates with vehicles it has edges to
        for vehicle in self.vehicles:
            # Get list of vehicles this vehicle can communicate with
            neighbors = []
            
            for edge in self.communication_graph.out_edges(vehicle.id):
                _, target_id = edge
                target_vehicle = next(v for v in self.vehicles if v.id == target_id)
                neighbors.append(target_vehicle)
            
            # Perform communication
            if neighbors:
                vehicle.communicate(neighbors)
    
    def _update_vehicle_beliefs(self, vehicle):
        """
        Update vehicle's beliefs based on observations of other vehicles.
        
        Args:
            vehicle (BayesianVehicle): Vehicle to update beliefs for
        """
        # For each other vehicle in observation range
        for other in self.vehicles:
            if other.id != vehicle.id:
                # Calculate distance
                distance = np.linalg.norm(
                    np.array(vehicle.position) - np.array(other.position)
                )
                
                # Check if other vehicle is within sensor range
                if distance <= vehicle.sensor_range:
                    # Create observation
                    observation = {
                        'position': other.position,
                        'velocity': other.velocity,
                        'acceleration': other.last_action[0] if hasattr(other, 'last_action') else 0,
                        'steering': other.last_action[1] if hasattr(other, 'last_action') else 0
                    }
                    
                    # Update belief based on observation
                    vehicle.update_belief(other.id, observation)
    
    def _compute_game_state(self):
        """
        Compute the current Bayesian game state.
        
        Returns:
            dict: Game state representation
        """
        # Construct game state
        game_state = {
            'vehicles': self.vehicles,
            'type_space': {},
            'action_space': {},
            'beliefs': {}
        }
        
        # Fill in type space
        for vehicle in self.vehicles:
            game_state['type_space'][vehicle.id] = ['standard', 'premium', 'emergency']
        
        # Fill in action space (simplified discrete actions)
        for vehicle in self.vehicles:
            game_state['action_space'][vehicle.id] = [0, 1, 2, 3, 4]  # 5 discrete actions
        
        # Fill in beliefs
        for vehicle in self.vehicles:
            game_state['beliefs'][vehicle.id] = vehicle.beliefs
        
        return game_state
    
    def _get_vehicle_state(self, vehicle):
        """
        Get the current state for a specific vehicle.
        
        Args:
            vehicle (BayesianVehicle): The vehicle to get state for
            
        Returns:
            dict: State information
        """
        # Get positions and velocities of nearby vehicles
        nearby_vehicles = []
        
        for other in self.vehicles:
            if other.id != vehicle.id:
                # Calculate distance
                distance = np.linalg.norm(
                    np.array(vehicle.position) - np.array(other.position)
                )
                
                # Include if within sensor range
                if distance <= vehicle.sensor_range:
                    nearby_vehicles.append({
                        'id': other.id,
                        'position': other.position,
                        'velocity': other.velocity,
                        'distance': distance
                    })
        
        # Create state dictionary
        state = {
            'position': vehicle.position,
            'velocity': vehicle.velocity,
            'goal': vehicle.goal,
            'nearby_vehicles': nearby_vehicles,
            'emergency_active': self.emergency_active,
            'time_step': self.current_step
        }
        
        return state
    
    def _get_observation(self):
        """
        Get full observation of the environment.
        
        Returns:
            dict: Observation dictionary
        """
        # Create observation dictionary with all vehicle states
        observation = {
            'vehicles': [],
            'emergency_active': self.emergency_active,
            'time_step': self.current_step
        }
        
        # Add vehicle states
        for vehicle in self.vehicles:
            vehicle_obs = {
                'id': vehicle.id,
                'position': vehicle.position,
                'velocity': vehicle.velocity,
                'goal': vehicle.goal,
                'type': vehicle.vehicle_type,
                'emergency_active': getattr(vehicle, 'emergency_active', False)
            }
            observation['vehicles'].append(vehicle_obs)
        
        return observation
    
    def render(self):
        """Render the current state of the environment."""
        if not self.config['visualization']:
            return
        
        # Only render at specified frequency
        if self.current_step % self.config['visualization_frequency'] != 0:
            return
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Draw city grid
        draw_city_grid(
            self.screen, 
            self.config['grid_size'], 
            self.config['block_size']
        )
        
        # Draw vehicles
        draw_vehicles(self.screen, self.vehicles)
        
        # Draw belief states
        draw_belief_states(self.screen, self.vehicles)
        
        # Get current metrics
        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
        else:
            current_metrics = {}
        
        # Create or update belief heatmap
        if self.belief_heatmap is None or self.current_step % self.update_heatmap_frequency == 0:
            self.belief_heatmap = create_belief_heatmap(self.vehicles, self.config['grid_size'])
        
        # Display belief heatmap in bottom right corner if available
        if self.belief_heatmap:
            heatmap_width, heatmap_height = 300, 225  # Fixed size for heatmap
            heatmap_x = self.config['screen_width'] - 320  # Position in bottom right
            heatmap_y = self.config['screen_height'] - 245
            
            # Draw background for heatmap
            pygame.draw.rect(
                self.screen, 
                (240, 240, 240), 
                (heatmap_x - 10, heatmap_y - 10, heatmap_width + 20, heatmap_height + 20)
            )
            
            # Draw title
            font = pygame.font.SysFont('Arial', 16, bold=True)
            title_surface = font.render("Belief Distribution", True, (0, 0, 0))
            self.screen.blit(title_surface, (heatmap_x, heatmap_y - 30))
            
            # Resize and display heatmap
            heatmap_scaled = pygame.transform.scale(self.belief_heatmap, (heatmap_width, heatmap_height))
            self.screen.blit(heatmap_scaled, (heatmap_x, heatmap_y))
        
        # Draw info panel
        draw_info_panel(self.screen, current_metrics, self.emergency_active)
        
        # Draw information flow
        visualize_information_flow(
            self.screen,
            self.vehicles,
            self.communication_graph
        )

        # Add a legend for vehicle types at the top
        self._draw_legend()
        
        # Draw explanation overlay
        self._draw_explanation_overlay()
        
        # Draw debug mode indicator
        if self.debug_mode:
            self._draw_debug_mode_indicator()
        
        # Update display
        pygame.display.flip()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                # Toggle debug mode with 'D' key
                if event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode {'ON' if self.debug_mode else 'OFF'}")
                # Step in debug mode with Space key
                elif event.key == pygame.K_SPACE and self.debug_mode:
                    self.step_requested = True
        
        # Control rendering speed
        self.clock.tick(10)  # 10 FPS
        
        return True

    def _draw_legend(self):
        """Draw a legend explaining the different vehicle types and symbols."""
        legend_x = 10
        legend_y = 10
        font = pygame.font.SysFont('Arial', 14)
        title_font = pygame.font.SysFont('Arial', 16, bold=True)
        
        # Draw legend background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (legend_x, legend_y, 260, 140),
            border_radius=5
        )
        
        # Draw title
        title_surface = title_font.render("Legend", True, (0, 0, 0))
        self.screen.blit(title_surface, (legend_x + 10, legend_y + 10))
        
        # Draw vehicle type examples
        y_offset = 40
        
        # Standard vehicle
        pygame.draw.circle(self.screen, (0, 0, 255), (legend_x + 20, legend_y + y_offset), 8)
        text_surface = font.render("Standard Vehicle", True, (0, 0, 0))
        self.screen.blit(text_surface, (legend_x + 40, legend_y + y_offset - 8))
        
        # Premium vehicle
        pygame.draw.circle(self.screen, (0, 255, 0), (legend_x + 20, legend_y + y_offset + 25), 8)
        text_surface = font.render("Premium Vehicle", True, (0, 0, 0))
        self.screen.blit(text_surface, (legend_x + 40, legend_y + y_offset + 17))
        
        # Emergency vehicle
        pygame.draw.circle(self.screen, (255, 0, 0), (legend_x + 20, legend_y + y_offset + 50), 8)
        pygame.draw.circle(self.screen, (255, 0, 0), (legend_x + 20, legend_y + y_offset + 50), 12, 2)
        text_surface = font.render("Emergency Vehicle", True, (0, 0, 0))
        self.screen.blit(text_surface, (legend_x + 40, legend_y + y_offset + 42))
        
        # Goal indicator
        pygame.draw.circle(self.screen, (0, 0, 255), (legend_x + 20, legend_y + y_offset + 75), 3, 1)
        pygame.draw.line(self.screen, (0, 0, 255), (legend_x + 20, legend_y + y_offset + 75), (legend_x + 35, legend_y + y_offset + 75), 1)
        text_surface = font.render("Goal & Path", True, (0, 0, 0))
        self.screen.blit(text_surface, (legend_x + 40, legend_y + y_offset + 67))

    def _draw_explanation_overlay(self):
        """Draw an explanation overlay to help users understand the simulation."""
        # Calculate position for the explanation box
        box_width = 300
        box_height = 350
        
        # Position in the bottom left corner
        box_x = 10
        box_y = self.config['screen_height'] - box_height - 10
        
        # Draw semi-transparent background
        explanation_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        explanation_surface.fill((240, 240, 255, 200))  # Light blue with transparency
        
        # Create fonts
        title_font = pygame.font.SysFont('Arial', 16, bold=True)
        text_font = pygame.font.SysFont('Arial', 12)
        
        # Draw title
        title_surface = title_font.render("Fleet Coordination Explanation", True, (0, 0, 0))
        explanation_surface.blit(title_surface, (10, 10))
        
        # Draw explanation text
        explanation_text = [
            "Bayesian Game Theory in Action:",
            "",
            "1. Vehicles with different types (blue=standard,",
            "   green=premium, red=emergency) navigate to goals.",
            "",
            "2. Vehicles have beliefs about others' types shown",
            "   as small pie charts near their position. Colors",
            "   indicate belief probabilities for each type.",
            "",
            "3. Dashed lines show information flow between",
            "   vehicles (V2V communication). Line color indicates",
            "   the sending vehicle's type.",
            "",
            "4. Vehicles update beliefs based on observations",
            "   of others' behavior and communication.",
            "",
            "5. The belief heatmap (bottom right) shows the",
            "   spatial distribution of vehicle type beliefs.",
            "",
            "6. Emergency scenario activates mid-simulation,",
            "   requiring other vehicles to adapt their behavior.",
            "",
            "7. Performance metrics (right panel) measure",
            "   overall fleet coordination effectiveness.",
            "",
            f"Current step: {self.current_step} / {self.config['max_steps']}",
            f"Emergency: {'Active' if self.emergency_active else 'Not active'}"
        ]
        
        y_pos = 40
        for line in explanation_text:
            text_surface = text_font.render(line, True, (0, 0, 40))
            explanation_surface.blit(text_surface, (10, y_pos))
            y_pos += 20
        
        # Draw a border
        pygame.draw.rect(explanation_surface, (0, 0, 100), 
                         (0, 0, box_width, box_height), 2, border_radius=5)
        
        # Blit the explanation surface to the main screen
        self.screen.blit(explanation_surface, (box_x, box_y))

    def _draw_debug_mode_indicator(self):
        """Draw an indicator showing debug mode is active and how to step"""
        # Create a semi-transparent overlay at the top
        debug_surface = pygame.Surface((300, 40), pygame.SRCALPHA)
        debug_surface.fill((255, 200, 200, 200))  # Light red with transparency
        
        # Add text
        font = pygame.font.SysFont('Arial', 16, bold=True)
        text1 = font.render("DEBUG MODE", True, (200, 0, 0))
        text2 = font.render("Press SPACE to step", True, (0, 0, 0))
        
        debug_surface.blit(text1, (10, 5))
        debug_surface.blit(text2, (10, 22))
        
        # Draw border
        pygame.draw.rect(debug_surface, (200, 0, 0), (0, 0, 300, 40), 2, border_radius=5)
        
        # Position at top center of screen
        pos_x = (self.config['screen_width'] - 300) // 2
        self.screen.blit(debug_surface, (pos_x, 10))

    def close(self):
        """Clean up resources."""
        if self.config['visualization']:
            pygame.quit()
    
    def analyze_results(self):
        """
        Analyze simulation results and generate plots.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        # Analyze belief convergence
        results['belief_convergence'] = analyze_belief_convergence(self.belief_history)
        
        # Create fleet performance plot
        fleet_performance_fig = plot_fleet_performance(self.metrics_history)
        results['fleet_performance_fig'] = fleet_performance_fig
        
        # Final metrics
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            results['final_metrics'] = final_metrics
        
        return results


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Fleet Coordination Simulation")
    parser.add_argument('--no-viz', action='store_true', help="Disable visualization")
    parser.add_argument('--steps', type=int, default=500, help="Number of simulation steps")
    parser.add_argument('--standard', type=int, default=3, help="Number of standard vehicles")
    parser.add_argument('--premium', type=int, default=1, help="Number of premium vehicles")
    parser.add_argument('--emergency', type=int, default=1, help="Number of emergency vehicles")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--debug', action='store_true', help="Start in debug mode")
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Configure environment
    config = {
        'visualization': not args.no_viz,
        'max_steps': args.steps,
        'num_standard_vehicles': args.standard,
        'num_premium_vehicles': args.premium,
        'num_emergency_vehicles': args.emergency
    }
    
    # Create environment
    env = FleetCoordinationEnvironment(config)
    
    # Set initial debug mode if provided as argument
    if args.debug:
        env.debug_mode = True
        print("Debug mode enabled - press SPACE to step through simulation")
    
    # Reset environment
    observation = env.reset()
    
    # Run simulation
    done = False
    while not done:
        # In debug mode, wait for step request
        if env.debug_mode:
            # Always render in debug mode
            if not env.render():
                break
                
            # Wait for step request
            if not env.step_requested:
                # Delay to prevent hogging CPU while waiting
                time.sleep(0.1)
                continue
            else:
                # Reset step request flag
                env.step_requested = False
        
        # Step environment
        observation, metrics, done, info = env.step()
        
        # Render
        if env.config['visualization']:
            if not env.render():
                break
    
    # Analyze results
    results = env.analyze_results()
    
    # Print some final metrics
    if 'final_metrics' in results:
        print("Final Metrics:")
        for key, value in results['final_metrics'].items():
            print(f"  {key}: {value}")
    
    # Clean up
    env.close()
    
    # Show fleet performance plot
    if 'fleet_performance_fig' in results:
        plt.show()

if __name__ == "__main__":
    main()