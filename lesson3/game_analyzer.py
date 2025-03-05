#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Game theory analysis tools for the Roundabout Traffic Management simulation.

This module provides tools for analyzing game states, computing Pareto frontiers,
and visualizing game theory concepts in the context of traffic management.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import nashpy as nash

class GameAnalyzer:
    """
    A class for analyzing game states and computing various game theory metrics
    in the context of traffic management.
    """
    
    def __init__(self):
        """Initialize the game analyzer."""
        self.pareto_history = []
        self.nash_history = []
    
    def find_pareto_optimal_solutions(self, points):
        """
        Find Pareto optimal points from a set of points representing utilities.
        
        Args:
            points (numpy.ndarray): Array of points, each representing utilities
                                  [utility1, utility2] for different objectives
                                  
        Returns:
            numpy.ndarray: Boolean mask indicating which points are Pareto optimal
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([], dtype=bool)
        elif n_points == 1:
            return np.array([True], dtype=bool)
        
        # Initialize all points as Pareto optimal
        is_pareto = np.ones(n_points, dtype=bool)
        
        # Compare each point with every other point
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        is_pareto[i] = False
                        break
        
        # Store for historical analysis
        self.pareto_history.append({
            'points': points.copy(),
            'pareto_mask': is_pareto.copy()
        })
        
        return is_pareto
    
    def plot_pareto_frontier(self, points, point_ids=None, show_dominated=True):
        """
        Plot the Pareto frontier for a set of points.
        
        Args:
            points (numpy.ndarray): Array of points to plot
            point_ids (list, optional): List of identifiers for the points
            show_dominated (bool): Whether to show dominated points
            
        Returns:
            matplotlib.figure.Figure: The figure object with the plot
        """
        if len(points) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Find Pareto optimal points
        is_pareto = self.find_pareto_optimal_solutions(points)
        
        # Plot all points if requested
        if show_dominated:
            dominated_points = points[~is_pareto]
            ax.scatter(
                dominated_points[:, 0],
                dominated_points[:, 1],
                c='gray',
                alpha=0.5,
                label='Dominated Solutions'
            )
        
        # Plot Pareto optimal points
        pareto_points = points[is_pareto]
        ax.scatter(
            pareto_points[:, 0],
            pareto_points[:, 1],
            c='red',
            label='Pareto Optimal'
        )
        
        # Add point labels if provided
        if point_ids is not None:
            for i, (x, y) in enumerate(points):
                if is_pareto[i] or show_dominated:
                    ax.annotate(
                        f'Vehicle {point_ids[i]}',
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
        
        # Plot Pareto frontier line
        if len(pareto_points) > 1:
            # Sort points by x-coordinate
            sorted_indices = np.argsort(pareto_points[:, 0])
            sorted_points = pareto_points[sorted_indices]
            
            # Plot line connecting Pareto optimal points
            ax.plot(
                sorted_points[:, 0],
                sorted_points[:, 1],
                'r--',
                label='Pareto Frontier'
            )
        
        # Customize plot
        ax.set_xlabel('Travel Time Utility (-)')
        ax.set_ylabel('Safety Margin')
        ax.set_title('Pareto Frontier of Vehicle Strategies')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def analyze_nash_equilibria(self, payoff_matrices):
        """
        Analyze Nash equilibria for a given game state.
        
        Args:
            payoff_matrices (dict): Dictionary containing payoff matrices for players
            
        Returns:
            tuple: (equilibria, is_pure) where equilibria is a list of Nash equilibria
                  and is_pure indicates whether they are pure strategy equilibria
        """
        try:
            # Create the game using Nashpy
            game = nash.Game(
                payoff_matrices['payoff_matrix_self'],
                payoff_matrices['payoff_matrix_other']
            )
            
            # Find pure strategy Nash equilibria
            pure_equilibria = list(game.support_enumeration())
            
            # Check if we found any pure strategy equilibria
            if pure_equilibria:
                return pure_equilibria, True
            
            # If no pure strategy equilibria, find mixed strategy equilibria
            mixed_equilibria = list(game.vertex_enumeration())
            
            # Store for historical analysis
            self.nash_history.append({
                'payoff_matrices': payoff_matrices,
                'pure_equilibria': pure_equilibria,
                'mixed_equilibria': mixed_equilibria
            })
            
            return mixed_equilibria, False
            
        except Exception as e:
            print(f"Error computing Nash equilibria: {e}")
            return [], False
    
    def compute_best_response_dynamics(self, initial_strategies, payoff_matrices, max_iterations=100):
        """
        Compute best response dynamics starting from given initial strategies.
        
        Args:
            initial_strategies (tuple): Initial strategies for both players
            payoff_matrices (dict): Dictionary containing payoff matrices
            max_iterations (int): Maximum number of iterations
            
        Returns:
            list: Sequence of strategy profiles visited during best response dynamics
        """
        current_strategies = list(initial_strategies)
        strategy_sequence = [current_strategies.copy()]
        
        for _ in range(max_iterations):
            old_strategies = current_strategies.copy()
            
            # Update player 1's strategy
            player1_payoffs = payoff_matrices['payoff_matrix_self']
            best_response1 = np.argmax(player1_payoffs[:, int(old_strategies[1])])
            current_strategies[0] = best_response1
            
            # Update player 2's strategy
            player2_payoffs = payoff_matrices['payoff_matrix_other']
            best_response2 = np.argmax(player2_payoffs[:, int(old_strategies[0])])
            current_strategies[1] = best_response2
            
            strategy_sequence.append(current_strategies.copy())
            
            # Check for convergence
            if (current_strategies == old_strategies).all():
                break
        
        return strategy_sequence
    
    def plot_best_response_dynamics(self, strategy_sequence):
        """
        Plot the sequence of strategies visited during best response dynamics.
        
        Args:
            strategy_sequence (list): List of strategy profiles
            
        Returns:
            matplotlib.figure.Figure: The figure object with the plot
        """
        strategy_array = np.array(strategy_sequence)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot strategy trajectories
        ax.plot(
            range(len(strategy_sequence)),
            strategy_array[:, 0],
            'b-o',
            label='Player 1 Strategy'
        )
        ax.plot(
            range(len(strategy_sequence)),
            strategy_array[:, 1],
            'r-o',
            label='Player 2 Strategy'
        )
        
        # Customize plot
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Strategy')
        ax.set_title('Best Response Dynamics')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def analyze_repeated_game_outcomes(self, interaction_history):
        """
        Analyze outcomes of repeated game interactions.
        
        Args:
            interaction_history (dict): Dictionary of interaction histories
            
        Returns:
            dict: Analysis results including cooperation rates and strategy adaptations
        """
        results = {
            'cooperation_rates': {},
            'collision_rates': {},
            'strategy_adaptations': {}
        }
        
        for pair_key, interactions in interaction_history.items():
            interactions_list = list(interactions)
            n_interactions = len(interactions_list)
            
            if n_interactions == 0:
                continue
            
            # Calculate cooperation rate
            cooperative_actions = sum(
                1 for interaction in interactions_list
                if interaction['type'] != 'collision'
                and (interaction['vehicle1']['action'] == 1 
                     or interaction['vehicle2']['action'] == 1)
            )
            results['cooperation_rates'][pair_key] = cooperative_actions / n_interactions
            
            # Calculate collision rate
            collisions = sum(
                1 for interaction in interactions_list
                if interaction['type'] == 'collision'
            )
            results['collision_rates'][pair_key] = collisions / n_interactions
            
            # Analyze strategy adaptations
            strategy_changes = {
                'vehicle1': self._analyze_strategy_changes(interactions_list, 'vehicle1'),
                'vehicle2': self._analyze_strategy_changes(interactions_list, 'vehicle2')
            }
            results['strategy_adaptations'][pair_key] = strategy_changes
        
        return results
    
    def _analyze_strategy_changes(self, interactions, vehicle_key):
        """Analyze how a vehicle's strategy changed over time."""
        action_sequence = [
            interaction[vehicle_key]['action']
            for interaction in interactions
        ]
        
        if len(action_sequence) < 2:
            return {'changes': 0, 'pattern': 'insufficient_data'}
        
        # Count strategy changes
        changes = sum(
            1 for i in range(1, len(action_sequence))
            if action_sequence[i] != action_sequence[i-1]
        )
        
        # Identify patterns
        if changes == 0:
            pattern = 'constant'
        elif all(action_sequence[i] == action_sequence[i-2] 
                for i in range(2, len(action_sequence))):
            pattern = 'alternating'
        elif changes / len(action_sequence) > 0.7:
            pattern = 'highly_variable'
        else:
            pattern = 'adaptive'
        
        return {
            'changes': changes,
            'pattern': pattern,
            'final_actions': action_sequence[-min(5, len(action_sequence)):]
        }