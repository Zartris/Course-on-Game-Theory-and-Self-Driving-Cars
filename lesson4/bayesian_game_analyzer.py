#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian Game Analyzer for autonomous vehicle fleet coordination.

This module provides tools for analyzing Bayesian games, finding Bayesian Nash
equilibria, and computing the value of information in autonomous vehicle coordination.
"""

import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt
import itertools
from scipy.spatial import ConvexHull

class BayesianGameAnalyzer:
    """
    Tools for analyzing Bayesian games and computing equilibria in the context
    of autonomous vehicle fleet coordination.
    """
    
    def __init__(self):
        """Initialize the Bayesian Game Analyzer."""
        pass
    
    def compute_expected_utility(self, action_profile, type_distribution):
        """
        Compute expected utility given an action profile and type distribution.
        
        Args:
            action_profile (dict): Mapping from player IDs to actions
            type_distribution (dict): Joint probability distribution over types
            
        Returns:
            dict: Expected utility for each player
        """
        # Extract players
        players = list(action_profile.keys())
        
        # Initialize expected utilities
        expected_utilities = {player_id: 0.0 for player_id in players}
        
        # Iterate over possible type combinations
        for type_profile, probability in type_distribution.items():
            # Calculate utility for each player given this type profile
            for player_id in players:
                # In a real implementation, this would call a utility function
                # that depends on actions and types
                utility = self._compute_utility(
                    player_id, 
                    action_profile, 
                    type_profile
                )
                
                # Accumulate weighted by type probability
                expected_utilities[player_id] += probability * utility
        
        return expected_utilities
    
    def _compute_utility(self, player_id, action_profile, type_profile):
        """
        Compute utility for a specific player given actions and types.
        
        Args:
            player_id: ID of the player
            action_profile (dict): Actions for all players
            type_profile (dict): Types of all players
            
        Returns:
            float: Utility value
        """
        # This is a placeholder for a utility function
        # In a real implementation, this would depend on the specific game
        return 0.0
    
    def find_bayesian_nash_equilibria(self, game_state):
        """
        Find Bayesian Nash equilibria for a given game state.
        
        Args:
            game_state (dict): Game state representation, including:
                - vehicles: List of vehicle objects
                - type_space: Dictionary mapping player IDs to possible types
                - action_space: Dictionary mapping player IDs to possible actions
                - beliefs: Dictionary of beliefs for each player about others
                
        Returns:
            list: List of equilibria, where each equilibrium is a dictionary with:
                - strategies: Dictionary mapping player IDs to strategies
                - utilities: Expected utilities for each player
        """
        # Simple case: Check if there are exactly 2 vehicles
        # If so, we can use the Nashpy library directly
        if len(game_state['vehicles']) == 2:
            vehicle1, vehicle2 = game_state['vehicles'][0], game_state['vehicles'][1]
            return self._find_2player_bayesian_nash(game_state, vehicle1.id, vehicle2.id)
        
        # For more than 2 vehicles, we would need more complex methods
        # This is a simplified placeholder that returns a heuristic equilibrium
        elif len(game_state['vehicles']) > 2:
            return self._find_multiplayer_heuristic_equilibrium(game_state)
        
        # No vehicles, no equilibria
        return []
    
    def _find_2player_bayesian_nash(self, game_state, player1_id, player2_id):
        """
        Find Bayesian Nash equilibria for a 2-player game using Nashpy.
        
        Args:
            game_state: Game state representation
            player1_id: ID of first player
            player2_id: ID of second player
            
        Returns:
            list: List of equilibria
        """
        # Extract action spaces
        action_space1 = game_state['action_space'][player1_id]
        action_space2 = game_state['action_space'][player2_id]
        
        # Extract type spaces
        type_space1 = game_state['type_space'][player1_id]
        type_space2 = game_state['type_space'][player2_id]
        
        # Extract beliefs
        beliefs1 = game_state['beliefs'][player1_id]
        beliefs2 = game_state['beliefs'][player2_id]
        
        # Compute expected payoff matrices
        # This is highly simplified and would be more complex in a real implementation
        payoff_matrix1 = np.zeros((len(action_space1), len(action_space2)))
        payoff_matrix2 = np.zeros((len(action_space1), len(action_space2)))
        
        # Populate matrices based on expected utilities
        for i, action1 in enumerate(action_space1):
            for j, action2 in enumerate(action_space2):
                # Expected utility for player 1
                utility1 = 0.0
                for type1 in type_space1:
                    for type2 in type_space2:
                        # Get probability of this type combination
                        # In a proper implementation, this would use the beliefs
                        prob = 1.0 / (len(type_space1) * len(type_space2))
                        
                        # Get utility for this scenario
                        scenario_utility1 = self._scenario_utility(
                            player1_id, action1, type1, 
                            player2_id, action2, type2,
                            game_state
                        )
                        
                        # Weight by probability
                        utility1 += prob * scenario_utility1
                
                # Expected utility for player 2
                utility2 = 0.0
                for type1 in type_space1:
                    for type2 in type_space2:
                        # Get probability of this type combination
                        prob = 1.0 / (len(type_space1) * len(type_space2))
                        
                        # Get utility for this scenario
                        scenario_utility2 = self._scenario_utility(
                            player2_id, action2, type2,
                            player1_id, action1, type1,
                            game_state
                        )
                        
                        # Weight by probability
                        utility2 += prob * scenario_utility2
                
                # Store in matrices
                payoff_matrix1[i, j] = utility1
                payoff_matrix2[i, j] = utility2
        
        # Create game using Nashpy
        game = nash.Game(payoff_matrix1, payoff_matrix2)
        
        # Compute equilibria
        equilibria = list(game.support_enumeration())
        
        # Convert to our equilibrium format
        result = []
        for eq in equilibria:
            # Extract strategies
            strategy1, strategy2 = eq
            
            # Create strategy dictionary
            strategies = {
                player1_id: strategy1,
                player2_id: strategy2
            }
            
            # Calculate expected utilities
            utilities = {
                player1_id: strategy1 @ payoff_matrix1 @ strategy2,
                player2_id: strategy1 @ payoff_matrix2 @ strategy2
            }
            
            # Add to results
            result.append({
                'strategies': strategies,
                'utilities': utilities
            })
        
        return result
    
    def _scenario_utility(self, player_id, action, player_type, 
                         other_id, other_action, other_type, game_state):
        """
        Calculate utility for a specific scenario.
        
        Args:
            player_id: ID of the player
            action: Action of the player
            player_type: Type of the player
            other_id: ID of the other player
            other_action: Action of the other player
            other_type: Type of the other player
            game_state: Game state representation
            
        Returns:
            float: Utility value
        """
        # This is a simplified placeholder for a utility calculation
        # In a real implementation, this would depend on the specific game
        
        # Find the vehicle objects
        player_vehicle = None
        other_vehicle = None
        for vehicle in game_state['vehicles']:
            if vehicle.id == player_id:
                player_vehicle = vehicle
            elif vehicle.id == other_id:
                other_vehicle = vehicle
        
        if player_vehicle is None or other_vehicle is None:
            return 0.0
        
        # Calculate distance between vehicles
        player_pos = player_vehicle.position
        other_pos = other_vehicle.position
        distance = np.linalg.norm(np.array(player_pos) - np.array(other_pos))
        
        # Basic utility components
        goal_progress = 0.5  # Placeholder
        safety = max(0.0, min(1.0, distance / 10.0))  # Higher distance = safer
        
        # Type-dependent weights
        if player_type == 'standard':
            weights = {'goal_progress': 0.5, 'safety': 0.5}
        elif player_type == 'premium':
            weights = {'goal_progress': 0.7, 'safety': 0.3}
        elif player_type == 'emergency':
            weights = {'goal_progress': 0.9, 'safety': 0.1}
        else:
            weights = {'goal_progress': 0.5, 'safety': 0.5}
        
        # Action-dependent modifications
        if action == 0:  # Maintain
            pass  # No change
        elif action == 1:  # Accelerate
            goal_progress += 0.2
        elif action == 2:  # Decelerate
            goal_progress -= 0.1
            safety += 0.2
        elif action == 3:  # Turn left
            goal_progress -= 0.1
        elif action == 4:  # Turn right
            goal_progress -= 0.1
        
        # Combine components
        utility = (
            weights['goal_progress'] * goal_progress +
            weights['safety'] * safety
        )
        
        return utility
    
    def _find_multiplayer_heuristic_equilibrium(self, game_state):
        """
        Find a heuristic equilibrium for games with more than 2 players.
        
        Args:
            game_state: Game state representation
            
        Returns:
            list: List containing the heuristic equilibrium
        """
        # This is a simplified placeholder that returns a basic equilibrium
        # In a real implementation, this would use more sophisticated methods
        
        # Extract vehicle IDs
        vehicle_ids = [v.id for v in game_state['vehicles']]
        
        # Create a simple heuristic strategy: all vehicles maintain current speed
        strategies = {}
        utilities = {}
        
        for vehicle_id in vehicle_ids:
            # Create a strategy that puts all probability on "maintain" action
            strategy = np.zeros(len(game_state['action_space'][vehicle_id]))
            strategy[0] = 1.0  # Assume 0 is "maintain"
            strategies[vehicle_id] = strategy
            
            # Placeholder utility
            utilities[vehicle_id] = 0.0
        
        return [{
            'strategies': strategies,
            'utilities': utilities
        }]
    
    def calculate_value_of_information(self, game, additional_info):
        """
        Calculate the value of additional information in a Bayesian game.
        
        Args:
            game (dict): Bayesian game representation
            additional_info (dict): Additional information to evaluate
            
        Returns:
            float: Value of the information (improvement in expected utility)
        """
        # Compute expected utility without additional information
        equilibria_before = self.find_bayesian_nash_equilibria(game)
        if not equilibria_before:
            return 0.0
            
        # Use the best equilibrium for each player
        utility_before = {}
        for player_id in game['beliefs'].keys():
            utility_before[player_id] = max(
                eq['utilities'].get(player_id, float('-inf')) 
                for eq in equilibria_before
            )
        
        # Create updated game with additional information
        updated_game = self._update_game_with_info(game, additional_info)
        
        # Compute expected utility with additional information
        equilibria_after = self.find_bayesian_nash_equilibria(updated_game)
        if not equilibria_after:
            return 0.0
            
        # Use the best equilibrium for each player
        utility_after = {}
        for player_id in updated_game['beliefs'].keys():
            utility_after[player_id] = max(
                eq['utilities'].get(player_id, float('-inf')) 
                for eq in equilibria_after
            )
        
        # Calculate improvement in utility
        value = {}
        for player_id in utility_before.keys():
            value[player_id] = utility_after.get(player_id, 0.0) - utility_before.get(player_id, 0.0)
        
        return value
    
    def _update_game_with_info(self, game, additional_info):
        """
        Update a game with additional information.
        
        Args:
            game (dict): Original game
            additional_info (dict): Additional information
            
        Returns:
            dict: Updated game
        """
        # Create a deep copy of the game
        updated_game = {
            'vehicles': game['vehicles'],
            'type_space': game['type_space'].copy(),
            'action_space': game['action_space'].copy(),
            'beliefs': {}
        }
        
        # Update beliefs based on additional information
        for player_id, beliefs in game['beliefs'].items():
            updated_beliefs = beliefs.copy()
            
            # If this player is the recipient of additional information
            if player_id in additional_info:
                info = additional_info[player_id]
                
                # Update beliefs about other players' types
                if 'observed_types' in info:
                    for other_id, observed_type in info['observed_types'].items():
                        if other_id in updated_beliefs:
                            # Update to be certain about this type
                            for type_name in updated_beliefs[other_id]:
                                updated_beliefs[other_id][type_name] = 1.0 if type_name == observed_type else 0.0
            
            updated_game['beliefs'][player_id] = updated_beliefs
        
        return updated_game
    
    def analyze_information_structure(self, game):
        """
        Analyze the information structure of a Bayesian game.
        
        Args:
            game (dict): Bayesian game representation
            
        Returns:
            dict: Analysis of information structure
        """
        # Number of players
        n_players = len(game['vehicles'])
        
        # Calculate entropy of beliefs for each player
        belief_entropies = {}
        for player_id, beliefs in game['beliefs'].items():
            player_entropy = {}
            
            for other_id, type_probs in beliefs.items():
                # Calculate entropy of belief about this player
                entropy = 0.0
                for prob in type_probs.values():
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
                player_entropy[other_id] = entropy
            
            belief_entropies[player_id] = player_entropy
        
        # Calculate information asymmetry
        asymmetry = {}
        for i, player1_id in enumerate(game['beliefs'].keys()):
            for j, player2_id in enumerate(game['beliefs'].keys()):
                if i < j:
                    # Calculate difference in beliefs
                    diff = 0.0
                    count = 0
                    
                    # Compare beliefs about each other
                    if player2_id in game['beliefs'][player1_id] and player1_id in game['beliefs'][player2_id]:
                        beliefs1 = game['beliefs'][player1_id][player2_id]
                        beliefs2 = game['beliefs'][player2_id][player1_id]
                        
                        # Sum up differences in probabilities
                        for type_name in set(beliefs1.keys()) | set(beliefs2.keys()):
                            prob1 = beliefs1.get(type_name, 0.0)
                            prob2 = beliefs2.get(type_name, 0.0)
                            diff += abs(prob1 - prob2)
                            count += 1
                        
                        if count > 0:
                            asymmetry[(player1_id, player2_id)] = diff / count
        
        return {
            'belief_entropy': belief_entropies,
            'information_asymmetry': asymmetry
        }
    
    def visualize_belief_updates(self, belief_history):
        """
        Visualize how beliefs have been updated over time.
        
        Args:
            belief_history (dict): History of belief updates
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with belief visualizations
        """
        # Number of players and time steps
        players = list(belief_history.keys())
        if not players:
            return plt.figure()
            
        times = list(range(len(belief_history[players[0]])))
        if not times:
            return plt.figure()
        
        # Create figure
        fig, axes = plt.subplots(len(players), 1, figsize=(10, 3*len(players)))
        if len(players) == 1:
            axes = [axes]
        
        # For each player, plot belief evolution
        for i, player_id in enumerate(players):
            ax = axes[i]
            history = belief_history[player_id]
            
            # For each time step, plot beliefs
            for t, beliefs in enumerate(history):
                # For each type, plot belief probability
                for type_name, prob in beliefs.items():
                    ax.scatter(t, prob, label=f"{type_name}" if t == 0 else "")
                    
                    # Connect with lines
                    if t > 0:
                        prev_prob = history[t-1].get(type_name, 0.0)
                        ax.plot([t-1, t], [prev_prob, prob], '-')
            
            ax.set_title(f"Player {player_id} Belief Evolution")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief Probability")
            ax.grid(True)
            ax.legend()
        
        fig.tight_layout()
        return fig
    
    def compute_efficiency_loss(self, full_info_outcome, partial_info_outcome):
        """
        Compute the efficiency loss due to incomplete information.
        
        Args:
            full_info_outcome (dict): Outcome with full information
            partial_info_outcome (dict): Outcome with partial information
            
        Returns:
            dict: Efficiency loss metrics
        """
        # Extract utilities
        utilities_full = full_info_outcome.get('utilities', {})
        utilities_partial = partial_info_outcome.get('utilities', {})
        
        # Calculate absolute and relative losses
        absolute_loss = {}
        relative_loss = {}
        
        for player_id in set(utilities_full.keys()) | set(utilities_partial.keys()):
            util_full = utilities_full.get(player_id, 0.0)
            util_partial = utilities_partial.get(player_id, 0.0)
            
            # Calculate losses
            abs_loss = util_full - util_partial
            absolute_loss[player_id] = abs_loss
            
            if util_full != 0:
                relative_loss[player_id] = abs_loss / abs(util_full)
            else:
                relative_loss[player_id] = 0.0
        
        # Calculate total loss
        total_abs_loss = sum(absolute_loss.values())
        
        # Calculate Price of Anarchy (PoA)
        # PoA = social welfare of optimal outcome / social welfare of Nash equilibrium
        social_welfare_full = sum(utilities_full.values())
        social_welfare_partial = sum(utilities_partial.values())
        
        if social_welfare_partial > 0:
            poa = social_welfare_full / social_welfare_partial
        else:
            poa = float('inf') if social_welfare_full > 0 else 1.0
        
        return {
            'absolute_loss': absolute_loss,
            'relative_loss': relative_loss,
            'total_absolute_loss': total_abs_loss,
            'price_of_anarchy': poa
        }