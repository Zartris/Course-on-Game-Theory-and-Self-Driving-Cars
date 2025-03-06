"""
Direct Reciprocity Strategies for Multi-Robot Cooperation

This module implements various direct reciprocity strategies for promoting
cooperation in multi-robot systems, including Tit-for-Tat, Generous Tit-for-Tat,
Win-Stay Lose-Shift, and memory-based approaches.
"""

import random
import numpy as np
from collections import defaultdict


def tit_for_tat(opponent_history, my_history):
    """
    Implement the Tit-for-Tat strategy.
    
    Args:
        opponent_history: List of opponent's past actions (0=defect, 1=cooperate)
        my_history: List of my past actions
    
    Returns:
        Next action (0=defect, 1=cooperate)
    """
    if not opponent_history:
        return 1  # Start with cooperation
    else:
        return opponent_history[-1]  # Copy opponent's last move


def generous_tit_for_tat(opponent_history, my_history, generosity=0.1):
    """
    Implement the Generous Tit-for-Tat strategy.
    
    Args:
        opponent_history: List of opponent's past actions
        my_history: List of my past actions
        generosity: Probability of cooperating after opponent defection
    
    Returns:
        Next action (0=defect, 1=cooperate)
    """
    if not opponent_history:
        return 1  # Start with cooperation
    
    if opponent_history[-1] == 1:  # If opponent cooperated
        return 1  # Cooperate
    else:  # If opponent defected
        if random.random() < generosity:
            return 1  # Occasionally forgive
        else:
            return 0  # Usually defect


def win_stay_lose_shift(opponent_history, my_history, payoff_matrix):
    """
    Implement the Win-Stay, Lose-Shift strategy.
    
    Args:
        opponent_history: List of opponent's past actions
        my_history: List of my past actions
        payoff_matrix: The game's payoff matrix
    
    Returns:
        Next action (0=defect, 1=cooperate)
    """
    if not opponent_history:
        return 1  # Start with cooperation
    
    # Get last actions
    my_last_action = my_history[-1]
    opponent_last_action = opponent_history[-1]
    
    # Calculate payoff from last round
    payoff = payoff_matrix[my_last_action][opponent_last_action]
    
    # Determine if it was a "win" or "lose"
    if (my_last_action == 1 and opponent_last_action == 1) or (my_last_action == 0 and opponent_last_action == 1):
        # Win: got R or T
        return my_last_action  # Stay with same action
    else:
        # Lose: got P or S
        return 1 - my_last_action  # Switch action


def memory_n_strategy(opponent_history, my_history, memory_length=3, strategy_table=None):
    """
    Implement a strategy based on the last n moves.
    
    Args:
        opponent_history: List of opponent's past actions
        my_history: List of my past actions
        memory_length: Number of past moves to consider
        strategy_table: Mapping from history patterns to actions
    
    Returns:
        Next action (0=defect, 1=cooperate)
    """
    if len(opponent_history) < memory_length:
        return 1  # Default to cooperation if not enough history
    
    # Create a key from recent history
    history_key = ""
    for i in range(memory_length):
        idx = -memory_length + i
        history_key += str(my_history[idx]) + str(opponent_history[idx])
    
    # Look up action in strategy table
    return strategy_table.get(history_key, 1)  # Default to cooperation if pattern not found


def calculate_cooperation_threshold(payoff_matrix):
    """
    Calculate the minimum discount factor needed for cooperation.
    
    Args:
        payoff_matrix: The game's payoff matrix [R, S, T, P]
    
    Returns:
        Minimum discount factor for cooperation
    """
    R, S, T, P = payoff_matrix
    return (T - R) / (T - P)


def error_tolerant_tit_for_tat(opponent_history, my_history, error_memory=3, forgiveness_threshold=0.3):
    """
    Implement an error-tolerant version of Tit-for-Tat.
    
    Args:
        opponent_history: List of opponent's past actions
        my_history: List of my past actions
        error_memory: Number of past moves to consider for error detection
        forgiveness_threshold: Threshold for forgiveness
    
    Returns:
        Next action (0=defect, 1=cooperate)
    """
    if not opponent_history:
        return 1  # Start with cooperation
    
    # Check if opponent's last move was defection
    if opponent_history[-1] == 0:
        # Look at recent history to detect if this might be an error
        history_length = min(error_memory, len(opponent_history) - 1)
        if history_length > 0:
            # Calculate cooperation rate in recent history
            recent_cooperation = sum(opponent_history[-history_length-1:-1]) / history_length
            
            # If opponent has been mostly cooperative, forgive this defection
            if recent_cooperation > forgiveness_threshold:
                return 1  # Forgive and cooperate
        
        # Otherwise, reciprocate the defection
        return 0
    else:
        # Reciprocate cooperation
        return 1


class RecognitionBasedCooperation:
    """Implement cooperation based on recognition of interaction partners."""
    
    def __init__(self, robot_id, strategy='tit_for_tat', payoff_matrix=None):
        """
        Initialize recognition-based cooperation.
        
        Args:
            robot_id: Unique identifier for this robot
            strategy: Strategy to use ('tit_for_tat', 'generous_tit_for_tat', 'win_stay_lose_shift')
            payoff_matrix: Payoff matrix for the game (required for win_stay_lose_shift)
        """
        self.robot_id = robot_id
        self.strategy = strategy
        self.payoff_matrix = payoff_matrix
        self.interaction_history = {}  # Maps partner IDs to interaction histories
    
    def decide_action(self, partner_id):
        """Decide whether to cooperate with a specific partner."""
        # If new partner, initialize history
        if partner_id not in self.interaction_history:
            self.interaction_history[partner_id] = {
                'my_actions': [],
                'their_actions': []
            }
            return 1  # Start with cooperation for new partners
        
        # Retrieve history with this partner
        history = self.interaction_history[partner_id]
        
        # Apply selected strategy
        if self.strategy == 'tit_for_tat':
            return tit_for_tat(history['their_actions'], history['my_actions'])
        elif self.strategy == 'generous_tit_for_tat':
            return generous_tit_for_tat(history['their_actions'], history['my_actions'])
        elif self.strategy == 'win_stay_lose_shift':
            if self.payoff_matrix is None:
                raise ValueError("Payoff matrix required for win_stay_lose_shift strategy")
            return win_stay_lose_shift(history['their_actions'], history['my_actions'], self.payoff_matrix)
        else:
            return 1  # Default to cooperation
    
    def update_history(self, partner_id, my_action, their_action):
        """Update interaction history after an interaction."""
        if partner_id not in self.interaction_history:
            self.interaction_history[partner_id] = {
                'my_actions': [],
                'their_actions': []
            }
        
        self.interaction_history[partner_id]['my_actions'].append(my_action)
        self.interaction_history[partner_id]['their_actions'].append(their_action)


class ResourceSharingRobot:
    """Robot that uses direct reciprocity for resource sharing."""
    
    def __init__(self, robot_id, resource_capacity=100):
        """
        Initialize resource sharing robot.
        
        Args:
            robot_id: Unique identifier for this robot
            resource_capacity: Maximum resource capacity
        """
        self.robot_id = robot_id
        self.resource_level = resource_capacity
        self.max_resource = resource_capacity
        self.sharing_history = {}  # Maps partner IDs to sharing history
    
    def request_resource(self, partner_id, amount):
        """Request resources from another robot."""
        # Calculate reciprocity score with this partner
        reciprocity_score = self.calculate_reciprocity(partner_id)
        
        # Make request with reciprocity information
        return {
            'requester_id': self.robot_id,
            'amount': amount,
            'reciprocity_score': reciprocity_score
        }
    
    def respond_to_request(self, request):
        """Decide whether to share resources based on reciprocity."""
        requester_id = request['requester_id']
        amount = request['amount']
        reciprocity_score = request['reciprocity_score']
        
        # Initialize history if new partner
        if requester_id not in self.sharing_history:
            self.sharing_history[requester_id] = {
                'given': 0,
                'received': 0
            }
        
        # Decide based on reciprocity and available resources
        available = self.resource_level - self.max_resource * 0.2  # Keep 20% reserve
        
        if available <= 0:
            # Cannot share if resources too low
            return 0
        
        if reciprocity_score >= 0.8:
            # High reciprocity: share fully
            share_amount = min(amount, available)
        elif reciprocity_score >= 0.5:
            # Medium reciprocity: share partially
            share_amount = min(amount * 0.7, available)
        elif reciprocity_score >= 0.2:
            # Low reciprocity: share minimally
            share_amount = min(amount * 0.3, available)
        else:
            # Very low or negative reciprocity: don't share
            share_amount = 0
        
        # Update resources and history
        if share_amount > 0:
            self.resource_level -= share_amount
            self.sharing_history[requester_id]['given'] += share_amount
        
        return share_amount
    
    def receive_resource(self, sender_id, amount):
        """Receive resources from another robot."""
        self.resource_level += amount
        
        # Update sharing history
        if sender_id not in self.sharing_history:
            self.sharing_history[sender_id] = {
                'given': 0,
                'received': 0
            }
        
        self.sharing_history[sender_id]['received'] += amount
    
    def calculate_reciprocity(self, partner_id):
        """Calculate reciprocity score with a partner."""
        if partner_id not in self.sharing_history:
            return 0  # Neutral score for new partners
        
        history = self.sharing_history[partner_id]
        
        # If no sharing has occurred yet
        if history['given'] == 0 and history['received'] == 0:
            return 0
        
        # If only received but never given
        if history['given'] == 0:
            return 1  # Maximum score (we owe them)
        
        # If only given but never received
        if history['received'] == 0:
            return -0.5  # Negative but not minimum (they owe us)
        
        # Calculate ratio of received to given
        ratio = history['received'] / history['given']
        
        # Convert to a score between -1 and 1
        if ratio >= 1:
            # They've given more than or equal to what we've given
            return min(ratio - 1, 1)  # Cap at 1
        else:
            # We've given more than we've received
            return max((ratio - 1) / 2, -1)  # Cap at -1, scale to be less negative


class ReciprocityBasedTaskAllocation:
    """Task allocation system based on direct reciprocity."""
    
    def __init__(self, robot_team):
        """
        Initialize reciprocity-based task allocation.
        
        Args:
            robot_team: List of robots in the team
        """
        self.robot_team = robot_team
        self.task_history = {}  # Maps (robot_i, robot_j) pairs to task history
    
    def allocate_task(self, task):
        """Allocate a task based on reciprocity and capabilities."""
        # Find capable robots
        capable_robots = [r for r in self.robot_team if r.can_perform(task)]
        
        if not capable_robots:
            return None  # No robot can perform this task
        
        # Calculate workload balance based on reciprocity
        workload_scores = []
        
        for robot in capable_robots:
            # Calculate how much this robot has helped others vs. been helped
            given_help = sum(self.get_task_history(robot.id, other.id)['given']
                            for other in self.robot_team if other.id != robot.id)
            
            received_help = sum(self.get_task_history(other.id, robot.id)['given']
                               for other in self.robot_team if other.id != robot.id)
            
            # Robots that have given more help than received should get fewer tasks
            if given_help > 0:
                reciprocity_score = received_help / given_help
            else:
                reciprocity_score = 2.0  # High score for robots that haven't helped yet
            
            # Combine with current workload
            workload_score = reciprocity_score / (1 + robot.current_workload)
            workload_scores.append((robot, workload_score))
        
        # Select robot with highest workload score
        selected_robot = max(workload_scores, key=lambda x: x[1])[0]
        
        # Update task history for all robot pairs
        for other in self.robot_team:
            if other.id != selected_robot.id:
                self.update_task_history(selected_robot.id, other.id, 1, 0)
        
        return selected_robot
    
    def get_task_history(self, robot_i_id, robot_j_id):
        """Get the task history between two robots."""
        pair_key = (min(robot_i_id, robot_j_id), max(robot_i_id, robot_j_id))
        
        if pair_key not in self.task_history:
            self.task_history[pair_key] = {
                'given': 0,  # Tasks done by robot_i that helped robot_j
                'received': 0  # Tasks done by robot_j that helped robot_i
            }
        
        # Return from perspective of robot_i
        if robot_i_id < robot_j_id:
            return self.task_history[pair_key]
        else:
            # Swap given and received for correct perspective
            history = self.task_history[pair_key]
            return {'given': history['received'], 'received': history['given']}
    
    def update_task_history(self, robot_i_id, robot_j_id, given_increment, received_increment):
        """Update the task history between two robots."""
        history = self.get_task_history(robot_i_id, robot_j_id)
        
        # Update from perspective of robot_i
        history['given'] += given_increment
        history['received'] += received_increment


class ReciprocityBasedExploration:
    """Collaborative exploration system based on direct reciprocity."""
    
    def __init__(self, robot_id, map_size):
        """
        Initialize reciprocity-based exploration.
        
        Args:
            robot_id: Unique identifier for this robot
            map_size: Size of the environment map
        """
        self.robot_id = robot_id
        self.explored_cells = set()  # Cells explored by this robot
        self.shared_data = {}  # Maps partner IDs to shared exploration data
        self.map_size = map_size
    
    def explore_cell(self, cell_coords):
        """Explore a cell and add to explored set."""
        # Simulate exploration of the cell
        self.explored_cells.add(cell_coords)
        
        # Return data about the cell (simplified)
        return {
            'coords': cell_coords,
            'content': self.simulate_cell_content(cell_coords),
            'explorer_id': self.robot_id
        }
    
    def simulate_cell_content(self, cell_coords):
        """Simulate content of a cell (placeholder)."""
        # In a real implementation, this would return actual sensor data
        return {
            'obstacle': random.random() < 0.2,
            'resource_value': random.random()
        }
    
    def request_data_sharing(self, partner_id):
        """Request exploration data from another robot."""
        # Calculate how much data we've shared with this partner
        if partner_id in self.shared_data:
            data_balance = len(self.shared_data[partner_id])
        else:
            data_balance = 0
        
        # Make request with data balance information
        return {
            'requester_id': self.robot_id,
            'data_shared_count': data_balance
        }
    
    def respond_to_data_request(self, request):
        """Share exploration data based on reciprocity."""
        requester_id = request['requester_id']
        data_they_shared = request['data_shared_count']
        
        # Initialize if new partner
        if requester_id not in self.shared_data:
            self.shared_data[requester_id] = set()
        
        # Calculate how much data we've received from them
        data_we_received = len(self.shared_data[requester_id])
        
        # Decide how much to share based on reciprocity
        if data_we_received >= data_they_shared:
            # They've shared at least as much as we have - share everything
            share_ratio = 1.0
        else:
            # They've shared less - share proportionally
            share_ratio = 0.5 + (data_they_shared / (2 * max(1, data_we_received)))
        
        # Select data to share
        sharable_cells = self.explored_cells - self.shared_data.get(requester_id, set())
        num_to_share = int(len(sharable_cells) * share_ratio)
        
        if num_to_share == 0 and sharable_cells:
            num_to_share = 1  # Always share at least one cell if possible
        
        cells_to_share = list(sharable_cells)[:num_to_share]
        
        # Update sharing record
        self.shared_data[requester_id].update(cells_to_share)
        
        # Return shared data
        return [{'coords': cell, 'explorer_id': self.robot_id} for cell in cells_to_share]
    
    def receive_shared_data(self, sender_id, shared_data):
        """Process exploration data shared by another robot."""
        # Initialize if new partner
        if sender_id not in self.shared_data:
            self.shared_data[sender_id] = set()
        
        # Extract cell coordinates from shared data
        shared_cells = {item['coords'] for item in shared_data}
        
        # Update sharing record
        self.shared_data[sender_id].update(shared_cells)
        
        # Update explored cells
        self.explored_cells.update(shared_cells)


# Example usage
if __name__ == "__main__":
    # Define a Prisoner's Dilemma payoff matrix
    # [R, S, T, P] = [3, 0, 5, 1]
    PAYOFF_MATRIX = [
        [1, 5],  # Defector's payoffs
        [0, 3]   # Cooperator's payoffs
    ]
    
    # Calculate minimum discount factor for cooperation
    min_discount = calculate_cooperation_threshold([3, 0, 5, 1])
    print(f"Minimum discount factor for cooperation: {min_discount:.3f}")
    
    # Simulate interaction between two robots using different strategies
    robot1 = RecognitionBasedCooperation(robot_id=1, strategy='tit_for_tat')
    robot2 = RecognitionBasedCooperation(robot_id=2, strategy='generous_tit_for_tat')
    
    # Interaction history
    history = []
    
    # Simulate 10 interactions
    for _ in range(10):
        # Decide actions
        robot1_action = robot1.decide_action(robot2.robot_id)
        robot2_action = robot2.decide_action(robot1.robot_id)
        
        # Update histories
        robot1.update_history(robot2.robot_id, robot1_action, robot2_action)
        robot2.update_history(robot1.robot_id, robot2_action, robot1_action)
        
        # Record interaction
        history.append((robot1_action, robot2_action))
    
    # Print interaction history
    print("Interaction history (Robot1, Robot2):")
    for i, (a1, a2) in enumerate(history):
        print(f"Round {i+1}: {'C' if a1 == 1 else 'D'}, {'C' if a2 == 1 else 'D'}")
