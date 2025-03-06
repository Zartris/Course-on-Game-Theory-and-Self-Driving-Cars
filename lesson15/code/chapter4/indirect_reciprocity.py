"""
Indirect Reciprocity Mechanisms for Multi-Robot Cooperation

This module implements various indirect reciprocity mechanisms for promoting
cooperation in multi-robot systems, including reputation systems, gossip
mechanisms, and distributed reputation management.
"""

import numpy as np
import random
from collections import Counter


def simulate_indirect_reciprocity_evolution(initial_frequencies, benefit, cost, 
                                           reputation_accuracy, n_generations=100):
    """
    Simulate the evolution of indirect reciprocity.
    
    Args:
        initial_frequencies: Initial frequencies of [cooperators, defectors, discriminators]
        benefit: Benefit of receiving cooperation
        cost: Cost of cooperating
        reputation_accuracy: Probability of correct reputation assessment
        n_generations: Number of generations to simulate
    
    Returns:
        History of frequencies over generations
    """
    # Initialize frequencies
    x_c, x_d, x_i = initial_frequencies
    frequency_history = [(x_c, x_d, x_i)]
    
    for _ in range(n_generations):
        # Calculate payoffs
        pi_c = benefit * (x_c + x_i) - cost
        pi_d = benefit * x_c
        pi_i = benefit * (x_c + reputation_accuracy * x_i) - cost * (x_c + reputation_accuracy * x_i)
        
        # Calculate average payoff
        avg_payoff = x_c * pi_c + x_d * pi_d + x_i * pi_i
        
        # Update frequencies using replicator dynamics
        x_c_new = x_c * (pi_c - avg_payoff)
        x_d_new = x_d * (pi_d - avg_payoff)
        x_i_new = x_i * (pi_i - avg_payoff)
        
        # Normalize to ensure sum = 1
        total = x_c + x_d + x_i + x_c_new + x_d_new + x_i_new
        
        if total > 0:
            x_c = x_c + x_c_new / total
            x_d = x_d + x_d_new / total
            x_i = x_i + x_i_new / total
        
        # Record history
        frequency_history.append((x_c, x_d, x_i))
    
    return frequency_history


class IndirectReciprocitySystem:
    """Implementation of indirect reciprocity in a robot system."""
    
    def __init__(self, num_robots=100, initial_reputation=0.5, 
                 cooperation_cost=0.1, cooperation_benefit=0.3,
                 reputation_noise=0.1, observation_probability=0.3):
        """
        Initialize indirect reciprocity system.
        
        Args:
            num_robots: Number of robots in the system
            initial_reputation: Initial reputation value for all robots
            cooperation_cost: Cost of cooperation
            cooperation_benefit: Benefit from receiving cooperation
            reputation_noise: Probability of errors in reputation assessment
            observation_probability: Probability that an interaction is observed
        """
        self.num_robots = num_robots
        self.cooperation_cost = cooperation_cost
        self.cooperation_benefit = cooperation_benefit
        self.reputation_noise = reputation_noise
        self.observation_probability = observation_probability
        
        # Initialize reputations
        self.reputations = np.ones(num_robots) * initial_reputation
        
        # Initialize strategies (threshold for cooperation)
        self.strategies = np.random.rand(num_robots)
        
        # History
        self.reputation_history = [self.reputations.copy()]
        self.strategy_history = [self.strategies.copy()]
        self.cooperation_rate_history = []
    
    def decide_cooperation(self, robot_idx, partner_idx):
        """Decide whether robot_idx cooperates with partner_idx."""
        # Get partner's reputation
        partner_reputation = self.reputations[partner_idx]
        
        # Add noise to reputation assessment
        if np.random.rand() < self.reputation_noise:
            partner_reputation = 1 - partner_reputation
        
        # Decide based on strategy threshold
        return partner_reputation >= self.strategies[robot_idx]
    
    def update_reputation(self, robot_idx, cooperated):
        """Update reputation based on cooperation decision."""
        # Reputation increases for cooperation, decreases for defection
        if cooperated:
            self.reputations[robot_idx] = min(1.0, self.reputations[robot_idx] + 0.1)
        else:
            self.reputations[robot_idx] = max(0.0, self.reputations[robot_idx] - 0.1)
    
    def interact(self, robot_i, robot_j):
        """Simulate an interaction between two robots."""
        # Decisions
        i_cooperates = self.decide_cooperation(robot_i, robot_j)
        j_cooperates = self.decide_cooperation(robot_j, robot_i)
        
        # Calculate payoffs
        i_payoff = 0
        j_payoff = 0
        
        if i_cooperates:
            i_payoff -= self.cooperation_cost
        if j_cooperates:
            j_payoff -= self.cooperation_cost
        
        if j_cooperates:
            i_payoff += self.cooperation_benefit
        if i_cooperates:
            j_payoff += self.cooperation_benefit
        
        # Update reputations if interaction is observed
        if np.random.rand() < self.observation_probability:
            self.update_reputation(robot_i, i_cooperates)
        
        if np.random.rand() < self.observation_probability:
            self.update_reputation(robot_j, j_cooperates)
        
        return i_payoff, j_payoff, i_cooperates, j_cooperates
    
    def evolve_strategies(self, selection_strength=1.0):
        """Evolve strategies based on accumulated payoffs."""
        # Initialize payoffs
        payoffs = np.zeros(self.num_robots)
        
        # Simulate interactions
        cooperation_count = 0
        total_interactions = 0
        
        for _ in range(self.num_robots * 5):  # Multiple interactions per generation
            # Select random pair
            i, j = np.random.choice(self.num_robots, 2, replace=False)
            
            # Interact
            i_payoff, j_payoff, i_cooperates, j_cooperates = self.interact(i, j)
            
            # Update payoffs
            payoffs[i] += i_payoff
            payoffs[j] += j_payoff
            
            # Count cooperation
            cooperation_count += i_cooperates + j_cooperates
            total_interactions += 2
        
        # Calculate cooperation rate
        cooperation_rate = cooperation_count / total_interactions
        self.cooperation_rate_history.append(cooperation_rate)
        
        # Selection and reproduction
        # Convert payoffs to selection probabilities
        selection_probs = np.exp(selection_strength * payoffs)
        selection_probs = selection_probs / np.sum(selection_probs)
        
        # Select parents
        parents = np.random.choice(
            self.num_robots, size=self.num_robots, p=selection_probs
        )
        
        # Create offspring with mutation
        new_strategies = self.strategies[parents].copy()
        mutations = np.random.normal(0, 0.02, self.num_robots)
        new_strategies += mutations
        new_strategies = np.clip(new_strategies, 0, 1)
        
        # Update strategies
        self.strategies = new_strategies
        
        # Update history
        self.reputation_history.append(self.reputations.copy())
        self.strategy_history.append(self.strategies.copy())
    
    def run_simulation(self, generations=100, selection_strength=1.0):
        """Run the simulation for a number of generations."""
        for _ in range(generations):
            self.evolve_strategies(selection_strength)
        
        return {
            'reputation_history': self.reputation_history,
            'strategy_history': self.strategy_history,
            'cooperation_rate_history': self.cooperation_rate_history
        }
    
    def analyze_results(self):
        """Analyze the results of the simulation."""
        # Calculate final cooperation rate
        final_cooperation_rate = self.cooperation_rate_history[-1]
        
        # Calculate average strategy threshold
        final_strategies = self.strategy_history[-1]
        avg_strategy = np.mean(final_strategies)
        
        # Calculate strategy diversity
        strategy_diversity = np.std(final_strategies)
        
        # Calculate reputation distribution
        final_reputations = self.reputation_history[-1]
        avg_reputation = np.mean(final_reputations)
        reputation_diversity = np.std(final_reputations)
        
        return {
            'final_cooperation_rate': final_cooperation_rate,
            'avg_strategy': avg_strategy,
            'strategy_diversity': strategy_diversity,
            'avg_reputation': avg_reputation,
            'reputation_diversity': reputation_diversity
        }


class DistributedReputationSystem:
    """Implementation of a distributed reputation system for robot teams."""
    
    def __init__(self, num_robots=50, communication_range=0.2, world_size=1.0,
                 initial_reputation=0.5, reputation_memory=0.9, trust_threshold=0.3):
        """
        Initialize distributed reputation system.
        
        Args:
            num_robots: Number of robots in the system
            communication_range: Maximum distance for direct communication
            world_size: Size of the world (square with side length world_size)
            initial_reputation: Initial reputation value for all robots
            reputation_memory: Weight given to previous reputation assessments
            trust_threshold: Minimum reputation needed for cooperation
        """
        self.num_robots = num_robots
        self.communication_range = communication_range
        self.world_size = world_size
        self.reputation_memory = reputation_memory
        self.trust_threshold = trust_threshold
        
        # Initialize positions
        self.positions = np.random.rand(num_robots, 2) * world_size
        
        # Initialize reputations (each robot's view of others)
        self.reputations = np.ones((num_robots, num_robots)) * initial_reputation
        
        # Set self-reputation to 1.0
        for i in range(num_robots):
            self.reputations[i, i] = 1.0
        
        # Initialize cooperation strategy (threshold-based)
        self.cooperation_thresholds = np.ones(num_robots) * trust_threshold
        
        # History
        self.average_reputation_history = [np.mean(self.reputations)]
        self.cooperation_rate_history = []
    
    def get_neighbors(self, robot_idx):
        """Get indices of robots within communication range."""
        neighbors = []
        pos = self.positions[robot_idx]
        
        for i in range(self.num_robots):
            if i != robot_idx:
                dist = np.linalg.norm(self.positions[i] - pos)
                if dist <= self.communication_range:
                    neighbors.append(i)
        
        return neighbors
    
    def decide_cooperation(self, robot_idx, partner_idx):
        """Decide whether robot_idx cooperates with partner_idx."""
        return self.reputations[robot_idx, partner_idx] >= self.cooperation_thresholds[robot_idx]
    
    def update_reputation(self, observer_idx, target_idx, cooperated):
        """Update reputation based on observed cooperation."""
        old_rep = self.reputations[observer_idx, target_idx]
        
        if cooperated:
            new_rep = old_rep * self.reputation_memory + (1 - self.reputation_memory) * 1.0
        else:
            new_rep = old_rep * self.reputation_memory + (1 - self.reputation_memory) * 0.0
        
        self.reputations[observer_idx, target_idx] = new_rep
    
    def share_reputations(self, robot_idx):
        """Share reputation information with neighbors."""
        neighbors = self.get_neighbors(robot_idx)
        
        for neighbor_idx in neighbors:
            # For each robot that both know about
            for target_idx in range(self.num_robots):
                if target_idx != robot_idx and target_idx != neighbor_idx:
                    # Share reputation information
                    shared_rep = self.reputations[robot_idx, target_idx]
                    neighbor_rep = self.reputations[neighbor_idx, target_idx]
                    
                    # Trust in the sharing robot affects how much to update
                    trust_in_sharer = self.reputations[neighbor_idx, robot_idx]
                    
                    # Update neighbor's reputation assessment
                    if trust_in_sharer > 0.5:  # Only trust information from reputable robots
                        weight = 0.2 * trust_in_sharer
                        self.reputations[neighbor_idx, target_idx] = (
                            (1 - weight) * neighbor_rep + weight * shared_rep
                        )
    
    def move_robots(self):
        """Move robots randomly in the environment."""
        # Simple random movement
        movement = (np.random.rand(self.num_robots, 2) - 0.5) * 0.1
        self.positions += movement
        
        # Ensure robots stay within bounds
        self.positions = np.clip(self.positions, 0, self.world_size)
    
    def simulate_interactions(self):
        """Simulate interactions between robots."""
        cooperation_count = 0
        interaction_count = 0
        
        # Each robot interacts with neighbors
        for i in range(self.num_robots):
            neighbors = self.get_neighbors(i)
            
            for j in neighbors:
                # Decisions
                i_cooperates = self.decide_cooperation(i, j)
                j_cooperates = self.decide_cooperation(j, i)
                
                # Update reputations based on observations
                self.update_reputation(j, i, i_cooperates)
                self.update_reputation(i, j, j_cooperates)
                
                # Count cooperation
                cooperation_count += i_cooperates + j_cooperates
                interaction_count += 2
        
        # Calculate cooperation rate
        if interaction_count > 0:
            cooperation_rate = cooperation_count / interaction_count
        else:
            cooperation_rate = 0
        
        self.cooperation_rate_history.append(cooperation_rate)
    
    def run_simulation(self, steps=100):
        """Run the simulation for a number of steps."""
        for _ in range(steps):
            # Move robots
            self.move_robots()
            
            # Simulate interactions
            self.simulate_interactions()
            
            # Share reputation information
            for i in range(self.num_robots):
                self.share_reputations(i)
            
            # Update history
            self.average_reputation_history.append(np.mean(self.reputations))
        
        return {
            'average_reputation_history': self.average_reputation_history,
            'cooperation_rate_history': self.cooperation_rate_history
        }
    
    def analyze_results(self):
        """Analyze the results of the simulation."""
        # Calculate final cooperation rate
        final_cooperation_rate = self.cooperation_rate_history[-1] if self.cooperation_rate_history else 0
        
        # Calculate reputation consensus
        reputation_std = np.std(self.reputations, axis=0)
        avg_reputation_std = np.mean(reputation_std)
        
        # Calculate reputation correlation with cooperation threshold
        reputation_means = np.mean(self.reputations, axis=0)
        threshold_reputation_corr = np.corrcoef(reputation_means, self.cooperation_thresholds)[0, 1]
        
        return {
            'final_cooperation_rate': final_cooperation_rate,
            'avg_reputation_std': avg_reputation_std,
            'threshold_reputation_corr': threshold_reputation_corr
        }


class NetworkReciprocityModel:
    """Implementation of network reciprocity for cooperation."""
    
    def __init__(self, network_type='lattice', network_size=100, initial_coop_freq=0.5,
                 benefit=4.0, cost=1.0):
        """
        Initialize network reciprocity model.
        
        Args:
            network_type: Type of network ('lattice', 'small_world', 'scale_free')
            network_size: Number of nodes in the network
            initial_coop_freq: Initial frequency of cooperators
            benefit: Benefit of receiving cooperation
            cost: Cost of cooperating
        """
        self.network_size = network_size
        self.benefit = benefit
        self.cost = cost
        
        # Create network
        if network_type == 'lattice':
            self.network = self.create_lattice_network()
        elif network_type == 'small_world':
            self.network = self.create_small_world_network()
        elif network_type == 'scale_free':
            self.network = self.create_scale_free_network()
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Initialize strategies (1 = cooperate, 0 = defect)
        self.strategies = np.random.choice(
            [0, 1], size=network_size, p=[1-initial_coop_freq, initial_coop_freq]
        )
        
        # History
        self.cooperation_history = [np.mean(self.strategies)]
        self.cluster_size_history = [self.calculate_cooperator_cluster_size()]
    
    def create_lattice_network(self, k=4):
        """Create a 2D lattice network with periodic boundary conditions."""
        # Determine grid dimensions
        side_length = int(np.sqrt(self.network_size))
        if side_length**2 != self.network_size:
            raise ValueError("Network size must be a perfect square for lattice network")
        
        # Create adjacency matrix
        adjacency = np.zeros((self.network_size, self.network_size))
        
        for i in range(self.network_size):
            # Convert to 2D coordinates
            x, y = i % side_length, i // side_length
            
            # Connect to neighbors (with periodic boundary conditions)
            neighbors = [
                ((x+1) % side_length) + y * side_length,  # Right
                ((x-1) % side_length) + y * side_length,  # Left
                x + ((y+1) % side_length) * side_length,  # Down
                x + ((y-1) % side_length) * side_length   # Up
            ]
            
            for neighbor in neighbors:
                adjacency[i, neighbor] = 1
        
        return adjacency
    
    def create_small_world_network(self, k=4, p=0.1):
        """Create a small-world network using the Watts-Strogatz model."""
        # Start with a ring lattice
        adjacency = np.zeros((self.network_size, self.network_size))
        
        # Connect each node to k nearest neighbors
        for i in range(self.network_size):
            for j in range(1, k//2 + 1):
                adjacency[i, (i+j) % self.network_size] = 1
                adjacency[i, (i-j) % self.network_size] = 1
        
        # Rewire edges with probability p
        for i in range(self.network_size):
            for j in range(self.network_size):
                if adjacency[i, j] == 1 and np.random.rand() < p:
                    # Remove this edge
                    adjacency[i, j] = 0
                    
                    # Add a new edge to a random node
                    new_neighbor = np.random.randint(0, self.network_size)
                    while new_neighbor == i or adjacency[i, new_neighbor] == 1:
                        new_neighbor = np.random.randint(0, self.network_size)
                    
                    adjacency[i, new_neighbor] = 1
        
        return adjacency
    
    def create_scale_free_network(self, m=2):
        """Create a scale-free network using the Barabási-Albert model."""
        # Start with a complete graph of m nodes
        adjacency = np.zeros((self.network_size, self.network_size))
        
        # Initial complete graph
        for i in range(m):
            for j in range(i+1, m):
                adjacency[i, j] = 1
                adjacency[j, i] = 1
        
        # Add remaining nodes with preferential attachment
        for i in range(m, self.network_size):
            # Calculate attachment probabilities
            degrees = np.sum(adjacency[:i, :i], axis=1)
            probs = degrees / np.sum(degrees)
            
            # Select m nodes to connect to
            targets = np.random.choice(i, size=m, replace=False, p=probs)
            
            # Add edges
            for target in targets:
                adjacency[i, target] = 1
                adjacency[target, i] = 1
        
        return adjacency
    
    def calculate_payoffs(self):
        """Calculate payoffs for each individual based on network interactions."""
        payoffs = np.zeros(self.network_size)
        
        for i in range(self.network_size):
            for j in range(self.network_size):
                if self.network[i, j] == 1:  # If i and j are connected
                    # Calculate payoff from this interaction
                    if self.strategies[i] == 1:  # i cooperates
                        payoffs[i] -= self.cost
                    if self.strategies[j] == 1:  # j cooperates
                        payoffs[i] += self.benefit
        
        return payoffs
    
    def update_strategies(self):
        """Update strategies based on payoffs and network structure."""
        payoffs = self.calculate_payoffs()
        new_strategies = self.strategies.copy()
        
        for i in range(self.network_size):
            # Get neighbors
            neighbors = np.where(self.network[i] == 1)[0]
            
            if len(neighbors) > 0:
                # Select a random neighbor
                neighbor = np.random.choice(neighbors)
                
                # Imitate neighbor's strategy with probability proportional to payoff difference
                payoff_diff = payoffs[neighbor] - payoffs[i]
                
                if payoff_diff > 0:
                    # Probability of imitation increases with payoff difference
                    imitation_prob = payoff_diff / (self.benefit + self.cost)
                    
                    if np.random.rand() < imitation_prob:
                        new_strategies[i] = self.strategies[neighbor]
        
        self.strategies = new_strategies
    
    def calculate_cooperator_cluster_size(self):
        """Calculate the average size of cooperator clusters."""
        # Create a graph of only cooperators
        cooperator_indices = np.where(self.strategies == 1)[0]
        
        if len(cooperator_indices) == 0:
            return 0
        
        cooperator_network = self.network[cooperator_indices][:, cooperator_indices]
        
        # Identify connected components (clusters)
        visited = set()
        clusters = []
        
        for i in range(len(cooperator_indices)):
            if i not in visited:
                # Start a new cluster
                cluster = set([i])
                visited.add(i)
                
                # Expand cluster
                frontier = [i]
                while frontier:
                    node = frontier.pop(0)
                    neighbors = np.where(cooperator_network[node] == 1)[0]
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            cluster.add(neighbor)
                            frontier.append(neighbor)
                
                clusters.append(cluster)
        
        # Calculate average cluster size
        if clusters:
            return np.mean([len(cluster) for cluster in clusters])
        else:
            return 0
    
    def run_simulation(self, generations=100):
        """Run the simulation for a number of generations."""
        for _ in range(generations):
            self.update_strategies()
            
            # Update history
            self.cooperation_history.append(np.mean(self.strategies))
            self.cluster_size_history.append(self.calculate_cooperator_cluster_size())
        
        return {
            'cooperation_history': self.cooperation_history,
            'cluster_size_history': self.cluster_size_history
        }
    
    def analyze_results(self):
        """Analyze the results of the simulation."""
        # Calculate final cooperation rate
        final_coop_rate = self.cooperation_history[-1]
        
        # Calculate change in cooperation rate
        coop_rate_change = final_coop_rate - self.cooperation_history[0]
        
        # Calculate final cluster size
        final_cluster_size = self.cluster_size_history[-1]
        
        # Calculate network metrics
        degrees = np.sum(self.network, axis=1)
        avg_degree = np.mean(degrees)
        
        # Calculate correlation between degree and strategy
        degree_strategy_corr = np.corrcoef(degrees, self.strategies)[0, 1]
        
        return {
            'final_coop_rate': final_coop_rate,
            'coop_rate_change': coop_rate_change,
            'final_cluster_size': final_cluster_size,
            'avg_degree': avg_degree,
            'degree_strategy_corr': degree_strategy_corr
        }


# Example usage
if __name__ == "__main__":
    # Simulate indirect reciprocity evolution
    initial_freqs = [0.33, 0.33, 0.34]  # [cooperators, defectors, discriminators]
    results = simulate_indirect_reciprocity_evolution(
        initial_freqs, benefit=3.0, cost=1.0, reputation_accuracy=0.8, n_generations=100
    )
    
    # Print final frequencies
    final_freqs = results[-1]
    print(f"Final frequencies: Cooperators={final_freqs[0]:.3f}, "
          f"Defectors={final_freqs[1]:.3f}, Discriminators={final_freqs[2]:.3f}")
    
    # Create and run a distributed reputation system
    rep_system = DistributedReputationSystem(num_robots=50, communication_range=0.3)
    rep_results = rep_system.run_simulation(steps=50)
    
    # Print final cooperation rate
    final_coop_rate = rep_results['cooperation_rate_history'][-1]
    print(f"Final cooperation rate with distributed reputation: {final_coop_rate:.3f}")
    
    # Create and run a network reciprocity model
    network_model = NetworkReciprocityModel(network_type='small_world', network_size=100)
    network_results = network_model.run_simulation(generations=50)
    
    # Print final cooperation rate
    final_coop_rate = network_results['cooperation_history'][-1]
    print(f"Final cooperation rate with network reciprocity: {final_coop_rate:.3f}")
