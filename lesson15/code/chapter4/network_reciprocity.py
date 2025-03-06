"""
Network Reciprocity Models for Multi-Robot Cooperation

This module implements various network reciprocity models for studying
how network structure affects the evolution of cooperation in multi-robot systems.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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
    
    def visualize_network(self, ax=None, node_size=100):
        """Visualize the network with cooperators and defectors."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create NetworkX graph from adjacency matrix
        G = nx.from_numpy_array(self.network)
        
        # Set node colors based on strategies
        node_colors = ['green' if s == 1 else 'red' for s in self.strategies]
        
        # Draw the network
        pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        # Add legend
        ax.plot([], [], 'go', markersize=10, label='Cooperator')
        ax.plot([], [], 'ro', markersize=10, label='Defector')
        ax.legend()
        
        ax.set_title(f'Network Structure (Cooperation Rate: {np.mean(self.strategies):.2f})')
        ax.axis('off')
        
        return ax


class SpatialGameModel:
    """Implementation of spatial games for studying cooperation."""
    
    def __init__(self, grid_size=10, initial_coop_freq=0.5, benefit=4.0, cost=1.0, 
                 game_type='prisoners_dilemma'):
        """
        Initialize spatial game model.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            initial_coop_freq: Initial frequency of cooperators
            benefit: Benefit of receiving cooperation
            cost: Cost of cooperating
            game_type: Type of game ('prisoners_dilemma', 'snowdrift', 'stag_hunt')
        """
        self.grid_size = grid_size
        self.benefit = benefit
        self.cost = cost
        self.game_type = game_type
        
        # Initialize grid with strategies (1 = cooperate, 0 = defect)
        self.grid = np.random.choice(
            [0, 1], 
            size=(grid_size, grid_size), 
            p=[1-initial_coop_freq, initial_coop_freq]
        )
        
        # Set up payoff matrix based on game type
        self.setup_payoff_matrix()
        
        # History
        self.cooperation_history = [np.mean(self.grid)]
        self.spatial_pattern_history = []
        self.save_spatial_pattern()
    
    def setup_payoff_matrix(self):
        """Set up payoff matrix based on game type."""
        if self.game_type == 'prisoners_dilemma':
            # Prisoner's Dilemma: T > R > P > S
            self.payoff_matrix = np.array([
                [0, self.benefit],             # Defector's payoffs
                [-self.cost, self.benefit - self.cost]  # Cooperator's payoffs
            ])
        elif self.game_type == 'snowdrift':
            # Snowdrift (Hawk-Dove): T > R > S > P
            self.payoff_matrix = np.array([
                [0, self.benefit],             # Defector's payoffs
                [self.benefit - self.cost, (self.benefit - self.cost) / 2]  # Cooperator's payoffs
            ])
        elif self.game_type == 'stag_hunt':
            # Stag Hunt: R > T > P > S
            self.payoff_matrix = np.array([
                [self.cost / 2, self.benefit],  # Defector's payoffs
                [0, self.benefit * 2]           # Cooperator's payoffs
            ])
        else:
            raise ValueError(f"Unknown game type: {self.game_type}")
    
    def get_neighbors(self, i, j):
        """Get the indices of neighboring cells (Moore neighborhood)."""
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip self
                
                # Apply periodic boundary conditions
                ni = (i + di) % self.grid_size
                nj = (j + dj) % self.grid_size
                
                neighbors.append((ni, nj))
        
        return neighbors
    
    def calculate_payoffs(self):
        """Calculate payoffs for each cell based on interactions with neighbors."""
        payoffs = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                strategy = self.grid[i, j]
                
                # Get neighbors
                neighbors = self.get_neighbors(i, j)
                
                # Calculate payoff from interactions with neighbors
                for ni, nj in neighbors:
                    neighbor_strategy = self.grid[ni, nj]
                    payoffs[i, j] += self.payoff_matrix[strategy, neighbor_strategy]
        
        return payoffs
    
    def update_strategies(self):
        """Update strategies based on payoffs and local interactions."""
        payoffs = self.calculate_payoffs()
        new_grid = self.grid.copy()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get neighbors
                neighbors = self.get_neighbors(i, j)
                
                # Add self to potential strategy sources
                all_sources = [(i, j)] + neighbors
                
                # Find strategy with highest payoff in neighborhood
                best_payoff = payoffs[i, j]
                best_strategy = self.grid[i, j]
                
                for ni, nj in neighbors:
                    if payoffs[ni, nj] > best_payoff:
                        best_payoff = payoffs[ni, nj]
                        best_strategy = self.grid[ni, nj]
                
                # Update strategy
                new_grid[i, j] = best_strategy
        
        self.grid = new_grid
    
    def save_spatial_pattern(self):
        """Save current spatial pattern to history."""
        self.spatial_pattern_history.append(self.grid.copy())
    
    def run_simulation(self, generations=100):
        """Run the simulation for a number of generations."""
        for _ in range(generations):
            self.update_strategies()
            
            # Update history
            self.cooperation_history.append(np.mean(self.grid))
            
            # Save spatial pattern every 10 generations
            if _ % 10 == 0:
                self.save_spatial_pattern()
        
        return {
            'cooperation_history': self.cooperation_history,
            'spatial_pattern_history': self.spatial_pattern_history
        }
    
    def analyze_results(self):
        """Analyze the results of the simulation."""
        # Calculate final cooperation rate
        final_coop_rate = self.cooperation_history[-1]
        
        # Calculate change in cooperation rate
        coop_rate_change = final_coop_rate - self.cooperation_history[0]
        
        # Calculate spatial autocorrelation (Moran's I)
        moran_i = self.calculate_spatial_autocorrelation()
        
        # Calculate cluster statistics
        cluster_stats = self.calculate_cluster_statistics()
        
        return {
            'final_coop_rate': final_coop_rate,
            'coop_rate_change': coop_rate_change,
            'spatial_autocorrelation': moran_i,
            'cluster_stats': cluster_stats
        }
    
    def calculate_spatial_autocorrelation(self):
        """Calculate spatial autocorrelation (Moran's I) for the current grid."""
        # Flatten grid
        flat_grid = self.grid.flatten()
        
        # Calculate mean and variance
        mean = np.mean(flat_grid)
        variance = np.var(flat_grid)
        
        if variance == 0:
            return 0  # No variation, return 0
        
        # Create weight matrix (1 for neighbors, 0 otherwise)
        n = self.grid_size * self.grid_size
        W = np.zeros((n, n))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                
                # Get neighbors
                neighbors = self.get_neighbors(i, j)
                
                for ni, nj in neighbors:
                    nidx = ni * self.grid_size + nj
                    W[idx, nidx] = 1
        
        # Calculate Moran's I
        numerator = 0
        for i in range(n):
            for j in range(n):
                if W[i, j] == 1:
                    numerator += (flat_grid[i] - mean) * (flat_grid[j] - mean)
        
        denominator = sum(sum(W)) * sum((flat_grid - mean) ** 2) / n
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def calculate_cluster_statistics(self):
        """Calculate statistics about cooperator and defector clusters."""
        # Identify clusters using connected components
        cooperator_clusters = self.identify_clusters(1)  # Cooperator clusters
        defector_clusters = self.identify_clusters(0)    # Defector clusters
        
        # Calculate statistics
        coop_cluster_sizes = [len(cluster) for cluster in cooperator_clusters]
        defect_cluster_sizes = [len(cluster) for cluster in defector_clusters]
        
        return {
            'cooperator_clusters': {
                'count': len(cooperator_clusters),
                'avg_size': np.mean(coop_cluster_sizes) if coop_cluster_sizes else 0,
                'max_size': max(coop_cluster_sizes) if coop_cluster_sizes else 0
            },
            'defector_clusters': {
                'count': len(defector_clusters),
                'avg_size': np.mean(defect_cluster_sizes) if defect_cluster_sizes else 0,
                'max_size': max(defect_cluster_sizes) if defect_cluster_sizes else 0
            }
        }
    
    def identify_clusters(self, strategy):
        """Identify clusters of cells with the given strategy."""
        visited = set()
        clusters = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == strategy and (i, j) not in visited:
                    # Start a new cluster
                    cluster = set([(i, j)])
                    visited.add((i, j))
                    
                    # Expand cluster
                    frontier = [(i, j)]
                    while frontier:
                        cell = frontier.pop(0)
                        neighbors = self.get_neighbors(*cell)
                        
                        for neighbor in neighbors:
                            if self.grid[neighbor] == strategy and neighbor not in visited:
                                visited.add(neighbor)
                                cluster.add(neighbor)
                                frontier.append(neighbor)
                    
                    clusters.append(cluster)
        
        return clusters
    
    def visualize_grid(self, ax=None):
        """Visualize the current state of the grid."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a colored grid
        cmap = plt.cm.RdYlGn  # Red for defectors, green for cooperators
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        ax.tick_params(which='both', size=0, labelsize=0, labelbottom=False, labelleft=False)
        
        # Add title
        ax.set_title(f'Spatial Game ({self.game_type})\nCooperation Rate: {np.mean(self.grid):.2f}')
        
        # Add legend
        ax.plot([], [], 'gs', markersize=10, label='Cooperator')
        ax.plot([], [], 'rs', markersize=10, label='Defector')
        ax.legend(loc='upper right')
        
        return ax


# Example usage
if __name__ == "__main__":
    # Compare different network types
    network_types = ['lattice', 'small_world', 'scale_free']
    results = {}
    
    for network_type in network_types:
        model = NetworkReciprocityModel(network_type=network_type, network_size=100)
        results[network_type] = model.run_simulation(generations=100)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for network_type, result in results.items():
        plt.plot(result['cooperation_history'], label=network_type)
    
    plt.xlabel('Generation')
    plt.ylabel('Cooperation Rate')
    plt.title('Evolution of Cooperation in Different Network Structures')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Run spatial game model
    spatial_model = SpatialGameModel(grid_size=20, game_type='snowdrift')
    spatial_results = spatial_model.run_simulation(generations=50)
    
    # Visualize final state
    plt.figure(figsize=(8, 8))
    spatial_model.visualize_grid()
    plt.show()
