"""
Genetic Algorithms for Strategy Evolution in Multi-Robot Systems

This module implements genetic algorithm components for evolving robot strategies,
including various encoding methods, selection mechanisms, crossover and mutation
operators, and population management approaches.
"""

import numpy as np
import random
import copy


class GeneticAlgorithm:
    """Base class for genetic algorithms in multi-robot systems."""
    
    def __init__(self, population_size=100, chromosome_length=10, 
                 crossover_rate=0.8, mutation_rate=0.1, elitism_count=2):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Size of the population
            chromosome_length: Length of each chromosome
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        # Initialize population with random chromosomes
        self.population = self.initialize_population()
        self.fitness_values = np.zeros(population_size)
        
        # Track best solution
        self.best_chromosome = None
        self.best_fitness = -float('inf')
        
        # History for analysis
        self.average_fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
    
    def initialize_population(self):
        """Initialize a random population. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_population")
    
    def evaluate_fitness(self, chromosome):
        """Evaluate fitness of a chromosome. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate_fitness")
    
    def evaluate_population(self):
        """Evaluate fitness of all chromosomes in the population."""
        for i, chromosome in enumerate(self.population):
            self.fitness_values[i] = self.evaluate_fitness(chromosome)
            
            # Update best solution
            if self.fitness_values[i] > self.best_fitness:
                self.best_fitness = self.fitness_values[i]
                self.best_chromosome = chromosome.copy()
        
        # Update history
        self.average_fitness_history.append(np.mean(self.fitness_values))
        self.best_fitness_history.append(self.best_fitness)
        self.diversity_history.append(self.calculate_diversity())
    
    def calculate_diversity(self):
        """Calculate population diversity. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement calculate_diversity")
    
    def selection(self, selection_method='tournament', tournament_size=3):
        """Select parents for reproduction."""
        if selection_method == 'tournament':
            return self.tournament_selection(tournament_size)
        elif selection_method == 'roulette':
            return self.roulette_wheel_selection()
        elif selection_method == 'rank':
            return self.rank_based_selection()
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def tournament_selection(self, tournament_size=3):
        """Tournament selection."""
        selected = []
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [self.fitness_values[i] for i in tournament_indices]
            
            # Select winner (best fitness)
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def roulette_wheel_selection(self):
        """Roulette wheel (fitness proportional) selection."""
        # Ensure all fitness values are positive
        fitness_min = min(0, np.min(self.fitness_values))
        adjusted_fitness = self.fitness_values - fitness_min + 1e-10
        
        # Calculate selection probabilities
        total_fitness = np.sum(adjusted_fitness)
        probabilities = adjusted_fitness / total_fitness
        
        # Select individuals
        selected_indices = np.random.choice(
            self.population_size, 
            size=self.population_size, 
            p=probabilities
        )
        
        return [self.population[i].copy() for i in selected_indices]
    
    def rank_based_selection(self, selection_pressure=1.5):
        """Rank-based selection."""
        # Sort indices by fitness
        sorted_indices = np.argsort(self.fitness_values)
        
        # Calculate rank-based probabilities
        ranks = np.arange(1, self.population_size + 1)
        probabilities = (2 - selection_pressure) / self.population_size + \
                        (2 * (ranks - 1) * (selection_pressure - 1)) / \
                        (self.population_size * (self.population_size - 1))
        
        # Select individuals
        selected_indices = np.random.choice(
            sorted_indices, 
            size=self.population_size, 
            p=probabilities
        )
        
        return [self.population[i].copy() for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement crossover")
    
    def mutation(self, chromosome):
        """Perform mutation on a chromosome. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement mutation")
    
    def elitism(self, new_population):
        """Preserve the best individuals."""
        if self.elitism_count <= 0:
            return new_population
        
        # Find indices of best individuals
        elite_indices = np.argsort(self.fitness_values)[-self.elitism_count:]
        
        # Replace the first few individuals with elites
        for i, elite_idx in enumerate(elite_indices):
            new_population[i] = self.population[elite_idx].copy()
        
        return new_population
    
    def evolve_generation(self):
        """Evolve the population for one generation."""
        # Evaluate current population
        self.evaluate_population()
        
        # Select parents
        parents = self.selection()
        
        # Create new population through crossover and mutation
        new_population = []
        
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[min(i+1, self.population_size-1)]
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Apply elitism
        new_population = self.elitism(new_population)
        
        # Update population
        self.population = new_population
    
    def run(self, generations=100):
        """Run the genetic algorithm for a number of generations."""
        for _ in range(generations):
            self.evolve_generation()
        
        return {
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_fitness,
            'average_fitness_history': self.average_fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }


class BinaryGA(GeneticAlgorithm):
    """Genetic algorithm with binary encoding."""
    
    def initialize_population(self):
        """Initialize a random population of binary chromosomes."""
        return [np.random.randint(0, 2, self.chromosome_length) 
                for _ in range(self.population_size)]
    
    def calculate_diversity(self):
        """Calculate population diversity as average Hamming distance."""
        total_distance = 0
        count = 0
        
        for i in range(self.population_size):
            for j in range(i+1, self.population_size):
                # Hamming distance
                distance = np.sum(self.population[i] != self.population[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def crossover(self, parent1, parent2):
        """Single-point crossover for binary chromosomes."""
        crossover_point = random.randint(1, self.chromosome_length - 1)
        
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return offspring1, offspring2
    
    def mutation(self, chromosome):
        """Bit-flip mutation for binary chromosomes."""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip bit
        
        return chromosome


class RealValuedGA(GeneticAlgorithm):
    """Genetic algorithm with real-valued encoding."""
    
    def __init__(self, population_size=100, chromosome_length=10, 
                 crossover_rate=0.8, mutation_rate=0.1, elitism_count=2,
                 gene_bounds=None):
        """
        Initialize the real-valued genetic algorithm.
        
        Args:
            population_size: Size of the population
            chromosome_length: Length of each chromosome
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            gene_bounds: List of (min, max) tuples for each gene
        """
        self.gene_bounds = gene_bounds or [(0.0, 1.0)] * chromosome_length
        super().__init__(population_size, chromosome_length, crossover_rate, mutation_rate, elitism_count)
    
    def initialize_population(self):
        """Initialize a random population of real-valued chromosomes."""
        population = []
        
        for _ in range(self.population_size):
            chromosome = np.zeros(self.chromosome_length)
            
            for i in range(self.chromosome_length):
                min_val, max_val = self.gene_bounds[i]
                chromosome[i] = random.uniform(min_val, max_val)
            
            population.append(chromosome)
        
        return population
    
    def calculate_diversity(self):
        """Calculate population diversity as average Euclidean distance."""
        total_distance = 0
        count = 0
        
        for i in range(self.population_size):
            for j in range(i+1, self.population_size):
                # Euclidean distance
                distance = np.sqrt(np.sum((self.population[i] - self.population[j])**2))
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def crossover(self, parent1, parent2, alpha=0.5):
        """Arithmetic crossover for real-valued chromosomes."""
        # Generate random weights for each gene
        weights = np.random.rand(self.chromosome_length)
        
        # Create offspring through weighted combination
        offspring1 = parent1 * weights + parent2 * (1 - weights)
        offspring2 = parent2 * weights + parent1 * (1 - weights)
        
        return offspring1, offspring2
    
    def mutation(self, chromosome, mutation_strength=0.1):
        """Gaussian mutation for real-valued chromosomes."""
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                min_val, max_val = self.gene_bounds[i]
                range_size = max_val - min_val
                
                # Scale mutation by gene range
                sigma = range_size * mutation_strength
                
                # Apply mutation
                chromosome[i] += random.gauss(0, sigma)
                
                # Ensure value stays within bounds
                chromosome[i] = max(min_val, min(max_val, chromosome[i]))
        
        return chromosome


class IslandModelGA(GeneticAlgorithm):
    """Genetic algorithm with island model population structure."""
    
    def __init__(self, num_islands=5, island_size=20, migration_rate=0.1, migration_interval=5,
                 chromosome_length=10, crossover_rate=0.8, mutation_rate=0.1, elitism_count=1,
                 base_ga_class=RealValuedGA, **base_ga_kwargs):
        """
        Initialize the island model genetic algorithm.
        
        Args:
            num_islands: Number of islands (subpopulations)
            island_size: Size of each island
            migration_rate: Proportion of individuals that migrate
            migration_interval: Number of generations between migrations
            chromosome_length: Length of each chromosome
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve per island
            base_ga_class: Base GA class to use for each island
            **base_ga_kwargs: Additional arguments for base GA class
        """
        self.num_islands = num_islands
        self.island_size = island_size
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        
        # Create islands
        self.islands = []
        for _ in range(num_islands):
            island = base_ga_class(
                population_size=island_size,
                chromosome_length=chromosome_length,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elitism_count=elitism_count,
                **base_ga_kwargs
            )
            self.islands.append(island)
        
        # Initialize tracking variables
        self.generation = 0
        self.best_chromosome = None
        self.best_fitness = -float('inf')
        
        # History
        self.average_fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
    
    def perform_migration(self):
        """Migrate individuals between islands."""
        # Calculate number of migrants per island
        migrants_count = max(1, int(self.island_size * self.migration_rate))
        
        # Collect migrants from each island (best individuals)
        all_migrants = []
        for island in self.islands:
            # Get indices of best individuals
            best_indices = np.argsort(island.fitness_values)[-migrants_count:]
            
            # Collect migrants and their fitness
            migrants = [(island.population[i].copy(), island.fitness_values[i]) 
                       for i in best_indices]
            all_migrants.append(migrants)
        
        # Distribute migrants to destination islands
        for i, island in enumerate(self.islands):
            # Destination is the next island (ring topology)
            dest_idx = (i + 1) % self.num_islands
            dest_island = self.islands[dest_idx]
            
            # Get migrants from source island
            migrants = all_migrants[i]
            
            # Replace worst individuals in destination island
            worst_indices = np.argsort(dest_island.fitness_values)[:migrants_count]
            
            for j, (migrant, fitness) in enumerate(migrants):
                dest_idx = worst_indices[j]
                dest_island.population[dest_idx] = migrant
                dest_island.fitness_values[dest_idx] = fitness
    
    def evolve_generation(self):
        """Evolve all islands for one generation."""
        # Evolve each island independently
        for island in self.islands:
            island.evolve_generation()
        
        # Periodic migration
        self.generation += 1
        if self.generation % self.migration_interval == 0:
            self.perform_migration()
        
        # Update best solution across all islands
        for island in self.islands:
            if island.best_fitness > self.best_fitness:
                self.best_fitness = island.best_fitness
                self.best_chromosome = island.best_chromosome.copy()
        
        # Update history
        avg_fitness = np.mean([island.average_fitness_history[-1] for island in self.islands])
        self.average_fitness_history.append(avg_fitness)
        self.best_fitness_history.append(self.best_fitness)
        
        # Calculate diversity as average of island diversities plus between-island diversity
        island_diversities = [island.diversity_history[-1] for island in self.islands]
        avg_island_diversity = np.mean(island_diversities)
        
        # Between-island diversity (average distance between best individuals)
        best_chromosomes = [island.best_chromosome for island in self.islands]
        between_diversity = 0
        count = 0
        
        for i in range(self.num_islands):
            for j in range(i+1, self.num_islands):
                # Euclidean distance between best chromosomes
                distance = np.sqrt(np.sum((best_chromosomes[i] - best_chromosomes[j])**2))
                between_diversity += distance
                count += 1
        
        between_diversity = between_diversity / count if count > 0 else 0
        total_diversity = 0.5 * avg_island_diversity + 0.5 * between_diversity
        self.diversity_history.append(total_diversity)
    
    def run(self, generations=100):
        """Run the island model GA for a number of generations."""
        for _ in range(generations):
            self.evolve_generation()
        
        return {
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_fitness,
            'average_fitness_history': self.average_fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }


# Example application: Robot navigation strategy evolution
class NavigationStrategyGA(RealValuedGA):
    """GA for evolving robot navigation strategies."""
    
    def __init__(self, population_size=100, crossover_rate=0.8, mutation_rate=0.1, elitism_count=2,
                 obstacle_map=None, start_position=(0, 0), goal_position=(10, 10)):
        """
        Initialize the navigation strategy GA.
        
        Args:
            population_size: Size of the population
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            obstacle_map: 2D array representing obstacles (1=obstacle, 0=free)
            start_position: Starting position (x, y)
            goal_position: Goal position (x, y)
        """
        # Navigation strategy parameters:
        # [obstacle_weight, goal_weight, speed_factor, turn_rate, sensor_range, ...]
        chromosome_length = 5
        gene_bounds = [
            (0.0, 1.0),  # obstacle_weight
            (0.0, 1.0),  # goal_weight
            (0.1, 1.0),  # speed_factor
            (0.1, 1.0),  # turn_rate
            (1.0, 10.0)  # sensor_range
        ]
        
        super().__init__(
            population_size=population_size,
            chromosome_length=chromosome_length,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            gene_bounds=gene_bounds
        )
        
        self.obstacle_map = obstacle_map
        self.start_position = start_position
        self.goal_position = goal_position
    
    def evaluate_fitness(self, chromosome):
        """Evaluate fitness of a navigation strategy."""
        # Extract strategy parameters
        obstacle_weight = chromosome[0]
        goal_weight = chromosome[1]
        speed_factor = chromosome[2]
        turn_rate = chromosome[3]
        sensor_range = chromosome[4]
        
        # Simulate robot navigation with these parameters
        completion_time, energy_used, collisions = self.simulate_navigation(
            obstacle_weight, goal_weight, speed_factor, turn_rate, sensor_range
        )
        
        # Calculate fitness (example weights)
        if completion_time == float('inf'):  # Did not reach goal
            return 0.0
        
        # Multi-objective fitness function
        time_fitness = 100.0 / (1.0 + completion_time)  # Higher for faster completion
        energy_fitness = 50.0 / (1.0 + energy_used)     # Higher for lower energy use
        collision_fitness = 200.0 / (1.0 + collisions)  # Higher for fewer collisions
        
        # Combined fitness
        fitness = 0.4 * time_fitness + 0.3 * energy_fitness + 0.3 * collision_fitness
        
        return fitness
    
    def simulate_navigation(self, obstacle_weight, goal_weight, speed_factor, turn_rate, sensor_range):
        """
        Simulate robot navigation with the given strategy parameters.
        
        This is a simplified simulation. In a real application, this would use
        a physics-based simulator or actual robot hardware.
        
        Returns:
            completion_time: Time to reach goal (inf if not reached)
            energy_used: Energy consumed during navigation
            collisions: Number of collisions with obstacles
        """
        # Simplified simulation (placeholder)
        # In a real implementation, this would be a detailed simulation
        
        # Example: Higher obstacle_weight reduces collisions but increases time
        # Higher goal_weight reduces time but may increase collisions
        # Higher speed_factor reduces time but increases energy and collision risk
        
        # Baseline values
        base_time = 100.0
        base_energy = 50.0
        base_collisions = 5.0
        
        # Adjust based on parameters
        completion_time = base_time * (1.0 / speed_factor) * (1.0 + 0.5 * obstacle_weight - 0.3 * goal_weight)
        energy_used = base_energy * speed_factor * (1.0 + 0.2 * turn_rate)
        collisions = base_collisions * (1.0 - 0.8 * obstacle_weight + 0.4 * goal_weight) * speed_factor
        
        # Ensure non-negative values
        completion_time = max(10.0, completion_time)
        energy_used = max(5.0, energy_used)
        collisions = max(0.0, collisions)
        
        return completion_time, energy_used, collisions


# Example usage
if __name__ == "__main__":
    # Create and run a navigation strategy GA
    nav_ga = NavigationStrategyGA(population_size=50)
    results = nav_ga.run(generations=50)
    
    print("Best navigation strategy:")
    print(f"Obstacle weight: {results['best_chromosome'][0]:.3f}")
    print(f"Goal weight: {results['best_chromosome'][1]:.3f}")
    print(f"Speed factor: {results['best_chromosome'][2]:.3f}")
    print(f"Turn rate: {results['best_chromosome'][3]:.3f}")
    print(f"Sensor range: {results['best_chromosome'][4]:.3f}")
    print(f"Fitness: {results['best_fitness']:.3f}")
