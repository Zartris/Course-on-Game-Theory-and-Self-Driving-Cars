# Lesson 3: Roundabout Traffic Management - Coding Exercise

## Pareto Efficiency and Best Response Strategies in Roundabouts

This coding exercise implements a roundabout traffic scenario where multiple vehicles interact using game theory concepts, particularly focusing on Pareto efficiency, best response strategies, and repeated game dynamics.

### Learning Objectives

By completing this exercise, you will:
- Understand how to model roundabout traffic as a game theory problem
- Learn to calculate and visualize Pareto optimal solutions and the Pareto frontier
- Implement best response strategies for vehicles navigating a roundabout
- Explore how repeated interactions can lead to the emergence of cooperation
- Apply game theory concepts to optimize traffic flow in a realistic scenario
- Analyze how different driver behaviors affect system-level efficiency

### Prerequisites

To run this code, you need:
- Python 3.6+
- NumPy
- Pygame
- Gymnasium
- Nashpy
- Matplotlib
- SciPy

You can install the required packages using pip:
```bash
pip install numpy pygame gymnasium nashpy matplotlib scipy
```

### Files

- `roundabout_game.py`: The main implementation of the roundabout environment
- `vehicle.py`: Implementation of the Vehicle class with different driver behaviors
- `game_analyzer.py`: Game theory analysis tools and visualizations
- `utils/visualization.py`: Visualization utilities for the simulation
- `utils/metrics.py`: Performance metrics for traffic flow analysis

### How to Run the Code

1. Make sure you have the required packages installed
2. Run the main script:
```bash
python roundabout_game.py
```
3. Follow the on-screen prompts to simulate different scenarios
4. Use the Pygame window to visualize the simulation
5. Use the analysis tools to explore Pareto efficiency and Nash equilibria

### Game Description

In this game, multiple vehicles navigate a roundabout with several entry and exit points. Each vehicle has its own origin, destination, and driver behavior type (aggressive, conservative, or cooperative). The drivers must make decisions about when to enter the roundabout, how to navigate within it, and when to exit.

#### Actions
Each vehicle can take the following actions:
- **Enter**: Attempt to enter the roundabout
- **Yield**: Wait at the entry point
- **Accelerate**: Increase speed while in the roundabout
- **Decelerate**: Decrease speed while in the roundabout
- **Change Lane**: Move between inner and outer lanes (in multi-lane roundabouts)
- **Exit**: Take an exit from the roundabout

#### Driver Types

1. **Aggressive**: Prioritizes minimizing travel time over safety
   - More willing to accept small gaps for entry
   - Less likely to yield to other vehicles
   - Maintains higher speeds within the roundabout

2. **Conservative**: Prioritizes safety over travel time
   - Requires larger gaps for entry
   - More likely to yield to other vehicles
   - Maintains moderate speeds within the roundabout

3. **Cooperative**: Balances individual goals with system efficiency
   - Adjusts behavior based on traffic conditions
   - May create gaps to allow others to enter
   - Adapts to repeated interactions with other vehicles

### Pareto Efficiency Analysis

The simulation includes tools to analyze the Pareto efficiency of different traffic scenarios:

1. **Pareto Frontier Visualization**: Plots the trade-offs between different objectives (e.g., travel time vs. safety vs. throughput)

2. **Pareto Improvements**: Identifies changes to vehicle behaviors that would result in Pareto improvements

3. **System Efficiency Metrics**: Calculates overall traffic flow efficiency and identifies bottlenecks

### Best Response Analysis

The code implements best response dynamics:

1. **Best Response Computation**: Calculates the optimal response for each vehicle given others' strategies

2. **Strategy Adaptation**: Allows vehicles to adapt their strategies based on observed behavior

3. **Nash Equilibrium Detection**: Identifies when the system reaches a stable state where no vehicle can benefit from changing its strategy

### Repeated Game Features

The simulation demonstrates repeated game concepts:

1. **History Tracking**: Records the interactions between vehicles across multiple encounters

2. **Strategy Evolution**: Shows how vehicles adapt their strategies over repeated interactions

3. **Emergence of Cooperation**: Demonstrates how cooperative behavior can emerge from repeated interactions

### What to Look For

- **Emergent Patterns**: Watch how different traffic patterns emerge based on the mix of driver types
- **Efficiency vs. Fairness**: Observe the trade-off between system efficiency and fairness to individual vehicles
- **Adaptation**: Notice how vehicles adapt their strategies after repeated interactions
- **Stability**: Look for stable states where traffic flows efficiently without excessive stop-and-go behavior

### Using the Gymnasium Environment

The `RoundaboutEnv` class follows the standard Gymnasium interface, allowing for:

- Integration with reinforcement learning algorithms
- Systematic comparison of different strategies
- Controlled experimentation with different parameters
- Compatible with standard tools for analyzing agent behavior

### Extension Activities

1. Modify driver behavior parameters to see how they affect system performance
2. Implement new driver types with different utility functions
3. Experiment with different roundabout designs (number of entry points, lanes, etc.)
4. Implement a learning algorithm that improves vehicle strategies over time
5. Add more sophisticated traffic patterns like rush hour or emergency vehicles
6. Analyze the impact of different percentages of autonomous vehicles in mixed traffic

### Connection to Real-World Applications

This exercise demonstrates several key challenges in autonomous vehicle design and traffic management:

- How to design effective decision-making algorithms for complex multi-agent scenarios
- How to balance individual vehicle objectives with system-level efficiency
- How repeated interactions in traffic can lead to the emergence of social norms
- How Pareto efficiency concepts can guide the design of traffic management systems
- How autonomous vehicles can potentially transform traffic flow through coordinated behavior

By exploring these concepts through simulation, we gain insights into how game theory can inform the design of more efficient, safe, and fair transportation systems.