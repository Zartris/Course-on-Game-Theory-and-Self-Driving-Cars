# Day 1: Introduction to Game Theory - Coding Exercise

## Intersection Game Simulation

This coding exercise implements a simple game theory scenario where two self-driving cars approach a four-way intersection and must decide whether to proceed or yield, built as a Gymnasium environment with Pygame visualization.

### Learning Objectives

By completing this exercise, you will:
- Understand how to model a real-world autonomous driving scenario as a game theory problem
- Implement and visualize a simple 2x2 game with Python using the Gymnasium framework
- Learn to analyze a game to identify dominant strategies and Nash equilibria
- See how payoff matrices represent the outcomes of different strategy combinations
- Gain familiarity with Gymnasium and Pygame for creating interactive simulations

### Prerequisites

To run this code, you need:
- Python 3.6+
- NumPy
- Pygame
- Gymnasium

You can install the required packages using pip:
```
pip install numpy pygame gymnasium
```

### Files

- `intersection_game.py`: The main implementation of the intersection game as a Gymnasium environment

### How to Run the Code

1. Make sure you have the required packages installed
2. Run the script:
```
python intersection_game.py
```
3. Follow the on-screen prompts to simulate different scenarios
4. Use the Pygame window to visualize the simulation
5. Close the Pygame window or press ESC to end the current simulation

### Game Description

In this game, two self-driving cars approach a four-way intersection from perpendicular directions. Each car must decide whether to proceed through the intersection or yield to the other car.

#### Actions
Each car has two possible actions:
- **Proceed (0)**: The car continues through the intersection without stopping
- **Yield (1)**: The car stops and lets the other car pass

#### Payoffs
The payoff structure is designed to reflect realistic priorities for self-driving cars:
- When both cars proceed, there's a high risk of collision resulting in large negative payoffs (-10, -10)
- When one car proceeds and the other yields, the proceeding car gains time (payoff 3), while the yielding car loses some time but avoids a collision (payoff 1)
- When both cars yield, they're being overly cautious, resulting in inefficiency but no safety risk (0, 0)

### What to Look For

- **Nash Equilibria**: Are there strategy combinations where no car can improve its outcome by unilaterally changing its strategy?
- **Dominant Strategies**: Does either car have a strategy that's always better regardless of what the other car does?
- **Collision Avoidance**: How does the payoff structure incentivize collision avoidance?
- **Visualization**: Watch how the decisions of both cars play out in the visual simulation

### Using the Gymnasium Environment

The `IntersectionGameEnv` class follows the standard Gymnasium interface:

- `reset()`: Resets the environment to its initial state
- `step(actions)`: Takes a step in the environment using the specified actions
- `render()`: Renders the current state of the environment using Pygame

The environment can be used with reinforcement learning algorithms by following the standard Gymnasium interaction pattern:

```python
env = IntersectionGameEnv(render_mode="human")
obs, info = env.reset()
done = False
terminated = False
truncated = False

while not (terminated or truncated):
    # Select an action (in this case, a tuple of actions for both cars)
    actions = (action_car1, action_car2)  # 0 = Proceed, 1 = Yield
    
    # Take a step in the environment
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Render the environment
    env.render()
```

### Extension Activities

1. Modify the payoff matrix to see how it affects the Nash equilibria
2. Add a third car to the intersection and expand the game
3. Implement a traffic signal that alternates which car has right-of-way
4. Create different driver profiles (aggressive, cautious, etc.) with different payoff structures
5. Use reinforcement learning to train an agent to find the optimal policy for navigating the intersection

### Connection to Real-World Self-Driving Cars

This simple game demonstrates one of the fundamental challenges in developing self-driving cars: how to navigate intersections safely and efficiently when interacting with other vehicles. In the real world, autonomous vehicles use sensors and prediction algorithms to estimate the intentions of other road users and make decisions that maximize safety while minimizing travel time.

Game theory provides a mathematical framework for understanding these interactions and designing decision-making algorithms that can handle complex multi-agent scenarios.

### Why Gymnasium and Pygame?

Using the Gymnasium framework with Pygame visualization provides several advantages:
1. **Standardized Interface**: Gymnasium provides a standardized environment interface widely used in reinforcement learning
2. **Interactive Visualization**: Pygame allows for real-time, interactive visualization that can be easily customized
3. **Event Handling**: Pygame supports event handling for keyboard and mouse input, making the simulations more interactive
4. **Separation of Concerns**: The environment dynamics are separate from the visualization, allowing for flexible rendering options
5. **Future Expandability**: Both frameworks are well-maintained and can be extended for more complex scenarios
6. **Cross-platform Compatibility**: Works on various operating systems with minimal configuration