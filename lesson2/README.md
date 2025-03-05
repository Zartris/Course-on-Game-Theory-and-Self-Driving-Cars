# Lesson 2: Lane Changing Game

This lesson implements a game theory scenario focused on lane-changing decisions for autonomous vehicles. The implementation demonstrates dominant strategies and Nash equilibria in the context of self-driving cars interacting with different driver behaviors.

## Overview

The game simulates two vehicles on a three-lane highway making strategic decisions about lane changes. Each vehicle can be one of three types:
- Aggressive: Prioritizes speed over safety
- Defensive: Prioritizes safety over speed
- Cooperative: Balances personal goals with overall traffic efficiency

### Key Features

1. **Different Driver Behaviors**: Each car is randomly assigned a behavior type that affects its payoff matrix
2. **Nash Equilibrium Analysis**: Uses the Nashpy library to calculate pure strategy Nash equilibria
3. **Dominant Strategy Detection**: Automatically identifies if any player has a dominant strategy
4. **Interactive Visualization**: Shows cars moving on a three-lane highway using Pygame

## Prerequisites

Make sure you have all required dependencies installed:
```bash
pip install -r ../requirements.txt
```

Required packages:
- gymnasium
- pygame
- nashpy
- numpy

## How to Run

1. Navigate to the lesson2 directory
2. Run the game:
```bash
python lane_changing_game.py
```

## Game Rules

### Actions
Each car has three possible actions:
- Change Left (0)
- Stay (1)
- Change Right (2)

### Payoffs
Payoff values depend on the combination of driver types and chosen actions. The payoff matrices are designed to reflect realistic driving scenarios:

1. **Aggressive vs Aggressive**:
   - High rewards for successful lane changes
   - Severe penalties for collisions
   - Moderate penalties for yielding

2. **Defensive vs Defensive**:
   - Higher penalties for risky maneuvers
   - Positive rewards for maintaining safe distances
   - Small penalties for staying in current lane

3. **Cooperative vs Cooperative**:
   - Balanced rewards for coordinated actions
   - Moderate penalties for collisions
   - Small rewards for mutually beneficial decisions

### Game Flow
1. The program randomly assigns behavior types to both cars
2. The current game configuration is analyzed for:
   - Nash equilibria
   - Dominant strategies
   - Payoff matrix for the current driver types
3. Players choose actions for both cars
4. The environment simulates the outcome and displays results

## Code Structure

- `LaneChangingGameEnv`: Main Gymnasium environment class
  - Implements the game logic and visualization
  - Manages different driver behaviors and payoff matrices
  - Provides methods for game analysis
- `analyze_game()`: Finds Nash equilibria and dominant strategies
- `print_payoff_matrix()`: Displays the current payoff matrix
- `main()`: Interactive game loop and user interface

## Learning Objectives

After completing this exercise, you should understand:
1. How different driver behaviors affect strategic decisions in traffic
2. How to identify dominant strategies in a multi-action game
3. How to calculate and interpret Nash equilibria using Nashpy
4. How to model realistic traffic scenarios using game theory

## Extension Activities

1. **Additional Driver Types**: Add new driver behavior types with different payoff matrices
2. **Mixed Strategy Analysis**: Implement calculation of mixed strategy Nash equilibria
3. **Multi-Car Scenarios**: Extend the game to handle more than two cars
4. **Learning Algorithms**: Add reinforcement learning agents that learn optimal strategies

## References

1. Nash, J. (1951). Non-Cooperative Games. Annals of Mathematics
2. Schwarting, W., et al. (2019). Social behavior for autonomous vehicles. PNAS
3. Gymnasium Documentation: https://gymnasium.farama.org/
4. Nashpy Documentation: https://nashpy.readthedocs.io/