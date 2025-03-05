# Lesson 4: Bayesian Games and Fleet Coordination

This lesson covers Bayesian games and their application to fleet coordination in autonomous vehicles. The implementation includes:

1. A `BayesianVehicle` class that represents an autonomous vehicle with belief-based decision making
2. A `BayesianGameAnalyzer` that finds Bayesian Nash equilibria and computes the value of information
3. A fleet coordination simulator that demonstrates multi-vehicle coordination under uncertainty

## Key Concepts

- **Bayesian Games**: Games with incomplete information where players have beliefs about other players' types
- **Type-based Decision Making**: Vehicles have different types (standard, premium, emergency) with different capabilities and priorities
- **Belief Updates**: Vehicles observe each other and update their beliefs using Bayes' rule
- **Bayesian Nash Equilibrium**: The solution concept for games with incomplete information
- **Value of Information**: Measuring how additional information improves decision quality

## Files

- `bayesian_vehicle.py`: Implementation of vehicles with Bayesian reasoning
- `bayesian_game_analyzer.py`: Tools for analyzing Bayesian games and finding equilibria
- `fleet_coordination.py`: Main simulator for fleet coordination
- `utils/`: Supporting utilities
  - `visualization.py`: Visualization tools for the simulation
  - `metrics.py`: Performance metrics calculation and analysis

## Running the Simulator

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run the fleet coordination simulator:
```bash
python fleet_coordination.py
```

3. Command line options:
```
--no-viz         # Disable visualization
--steps NUM      # Set number of simulation steps (default: 500)
--standard NUM   # Number of standard vehicles (default: 3)
--premium NUM    # Number of premium vehicles (default: 1)
--emergency NUM  # Number of emergency vehicles (default: 1)
--seed NUM       # Set random seed for reproducibility
```

Example with custom configuration:
```bash
python fleet_coordination.py --steps 300 --standard 5 --premium 2 --emergency 1
```

## Visualization

The visualization shows:
- Vehicles with different colors (blue: standard, green: premium, red: emergency)
- Belief states as pie charts between vehicles
- Communication links shown as green lines with arrows
- A city grid with roads and blocks
- Performance metrics in a side panel

## Learning Exercises

1. **Belief Updates**: Observe how vehicles update their beliefs as they observe each other's behavior.
2. **Emergency Response**: Watch how vehicles respond to emergency vehicles by giving them priority.
3. **Nash Equilibria**: Analyze the Bayesian Nash equilibria that emerge in different scenarios.
4. **Communication Effects**: Try modifying vehicle communication ranges and reliability to see how it affects coordination.
5. **Value of Information**: Compare scenarios with different levels of information sharing to measure its value.

## Game Theory Analysis

The `BayesianGameAnalyzer` provides tools for:

1. Computing expected utility given beliefs
2. Finding Bayesian Nash equilibria
3. Analyzing the value of information
4. Computing the Price of Anarchy (comparing optimal vs. Nash equilibrium outcomes)
5. Visualizing belief updates over time

Try modifying the belief initialization in the fleet coordination simulator to see how it affects the vehicles' decisions.