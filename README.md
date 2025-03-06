# Course-on-Game-Theory-and-Self-Driving-Cars

## Course: Game Theory and Decision-Making in Multi-Robot Systems and Self-Driving Cars

This course focuses on game theory and decision-making in multi-robot systems and autonomous driving contexts. You'll learn through practical examples and theoretical foundations derived from real-world autonomous driving scenarios. Each lesson introduces key concepts alongside coding exercises designed to deepen your understanding.

## Setup

### Virtual Environment

Set up the virtual environment:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# For bash/zsh:
source .venv/bin/activate
# Or simply:
source activate.sh

# For fish:
. .venv/bin/activate.fish
# Or simply:
. activate.fish

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

After setting up the environment, run examples as follows:

```bash
# Example: Run the fleet coordination simulation
cd lesson4
python fleet_coordination.py
```

## Course Outline:

### Lessons 1-3: Introduction to Game Theory
- **Lesson 1:** Introduction to Game Theory, types of games (cooperative/non-cooperative, static/dynamic), key concepts (players, actions, payoffs), intersection decision-making example.
- **Lesson 2:** Dominant strategies, Nash Equilibrium (pure and mixed strategies), lane-changing scenario analysis.
- **Lesson 3:** Pareto efficiency, best responses, repeated games, traffic flow optimization example.

### Lessons 4-7: Game Theory in Multi-Robot Systems
- **Lesson 4:** Bayesian games and fleet coordination, autonomous fleet management with incomplete information.
- **Lesson 5:** Task allocation strategies in multi-robot systems, search and rescue operations example.
- **Lesson 6:** Cooperative game theory, collaboration strategies for robot teams.
- **Lesson 7:** Auction mechanisms, resource allocation, robot bidding systems.

### Lessons 8-12: Game Theory in Self-Driving Cars
- **Lesson 8:** Modeling autonomous vehicle interactions, intersection collision avoidance, lane merging.
- **Lesson 9:** Predicting human driver behavior in mixed traffic scenarios.
- **Lesson 10:** Game-theoretic planning for navigation, trajectory optimization in traffic.
- **Lesson 12:** Liability and responsibility analysis in autonomous vehicle accidents.

### Lessons 13-16: Advanced Topics and Applications
- **Lesson 13:** Potential games, swarm robotics, spatial distribution of robots.
- **Lesson 14:** Pursuit-evasion games, robotics applications, strategies for capturing or evading.
- **Lesson 15:** Evolutionary game theory, adaptive learning strategies for robots.
- **Lesson 16:** Mechanism design, incentivizing collaboration in multi-robot tasks.

### Lessons 17-19: Simulation and Research
- **Lesson 17:** Introduction to 2D multi-robot simulators (Player/Stage), basic simulation setup.
- **Lesson 18:** Robot sensors and actuators, practical simulation using Player/Stage.
- **Lesson 19:** Autonomous lane merging without communication, decentralized decision-making.

This structured course combines theoretical concepts, coding exercises, and realistic simulations, offering a robust foundation in game theory and decision-making for research in multi-robot systems and self-driving cars.

