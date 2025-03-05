# Lesson 5: Task Allocation and Auction Mechanisms in Multi-Robot Systems

## Overview
This lesson explores task allocation in multi-robot systems using auction-based approaches. You'll learn how robots can efficiently distribute tasks among themselves using economic principles and game theory.

## Learning Objectives
- Understand multi-robot task allocation (MRTA) concepts
- Learn different auction mechanisms for distributing tasks
- Implement strategic bidding behaviors
- Analyze the efficiency of different allocation strategies

## Contents
- `Task_allocation_and_auction_mechanisms.md`: Theoretical background
- `task_allocation.py`: Main simulation environment implementation
- `bidding_robot.py`: Robot class with bidding capabilities
- `auction_mechanism.py`: Framework for various auction types
- `task_manager.py`: Task generation and management system
- `utils/`: Helper modules for visualization, metrics, and learning
- `auction_comparison.py`: Script to compare different auction mechanisms
- `bidding_strategy_comparison.py`: Script to compare different bidding strategies
- `run_experiments.py`: Comprehensive tool to run multiple experiments

## Running the Simulation

### Basic Task Allocation Simulation

```bash
python task_allocation.py --num_robots 5 --num_tasks 10 --grid_size 20 --auction_type sequential --payment_rule first_price --render_mode pygame
```

#### Options:
- `--num_robots`: Number of robots in the simulation (default: 5)
- `--num_tasks`: Initial number of tasks to generate (default: 10)
- `--grid_size`: Size of the environment grid (default: 20)
- `--auction_type`: Type of auction mechanism ("sequential", "parallel", "combinatorial")
- `--payment_rule`: Payment rule for auction ("first_price", "second_price", "vcg")
- `--dynamic_tasks`: Whether new tasks arrive during simulation (default: True)
- `--render_mode`: Visualization mode ("pygame", "matplotlib", "none")
- `--max_steps`: Maximum simulation steps (default: 100)
- `--compare`: Run multiple auction types and compare results

### Comparing Auction Mechanisms

```bash
python auction_comparison.py --num_robots 8 --num_tasks 30 --max_steps 150 --output_dir results
```

#### Options:
- `--num_robots`: Number of robots in the simulation (default: 5)
- `--num_tasks`: Number of tasks to generate (default: 20)
- `--grid_size`: Size of the environment grid (default: 20)
- `--dynamic_tasks`: Whether new tasks arrive during simulation (default: True)
- `--max_steps`: Maximum simulation steps (default: 100)
- `--output_dir`: Directory to save output files (default: "results")

### Comparing Bidding Strategies

```bash
python bidding_strategy_comparison.py --auction_type combinatorial --payment_rule first_price --robots_per_strategy 3
```

#### Options:
- `--robots_per_strategy`: Number of robots per strategy type (default: 3)
- `--num_tasks`: Number of tasks to generate (default: 20)
- `--grid_size`: Size of the environment grid (default: 20)
- `--auction_type`: Type of auction mechanism ("sequential", "parallel", "combinatorial")
- `--payment_rule`: Payment rule for auction ("first_price", "second_price", "vcg")
- `--dynamic_tasks`: Whether new tasks arrive during simulation (default: True)
- `--max_steps`: Maximum simulation steps (default: 100)
- `--output_dir`: Directory to save output files (default: "results")

### Running Multiple Experiments

```bash
python run_experiments.py --verbose
```

#### Options:
- `--output_dir`: Directory to save output files (default: "results")
- `--verbose`: Show detailed output during execution
- `--auction_only`: Run only auction mechanism comparisons
- `--bidding_only`: Run only bidding strategy comparisons

## Features

### Auction Mechanisms
- **Sequential Auctions**: Tasks are auctioned one at a time
- **Parallel Auctions**: All tasks are auctioned simultaneously
- **Combinatorial Auctions**: Robots can bid on bundles of tasks

### Payment Rules
- **First Price**: Winner pays its bid amount
- **Second Price**: Winner pays the second-highest bid amount
- **VCG**: Winner pays the opportunity cost to the rest of the system

### Bidding Strategies
- **Truthful**: Robots bid their true utility
- **Strategic**: Robots shade their bids based on competition
- **Learning**: Robots adjust bidding strategy based on past outcomes
- **Cooperative**: Robots consider team goals in their bids

### Coalition Formation
- Support for complex tasks requiring multiple robots
- Role-based coalition formation
- Coordinated task execution by coalition members

## Output

All comparison scripts produce visualizations in the specified output directory:
- Bar charts comparing performance metrics
- Line charts showing metrics over time
- Radar charts for multi-dimensional comparison
- Detailed logs of execution results

## Experiment Results

Results are saved in the `results` directory with subdirectories for each experiment:
- `auction_comparison_*`: Results from auction mechanism comparisons
- `bidding_comparison_*`: Results from bidding strategy comparisons

Each directory contains PNG images visualizing the experimental results.

## Exercises
1. Compare the efficiency of different auction mechanisms 
2. Implement and test a new bidding strategy
3. Create a coalition formation strategy for complex tasks
4. Analyze how communication constraints affect allocation efficiency

## References
- "Market-Based Multirobot Coordination: A Survey and Analysis" by M.B. Dias et al.
- "A Taxonomy for Multi-Robot Task Allocation" by B.P. Gerkey and M.J. Matarić
- "Auction Theory" by Vijay Krishna