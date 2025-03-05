# What is Game Theory?

Game theory is a fascinating field of study that deals with **strategic decision-making**. It provides a formal framework for analyzing situations where the outcome of a choice made by one individual (or agent) depends not only on their own actions but also on the actions of others. 

Think of it like this: in many real-world scenarios, we're not making decisions in a vacuum. Our choices are intertwined with the choices of others, creating a complex web of interactions. Game theory helps us understand and navigate this web by providing tools to analyze these interactions and predict their outcomes.

**Why is this relevant to self-driving cars and multi-robot systems?**

Imagine a busy intersection with multiple self-driving cars. Each car needs to decide when to proceed, yield, or change lanes. The safety and efficiency of the entire system depend on how these cars interact and coordinate their actions. Game theory provides the mathematical language and tools to model these interactions, predict potential conflicts, and design algorithms that enable safe and efficient navigation.

**Key Characteristics of Game Theory:**

* **Interdependence:** The core idea is that the outcome for each participant (player) depends on the actions of all the players involved.
* **Strategic Thinking:** Players need to anticipate the actions of others and make decisions that are in their own best interest, considering the potential responses of other players.
* **Rationality:** Game theory often assumes that players are rational, meaning they aim to maximize their own benefit or minimize their losses.
* **Mathematical Modeling:** Game theory uses mathematical tools to represent games, analyze strategies, and predict outcomes.

**Historical Context:**

Game theory as a formal discipline emerged in the mid-20th century, with the publication of John von Neumann and Oskar Morgenstern's seminal work "Theory of Games and Economic Behavior" in 1944. However, strategic thinking about interactive decision-making can be traced back much further, appearing in military strategy, economics, and even in philosophical discussions dating back to ancient civilizations.

The field gained significant momentum in the 1950s with John Nash's groundbreaking work on non-cooperative games and equilibrium points (now known as Nash equilibria). Since then, game theory has expanded into numerous disciplines and has become an essential tool for understanding complex interactions in various domains.

**Beyond Self-Driving Cars:**

While our focus is on autonomous driving, game theory has applications in a wide range of fields, including:

* **Economics:** Analyzing market competition, auctions, and bargaining.
* **Political Science:** Understanding voting behavior, political campaigns, and international relations.
* **Biology:** Studying animal behavior, evolution, and ecological interactions.
* **Computer Science:** Designing algorithms for artificial intelligence, multi-agent systems, and network security.

In the next sections, we'll delve deeper into the different types of games, key concepts, and specific examples related to self-driving cars and multi-robot systems.

# Types of Games

In game theory, games can be classified based on various characteristics, which help us understand the nature of the strategic interactions involved. Here are some of the most common classifications, particularly relevant to self-driving cars and multi-robot systems:

## 1. Cooperative vs. Non-cooperative Games

This classification focuses on whether players can form binding agreements and work together to achieve a common goal.

* **Cooperative Games:** In cooperative games, players can form coalitions, negotiate, and make binding agreements to coordinate their actions and share the resulting payoffs. This is often seen in scenarios where collaboration leads to mutual benefits [(web)](https://en.wikipedia.org/wiki/Cooperative_game_theory).
    * **Example:** A group of self-driving cars coordinating their speeds and lane changes to optimize traffic flow and minimize congestion on a highway [(paper)](https://www.researchgate.net/publication/220660863_Game_Theory_and_Decision_Theory_in_Multi-Agent_Systems).
* **Non-cooperative Games:** In non-cooperative games, players act in their own self-interest, and there are no binding agreements. Each player chooses their strategy independently, aiming to maximize their own payoff [(web)](https://en.wikipedia.org/wiki/Non-cooperative_game_theory).
    * **Example:** Two self-driving cars approaching an intersection, each deciding whether to yield or proceed based on their individual assessment of the situation.

## 2. Simultaneous vs. Sequential Games
This classification distinguishes between games where players make decisions at the same time versus those where they take turns.

* **Simultaneous Games (Static Games):** In simultaneous games, players make their decisions without knowing the choices of the other players. They act simultaneously or, equivalently, in isolation from each other.
    * **Example:**  Self-driving cars at a four-way stop, where each car must decide whether to "stop" or "go" without knowing the decisions of the other cars.
* **Sequential Games (Dynamic Games):** In sequential games, players take turns making decisions, and they can observe the actions of the previous players. This allows for more complex strategic reasoning, as players can adapt their strategies based on the observed history of the game.
    * **Example:** A self-driving car merging onto a highway, where it must anticipate the reactions of the cars already on the highway and adjust its speed and trajectory accordingly.

## 3. Zero-Sum vs. Non-Zero-Sum Games

This classification focuses on the relationship between the gains and losses of the players.

* **Zero-Sum Games:** In zero-sum games, the total benefit to all players in the game sums to zero (or a constant amount). One player's gain is exactly balanced by the losses of the other players. These are strictly competitive games.
    * **Example:** Two self-driving cars competing for a single parking spot—if one car gets the spot, the other necessarily doesn't.
* **Non-Zero-Sum Games:** In non-zero-sum games, the sum of outcomes is not constant. Players' interests are not strictly opposed, and there can be outcomes that are beneficial (or detrimental) to all players.
    * **Example:** Self-driving cars navigating an intersection—multiple cars can safely cross without collision, resulting in positive outcomes for all participants.

## 4. Perfect vs. Imperfect Information Games

This classification relates to how much each player knows about the state of the game.

* **Perfect Information Games:** In games of perfect information, each player knows the complete history of the game's play and all possible future states.
    * **Example:** A self-driving car with complete sensor coverage of its surroundings, able to detect and track all relevant vehicles and obstacles.
* **Imperfect Information Games:** In games of imperfect information, players don't have complete knowledge of the game's history or of the other players' previous moves.
    * **Example:** A self-driving car in foggy conditions, where sensor data is limited and there is uncertainty about the exact positions or intentions of other road users.

## Importance in Autonomous Driving

Understanding these classifications is crucial for designing effective algorithms for self-driving cars and multi-robot systems. For instance, in scenarios like intersection navigation or lane merging, where simultaneous decision-making is involved, game-theoretic models can help predict potential conflicts and design strategies to avoid them. In more complex situations, such as highway driving with multiple cars, sequential game models can capture the dynamic interactions and enable more sophisticated planning and coordination.

The challenges of autonomous driving often involve imperfect information (due to sensor limitations, occlusions, or communication constraints) and non-zero-sum outcomes (where safe and efficient navigation benefits all participants). Recognizing these characteristics helps in developing appropriate models and algorithms.

# Key Concepts in Game Theory

To effectively analyze strategic interactions, we need to understand the fundamental building blocks of game theory. Here are some key concepts, illustrated with examples relevant to self-driving cars and multi-robot systems:

## 1. Players

Players are the **decision-makers** involved in a game. They can be individuals, organizations, or even autonomous agents like self-driving cars or robots. Each player has a set of possible actions they can choose from, and their goal is to select the action that will lead to the best outcome for themselves.

* **Example:** In a lane-merging scenario, the players could be two self-driving cars, each trying to merge into the same lane.

## 2. Actions

Actions are the **choices** available to each player in a game. These actions can be discrete (e.g., "yield" or "proceed" at an intersection) or continuous (e.g., adjusting the speed and steering angle of a self-driving car).

* **Example:** In a cooperative task allocation scenario, the actions for a robot could be to choose which task to perform from a set of available tasks.

## 3. Payoffs
Payoffs represent the **outcomes** or **rewards** associated with each action, given the actions of the other players. Payoffs can be numerical values (e.g., time taken to reach a destination, amount of fuel consumed) or more abstract representations of preferences (e.g., "win" or "lose").

* **Example:** In a collision avoidance scenario, the payoff for a self-driving car could be a high value for successfully avoiding a collision and a low value for colliding with another vehicle.

### Payoff Matrices

For simultaneous games with discrete actions, payoffs are often represented in a **payoff matrix**. This matrix shows the outcome for each player for every possible combination of actions.

**Example: Intersection Game Payoff Matrix**

Consider a simple intersection game where two self-driving cars (Car 1 and Car 2) must decide whether to proceed or yield at an intersection:

| | Car 2: Proceed | Car 2: Yield |
|---|---|---|
| **Car 1: Proceed** | (-10, -10) | (3, 1) |
| **Car 1: Yield** | (1, 3) | (0, 0) |

In this matrix:
- The first value in each cell represents Car 1's payoff
- The second value represents Car 2's payoff
- If both cars proceed, they collide, resulting in large negative payoffs (-10, -10)
- If one car proceeds while the other yields, the proceeding car gains time (payoff 3) and the yielding car avoids a collision but loses some time (payoff 1)
- If both cars yield, they're being overly cautious, resulting in a small inefficiency (0, 0)

## 4. Strategies

A strategy is a **complete plan of action** that specifies what a player will do in every possible situation. It outlines how a player will respond to every possible action of the other players.

* **Example:** A strategy for a self-driving car navigating a roundabout could be to yield to any car already in the roundabout and then proceed when there is a safe gap.

**Types of Strategies:**

* **Pure Strategy:** A player consistently chooses a single action every time they face a particular situation.
  * **Example:** A self-driving car always yields at a four-way stop if another car arrives first.
  
* **Mixed Strategy:** A player randomly selects from multiple possible actions according to a probability distribution.
  * **Example:** A self-driving car might choose to merge lanes with 70% probability if the gap is of medium size, and wait for a larger gap with 30% probability.

## 5. Rationality

Rationality is a core assumption in game theory. It implies that players are **self-interested** and aim to **maximize their own payoffs**. Rational players will analyze the game, anticipate the actions of others, and choose the strategy that they believe will lead to the best outcome for themselves.

* **Example:** A rational self-driving car will choose a lane-changing strategy that minimizes its travel time, even if it means slightly inconveniencing other drivers.

## 6. Common Knowledge

Common knowledge refers to information that is known by all players, and all players know that all players know it, and so on. In other words, it's not just that everyone knows something, but everyone knows that everyone else knows it, and everyone knows that everyone knows that everyone else knows it, and so on ad infinitum. This is a crucial assumption in many game-theoretic analyses, as it ensures that players have a shared understanding of the game's rules and structure.

* **Example:** In a traffic scenario with standardized rules (e.g., right-of-way at intersections), these rules are common knowledge among all drivers, including self-driving cars.

## 7. Utility Functions

A utility function is a mathematical representation of a player's preferences over outcomes. It assigns a numerical value (utility) to each possible outcome, reflecting how desirable that outcome is to the player. In self-driving car applications, utility functions might incorporate factors like safety, time efficiency, comfort, and energy consumption.

* **Example:** A self-driving car's utility function might heavily weight safety (avoiding collisions) while also considering travel time, passenger comfort (avoiding sharp accelerations), and fuel efficiency.

These key concepts provide the foundation for understanding and analyzing strategic interactions in various scenarios, including those involving self-driving cars and multi-robot systems.

# Game Representations

Games can be represented in different ways, depending on their structure and complexity. Here are the two most common representations:

## 1. Normal Form (Strategic Form)

Normal form, also known as strategic form, represents a game as a matrix of payoffs. It's particularly useful for simultaneous games with a finite number of players and strategies.

**Components:**
- A list of players
- For each player, a list of strategies
- For each combination of strategies, a list of payoffs (one for each player)

**Example:** The intersection game above is represented in normal form.

## 2. Extensive Form

Extensive form represents a game as a tree, showing the sequence of moves, the information available to each player at each decision point, and the payoffs for all possible outcomes. This representation is particularly useful for sequential games.

**Components:**
- Nodes (decision points)
- Edges (actions)
- Information sets (grouping nodes where a player cannot distinguish which node they are at)
- Payoffs (at terminal nodes)

**Example:** A sequential lane-changing scenario could be represented in extensive form, showing first the decision of one car to signal a lane change, followed by another car's decision to yield or maintain speed.

# Day 1 Exercise: Intersection Game

## Exercise Overview

In this exercise, you'll implement a simple game theory scenario where two self-driving cars approach an intersection and must decide whether to proceed or yield. This exercise demonstrates the concepts of strategic interactions, payoff matrices, and decision-making in a multi-agent scenario.

## The Scenario

Two autonomous vehicles are approaching a four-way intersection from perpendicular directions. Each vehicle must decide whether to proceed through the intersection or yield to the other vehicle. The outcome depends on the decisions of both vehicles:

- If both vehicles proceed, they will collide, resulting in a severe negative outcome for both.
- If one vehicle proceeds and the other yields, the proceeding vehicle gains time, while the yielding vehicle loses some time but avoids a collision.
- If both vehicles yield, they are being overly cautious, resulting in slight inefficiency but no safety risk.

## The Payoff Matrix

| | Car 2: Proceed | Car 2: Yield |
|---|---|---|
| **Car 1: Proceed** | (-10, -10) | (3, 1) |
| **Car 1: Yield** | (1, 3) | (0, 0) |

## Implementation

The exercise involves creating a Python simulation of this interaction using Gymnasium (OpenAI Gym's successor) and Pygame for visualization. The implementation includes:

1. **Environment Setup:** Creating a Gymnasium environment that models the intersection scenario.
2. **State Representation:** Defining the state space to represent the positions of both vehicles.
3. **Action Space:** Defining the actions available to each vehicle (Proceed or Yield).
4. **Transition Dynamics:** Implementing how the state evolves based on the chosen actions.
5. **Reward Structure:** Implementing the payoff matrix as rewards.
6. **Visualization:** Using Pygame to visualize the intersection, vehicles, and outcomes.
7. **Game Analysis:** Analyzing the game to identify dominant strategies and Nash equilibria.

## Learning Objectives

By completing this exercise, you will:

1. Understand how to model a real-world autonomous driving scenario using game theory concepts.
2. Implement a simulation that captures strategic interactions between two agents.
3. Visualize and analyze different outcomes based on chosen strategies.
4. Identify dominant strategies and Nash equilibria in a simple game.
5. Gain practical experience with Gymnasium and Pygame for creating interactive simulations.

## Extension Ideas

After completing the basic exercise, consider these extensions:

- Modify the payoff values to see how they affect the strategic considerations.
- Add uncertainty about the other car's intentions.
- Implement learning algorithms that allow the cars to adapt their strategies over time.
- Add more cars to the intersection, creating a more complex multi-agent scenario.

This exercise serves as a foundation for understanding how game theory concepts apply to autonomous driving scenarios, setting the stage for more complex interactions in future lessons.