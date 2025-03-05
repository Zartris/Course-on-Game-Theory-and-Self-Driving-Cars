# Dominant Strategies and Nash Equilibrium in Self-Driving Cars

In this lesson, we delve deeper into key game theory concepts that are fundamental to understanding strategic interactions in autonomous driving scenarios. We'll explore dominant strategies, Nash equilibrium, and how these concepts apply specifically to lane-changing decisions for self-driving cars.

## 1. Dominant Strategies

### Definition and Types

A **dominant strategy** is a strategy that provides a player with the best outcome regardless of the strategies chosen by other players. It's essentially a "no-brainer" decision that a rational player would always choose. There are two types of dominant strategies:

1. **Strictly Dominant Strategy:** A strategy that provides a strictly higher payoff than any other strategy, regardless of what other players do.

2. **Weakly Dominant Strategy:** A strategy that provides at least as good a payoff as any other strategy (and sometimes better), regardless of what other players do.

### Mathematical Formulation of Dominance

Let's formalize the concept of dominant strategies mathematically:

- Let $S_i$ be the set of possible strategies for player $i$
- Let $s_i \in S_i$ be a specific strategy for player $i$
- Let $S_{-i}$ be the set of all possible strategy combinations for all players except $i$
- Let $s_{-i} \in S_{-i}$ be a specific combination of strategies for all players except $i$
- Let $u_i(s_i, s_{-i})$ be the payoff function for player $i$ when player $i$ chooses strategy $s_i$ and all other players choose strategies $s_{-i}$

**Strict Dominance:**
Strategy $s_i^*$ strictly dominates strategy $s_i'$ if:

$$u_i(s_i^*, s_{-i}) > u_i(s_i', s_{-i}) \quad \forall s_{-i} \in S_{-i}$$

This means that regardless of what other players do, strategy $s_i^*$ always yields a higher payoff than strategy $s_i'$.

**Weak Dominance:**
Strategy $s_i^*$ weakly dominates strategy $s_i'$ if:

$$u_i(s_i^*, s_{-i}) \geq u_i(s_i', s_{-i}) \quad \forall s_{-i} \in S_{-i}$$

and

$$\exists s_{-i} \in S_{-i} \text{ such that } u_i(s_i^*, s_{-i}) > u_i(s_i', s_{-i})$$

This means that strategy $s_i^*$ is never worse than strategy $s_i'$ and is sometimes better.

### How to Identify Dominant Strategies

To identify a dominant strategy for a player:

1. Compare the payoffs for each of the player's strategies against every possible combination of strategies chosen by the other players.
2. If one strategy consistently yields higher payoffs than all other strategies, it is strictly dominant.
3. If one strategy yields payoffs that are at least as high as all other strategies (and sometimes higher), it is weakly dominant.

### Iterated Elimination of Dominated Strategies

In games where there isn't an immediate dominant strategy, we can sometimes simplify the game through **iterated elimination of dominated strategies**:

1. Identify and eliminate all dominated strategies for each player.
2. In the reduced game, check again for newly dominated strategies.
3. Continue eliminating dominated strategies until no more can be eliminated.

This process can sometimes lead to a unique solution or at least simplify the analysis of complex games.

Mathematically, we define the sets of surviving strategies in each round of elimination:

Round 0: $S_i^0 = S_i$ (original strategy set for player $i$)

Round $k+1$: $S_i^{k+1} = \{s_i \in S_i^k | s_i \text{ is not strictly dominated by any other } s_i' \in S_i^k\}$

The process continues until $S_i^k = S_i^{k+1}$ for all players $i$ (i.e., no more strategies can be eliminated).

### Examples in Autonomous Driving

**Example 1: Emergency Braking Scenario**

Consider a scenario where a pedestrian suddenly appears in front of a self-driving car. The car has two strategies:
- Brake immediately
- Maintain speed

Regardless of what other road users do, braking is strictly dominant because avoiding a pedestrian collision is always preferred to maintaining speed and risking a collision.

**Example 2: Lane Selection with Traffic**

Imagine a self-driving car approaching a highway split with two lanes:
- Lane A: Has a 60% chance of being slower due to a potential accident
- Lane B: Has consistent but slightly slow traffic

For a risk-averse autonomous vehicle optimizing for predictable travel time, choosing Lane B would be a weakly dominant strategy because it provides more consistent performance across different scenarios.

## 2. Nash Equilibrium

### Definition and Importance

A **Nash equilibrium** is a set of strategies, one for each player, such that no player can benefit by changing their strategy unilaterally while the other players keep their strategies unchanged. Nash equilibria represent stable points in strategic interactions where no player has an incentive to deviate from their chosen strategy.

### Mathematical Formulation of Nash Equilibrium

Consider a game with $n$ players:
- Each player $i$ has a strategy set $S_i$
- Let $s = (s_1, s_2, ..., s_n)$ denote a strategy profile, where $s_i \in S_i$ is player $i$'s strategy
- Let $s_{-i}$ denote the strategies of all players except player $i$
- Let $u_i(s_i, s_{-i})$ be player $i$'s payoff function

A strategy profile $s^* = (s_1^*, s_2^*, ..., s_n^*)$ is a **Nash equilibrium** if, for all players $i$ and all strategies $s_i \in S_i$:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*)$$

This inequality states that no player can unilaterally improve their payoff by deviating from their equilibrium strategy.

The concept is crucial because:
- It predicts likely outcomes in strategic interactions
- It represents stable states where players' expectations about each other are confirmed
- It helps in designing robust decision-making algorithms for autonomous systems

### Pure Strategy Nash Equilibria

A **pure strategy Nash equilibrium** occurs when each player selects a single strategy (rather than a probabilistic mix of strategies) and no player can benefit by unilaterally changing their strategy.

**How to find Pure Strategy Nash Equilibria:**

1. For each possible combination of strategies, check if any player could improve their payoff by unilaterally switching to a different strategy.
2. If no player can improve by switching, that combination is a Nash equilibrium.

For a two-player game represented as a payoff matrix, a common method to identify pure strategy Nash equilibria is:

1. For each cell in the matrix, check if player 1 can improve by deviating (compare with other cells in the same column)
2. Check if player 2 can improve by deviating (compare with other cells in the same row)
3. If neither player can improve by deviating, that cell represents a Nash equilibrium

### Mixed Strategy Nash Equilibria

A **mixed strategy Nash equilibrium** involves players choosing probability distributions over their available strategies rather than selecting a single strategy.

Let's denote:
- $\Delta(S_i)$ as the set of all probability distributions over player $i$'s strategy set $S_i$
- $\sigma_i \in \Delta(S_i)$ as a mixed strategy for player $i$
- $\sigma_i(s_i)$ as the probability that player $i$ assigns to pure strategy $s_i$
- $u_i(\sigma_i, \sigma_{-i})$ as player $i$'s expected payoff when players use mixed strategies

A mixed strategy profile $\sigma^* = (\sigma_1^*, \sigma_2^*, ..., \sigma_n^*)$ is a **Nash equilibrium** if, for all players $i$ and all mixed strategies $\sigma_i \in \Delta(S_i)$:

$$u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*)$$

Mixed strategies become important when:
- No pure strategy Nash equilibrium exists
- Players want to be unpredictable
- The game involves uncertainty or risk management

### How to Find Nash Equilibria Mathematically

**For Pure Strategy Nash Equilibria:**
1. Examine each cell in the payoff matrix
2. Check if any player could improve their payoff by unilaterally switching strategies
3. Mark combinations where no player can improve as Nash equilibria

**For Mixed Strategy Nash Equilibria:**

Consider a 2×2 game where player 1 has strategies $A$ and $B$, and player 2 has strategies $C$ and $D$. Let player 1 play $A$ with probability $p$ and player 2 play $C$ with probability $q$.

The expected payoff for player 1 playing $A$ is:
$$E_1(A) = q \cdot u_1(A,C) + (1-q) \cdot u_1(A,D)$$

The expected payoff for player 1 playing $B$ is:
$$E_1(B) = q \cdot u_1(B,C) + (1-q) \cdot u_1(B,D)$$

For player 1 to be indifferent between $A$ and $B$ (a necessary condition for mixed strategy equilibrium):
$$E_1(A) = E_1(B)$$

Solving this equation gives us the equilibrium value of $q$.

Similarly, we can calculate the equilibrium value of $p$ by setting player 2's expected payoffs equal:
$$E_2(C) = E_2(D)$$

Using the Nashpy library in Python, we can calculate Nash equilibria programmatically, which we'll explore in our coding exercise.

### Examples in Self-Driving Cars

**Example 1: Lane-Change Game**

Consider two cars approaching a lane merge, with the following payoff matrix:

$$
\begin{array}{c|cc}
\text{Car 1 \textbackslash Car 2} & \text{Accelerate} & \text{Yield} \\
\hline
\text{Accelerate} & (-5,-5) & (2,-1) \\
\text{Yield} & (-1,2) & (-2,-2)
\end{array}
$$

This game has two pure strategy Nash equilibria: (Accelerate, Yield) and (Yield, Accelerate), representing situations where one car yields while the other takes advantage.

**Example 2: Traffic Light Approach**

A self-driving car approaching a yellow light faces a decision to brake or proceed, while considering other vehicles. This scenario often has mixed strategy Nash equilibria, especially in cases where the optimal decision depends on predicting other drivers' behaviors.

Consider the following payoff matrix for a car and a pedestrian at a crosswalk:

$$
\begin{array}{c|cc}
\text{Car \textbackslash Pedestrian} & \text{Cross} & \text{Wait} \\
\hline
\text{Proceed} & (-10,-10) & (2,0) \\
\text{Brake} & (0,1) & (-1,-1)
\end{array}
$$

If we solve for the mixed Nash equilibrium:

For the pedestrian to be indifferent between crossing and waiting:
$$p \cdot (-10) + (1-p) \cdot (1) = p \cdot (0) + (1-p) \cdot (-1)$$

Solving for $p$ (probability the car proceeds):
$$-10p + (1-p) = -1(1-p)$$
$$-10p + 1-p = -1+p$$
$$-11p + 1 = -1+p$$
$$-12p = -2$$
$$p = \frac{1}{6}$$

For the car to be indifferent between proceeding and braking:
$$q \cdot (-10) + (1-q) \cdot (2) = q \cdot (0) + (1-q) \cdot (-1)$$

Solving for $q$ (probability the pedestrian crosses):
$$-10q + 2(1-q) = -1(1-q)$$
$$-10q + 2-2q = -1+q$$
$$-12q + 2 = -1+q$$
$$-13q = -3$$
$$q = \frac{3}{13}$$

So the mixed Nash equilibrium has the car proceeding with probability $\frac{1}{6}$ and the pedestrian crossing with probability $\frac{3}{13}$.

## 3. Lane-Changing Scenarios

Lane-changing scenarios present rich opportunities to apply game theory concepts in autonomous driving, as they involve strategic interactions between multiple vehicles with potentially different objectives.

### Game-Theoretic Modeling of Lane-Changing Decisions

When modeling lane-changing decisions using game theory, we consider:

1. **Players:** Vehicles involved in the potential lane change (the lane-changer and affected vehicles)
2. **Actions:** Possible maneuvers (change lanes, maintain speed, accelerate, decelerate)
3. **Payoffs:** Outcomes for each vehicle (time saved, safety margin, comfort, fuel efficiency)
4. **Information:** What each vehicle knows about others' positions, speeds, and intentions

We can mathematically formulate a lane-changing game as follows:

- Players: $N = \{1, 2, ..., n\}$ (vehicles in the scenario)
- Action sets: $A_i$ for each player $i \in N$
- State space: $S$ (positions and velocities of all vehicles)
- Transition function: $T: S \times A_1 \times A_2 \times ... \times A_n \rightarrow S$
- Reward functions: $R_i: S \times A_1 \times A_2 \times ... \times A_n \times S \rightarrow \mathbb{R}$ for each player $i \in N$

The lane-changing game can be represented in normal form (as a payoff matrix) or extensive form (as a decision tree), depending on whether we're modeling simultaneous or sequential decisions.

### Different Driver Behaviors

Driver behavior significantly impacts strategic interactions in lane-changing scenarios:

1. **Aggressive Drivers:**
   - Prioritize time savings over safety margins
   - More likely to accelerate when another car signals a lane change
   - Higher tolerance for close-proximity maneuvers
   - Payoff matrix heavily weights progress and speed
   - Mathematically: $u_{\text{aggressive}}(s) = w_t \cdot \text{time\_saved} - w_s \cdot \text{safety\_margin}$ where $w_t \gg w_s$

2. **Defensive Drivers:**
   - Prioritize safety margins over time savings
   - More likely to yield when another car signals a lane change
   - Prefer larger gaps between vehicles
   - Payoff matrix heavily weights collision avoidance
   - Mathematically: $u_{\text{defensive}}(s) = w_t \cdot \text{time\_saved} - w_s \cdot \text{safety\_margin}$ where $w_s \gg w_t$

3. **Cooperative Drivers:**
   - Balance personal goals with system-level efficiency
   - Willing to yield if it improves overall traffic flow
   - May alternate between yielding and proceeding in repeated interactions
   - Payoff matrix includes benefits to other players
   - Mathematically: $u_{\text{cooperative}}(s) = w_t \cdot \text{time\_saved} - w_s \cdot \text{safety\_margin} + w_c \cdot \sum_{j \neq i} u_j(s)$

### Strategic Considerations for Autonomous Vehicles

When designing lane-changing strategies for autonomous vehicles that interact with human drivers, key considerations include:

1. **Intention Signaling:** Clearly communicating planned maneuvers to surrounding vehicles
2. **Behavior Prediction:** Classifying surrounding drivers as aggressive, defensive, or cooperative based on observed behavior
3. **Adaptation:** Adjusting strategy based on the predicted behavior of surrounding drivers
4. **Risk Management:** Balancing efficiency with safety margins appropriate to the situation
5. **Social Norms:** Adhering to unwritten traffic conventions that human drivers expect
6. **Ethical Considerations:** Determining how to balance the vehicle's passengers' interests against those of other road users

We can model the decision-making process for lane-changing as a sequential game with incomplete information, where the autonomous vehicle must estimate the types and payoffs of surrounding human drivers. Using Bayesian game theory:

- Let $\Theta_i$ be the set of possible types for player $i$ (e.g., aggressive, defensive, cooperative)
- Let $p(\theta)$ be the prior probability distribution over the type profiles
- Let $u_i(s,\theta)$ be the payoff to player $i$ when the strategy profile is $s$ and the type profile is $\theta$

A Bayesian Nash equilibrium is a strategy profile $s^*$ such that for each player $i$, each type $\theta_i$, and each alternative strategy $s_i'$:

$$\sum_{\theta_{-i}} p(\theta_{-i}|\theta_i)u_i(s_i^*(\theta_i), s_{-i}^*(\theta_{-i}), \theta) \geq \sum_{\theta_{-i}} p(\theta_{-i}|\theta_i)u_i(s_i', s_{-i}^*(\theta_{-i}), \theta)$$

By understanding the strategic landscape of lane-changing decisions through the lens of dominant strategies and Nash equilibria, we can design more effective decision-making algorithms for autonomous vehicles that navigate complex traffic scenarios safely and efficiently.

## Conclusion

Dominant strategies and Nash equilibria provide powerful analytical tools for understanding and designing autonomous driving systems. These concepts help us predict how vehicles will interact in traffic scenarios and develop algorithms that make rational decisions while accounting for the strategic behavior of other road users.

The mathematical formulations presented in this lesson give us a rigorous framework for analyzing strategic interactions in autonomous driving scenarios. By applying these concepts to lane-changing decisions, we can create autonomous vehicles that make optimal decisions while accounting for the behavior of human drivers and other autonomous vehicles.

In the accompanying coding exercise, we'll implement a lane-changing scenario that demonstrates these concepts, using the Nashpy library to calculate Nash equilibria and visualize different outcomes based on driver behaviors.