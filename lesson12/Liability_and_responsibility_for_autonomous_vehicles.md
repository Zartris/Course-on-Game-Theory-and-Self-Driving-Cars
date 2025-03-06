# Liability and Responsibility in Accidents Involving Self-Driving Cars

## Objective

This lesson examines the complex landscape of liability and responsibility frameworks for autonomous vehicle accidents. We will explore how game theory can be applied to determine fault in multi-agent collisions, analyze the legal and insurance implications of autonomous systems, and develop models for responsibility allocation in mixed-autonomy traffic environments.

## 1. The Liability Dilemma: A Game of Uncertainty

The introduction of autonomous vehicles (AVs) fundamentally disrupts traditional legal frameworks for accident liability.  Existing laws, designed for human drivers, struggle to address scenarios where decisions are made by complex algorithms.  This creates a "liability dilemma," a situation characterized by uncertainty and strategic interactions between multiple stakeholders, making it a prime candidate for analysis through the lens of game theory.

The core problem lies in assigning responsibility when an AV is involved in an accident.  Is it the fault of the manufacturer, the software developer, the "driver" (who may be merely a passenger), or even another human driver interacting with the AV?  Unlike simple accidents involving only human drivers, where negligence can often (though not always) be determined through established legal principles, AV accidents introduce layers of complexity:

*   **Algorithmic Decision-Making:**  AVs rely on complex algorithms, often involving machine learning, that make decisions in ways that may not be fully transparent or predictable, even to their creators.  This "black box" nature makes it difficult to apply traditional notions of foreseeability and fault.
*   **Multiple Potential Contributors:**  An AV's behavior is the result of interactions between hardware, software, sensor data, and potentially, the actions of other road users.  Disentangling these contributions to determine causation is a significant challenge.
*   **Strategic Interactions:**  Human drivers, aware of the presence of AVs (and potentially their limitations), may alter their own driving behavior.  This creates a strategic environment where the actions of one agent (human or AV) influence the risks and outcomes for others.  For example, a human driver might become more aggressive, knowing that an AV is programmed to be cautious. This is analogous to a multi-player game.
*   **Evolving Technology:**  AV technology is constantly evolving through software updates and machine learning.  This means that the "rules of the game" are not static, further complicating liability assessment.
*   **Unclear Legal Precedents:** The lack of extensive legal precedent specific to AV accidents creates uncertainty, encouraging strategic behavior by stakeholders (e.g., manufacturers might try to limit their liability through contractual agreements or by influencing regulatory frameworks).

These uncertainties can be framed as a *game* with multiple *players* (manufacturers, operators, other drivers, pedestrians, etc.), each with their own *strategies* (design choices, driving behaviors, legal arguments) and *payoffs* (avoiding liability, minimizing costs, maximizing safety). The "rules" of this game are currently ill-defined and subject to interpretation, leading to strategic maneuvering and potentially suboptimal outcomes.

For instance, consider a simple scenario: an AV collides with a human-driven vehicle.

*   **The AV Manufacturer's Strategy:** Might involve arguing that the AV's software was not defective, and the accident was caused by the unpredictable actions of the human driver.
*   **The Human Driver's Strategy:** Might involve claiming that the AV behaved erratically, forcing them into an unavoidable collision.
*   **The Insurer's Strategy:** Might involve minimizing payouts by contesting liability or seeking contribution from multiple parties.

The outcome of this "game" depends on the available evidence, the interpretation of existing laws, and the strategic choices of each player.  Traditional legal approaches, based on assigning fault to a single party, often fail to capture the complexities of these interactions. This is where game theory becomes essential.  By modeling accident scenarios as strategic games, we can analyze the incentives of different stakeholders, predict potential outcomes, and design more robust and equitable liability frameworks. The next section will delve into these game-theoretic approaches.
Okay, here's the complete text for Sections 2 and 3, incorporating the LaTeX math formatting and all the refinements we've discussed. This provides a cohesive and detailed treatment of the game-theoretic aspects of AV liability.

---

## 2. Game-Theoretic Approaches to Fault Determination

As established in Section 1, accidents involving autonomous vehicles (AVs) often involve complex interactions and uncertainties, making traditional fault determination methods inadequate. Game theory provides a powerful framework for analyzing these situations by modeling them as strategic interactions between rational agents.

### 2.1 Modeling Accident Scenarios as Games

The first step is to represent an accident scenario as a *game*. This involves defining the following key elements:

*   **Players (N):** The entities involved in the interaction. This could include:
    *   Autonomous Vehicles (represented by their controlling algorithms)
    *   Human Drivers
    *   Pedestrians
    *   Cyclists
    *   Potentially even infrastructure elements (e.g., a malfunctioning traffic light)

*   **Actions (A):** The set of possible actions each player can take. Examples include:
    *   Accelerating, braking, steering (for vehicles)
    *   Changing lanes
    *   Entering a crosswalk (for pedestrians)
    *   Signaling (or failing to signal)
    *   Following the rules/ Disregarding the rules of traffic.

*   **Payoffs (u):** A numerical representation of the outcome for each player, given the actions taken by all players. Payoffs can represent:
    *   Avoiding an accident (positive payoff)
    *   Being involved in an accident (negative payoff, potentially varying with severity)
    *   Time saved/lost
    *   Legal liability (negative payoff)

*   **Information Structure:**
    *   **Complete Information:** All players know the rules of the game, other players' actions, and payoffs.
    *   **Incomplete Information:** Players might have private information about their own intentions, capabilities, or risk preferences. For instance, a human driver might know they are distracted, but the AV might not.
    *   **Perfect Information:** Players know each other's previous moves.
    *   **Imperfect Information:** Players do not know each other's previous move. For example in the game Rock-Paper-Scissors.

*   **Game Representation:**
    *   **Normal Form (Strategic Form):** Represented by a matrix showing players, strategies, and payoffs. Suitable for simultaneous-move games (where players choose actions without knowing the choices of others). Example:

        |          | Driver B: Swerve Left | Driver B: Swerve Right | Driver B: Brake |
        |----------|-----------------------|------------------------|-----------------|
        | Driver A: Swerve Left | (-1, -1)             | (0, -5)              | (-2, -3)        |
        | Driver A: Swerve Right| (-5, 0)              | (-1, -1)              | (-3, -2)       |
        | Driver A: Brake      | (-3, -2)             |   (-2,-3)            | (-4, -4)        |

        This simple example shows a two-player game where each driver has three choices. The payoffs represent the negative consequences of each combination of actions (e.g., both swerving left results in a minor collision, -1, -1).

    *   **Extensive Form:** Represented by a game tree, showing the sequence of moves, decision nodes, and payoffs. Suitable for sequential-move games. This is more realistic for many accident scenarios, where actions unfold over time.
    *   **Other Forms:** Bayesian games for describing incomplete information, stochastic games for modeling probabilistic transitions.

**Mathematical Formulation (Example - Normal Form):**

Let:

*   $N = \{1, 2, ..., n\}$ be the set of players.
*   $A_i$ be the set of actions available to player $i$.
*   $A = A_1 \times A_2 \times ... \times A_n$ be the set of all possible action profiles (combinations of actions).
*   $u_i : A \rightarrow \mathbb{R}$ be the payoff function for player $i$, mapping each action profile to a real-valued payoff.

### 2.2 Nash Equilibria in Accident Avoidance

A central concept in game theory is the *Nash Equilibrium*. A Nash Equilibrium is a set of strategies (one for each player) such that no player can improve their payoff by *unilaterally* changing their strategy, assuming all other players keep their strategies the same.

*   **Definition:** An action profile $a^* = (a_1^*, a_2^*, ..., a_n^*)$ is a Nash Equilibrium if, for every player $i$:

    $$u_i(a^*) \geq u_i(a_i, a_{-i}^*), \quad \forall a_i \in A_i$$

    Where $a_{-i}^*$ represents the strategies of all players *except* player $i$.

*   **Accident Avoidance as an Equilibrium:** Ideally, we want accident avoidance to be a Nash Equilibrium. This means that if all agents are acting rationally to maximize their own payoffs (which includes avoiding accidents), they will choose actions that lead to a safe outcome.
*   **Non-Equilibrium Outcome:** If accident avoidance is *not* a Nash Equilibrium, it means that at least one player has an incentive to deviate from a safe strategy, potentially leading to an accident.
*   **Multiple Equilibria:** Many games have multiple Nash Equilibria. For example, in a scenario where two cars are approaching an intersection, both swerving left *or* both swerving right might be Nash Equilibria (avoiding a head-on collision), but one might be safer or more efficient than the other. This is the *equilibrium selection problem*.
*   **Refinements:**
    *   **Trembling Hand Perfection:** Accounts for the possibility of small mistakes (trembles). An equilibrium is trembling-hand perfect if it remains an equilibrium even when there's a small probability that players will choose a non-intended action.
    *   **Subgame Perfection:** Relevant for extensive-form games. It requires that the strategies form a Nash Equilibrium not just in the overall game, but also in every subgame (every possible continuation of the game from a given point onward).

### 2.3 Counterfactual Reasoning for Blame Attribution

Counterfactual reasoning is crucial for determining fault after an accident: "If player *i* had acted differently, would the accident have been avoided?"

*   **Definition:** A counterfactual statement takes the form: "If *X* had been different, then *Y* would have been different." In the context of accidents: "If the AV had braked earlier, the collision would not have occurred."

*   **Mathematical Framework (using causal models):**
    *   Represent the accident scenario as a causal graph, where nodes represent variables (e.g., vehicle speed, braking time, road conditions) and edges represent causal relationships.
    *   Use structural equations to model how variables influence each other. For example:

        $Collision = f(AV\_Speed, Human\_Speed, AV\_Braking\_Time, ...)$

    *   To evaluate a counterfactual, we intervene on the causal graph, changing the value of a variable (e.g., setting *AV\_Braking\_Time* to an earlier value) and then recalculating the outcome (Collision).

*   **Necessary and Sufficient Causes:**
    *   **Necessary Cause:** A factor without which the accident would *not* have occurred.
    *   **Sufficient Cause:** A factor that, by itself, is enough to guarantee the accident.

*   **Epistemological Challenges:**
    *   **Model Uncertainty:** Our causal models are always simplifications of reality. We may not know all the relevant factors or their precise relationships.
    *   **Counterfactual Indeterminacy:** With complex systems like AVs, it may be impossible to definitively say what *would* have happened in a counterfactual scenario, especially if the AI's decision-making process is not fully transparent.

### 2.4 Shapley Value and Responsibility Allocation

When multiple agents contribute to an accident, the Shapley Value provides a way to fairly allocate responsibility. It's based on cooperative game theory.

*   **Cooperative Game:** A game where players can form coalitions and share payoffs.
*   **Characteristic Function (v):** Assigns a value (e.g., the cost of the accident) to each possible coalition of players. $v(S)$ represents the total cost that coalition $S$ would incur if they were solely responsible.

*   **Shapley Value (φ):** For each player, compute the value by considering all possible coalitions. For each coalition, determine the marginal contribution of the player which is $v(S \cup \{i\}) - v(S)$. Then average all the marginal contributions.

*   **Mathematical Definition:** The Shapley Value for player *i*, denoted $\phi_i(v)$, is:

    $$\phi_i(v) = \frac{1}{n!} \sum_{R} [v(S_i(R) \cup \{i\}) - v(S_i(R))]$$

    Where:
    *   $n$ is the total number of players.
    *   $R$ is a permutation (ordering) of the players.
    *   $S_i(R)$ is the set of players that precede player *i* in the ordering $R$.
    *   The sum is taken over all possible orderings of the players.

*   **Fairness Properties:** The Shapley Value satisfies several desirable axioms, including:
    *   **Efficiency:** The sum of the Shapley Values equals the total value of the game (the total cost of the accident).
    *   **Symmetry:** Players with equal contributions receive equal shares of responsibility.
    *   **Dummy Player:** A player who contributes nothing to any coalition receives a Shapley Value of zero.
    *   **Additivity:** If two games are combined, the Shapley Value in the combined game is the sum of the Shapley Values in the individual games.

### 2.5 Strategic Behavior and Moral Hazard in Mixed-Autonomy Traffic

Mixed autonomy (human and AVs sharing the road) introduces new strategic challenges:

*   **Moral Hazard:** When one party (e.g., a human driver) takes on more risk because they know another party (e.g., an AV) will act to mitigate that risk. For example, a human driver might drive more aggressively, knowing that the AV is programmed to be cautious and avoid collisions.
*   **Exploitation of AV Behavior:** Humans might learn to "game" the system, predicting how AVs will react and exploiting that knowledge to their advantage (e.g., cutting off an AV, knowing it will brake).
*   **Adaptation and Counter-Adaptation:** AVs and human drivers will likely engage in a dynamic process of adaptation and counter-adaptation. As AVs become more sophisticated, humans may adapt their behavior, and vice versa, leading to a constantly shifting strategic landscape.

**Game-Theoretic Models:**

*   **Stackelberg Games:** One player (the leader) moves first, and the other player (the follower) responds. This can model situations where an AV's known behavior influences the actions of human drivers.
*   **Repeated Games:** The same interaction occurs multiple times, allowing for learning and adaptation. This can model the long-term evolution of driving behavior in mixed-autonomy traffic.
*   **Evolutionary Game Theory:** Models how strategies evolve over time in a population of interacting agents. This can be used to study the emergence of norms and conventions in mixed-autonomy traffic.


## 3. Allocating Responsibility in Multi-Agent Interactions

This section builds upon the Shapley Value and counterfactual reasoning concepts introduced earlier, applying them to the specific context of multi-agent collisions and shared responsibility in autonomous vehicle (AV) accidents.

### 3.1 Contribution Analysis and the Shapley Value

Recall from Section 2.4 that the Shapley Value provides a method for fairly distributing the "cost" (or "value") of a cooperative game among its players, based on their marginal contributions to all possible coalitions. In the context of AV accidents, the "cost" is typically the damage caused by the accident, and the "players" are the various entities that may have contributed to it (AVs, human drivers, infrastructure, etc.).

*   **Reiteration of Shapley Value:** The Shapley Value for player *i*, denoted $\phi_i(v)$, is:

    $$\phi_i(v) = \frac{1}{n!} \sum_{R} [v(S_i(R) \cup \{i\}) - v(S_i(R))]$$

    Where:
    *   $n$ is the total number of players.
    *   $R$ is a permutation (ordering) of the players.
    *   $S_i(R)$ is the set of players that precede player *i* in the ordering $R$.
    *   $v(S)$ is the characteristic function, representing the cost associated with coalition $S$.

*   **Quantitative Measures of Contribution:** To apply the Shapley Value, we need a way to quantify the contribution of each agent to the accident. This is where counterfactual reasoning and data analysis become crucial. Several approaches can be used:

    *   **Counterfactual Measures:**  Using simulations or causal models (as discussed in Section 2.3), we can estimate the probability of the accident occurring *with* and *without* a particular agent's action.  For example, we might simulate the accident scenario with the AV braking earlier and determine if the collision would have been avoided. The difference in probability can be used as a measure of contribution.
    *   **Responsibility Scores:**  Develop metrics based on observable factors like:
        *   **Proximity:**  How close was the agent to the point of impact?
        *   **Speed:**  Was the agent exceeding the speed limit or driving at an unsafe speed for the conditions?
        *   **Reaction Time:**  How quickly did the agent react to the hazard?
        *   **Visibility:** What was each agent view and were there any obstructions.
        *   **Adherence to Traffic Rules:**  Did the agent violate any traffic laws (e.g., running a red light, failing to yield)?

    *   **Connecting to Shapley:** These quantitative measures can inform the characteristic function *v(S)*.  For example:
        *   $v(\emptyset)$ = 0 (no accident if no agents are involved).
        *   $v(\{A\})$ might represent the expected cost (or probability) of an accident if only agent A were present and acting as they did.
        *   $v(\{A, B\})$ might represent the expected cost of the accident with both A and B present.

*   **Mathematical Example:**
    Consider a scenario where the probability that agent A (a human driver) alone will cause an accident is 0.1, agent B (an AV) alone is 0.2, and if they both act as they did, it is 0.5. This gives:

    $$v(\{\}) = 0$$
    $$v(\{A\}) = -0.1$$
    $$v(\{B\}) = -0.2$$
    $$v(\{A, B\}) = -0.5$$

The Shapley value gives:

Agent A: $$0.5(-0.1 - 0) + 0.5(-0.5 -(-0.2)) = -0.2$$  
Agent B: $$0.5(-0.2 - 0) + 0.5(-0.5 -(-0.1)) = -0.3$$

This calculation assigns a responsibility "cost" of -0.2 to agent A and -0.3 to agent B, reflecting their respective contributions to the accident.

### 3.2 Shared Responsibility Frameworks as Strategic Games

Different legal frameworks for shared responsibility create different strategic incentives for the involved parties.

*   **Joint and Several Liability:** Under this framework, each party involved in an accident can be held fully responsible for *all* the damages, regardless of their individual contribution.
    *   **Game-Theoretic Framing:** This can be viewed as a game where players have an incentive to shift blame to others to avoid paying the full cost. It can lead to a "race to the bottom" in terms of safety, as each party might assume others will take precautions. It can also lead to strategic litigation, as parties try to prove the other's negligence.
*   **Proportional Liability:** Each party is responsible only for their share of the damages, proportional to their contribution to the accident.
    *   **Game-Theoretic Framing:** This aligns more closely with the Shapley Value concept. It incentivizes each party to minimize their own contribution to the accident risk, as they will only be liable for their share.
*   **Negotiation and Bargaining:** In many cases, the allocation of responsibility is not determined by a strict formula but through negotiation and bargaining between stakeholders (e.g., insurance companies, manufacturers, individuals).
    *   **Game-Theoretic Framing:** This can be modeled as a bargaining game, where players have different bargaining power (based on resources, legal representation, etc.) and information asymmetry (one party might know more about the accident circumstances than another). The outcome of the bargaining game will depend on factors like the players' risk aversion, their outside options, and their beliefs about the likely outcome of litigation.
*   **Mechanism Design:** Liability rules can be viewed as a *mechanism* designed to achieve specific societal goals, such as:
    *   **Incentivizing Safety:** Designing rules that encourage manufacturers to develop safer AVs and drivers to behave more cautiously.
    *   **Ensuring Fair Compensation:** Designing rules that ensure victims of accidents receive adequate compensation.
    *  **Promoting Innovation:** Finding a balance between holding manufacturers accountable and not stifling innovation with excessive liability.

    Mechanism design aims to create "rules of the game" that lead to desirable outcomes, even when players are acting in their own self-interest.

### 3.3 The Roles of Different Stakeholders

The interaction between manufacturers, operators, regulators, and other drivers can be modeled as a multi-player game with distinct roles and incentives.

*   **Multi-Player Game:**  Each stakeholder has a set of possible actions (strategies) and payoffs (which could include profits, safety, legal liability, reputation).
*   **Manufacturer Incentives:**
    *   **Actions:** Design choices, testing procedures, software updates, warnings, and disclaimers.
    *   **Payoffs:** Profit maximization, market share, avoiding lawsuits, maintaining reputation. Liability rules influence these incentives.  Strict liability might encourage more conservative designs and extensive testing, while negligence-based rules might allow for more risk-taking.
*   **Operator Incentives:** (This could be a human driver, a fleet operator, or even the AV itself, represented by its control algorithm).
    *   **Actions:** Driving behavior, maintenance, software updates, monitoring the AV's performance, intervening when necessary.
    *   **Payoffs:**  Avoiding accidents, minimizing travel time, avoiding legal liability.
*   **Regulatory Role:**
    *   **Actions:** Setting safety standards, certification requirements, data recording mandates, rules for accident investigation, and establishing liability frameworks.
    *   **Payoffs:** Promoting public safety, fostering innovation, balancing competing interests.
    *   **Stackelberg Game:** The regulatory framework can often be modeled as a Stackelberg game. The regulator (leader) sets the rules, and then the manufacturers and operators (followers) react to those rules, choosing their actions to maximize their own payoffs within the established constraints.

## 4. Data, Machine Learning, and Game-Theoretic Analysis

This section explores how data from AVs and machine learning techniques can be integrated with the game-theoretic framework.

### 4.1 Data-Driven Game Representation
* **Using Data to inform the game:**
Data from Event Data Recorders (EDRs), sensors, and other sources can be used to construct more realistic and accurate game-theoretic models of accident scenarios.
* **Causal Inference from Observational Data:**
Techniques from causal inference can help determine cause-and-effect.
* **Simulations from Counterfactuals**
Counterfactual can be used to determine what would have happened in a different situation.

### 4.2 Machine Learning for Strategic Analysis

*   **Reinforcement Learning for Counterfactuals:** Reinforcement learning (RL) can be used to explore "what if" scenarios, which are essential for counterfactual reasoning and blame attribution.
    *   **Training RL Agents:**  RL agents can be trained in simulated environments that mimic real-world driving conditions. These agents can learn to perform driving tasks and respond to various situations.
    *   **Testing Counterfactuals:** By modifying the environment or the agent's parameters, we can simulate counterfactual scenarios. For example, we can change the agent's braking behavior and observe how this affects the outcome of a potential collision.
    *   **Estimating Causal Effects:**  By comparing the outcomes of different scenarios, we can estimate the causal effect of specific actions or factors on the probability or severity of an accident.

*   **Multi-Agent Reinforcement Learning (MARL):** MARL is particularly relevant for modeling interactions between multiple agents (e.g., human drivers and AVs).
    *   **Modeling Strategic Interactions:** MARL allows us to study how agents learn and adapt to each other's behavior in a dynamic environment.
    *   **Emergent Behavior:**  MARL can reveal emergent behaviors that might not be obvious from analyzing individual agents in isolation. For example, it can show how human drivers might learn to exploit the predictable behavior of AVs.
    *   **Game Theory in RL:** Use Nash-Equilibrium to find optimal policies.

## 5. Broader Context and Future Directions (Brief)

This section provides a concise overview of the broader context and future directions.

*   **Evolving Liability:** The legal frameworks for AV liability are still evolving. As technology advances and more AVs are deployed, new challenges and legal questions will continue to arise.  The interaction between legal precedent, technological development, and societal acceptance will shape the future of AV liability.
*   **Proactive Safety:** There's a growing emphasis on shifting from *reactive* liability (determining fault after an accident) to *proactive* safety assurance (preventing accidents in the first place). Game theory can play a crucial role in designing incentive structures that reward proactive safety investments.  This includes:
    *   **Continuous Monitoring:** Using data from AVs to continuously monitor their performance and identify potential safety risks.
    *   **Preventive Intervention:** Developing systems that can automatically intervene to prevent accidents (e.g., by taking control of the vehicle or issuing warnings).
* **Ethical Consideration:** There needs to be a balance between fairness, transparency and explainability when creating these models.

This completes the content for the revised sections 3, 4, and 5. We've maintained a strong focus on game-theoretic concepts and their application to AV liability, while also incorporating relevant connections to data analysis, machine learning, and the broader legal and ethical context. The examples and mathematical formulations provide concrete illustrations of the concepts, making them more accessible to students.


## Conclusion

This lesson has examined the complex landscape of liability and responsibility for autonomous vehicles. We have explored legal frameworks, game-theoretic approaches to fault determination, responsibility allocation in multi-agent collisions, and practical models for implementing liability systems. As autonomous vehicles become more prevalent, these frameworks will continue to evolve, blending traditional legal concepts with new approaches that acknowledge the unique characteristics of autonomous systems. The most effective liability frameworks will balance innovation and safety, providing clear rules while remaining adaptable to technological developments.

## References

1. Gurney, J.K. (2017). Autonomous Vehicles: Regulatory Challenges and Models for Liability. *Harvard Journal of Law & Technology*, *31*(2), 949-988.

2. Surden, H., & Williams, M.A. (2016). Technological Opacity, Predictability, and Self-Driving Cars. *Cardozo Law Review*, *38*, 121-181.

3. Glancy, D.J. (2015). Autonomous and Automated and Connected Cars—Oh My! First Generation Autonomous Cars in the Legal Ecosystem. *Minnesota Journal of Law, Science & Technology*, *16*, 619-691.

4. Halpern, J.Y., & Pearl, J. (2005). Causes and Explanations: A Structural-Model Approach. Part I: Causes. *The British Journal for the Philosophy of Science*, *56*(4), 843-887.

5. Shapley, L.S. (1953). A Value for n-person Games. *Contributions to the Theory of Games*, *2*(28), 307-317.

6. Klischat, M., Langer, M., & Krause, J. (2019). Using Counterfactual Reasoning and Reinforcement Learning for Decision-Making in Autonomous Driving. *Proceedings of the Conference on Robot Learning*, 749-758.

7. Nyholm, S. (2018). The Ethics of Crashes with Self-Driving Cars: A Roadmap, II. *Philosophy Compass*, *13*(7), e12506.

8. Javdani, S., Srinivasa, S.S., & Bagnell, J.A. (2015). Shared Autonomy via Hindsight Optimization. *Robotics: Science and Systems*, *11*.

9. Lagnado, D.A., Gerstenberg, T., & Zultan, R. (2013). Causal Responsibility and Counterfactuals. *Cognitive Science*, *37*(6), 1036-1073.

10. Schellekens, M. (2018). No-Fault Compensation Schemes for Self-Driving Vehicles. *Law, Innovation and Technology*, *10*(2), 314-333.

11. Vladeck, D.C. (2014). Machines Without Principals: Liability Rules and Artificial Intelligence. *Washington Law Review*, *89*, 117-150.

12. Elvy, S.A. (2018). Contracting in the Age of the Internet of Things: Article 2 of the UCC and Beyond. *Hofstra Law Review*, *44*(3), 839-941.

13. Marchant, G.E., & Lindor, R.A. (2012). The Coming Collision Between Autonomous Vehicles and the Liability System. *Santa Clara Law Review*, *52*(4), 1321-1340.

14. Abraham, K.S., & Rabin, R.L. (2019). Automated Vehicles and Manufacturer Responsibility for Accidents: A New Legal Regime for a New Era. *Virginia Law Review*, *105*(1), 127-171.

15. Choi, J.K., & Ji, Y.G. (2015). Investigating the Importance of Trust on Adopting an Autonomous Vehicle. *International Journal of Human-Computer Interaction*, *31*(10), 692-702.