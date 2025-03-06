# Introduction to 2D Multi-Robot Simulators - Player/Stage

## Objective

This lesson introduces the Player/Stage 2D multi-robot simulator, bridging the gap between abstract game-theoretic models and practical implementation. You'll learn to set up simulations, control robots, and integrate decision-making algorithms, enabling you to test and validate game-theoretic strategies in a controlled environment.

## 1. Why Simulate? Bridging Theory and Reality

*   **Simulation's Role:** Simulation is essential for robotics research. It allows you to test algorithms without the cost and risk of using real robots.  Critically, it lets you validate game-theoretic strategies *before* deploying them on physical hardware.
*   **2D vs. 3D:**  2D simulators (like Player/Stage) are computationally efficient and suitable for many multi-agent problems (navigation, coordination). 3D simulators (like Gazebo) offer higher realism but are more complex. Player/Stage is a good starting point for many game-theoretic experiments.
*   **Kinematic vs Dynamic:** Kinematic simulators (like Stage) focus on position and velocity, ignoring forces. Dynamic simulators model forces and accelerations, offering higher fidelity but at a computational cost.

## 2. Player/Stage: Core Concepts

*   **Client-Server Architecture:** Player/Stage uses a client-server model.
    *   **Player:** The *server* manages the simulation and provides an interface to the robots (real or simulated).
    *   **Stage:**  The 2D simulator (often used with Player).
    *   **Client Programs:** Your algorithms, written in C++, Python, or other languages, connect to the Player server to control the robots.

*   **Devices and Interfaces:** Player uses *devices* to represent robot components (sensors, actuators).  *Interfaces* provide a standardized way to interact with these devices.  This is *key* for code reusability: you can write control code that works with any robot providing a `position2d` interface, for example.
    *   `position2d`:  Controls the position and velocity of a mobile robot.
    *   `laser`:  Provides data from a laser range finder.
    *  `bumper`: Provides collision information.

*   **World Representation (.world file):**  The `.world` file defines the simulation environment:
    *   **Robots:**  Their physical properties (size, shape) and initial position.
    *   **Sensors:**  Their type and placement on the robots.
    *   **Environment:**  Walls, obstacles, and other features.

## 3. Setting Up a Player/Stage Simulation

*   **Installation:**  We'll assume a Linux environment with ROS (Robot Operating System) installed. Player/Stage can be easily installed via ROS:

    ```bash
    sudo apt-get install ros-<distro>-stage-ros
    ```
    (Replace `<distro>` with your ROS distribution, e.g., `noetic`).  For other installation methods, see the Player/Stage documentation.

*   **World File Structure:** A basic `.world` file looks like this:

    ```
    # Define the world size and properties
    define floorplan model
    (
      size [20 20 1]  # x, y, z dimensions
      boundary 1      # Enable world boundaries
    )

    # Define a robot (using the "pioneer2dx" model)
    define robot model
    (
      name "robot1"
      pose [2 2 0 0]  # x, y, z, yaw (initial position)
      size [0.4 0.4 0.2] # x, y, z dimensions
      drive "diff"      # Differential drive
      localization "odom"
      localization_origin [0 0 0 0]
    )
    # Define a laser range finder for the robot
     define laser model
     (
        name "laser1"
        pose [0.2 0 0 0] # x, y, z, yaw (relative to robot)
        range_max 5.0
        fov 180
        samples 180
     )

    # Create an instance of the floorplan
    floorplan( name "my_world" )

    # Create an instance of the robot
    robot( name "robot1" pose [2 2 0 0] )
    
    #Attach the laser
    laser( pose [0.1 0 0 0] name "laser1" robot "robot1")
    ```

    *   `define`:  Creates a reusable model definition.
    *   `model`: Defines properties of the defined model.
    *   `pose`:  Specifies the position and orientation (x, y, z, yaw).
    *   `size`: Defines the dimensions of the object.
    *  `drive "diff"` : Specifies that it has a differential drive system.

*   **Running a Simulation:**

    ```bash
    roslaunch stage_ros stageros.launch world_file:=/path/to/your/worldfile.world
    ```
    Replace `/path/to/your/worldfile.world` with the actual path to your `.world` file.

## 4. Controlling Robots with Player

*   **Client Libraries:**  We'll use the Python client library (`libplayercpp`).

*   **Basic Control Loop (Python):**

    ```python
    import libplayercpp

    # Connect to Player
    client = libplayercpp.PlayerClient("localhost", 6665)

    # Create a position2d proxy
    position = libplayercpp.Position2dProxy(client, 0)
    #Create a bumper proxy
    bumper = libplayercpp.BumperProxy(client, 0)

    # Create a laser proxy
    laser = libplayercpp.LaserProxy(client, 0)


    # Main control loop
    try:
        while True:
            # Read sensor data
            client.Read()

            # Print some laser readings
            #print("Laser ranges:", laser.GetRanges())

            # Simple obstacle avoidance:
            if bumper[0] or bumper[1]:
                # Stop and turn if bumper is pressed.
                position.SetSpeed(0.0, 1.0) # Turn
            elif laser.GetMinRange() < 0.5:
                # Stop and turn if something is close
                position.SetSpeed(0.0, 1.0)
            else:
                # Otherwise, move forward
                position.SetSpeed(0.5, 0.0)

    except libplayercpp.PlayerError as e:
        print(e)
    ```

    This example connects to Player, creates proxies for the `position2d`, `bumper`, and `laser` interfaces, and implements a simple obstacle avoidance behavior.  The robot moves forward unless an obstacle is detected by the laser (within 0.5 meters), in which case it turns.

*   **Essential Interfaces:**
    *   `position2d`:  Fundamental for controlling robot movement.  `SetSpeed(linear_velocity, angular_velocity)` is the key method.
    *   `laser`:  Provides range data from a laser scanner.  `GetRanges()` returns a list of distances. `GetMinRange()` returns minimum distances.
    *    `bumper`: Provides collision information.

## 5. Integrating Game-Theoretic Strategies

*   **Mapping Concepts:**
    *   **Players:**  Each robot in the simulation is a player.
    *   **Actions:**  Robot control commands (e.g., `position.SetSpeed(x_vel, yaw_rate)`).
    *   **Strategies:**  The algorithms you implement in your client program to determine the robot's actions.
    *   **Payoffs:**  Calculated based on simulation outcomes (e.g., distance to a goal, number of collisions, time elapsed).  You'll need to define functions in your client code to calculate these.
    *   **Information:**  Obtained through sensor readings (e.g., laser ranges, position).

*   **Example: Simple Collision Avoidance Game (Two Robots):**

    *   **Abstract Game:**
        *   **Players:** Robot 1, Robot 2.
        *   **Actions:**  Each robot can choose "Move Forward" or "Turn."
        *   **Payoffs:**
            *   Collision: -10 for both robots.
            *   No Collision: +1 for each robot.
            * Reaching goal: + 10

    *   **Implementation (Conceptual - Python):**

        ```python
        # ... (Player connection and proxy setup as before) ...

        def get_payoff(robot1_pos, robot2_pos, collision):
            payoff = 0
            if collision:
                payoff -= 10
            # Example: Add a reward for getting closer to a goal
            goal_pos = [10, 10]  # Example goal position
            distance_to_goal = ((robot1_pos[0] - goal_pos[0])**2 + (robot1_pos[1] - goal_pos[1])**2)**0.5
            if distance_to_goal<0.1:
                payoff +=10 # Reaching goal.
            payoff += 1 / (distance_to_goal + 0.0001) #Avoid division by zero.
            return payoff

        # Main loop (for each robot)
        while True:
            client.Read()

            # Get robot positions (you'd need to access position data)
            robot1_pos = [position.GetXPos(), position.GetYPos()]
            # ... (get robot2_pos similarly) ...

            # Implement strategies (very simplified example)
            if laser.GetMinRange() < 0.5:
                action = "Turn"
            else:
                action = "Move Forward"

            # Execute actions
            if action == "Turn":
                position.SetSpeed(0.0, 1.0)
            else:
                position.SetSpeed(0.5, 0.0)

            # Check for collisions (simplified example)
            collision = False
            if ((robot1_pos[0] - robot2_pos[0])**2 + (robot1_pos[1] - robot2_pos[1])**2)**0.5 < 0.8: # Assuming robot radius of 0.4
              collision = True

            # Calculate payoffs
            robot1_payoff = get_payoff(robot1_pos, robot2_pos, collision)
            # ... (calculate robot2_payoff similarly) ...

            #print("Robot 1 Payoff:", robot1_payoff) #For debugging
        ```

* **Data Collection:**
    * Use `print()` statements to write data to the console.
    * Redirect the console output to a file.
    * Use Python libraries like `csv` or `pandas` to write data to structured files (CSV, etc.) for later analysis.

## 6. Limitations and Next Steps (Very Brief)

*   **Stage's Limitations:** Stage is a 2D kinematic simulator.  It doesn't model complex physics or 3D environments.
*   **Further Exploration:**
    *   **Gazebo:**  For 3D, physics-based simulations, explore Gazebo.
    *   **Hardware-in-the-Loop:**  Consider connecting your Player client to a real robot for more realistic testing.
