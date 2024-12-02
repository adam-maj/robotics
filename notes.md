# Papers

Created: October 31, 2024 12:49 PM

# ZMP

<aside>
📜

[Zero-Moment Point - Thirty Five Years of Its Life](https://www.cs.cmu.edu/~cga/legs/vukobratovic.pdf)

</aside>

ZMP is concerned with biped locomotion.

The basic characteristics of all biped locomotions systems:

- The entire system can rotate around a single foot due to an outside disturbance (which can shift the system off balance)
- Achieving gait repeatability (consistently applying the same walking patterns)
- Regular interchange between single and double foot phases (one foot supporting vs. two feet supporting). Locomotion changes from an open to closed chain as these switch.

The foot contact is essential. The position of the robot with respect to the environment is determined by the position of the feet with respect to the ground.

> The foot cannot be controlled directly but in an indirect way, by ensuring the appropriate dynamics of the mechanism above the foot. Thus, the overall indicator of the mechanism behavior is the point where the influence of all forces acting on the mechanism can be replaced by one single force.

> Recognition of the significance and role of ZMP in the biped artificial walk was a turning point in gait planning and control.

> The aim of this work is primarily to remind the reader of the seminal results related to ZMP.

### The ZMP Notion

> Apart from the realization of the relative motion of the mechanism’s links, the most important task of a locomotion mechanism during the gait is to preserve its dynamic balance (some “new” authors use the term “stability”), which is achieved by ensuring the foot’s whole area, and not only the edge, is in contact with the ground.

The purpose of the locomotion is [1] to move the robot and [2] to maintain balance.

Balance can be defined by keeping the full robot foot in contact with the ground.

![Screenshot 2024-11-12 at 2.49.01 PM.png](../images/Screenshot_2024-11-12_at_2.49.01_PM.png)

For locomotion, we are only concerned with the forces acting on the robots foot. The forces come from the force/moment exerted on the foot by the above robot and the force/moment exerted on the foot by the ground pushing up on the foot + friction.

The friction forces $F_x$ and $F_y$ balance the horizontal force components on the foot, and the upward force $F_z$ balances the vertical moment and force from the robot.

Since the ground can only act directly up on the foot, the ground reaction force will have to shift horizontally on the foot in order to counteract the horizontal moment on the ankle.

> The moment $M_{A_X}$ is balanced by shifting the acting point of the force $R_z$, whose intensity is determined by the equation of balance of all the forces acting on the foot, by the corresponding distance $y$.

An ankle moment can only be compensated by changing the position of the ground reaction force.

If the ankle moment can’t be compensated by the ground reaction force because the point to compensate the moment is outside the foot, then the robot will rotate around the ankle and can fall. The task of balancing the robot is about maintaining this constraint.

This equilibrium requires that the $M_x = 0$ and $M_y = 0$ on the foot, hence it is known as the **zero-moment point**.

We have an equation for calculating the ground reaction point of the robot based on the following:

![Screenshot 2024-11-12 at 3.02.10 PM.png](../images/Screenshot_2024-11-12_at_3.02.10_PM.png)

Equation 2 requires that the robot foot doesn’t move vertically (the ground reaction force up balances the force of the robot and gravity down)

Equation 3 requires that the moment caused by the ground reaction force balances out the moments caused by gravity and by the moment/force caused by the robot on the foot.

Then they get an equation which gives the position of the ground reaction force acting point $P$.

![Screenshot 2024-11-12 at 3.04.50 PM.png](../images/Screenshot_2024-11-12_at_3.04.50_PM.png)

Then they need to use this to calculate whether the system is in dynamic equilibrium.

The ZMP check is then: in order for the system to be in dynamic equilibrium, a point P that satisfies equation 4 must be within the polygon on of the robot foot.

In operating a robot, ZMP plays a role in [1] determining the dynamics of the mechanism above the foot to ensure a valid ZMP position [2] determining the ZMP position for the mechanism in motion.

Case 1 is called gait synthesis. Case 2 is called gait control.

They can use ground sensors to measure the current ZMP of the robot.

We calculate the ZMP. If it’s inside the polygon, the robot is balanced, otherwise it’s not.

> The ZMP concept has been properly comprehended by researchers, widely used, and very frequently cited. It can be noted that, although being essentially correct, all the ZMP definitions differ significantly in the extent of their detail.

> ZMP is defined as that point on the ground at which the net moment of the inertial forces and the gravity forces has no component along the horizontal axes.

There is an important difference between the ZMP and the center of pressure (CoP). The CoP is the place where the ground acts on the foot. In dynamic balance, the ZMP and the CoP are the same. In states where the system isn’t balanced, the ZMP doesn’t exist whereas the CoP does.

The FZMP is the hypothetical ZMP that would balance the robot (outside the polygon of the foot).

> Preventing the robots’s overturning can also be achieved by temporary reconfiguration into a quadruped using the upper extremities, followed by re-establishing the motion in the form of regular dynamically balanced biped gait.

Humanoids may become slightly more complex to become more general purpose in the human world with softer/multi-link feet, elastic joints, transitions between walking patterns, etc.

The ground usually means something immobile, but the robot also needs to be able to walk on something deformable/mobile as well. The supports shouldn’t be considered static but should be considered dynamic systems.

### Conclusion

> The concept of ZMP has and will have an essential role in both theoretical considerations and the practical development of humanoid robots and biped locomotion.

Below is a model of the general task of biped balancing.

![Screenshot 2024-11-12 at 3.29.55 PM.png](../images/Screenshot_2024-11-12_at_3.29.55_PM.png)

# Preview Control

<aside>
📜

[Biped Walking Pattern Generation by using Preview
Control of Zero-Moment Point](https://mzucker.github.io/swarthmore/e91_s2013/readings/kajita2003preview.pdf)

</aside>

> Research on biped humanoid robots is currently one of the most exciting topics in the field of robotics.

There are 2 approaches to locomotion at the time. One approach requires precise knowledge of the robot dynamics and uses perfect models. The other approach uses limited knowledge and feedback control (the inverted pendulum approach).

> In this paper we introduce a novel walking pattern generation that allows arbitrary foot placements as a mixture of the ZMP based and the inverted pendulum based approaches.

### Walking pattern generation for a given ZMP

They use FFT and inverse FFT to solve the desired CoM trajectory given the desired ZMP point.

They use a ZMP control output that takes the future input as its input.

> The walking pattern is calculated by solving an inverse kinematics such that the CoM of the robot follows the output of the preview controller.

### Conclusion

> We proposed a new method for biped walking pattern generation. First we introduced a cart-table model, which is a convenient representation to design a ZMP controller.

After reviewing conventional methods that uses ZMP to generate walking pattern, we formalized the problem as the design of a ZMP tracking servo controller. It was shown that we can design such controller by adopting the preview control that uses the future ZMP reference.

>

# A\*

A\* is basically Dijkstra’s but with a heuristic that you’re moving in the correct direction, rather than just taking the shortest path.

These algorithms require a 2D/3D grid and don’t work well in higher dimensional configuration spaces like the real world.

# RRT

<aside>
📜

[Rapidly-Exploring Random Trees: A New Tool for Path Planning](https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf)

</aside>

RRT is a path planning algorithm good for non-holonomic constraints (the robot can’t just point and go wherever it wants to but is actually constrained by its dynamics to follow valid paths) and high degrees of freedom (can handle higher dimensional spaces characteristic of the real world).

> There is little hope of ever obtaining an efficient, general path planning algorithm.

> The primary difficulty with existing (sampling based) techniques is that, although powerful for standard path planning, they do not naturally extend to general non-holonomic planning problems.

> In the probabilistic roadmap approach, a graph is constructed in the configuration space by generating random configurations and attempting to connect pairs of nearby configurations with a local planner that will connect pairs of configurations.

PRM gets challenging in non-holonomic systems where it may need to make thousands of random samples to find a valid path.

> In this paper, we introduce a randomized data structure for path planning that is designed for problems that have non-holonomic constraints.

PRM and RRT are designed with as few heuristics and arbitrary parameters as possible, leading to consistent behavior.

### Rapidly-Exploring Random Trees

We deal with a state space $X$, practically the configuration space of a robot in a 3D environment $X = \mathcal{C}$ where we want to find a path from some $x_\textrm{init}$ to some goal region $X_\textrm{goal} \sub X$ or goal state $x_\textrm{goal}$.

We need to find the tangent bundle of the configuration space to solve this problem $X = T(\mathcal{C})$.

There are some states in the environment that are obstacles that must be avoided $X_\textrm{obs}$ either due to actual obstacles in the environment or constraints in robot dynamics. The solution vertices and edges must be all states in $X_\textrm{obs}$.

Importantly, you don’t know where all of $X_\textrm{obs}$ is and may only be able to check in real time whether a state is in the set.

We have the state transition equation $\dot{x} = f(x, u)$ giving the derivative of the state from the current state $x$ and specific control inputs $u$. In holonomic problems, $x$ can change in any direction. In nonholonomic problems, $x$ is constrained to certain changes.

![Screenshot 2024-11-12 at 4.58.34 PM.png](../images/Screenshot_2024-11-12_at_4.58.34_PM.png)

RRTs have the following advantages:

- RRTs are heavily biased toward expanding into unexplored regions
- Distribution of RRT vertices approaches sampling distribution
- Simple algorithm
- RRT is always connected and can be used as a path planning module

Because RRT selects the nearest neighbor in the graph, it prompts the tree to expand outward and explored regions it hasn’t previously explored.

> Based on simulation experiments such as the one shown above, we have concluded that the generated paths are not far from optimal and that the vertices will eventually become uniformly distributed.

> Based on our preliminary experiments, it appears that RRTs might be faster than the basic probabilistic roadmap approach for holonomic planning problems.

> Collision detection is a key bottleneck in path planning, and an RRT is completely suited for incremental collision detection.

![Screenshot 2024-11-12 at 5.17.34 PM.png](../images/Screenshot_2024-11-12_at_5.17.34_PM.png)

Efficient nearest neighbor techniques are needed to make RRT good.

> At the present time, we believe we have barely scratched the surface of potential applications of RRTs.

# CHOMP

<aside>
📜

[CHOMP: Gradient Optimization Techniques for Efficient Motion Planning](https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf)

</aside>

> In domains sparsely populated by obstacles, the heuristics used by sampling-based planners to navigate “narrow passages” can be needlessly complex; furthermore, additional post-processing is required to remove the jerky or extraneous motions from the paths that such planners generate.

> In recent years, sampling-based planning algorithms have met with widespread success due to their ability to rapidly discover the connectivity of high-dimensional configuration spaces.

PRM and RRT are often deployed in 2 phase: [1] first find a feasible path [2] optimize the path to remove jerky motion.

Shortest paths are found by slicing paths into shorter parts, and gradient based methods are used to find minimum-energy paths.

CHOMP is an algorithm for improving paths that can start with a naive path that has collisions and improve it from there.

> Instead of merely finding feasible paths, our goal is to directly construct trajectories which optimize over a variety of dynamic and task-based criteria.

### The CHOMP Algorithm

> In this section, we present CHOMP, a new trajectory optimization procedure based on covariant gradient descent.

They use geometrical relations to: [1] encourage smoothness by measuring update size in terms of changes of a dynamical quality to the path [2] measurements of obstacle costs should be taken in the environment to respect real geometries [3] trajectory update considerations should also be used to update from joint limit violations.

**1. Covariant Gradient Descent**

The goal is to find a smooth collision-free trajectory through configuration space from start to finish.

The trajectory cost is measured by dynamical terms and the cost of being near obstacles.

Covariant gradient descent encodes knowledge about the environments gradient into the vector so gradient descent is aware of the geometries.

They perform updates after each step of the trajectory that ensure that the trajectory is smooth.

**2. Understanding the update rule**

> CHOMP is covariant in the sense that the change to the trajectory that results from the update is a function only of the trajectory itself, and not the particular representation used - at least in the limit of small step size and fine discretization.

> When applying CHOMP, we typically use a simplified geometric description of our robots, approximating the robot as a “skeleton” of spheres and capsules, or line-segment swept spheres.

> However, there is an important difference that substantially improves performance in practice. Rather than integrating with respect to arc-length through configuration space, we integrate with respect to arc-length in the workspace.

> This simple modification represents a fundamental change: instead of assuming the geometry in the configuration space is Euclidean, we compute geometrical quantities directly in the workspace where Euclidean assumptions are more natural.

> Joint limits are traditionally handled by either adding a new potential to the objective function which penalizes the trajectory for approaching the limits, or by performing a simple projection back onto the set of feasible joint values when a violation of the limits is detected. In our experiments, we follow the latter approach.

CHOMP computes trajectories as linked waypoints. It uses the gradient to optimize the waypoints toward smoothness and away from obstacles.

### Experiments on a robotic arm

> Surprisingly, when CHOMP successfully finds a collision free trajectory, straight-line initialization typically outperforms RRT initialization.

> Our approach to robotic legged locomotion decomposes the problem into a footstep planner which informs the robot where to place its feet as it traverses the terrain [6], and a footstep controller which generates full-body trajectories to realize the planned footsteps.

This is what the split looks like in an actual motion planning scenario for a quadruped.

> Footsteps for the LittleDog robot consist of a stance phase, where all four feet have ground contact, and a swing phase, where the swing leg is moved to the next support location.

> For a given footstep, we run CHOMP as coordinate descent.

> The initial trunk trajectory is given by a Zero Moment Point (ZMP) preview controller.

They add priors to penalize collisions and kinematic reachability errors.

> Unlike many of the other teams who seemed to focus on feedback control, operational control, and other reactive behaviors, our strategy has been to strongly leverage optimization.

### Conclusions

> This work presents a powerful new trajectory optimization procedure that solves a much wider range of problems than previous optimizers, including many to which randomized planners are traditionally applied.

> Finally, this algorithm is amenable to new machine learning techniques.

# TrajOpt

<aside>
📜

[Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization](https://www.roboticsproceedings.org/rss09/p31.pdf)

</aside>

> Our algorithm was faster than the alternatives, solved more problems, and yielded higher quality paths.

The best trajectory optimization problem that consider smoothness and collision avoidance.

> Trajectory optimization algorithms have two roles in robotic motion planning. First, they can be used to smooth and shorten trajectories generated by some other method. Second, they can be used to plan from scratch: one initializes with a trajectory that contains collisions and perhaps violates constraints, and one hopes that the optimization converges to a high-quality trajectory satisfying constraints.

The 2 important elements of an optimization algorithm are [1] the numerical optimization method and [2] the method for checking for and penalizing collisions.

TrajOpt breaks down the cost function into a series of convex (quadratic) optimization problems that approximate the true cost function, allowing the model to train more easily.

Their algorithm is very fast, which is enabled by their formulation of the collision penalty.

Their method is also more reliable, able to solve many path planning problems that take a long time for other methods to solve.

Their algorithm is also better for path quality, converging to locally optimal solutions for avoiding obstacles.

Their approach is also flexible - it can easily update to adding new costs and obstacles.

> Overall, our algorithm was not only faster than the alternatives, but it solved a larger fraction of the problems.

### Background

> Sequential convex optimization solves a non-convex optimization problem by repeatedly constructing a convex subproblem - an approximation to the problem around the current iterate $x$.

Sequential convex needs a method to make the step size small and a strategy to turn infeasible constraints into penalties.

### Motion Planning Benchmark

> We tested both our algorithm and CHOMP under two conditions: single initialization and multiple initializations. For the single initialization, we used a straight line in configuration space from the start to the goal.

> Our algorithm with multiple initializations substantially outperformed the other approaches in both sets of problems.

![Screenshot 2024-11-12 at 6.19.44 PM.png](../images/Screenshot_2024-11-12_at_6.19.44_PM.png)

> Using this approach, we plan a sequence of steps across a room, as shown in figure 7. Each step is planned separately using the phases described above. The robot is able to obey these stability and footstep placement constraints while ducking under an obstacle.

### Real-World Experiments

> One of the main challenges in taking motion planning from simulation to reality is creating a useful representation of the environment’s geometry.

> The point clouds we used were obtained by mapping out the environment using SLAM and then preprocessing the map to obtain a convex decomposition.

### Discussion

> While the motivation of this work is similar to CHOMP, our approach differs from CHOMP in several important dimensions, most notably that (1) we use a different approach for collision detection, and (2) we use a different numerical optimization scheme.

> Spheres and distance fields are arguably not very well suited to situations where one needs to accurately model geometry, which is why collision-detection methods based on meshes and convex primitives are more prevalent in applications like realtime physics simulation, which require speed and accuracy.

> Another advantage of sequential quadratic programming is that it can handle deeply infeasible initializations using penalties and merit functions.

### Conclusion

> We presented a novel algorithm that uses trajectory optimization for robotic motion planning.

> Our algorithm was faster than the alternatives, solved a larger fraction of problems, and produced better paths.

# SLAM: Part 1

<aside>
📜

[Simultaneous Localization and Mapping: Part I](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1638022)

</aside>

> The simultaneous localization and mapping (SLAM) problem asks if it is possible for a mobile robot to be placed at an unknown location in an unknown environment and for the robot to incrementally build a consistent map of this environment while simultaneously determining its location within this map.

A robot has to build a map of the world and figure out where it is in the map when dropped in any new environment.

> A solution to the SLAM problem has been seen as a “holy grail” for the mobile robotics community as it would provide the means to make a robot truly autonomous.

Good SLAM is essential for an actually autonomous/intelligent humanoid robot.

> Substantial issues remain in practically realizing more general SLAM solutions and notably in building and using perceptually rich maps as part of a SLAM algorithm.

SLAM is a theoretically “solved” problem but still has room for improvement in practice.

### History of the SLAM Problem

> There must be a high degree of correlation between estimates of the location of different landmarks in a map and that, indeed, these correlations would grow with successive observations.

Estimates of different landmarks would all result from the same prediction errors on the robot.

Early work in visual navigation and sonar-based navigation with Kalman filters led to initial progress.

> As a mobile robot moves through an unknown environment taking relative observations of landmarks, the estimates of these landmarks are all necessarily correlated with each other because of the common error in estimated vehicle location.

All the landmark locations should be correlated because of the same prediction error. This can be used to update the map over time.

> A consistent full solution to the combined localization and mapping problem would require a joint state composed of the vehicle pose and every landmark position, to be updated following each landmark observation.

A SLAM solution would involve maintaining the state of every single observed landmark (point) and updating this state after each new observation.

Because of this, computation scales with the square of the number of landmarks.

> The conceptual breakthrough came with the realization that the combined mapping and localization problem, once formulated as a single estimation problem, was actually convergent.

People realized that the problem converges if you do mapping and localization together, since mapping makes the localization problem converge.

> Most importantly, it was recognized that the correlations between landmarks, which most researchers had tried to minimize, were actually the critical part of the problem and that, on the contrary, the more these correlations grew, the better the solution.

It wasn’t initially obvious that the problem converged. However, they eventually realized that as correlations between landmarks grow with new observations, the problem converges toward a solution.

### Formulation and Structure of the SLAM Problem

**1. Preliminaries**

SLAM lets a robot build a map of the world and deduce it’s location, while estimating the location of landmarks and trajectory of the robot without any prior knowledge.

We define the following sets of information for the SLAM problem

- $X_{0:k} = \{ x_0, x_1, …, x_k \}$ - the history of all robot locations
- $U_{0:k} = \{ u_0, u_1, …, u_k \}$ - the history of control inputs
- $m = \{ m_0, m_1, …, m_k\}$ - the set of all landmark locations
- $Z_{0:k} = \{ z_1, z_2, …, z_k \}$ - the set of all landmark observations

**2. Probabilistic SLAM**

The SLAM problem requires the following probability distribution to be computed for all times $k$:

$$
P(x_k, m|Z_{0:k}, U_{0:k}, x_0)
$$

Given all prior landmark observations $Z_{0:k}$ and control inputs $U_{0:k}$, and the starting position $x_0$ of the robot as a reference point, the robot has to compute it’s own current position $x_k$ in the environment and the locations of all landmarks $m$ as a solution to the SLAM problem.

We ideally want a recursive solution that starts with the information and prediction from the prior time step:

$$
P(x_{k-1}, m|Z_{0:k-1}, U_{0:k-1})
$$

And uses the new control $u_k$ and observation $z_k$ to compute the new distribution.

The **observation model** describes the probability of making an observation given the map and vehicle position:

$$
P(z_k|x_k, m)
$$

The **motion model** describes the probability of the robot location given prior location and control inputs:

$$
P(x_k|x_{k-1}, u_k)
$$

The SLAM algorithm uses a two step process.

First, it makes the **time update** which updates the prediction of the position and map given all prior landmark observations $Z_{0:{k-1}}$ and all motion commands including the most recent command $U_{0:k}$:

$$
P(x_k, m|Z_{0:k-1}, U_{0:k}, x_0) = \int P(x_k| x_{k-1}, u_k) \times P(x_{k-1}, m|Z_{0:k-1}, U_{0:k-1}, x_0) \: dx_{k-1}
$$

Here, we simply predict the new robot location $x_k$, given our prior beliefs about landmark positions before moving $P(x_{k-1}, m|Z_{0:k-1}, U_{0:k-1}, x_0)$, and our prediction about where we have moved given control inputs (the motion model) $P(x_k| x_{k-1}, u_k)$.

By multiplying these distributions here, we allow the uncertainty about position $x_{k-1}$ from the previous step to influence our prediction about $x_k$, in addition to the new motion command $u_k$.

We use the entire prior distribution since our prediction about $x_{k-1}$ (and the implicitly related landmark predictions $m$) depend on all the priors, so our new prediction of $x_k$ does as well since it requires $x_{k-1}$.

Through this calculation, we recursively maintain all the relationships of our prediction $x_k$ on all the prior information influencing it.

Notably, $P(x_k, m)$ must be a joint distribution since our predictions about the robot position and map are directly related since they are naturally relative to each other. Uncertainty about the robot position increase our uncertainty about where landmarks are relative to the robot.

After the time update, we use the new predictions to make the **measurement update** which is what corrects for prediction error:

$$
P(x_k, m|Z_{0:k}, U_{0:k}, x_0) = \frac{P(z_k|x_k,m)P(x_k, m|Z_{0:k-1}, U_{0:k}, x_0)}{P(z_k|Z_{0:k-1},U_{0:k})}
$$

Specifically, we use the newly updated joint distribution (with updated position prediction) from the time-update $P(x_k, m|Z_{0:k-1}, U_{0:k}, x_0)$, and we merge this (using Bayes’ Rule) with the distribution over robot positions $x_k$ and landmark positions $m$ that would make $z_k$ most likely (the observation model) $P(z_k|x_k, m)$.

This new input from the observation model encodes new information about the correlations of robot and map positions which gets factored into our full joint distribution.

Then, we normalize by $P(z_k|Z_{0:k-1}, U_{0:k})$ to keep this as a probability distribution.

In this entire process, we see that the joint distribution is recursively integrating information from the motion model $P(x_k| x_{k-1}, u_k)$ and observation model $P(z_k|x_k, m)$.

We can view the map building problem as computing the distribution $P(m|X_{0:k}, Z_{0:k}, U_{0:k})$, and we can view the localization problem as computing the distribution $P(x_i|Z_{0:k}, U_{0:k}, m)$.

The dependence of these predictions on each other is reflected in the joint distribution.

### Structure of Probabilistic SLAM

![Screenshot 2024-11-07 at 11.50.40 AM.png](../images/Screenshot_2024-11-07_at_11.50.40_AM.png)

> It can be seen that much of the error between estimated and true landmark locations is common between landmarks and is in fact due to a single source; errors in knowledge of where the robot is when landmark observations are made.

Most of the error accumulated over time in robotic mapping comes from errors in knowledge about the robots position.

This means that the relative locations between landmarks is generally known, but just the absolute location has error due to incorrect beliefs about the robot position.

> The most important insight in SLAM was to realize that the correlations between landmark estimates increase monotonically as more and more observations are made.

This means that predictions about landmark positions $P(m)$ always improves over time (becomes more concentrated).

> The relative location of observed landmarks is clearly independent of the coordinate frame of the vehicle, and successive observations from this fixed location would yield further independent measurements of the relative relationship between landmarks.

![Screenshot 2024-11-07 at 11.57.54 AM.png](../images/Screenshot_2024-11-07_at_11.57.54_AM.png)

> This process can be visualized (Figure 2) as a network of
> springs connecting all landmarks together

> As the robot moves through this environment and takes observations of the landmarks, the springs become increasingly (and monotonically) stiffer.

We can think of this as increasing the strength of correlations between points as certainty increases, causing points to further update our beliefs about where they are.

> In the limit, a rigid map of landmarks or an accurate relative map of the environment is obtained.

As our map improves, are certainty about the robot position also improves.

### Solutions to the SLAM Problem

> Solutions to the probabilistic SLAM problem involve finding an appropriate representation for both the observation model and motion model that allows efficient and consistent computation of the prior and posterior distributions.

Solving the SLAM problem is about finding a good representation for the observation model and motion model, which allows us to accurately calculate the time update and measurement update.

All the SLAM solutions like EKF-SLAM, FastSLAM, etc. are all different ways to represent the distributions of the observation and motion model.

**1. EKF-SLAM**

EKF-SLAM represents the motion and observation models with Gaussian noise.

Specifically, it describes the vehicle motion as:

$$
P(x_i|x_{k-1}, u_k) \Longleftrightarrow x_k = f(x_{k-1}, u_k) + w_k
$$

Adding the prediction in a perfect scenario modeled by the expected vehicle kinematics $f(x_{k-1}, u_k)$ with zero mean Gaussian noise $w_k$ with covariance $Q_k$ to model motion noise.

Similarly, the observation model is described as:

$$
P(z_k|x_k, m) \Longleftrightarrow z_k = h(x_k, m) + v_k
$$

Adding the expected geometry of the observation $h(\cdot)$ with zero mean Gaussian noise $v_k$ with covariance $R_k$ to model observation errors.

The output of these distributions is a Gaussian distribution with some variance centered around the expected robot position and expected observations.

This distribution is re-centered over time given new observations and motion inputs, and the noise should converge to lower variance over time, leading to a higher certainty SLAM prediction.

Both of these use Extended Kalman Filters (EKF). They treat the distributions as if error can be linearized around the central point (using the $w_k$ and $v_k$ Gaussian noise, which is characteristic of a Kalman Filter.

Such a setup can only handle linear systems, but they make it compatible with non-linearities by introducing the functions $f(\cdot)$ and $h(\cdot)$ which compute an individual point to linearize the prediction around at each point in time.

This method then uses the standard EKF procedure to update the mean and covariance matrices over time through calculations during the time-update and observation-update using the Jacobian of $h$ and $f$.

![Screenshot 2024-11-07 at 1.38.39 PM.png](../images/Screenshot_2024-11-07_at_1.38.39_PM.png)

The standard EKF functions consist of a time-update and observation-update which simplifies are prior time-update and observation-update equations into Gaussian distributions which can be calculated using closed-form equations (with a long derivation) from EKF.

The whole reason we use Gaussians to model the noise is so that this simplification is possible which makes calculation computationally tractable.

The primary weakness of EKF is that they assume the distributions are linear and Gaussian, which is a weakness in many scenarios.

EKF-SLAM converges as the determinant of the covariance matrix and landmark pair sub-matrices converges toward zero.

> The observation update step requires that all landmarks and
> the joint covariance matrix be updated every time an observation is made. Naively, this means computation grows quadratically with the number of landmarks.

> The standard formulation of the EKF-SLAM solution is especially fragile to incorrect association of observations to landmarks.

Attributing observations to the correct landmarks is challenging.

> EKF-SLAM employs linearized models of nonlinear motion and observation models and so inherits many caveats. Nonlinearity can be a significant problem in EKF-SLAM and leads to inevitable, and sometimes dramatic, inconsistency in solutions.

**2. FastSLAM: Rao-Blackwellized Filter**

> FastSLAM, with its basis in recursive Monte Carlo sampling, or particle filtering, was the first to directly represent the nonlinear process model and non-Gaussian pose distribution.

> The high dimensional state-space of the SLAM problem makes direct application of particle filters computationally infeasible.

[…]

> This is a key property of FastSLAM and the reason for its speed; the map is represented as a set of independent Gaussians, with linear complexity, rather than a joint map covariance with quadratic complexity.

[…]

> Updating the map, for a given trajectory particle $X_{0:k}^{(i)}$, is trivial. Each observed landmark is processed individually as an EKF measurement update from a known pose.

[…]

> Statistically, FastSLAM suffers degeneration due to its inability to forget the past.

### Implementation of SLAM

![Screenshot 2024-11-07 at 12.44.30 PM.png](../images/Screenshot_2024-11-07_at_12.44.30_PM.png)

> The “explore and return’’ experiment by Newman et al. was a moderate-scale indoor implementation.

> The experiment is remarkable because its return trip was fully autonomous.

![Screenshot 2024-11-07 at 12.44.44 PM.png](../images/Screenshot_2024-11-07_at_12.44.44_PM.png)

> Guivant and Nebot pioneered the application of SLAM in very large outdoor environments. They addressed computational issues of
> real-time operation, while also dealing with high-speed vehicle motion, non-flat terrain, and dynamic clutter.

### Conclusion

> This article has described the SLAM problem and the essential methods for solving the SLAM problem and has summarized key implementations and demonstrations of the method.

# ORB-SLAM

<aside>
📜

[ORB-SLAM: A Versatile and Accurate Monocular SLAM System](https://arxiv.org/pdf/1502.00956)

</aside>

> Visual SLAM has the goal of estimating the camera trajectory while reconstructing the environment.

Visual SLAM is challenging as it requires efficient usage of a subset of observations and keyframes to prevent redundancy as complexity grows, a strong network of observations to produce accurate results, sufficient loop-closure abilities, handling occlusions, etc.

ORB-SLAM is a new monocular SLAM algorithm that:

- Uses a single set of ORB features for all tasks: tracking, mapping, re-localization, and loop closing.
- Operates in real-time in large environments using a co-visibility graph.
- Real-time loop closing based on pose graph optimization.
- Real-time camera relocalization which allows recovery from tracking failure.
- A new initialization procedure.
- A survival of the fittest map point and keyframe selection approach.

> To the best of our knowledge, this is the most complete and reliable solution to monocular SLAM, and for the benefit of the community we make the source code public.

### Related Work

**1. Place Recognition**

> [Place recognition approaches] based on appearance, that is image to image matching, scale better in large environments than map to map or image to map methods.

> With appearance based methods, bag of words techniques, are to the fore because of their high efficiency.

**2. Map Initialization**

> Monocular SLAM requires a procedure to create an initial map because depth cannot be recovered from a single image.

**3. Monocular SLAM**

> Monocular SLAM was initially solved by filtering, [where] every frame is processed by the filter to jointly estimate the map feature locations and the camera pose.

> It has the drawbacks of wasting computation in processing consecutive frames with little new information and the accumulation of linearization errors.

> On the other hand keyframe-based approaches, estimate the map using only selected frames (keyframes) allowing to perform more costly but accurate bundle adjustment optimizations.

> Keyframe-based techniques are more accurate than filtering for the same computational cost.

> [PTAM] was the first work to introduce the idea of splitting camera tracking and mapping in parallel threads, and demonstrated to be successful for real time augmented reality applications in small environments.

> In our system we take advantage of the excellent ideas of using a local map based on co-visibility, and building the pose graph from the
> co-visibility graph, but apply them in a totally redesigned frontend and back-end.

> Another difference is that, instead of using specific features for loop detection (SURF), we perform the place recognition on the same tracked and mapped features, obtaining robust frame-rate relocalization and loop detection.

> All visual SLAM works in the literature agree that running BA with all the points and all frames is not feasible.

> The most cost effective approach is to keep as much points as possible, while keeping only non-redundant keyframes.

> Our survival of the fittest strategy achieves unprecedented robustness in difficult scenarios by inserting keyframes as quickly as possible, and removing later the redundant ones, to avoid the extra cost.

### System Overview

**1. Feature Choice**

> One of the main design ideas in our system is that the same features used by the mapping and tracking are used for place recognition to perform frame-rate relocalization and loop detection.

They use the same features for all tasks which makes their algorithm far more computationally efficient.

> [ORB features] are extremely fast to compute and match, while they have good invariance to viewpoint.

They use ORB to extract features which has high performance.

**2. Three Threads: Tracking, Local Mapping, and Loop Closing**

![Screenshot 2024-11-07 at 2.14.14 PM.png](../images/Screenshot_2024-11-07_at_2.14.14_PM.png)

> Our system, see an overview in Fig. 1, incorporates three threads that run in parallel: tracking, local mapping and loop closing.

> The tracking is in charge of localizing the camera with every frame and deciding when to insert a new keyframe.

> The local mapping processes new keyframes and performs local BA to achieve an optimal reconstruction in the surroundings of the camera pose.

> The local mapping is also in charge of culling redundant keyframes.

> The loop closing searches for loops with every new keyframe. If a loop is detected, we compute a similarity transformation that informs about the drift accumulated in the loop. Then both sides of the loop are aligned and duplicated points are fused.

**3. Map Points, Key Frames, and their Selection**

Each map point $p_i$ stores:

- Its 3D position within world coordinates
- It’s viewing direction
- A representative ORB descriptor for the point
- The max and min distances from which it can be observed

Each keyframe $K_i$ stores:

- The camera pose that transforms points from the world to camera coordinates
- The camera intrinsics like focal length and principle point
- All the ORB features extracted in the frame.

> Map points and keyframes are created with a generous policy, while a later very exigent culling mechanism is in charge of detecting redundant keyframes and wrongly matched or not trackable map points.

> This permits a flexible map expansion during exploration, which promotes robustness under hard conditions.

**4. Covisibility Graph and Essential Graph**

> Covisibility information between keyframes is very useful in several tasks of our system, and is represented as an undirected weighted graph.

> Each node is a keyframe and an edge between two keyframes exists if they share observations of the same map points.

They use this covisibility graph for loop closure.

**5. Bag of Words Place Recognition**

> The system has embedded a bags of words place recognition module, to perform loop detection and localization.

> Visual words are just a discretization of the descriptor space, which is known as the visual vocabulary. The vocabulary is created offline with the ORB descriptors extracted from a large set of images.

> If the images are general enough, the same vocabulary can be used for different environments getting a good performance.

### Automatic Map Initialization

> The goal of the map initialization is to compute the relative pose between two frames to triangulate an initial set of map points.

![Screenshot 2024-11-07 at 2.26.35 PM.png](../images/Screenshot_2024-11-07_at_2.26.35_PM.png)

### Tracking

The tracking thread steps are performed at every camera frame.

**1. ORB Extraction**

They first extract FAST corners from each section of the grid. Then they compute ORB descriptors and orientations using the FAST corners.

**2. Initial Pose Estimation from Previous Frame**

If tracking in the last frame was successful, they use constant velocity to predict the camera pose and search for observed map points that should be visible.

**3. Initial Pose Estimation via Global Relocalization**

If they lose tracking, they convert the frame into a bag of words and query the recognition database for keyframes that they can use for relocalization.

**4. Track Local Map**

> Once we have an estimation of the camera pose and an initial set of feature matches, we can project the map into the frame and search more map point correspondences.

They only use a local map to prevent complexity in large environments.

> The camera pose is finally optimized with all the map points found in the frame

**5. New Keyframe Decision**

Then the system has to decide whether to insert the current frame as a keyframe.

> As there is a mechanism in the local mapping to cull redundant keyframes, we will try to insert keyframes as fast as possible, because that makes the tracking more robust to challenging camera movements, typically rotations.

They want to be generous with storing keyframes because it helps with fast camera movements that are otherwise hard to recover.

### Local Mapping

These are steps performed with every new keyframe.

**1. Keyframe Insertion**

They add the keyframe to the covisibility graph and store the bag of words representation of the keyframe.

**2. Recent Map Points Culling**

> Map points, in order to be retained in the map, must pass a restrictive test during the first three keyframes after creation.

**3. New Map Point Creation**

> New map points are created by triangulating ORB from connected keyframes in the covisibility graph.

**4. Local Bundle Adjustment**

> The local BA optimizes the currently processes keyframe, all the keyframes connected to it in the covisibility graph, and all the map points seen by those keyframes.

**5. Local Keyframe Culling**

> In order to maintain a compact reconstruction, the local mapping tries to detect redundant keyframes and delete them.

> We discard all the keyframes in Kc whose 90% of the map points have been seen in at least other three keyframes in the same or finer scale.

### Loop Closing

**1. Loop Candidates Detection**

> At first we compute the similarity between the bag of words vector of $K_i$ and all its neighbors in the covisibility graph.

They query the recognition database to find similar keyframes and delete frames whose score is lower than some threshold.

> To accept a loop candidate we must detect consecutively three
> loop candidates that are consistent.

**2. Compute the Similarity Transformation**

They compute a similarity transformation between the frames.

**3. Loop Fusion**

> The first step in the loop correction is to fuse duplicated map points and insert new edges in the covisibility graph.

**4. Essential Graph Optimization**

> To effectively close the loop, we perform a pose graph optimization over the Essential Graph that distributes the loop closing error along the graph.

### Experiments

> Our system runs in real time and processes the images exactly at the frame rate they were acquired.

> ORB-SLAM has three main threads, that run in parallel with other tasks from ROS and the operating system.

**1. System Performance in the NewCollege Dataset**

![Screenshot 2024-11-07 at 2.51.05 PM.png](../images/Screenshot_2024-11-07_at_2.51.05_PM.png)

> The NewCollege dataset contains a 2.2km sequence from a robot traversing a campus and adjacent parks.

Since ORB-SLAM depends purely on visual data, they can construct the environment just from a video. Very cool.

> It contains several loops and fast rotations that makes the sequence quite challenging for monocular vision. To the best of our knowledge there is no other monocular system in the literature able to process this whole sequence.

Loop closures and fast turning make video data difficult to process.

**2. Localization Accuracy in the TUM RGB-D Benchmark**

> The TUM RGB-D benchmark is an excellent dataset to evaluate the accuracy of camera localization as it provides several sequences with accurate ground truth obtained with an external motion capture system.

This dataset is good for evaluating performance because it comes with location data.

> In terms of accuracy ORB-SLAM and PTAM are similar in open trajectories, while ORB-SLAM achieves higher accuracy when detecting large loops.

**3. Relocalization in the TUM RGB-D Benchmark**

> ORB-SLAM accurately relocalizes more than the double of frames than PTAM.

**4. Lifelong Experiment in the TUM RGB-D Benchmark**

> Previous relocalization experiments have shown that our system is able to localize in a map from very different viewpoints and robustly under moderate dynamic changes.

> This property in conjunction with our keyframe culling procedure allows to operate lifelong in the same environment under different viewpoints and some dynamic changes.

> While the lifelong operation in a static scenario should be a requirement of any SLAM system, more interesting is the case where dynamic changes occur.

**5. Large Scale and Large Loop Closing in the KITTI Dataset**

> 11 sequences from a car driven around a residential area with accurate ground truth from GPS and a Velodyne laser scanner.

> This is a very challenging dataset for monocular vision due to fast rotations, areas with lot of foliage, which make more difficult data association, and relatively high car speed.

![Screenshot 2024-11-07 at 2.56.51 PM.png](../images/Screenshot_2024-11-07_at_2.56.51_PM.png)

### Conclusions and Discussion

**1. Conclusion**

> In this work we have presented a new monocular SLAM system with a detailed description of its building blocks and an exhaustive evaluation in public datasets.

> The accuracy of the system is typically below 1 cm.

> The main contribution of our work is to expand the versatility of PTAM to environments that are intractable for that system.

> To the best of our knowledge, no other system has demonstrated to work in as many different scenarios and with such accuracy. Therefore our system is currently the most reliable and complete solution for monocular SLAM.

> Finally we have also demonstrated that ORB features have enough recognition power to enable place recognition from severe viewpoint change.

**2. Sparse/Feature-based vs. Dense/Direct Methods**

This is a sparse/feature-based SLAM method. There are also dense methods that perform dense reconstruction of the environment and localizing the camera by optimizing over image pixel intensities.

> In contrast, feature-based methods are able to match features with a wide baseline, thanks to their good invariance to viewpoint and illumination changes.

> We consider that the future of monocular SLAM should incorporate the best of both approaches.

**3. Future Work**

> The accuracy of our system can still be improved incorporating points at infinity in the tracking.

> Another open way is to upgrade the sparse map of our system to a denser and more useful reconstruction.

# DROID SLAM

<aside>
📜

[DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/pdf/2108.10869)

</aside>

Prior to this paper, there have already been many approaches to SLAM.

Early SLAM used probabilistic and filtering approaches, and alternating optimization of the map and camera poses.

> More recently, modern SLAM systems have leveraged least-squares
> optimization. A key element for accuracy has been full Bundle Adjustment (BA), which jointly optimizes the camera poses and the 3D map in a single optimization problem.

> One advantage of the optimization-based formulation is that a SLAM system can be easily modified to leverage different sensors.

ORB-SLAM3 supports monocular, stereo, RGB-D, and IMU sensors.

> Despite significant progress, current SLAM systems lack the robustness demanded for many real-world applications.

SLAM was still not ready. There are errors like lost feature tracks, divergence in optimization, and accumulation of drift.

They introduce the deep learning based DROID-SLAM in this paper.

> It has state-of-the-art performance, outperforming existing SLAM systems, classical or learning-based, on challenging benchmarks with very large margins.

DROID-SLAM has high accuracy, high robustness to failure, and strong generalization.

The “Differentiable Recurrent Optimization-Inspired Design” (DROID)

> is an end-to-end differentiable architecture that combines the strengths of both classical approaches and deep networks.

It uses recurrent iterative updates like RAFT. Unlike RAFT, DROID-SLAM iteratively updates camera poses and depth operating on any number of frames, rather than RAFT operating on optical flow in just two frames.

DROID-SLAM also uses a Dense Bundle Adjustment (DBA) layer.

### Related Work

**1. Visual SLAM**

Visual SLAM uses observations from monocular, stereo, or RGB-D images. Indirect approaches process the image into intermediate representations with points of interest and feature descriptors, and then match features between images.

They are optimized by minimizing re-projection error, the error from projecting a predicted feature from the 3D map approximation into the camera field of few based on the estimated camera pose and checking the actual difference from the expectation in the image.

Direct methods instead optimize over photometric error and skip image processing. This allows them to process more information about the image but leads to more difficult optimization problems.

DROID-SLAM takes an in between approach; it doesn’t use intermediate representations and passes images directly to the neural network like direct approaches, but uses the re-projection error optimization problem like indirect approaches. This way, it gets the rich features of direct and the faster optimization of indirect.

**2. Deep Learning**

Prior deep learning SLAM attempts have often attempted to implement specific features of the SLAM problem with deep learning.

There have been few end-to-end approaches and many have been incomplete.

DROID-SLAM optimizes the depth of each pixel with deep learning to make it a more flexible approach.

### Approach

> We take a video as input with two objectives: estimate the trajectory of the camera and build a 3D map of the environment.

The network operates on a collection of images $\{ I \}_{t=0}^N$ with each image have two state variables: a camera pose $G+t \in SE(3)$ and inverse depth $d_t \in R_+^{H \times W}$. Inverse depth is used for numerical stability for large depth values that approach 0.

They also use a frame graph $(\mathcal{V}, \mathcal{E})$ that’s updated with edges $(i, j) \in \mathcal{E}$ for any pair of images $I_i$ and $I_j$ that have overlapping fields of view.

**1. Feature Extraction and Correlation**

Features are extracted as images are added to the system.

There’s a feature extraction network that processes images with a feature network and a context network. The feature network builds correlation volumes and the context network is used by the network at each update.

For each edge of the frame graph, correlations between the two images are calculated by taking the dot products of the feature vectors.

The correlation volumes are then indexed for usage in later search.

**2. Update Operator**

![Screenshot 2024-11-07 at 5.56.17 PM.png](../images/Screenshot_2024-11-07_at_5.56.17_PM.png)

The core part of DROID-SLAM is an update operator that updates the camera poses and true depth map on every iteration, as well as updating the hidden state $h$.

Specifically, the GRU uses hidden state $h$ to compute a pose update $\Delta \xi^{(k)}$ and a depth update $\Delta d^{(k)}$. Then it computes the new post values and depth values as follows:

$$
\textrm{G}^{(k+1)} = \textrm{Exp}(\Delta\xi^{(k)}) \circ \textrm{G}^{(k)} \\
\textrm{d}^{(k+1)} = \Delta \textrm{d}^{(k)} + \textrm{d}^{(k)}
$$

This should eventually converge to a fixed point representing the true construction $\{ \textrm{G}^{(k)}\} \rarr \textrm{G}^*, \{ \textrm{d}^{(k)} \} \rarr \textrm{d}^*$.

At the start of each iteration, we use the current poses and depths to estimate correspondence with the current image. This gives a map of where the pixels in prior images are predicted to be.

Then, they use this correspondence field to index the correlation volumes by specifically searching in the correlation volumes in pixels predicted by the correspondence field.

This gives the network correlation correlation features and flow features that allows the network to learn to align visually similar image regions.

These correlation and flow features (which are represented across the image) pass through 2 convolutional layers and then enter the GRU, along with context features being added.

Instead of predicting depth or pose updates directly, the network instead predicts updates to dense flow fields with a revision flow field $r_{ij} \in \mathbb{R}^{H\times W \times 2}$ and associated confidence map $w_{ij} \in \mathbb{R}_+^{H \times W \times 2}$. This predicts where pixels in one image should move to based on estimation.

The dense bundle adjustment layer (DBA) then maps these flow revisions into pose and pixel-wise depth updates (basically a projection layer).

We want an updated pose $G’$ and depth $d’$ such that reprojected points match the revised correspondence $\textrm{p}_{ij}^*$. The DBA layer is part of the computation graph and back-propagation goes through the layer during draining.

**3. Training**

Training examples are made of 7-frame video sequences.

They use a pose loss and flow loss predicted from the ground truth depths and poses vs. the predicted depths and poses.

**4. SLAM System**

The SLAM system takes in video and performs localization and mapping. It has two threads: a **frontend** thread which takes new frames, extracts features, selects keyframes, and performs local bundle adjustment, and a **backend** which performs global bundle adjustment over the history of keyframes.

They initialize the network by collecting 12 frames and making a frame graph. Frames are only kept if there is sufficient optical flow between them.

The frontend operates directly on the incoming video frames by maintaining the keyframes and frame graph, updating pose and depth estimations, and removing redundant keyframes.

The backend performs full global bundle adjustment on all keyframes and updates the frame graph.

The system can also use Stereo and RGB-D, treating depth as a variable which can still have error. It can be trained on and the network learns to remove this error.

### Experiments

> We compare to both deep learning and established classical SLAM algorithms and put specific emphasis on cross-dataset generalization.

> Our network is trained entirely on monocular video from the synthetic TartanAir dataset. Training takes 1 week on 4 RTX-3090 GPUs.

Goes to show how much room for improvement there is if models still have room to improve by scaling up parameters and training them with more compute.

**1. TartanAir**

![Screenshot 2024-11-07 at 6.28.12 PM.png](../images/Screenshot_2024-11-07_at_6.28.12_PM.png)

> On most sequences, we outperform existing methods by an order-of-magnitude and achieve 8x lower average error than TartanVO and 20x lower than DeepV2D.

**2. EuRoC**

![Screenshot 2024-11-07 at 6.29.04 PM.png](../images/Screenshot_2024-11-07_at_6.29.04_PM.png)

**3. TUM-RGBD**

![Screenshot 2024-11-07 at 6.30.23 PM.png](../images/Screenshot_2024-11-07_at_6.30.23_PM.png)

> The RGBD dataset consists of indoor scenes captured with handheld camera. This is a notoriously difficult dataset for monocular methods due to rolling shutter artifacts, motion blur, and heavy rotation.

> It successfully tracks all 9 sequences while achieving 83% lower ATE than DeepFactors and which succeeds on all videos and 90% lower ATE than DeepV2D.

**4. ETH3D-SLAM**

![Screenshot 2024-11-07 at 6.30.35 PM.png](../images/Screenshot_2024-11-07_at_6.30.35_PM.png)

> Our system can run in real-time with 2 3090 GPUs. Tracking and local BA is run on the first GPU, while global BA and loop closure is run on the second.

### Conclusion

> We introduce DROID-SLAM, an end-to-end neural architecture for visual SLAM. DROID-SLAM is accurate, robust, and versatile and can be used on monocular, stereo, and RGB-D video. It outperforms prior work by large margins on challenging benchmarks.

---

# TRPO

<aside>
📜

[Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477)

</aside>

> We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm.

They first describe a theoretical optimization process to improve policies monotonically, and then create a tractable approximation of this which is TRPO.

Policy optimization algorithms can be classified into 3 categories:

1. **Policy iteration** methods which estimate value function under the current policy and then improve the policy
2. **Policy gradient** methods which estimate the gradient of expected return using sample trajectories
3. **Derivative-free optimization** methods like cross-entropy method (CEM) and covariant-matrix adaptation (CMA)

> In our experiments, we show that the same TRPO methods can learn complex policies for swimming, hopping, and walking, as well as playing Atari games directly from raw images.

TRPO is practically effective.

### Preliminaries

Given an MDP defined by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \rho_0, \gamma)$ defining the finite set of states $\mathcal{S}$, the finite set of actions $\mathcal{A}$, the transition probability distribution $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rarr \mathbb{R}$, the reward function $r: \mathcal{S} \rarr \mathbb{R}$, the distribution of the initial state $s_0$ given by $\rho_0: \mathcal{S} \rarr \mathbb{R}$, and the discount factor $\gamma \in (0,1)$.

We can represent the expected discounted reward of a policy $\pi$ as:

$$
\eta(\pi) = \mathbb{E}_{s_0,a_0,...}\left[ \sum_{t=0}^\infty \gamma^t r(s_t) \right]
$$

With dynamics defined by:

$$
s_0 \sim \rho_0(s_0), a_t \sim \pi(a_t|s_t), s_{t+1} \sim P(s_{t+1}|s_t, a_t)
$$

In other words, the starting state comes from the distribution of starting states, all actions are sampled from the policy $\pi$ given the current state, and the next state is given by the probability dynamics of the environment given the current state and action.

Then the value function gives the expected value of a given state for a policy (summing over all the possible actions in that state):

$$
V_\pi(s_t) = \mathbb{E}_{a_t,s_{t+1},...}\left[  \sum_{t=0}^\infty \gamma^lr(s_{t+l}) \right]
$$

Meanwhile, the action value function (Q-function) specifies the expected reward of taking a specific action in a given state:

$$
Q_\pi(s_t) = \mathbb{E}_{s_{t+1},a_{t+1},...}\left[  \sum_{t=0}^\infty \gamma^lr(s_{t+l}) \right]
$$

So the advantage function represents the difference between the Q-function and the value function, indicating how much better or worse an action is than the expected value of a state:

$$
A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)
$$

Then during optimization, the policy can update to increasingly use actions that have an advantage over the current policies value function.

We can then frame the expected return of another policy $\hat{\pi}$ in terms of its advantage over $\pi$:

$$
\eta(\hat{\pi}) = \eta({\pi}) + \mathbb{E}_{s_0,a_0,...}\left[ \sum_{t=0}^\infty \gamma^t A_\pi(s_t, a_t) \right]
$$

Instead of calculating the expected discounted reward for $\hat{\pi}$ the classic way by accumulating expected rewards, we instead just accumulate expected advantages over $\pi$ and add that to the expected discounted reward of $\pi$.

Then substituting in the visitation frequencies of each state $\rho_\pi$ and expanding out the expectation, we get that:

$$
\eta(\hat{\pi}) = \eta(\pi) + \sum_s \rho_{\hat{\pi}}(s) \sum_a \hat{\pi}(a|s)A_\pi(s, a)
$$

So we see that as long as every advantage is positive or zero, then $\hat{\pi}$ is guaranteed to improve the policies performance overall.

However, $\hat{\rho}_\pi$ makes this quantity difficult to optimize directly, so they instead use an approximation where they instead use the state probability densities from $\rho$ instead of $\hat{\pi}$:

$$
L_\pi(\hat{\pi}) = \eta(\pi) + \eta(\pi) + \sum_s \rho_{\pi}(s) \sum_a \hat{\pi}(a|s)A_\pi(s, a)
$$

However, this assumes that the probability densities are similar, which requires that $\pi$ and $\hat{\pi}$ are not so far apart such that the densities become inaccurate and ruin the optimization.

### Monotonic Improvement Guarantee

They prove that the procedure they suggest is guaranteed to monotonically improve the policy over time.

Then they suggest the following policy iteration algorithm that guarantees decreasing expected return $\eta$.

![Screenshot 2024-11-05 at 6.43.36 PM.png](../images/Screenshot_2024-11-05_at_6.43.36_PM.png)

### Optimization for Parameterized Policies

In practice, they find that the equivalent form to their algorithm that’s far easier to optimize is the following:

$$
\textrm{maximize}_\theta L_{\theta_\textrm{old}(\theta)} \\
\textrm{subject to} D_{KL}^{\textrm{max}}(\theta_\textrm{old}, \theta) \leq \delta
$$

And they use a simplified KL divergence as the constraint instead:

![Screenshot 2024-11-05 at 6.46.20 PM.png](../images/Screenshot_2024-11-05_at_6.46.20_PM.png)

### Sample-Based Estimation of the Objective and Constraint

They need to solve the following optimization problem:

$$
\textrm{maximize}_\theta \sum_s \rho_{\theta_\textrm{old}}(s) \sum_a \pi_\theta(a|s) A_{\theta_{\textrm{old}}}(s, a)
$$

subject to $\overline{D}_KL^{\rho_{\theta_\textrm{old}}}(\theta_\textrm{old}, \theta) \leq \delta$.

They then simplify the optimization function to the following and sample across trajectories in $\pi$ to approximate the optimization and constraints, which they then use to update the function.

$$
\textrm{maximize}_\theta \mathbb{E}_{s \sim \rho_{\theta_\textrm{old}}, a \sim q} \left[ \frac{\pi_\theta(a|s)}{q(a|s)} Q_{\theta_\textrm{old}(s, a)} \right]
$$

> All that remains is to replace the expectations by sample averages and replace the Q value by an empirical estimate.

![Screenshot 2024-11-05 at 6.56.40 PM.png](../images/Screenshot_2024-11-05_at_6.56.40_PM.png)

They use two different sampling methods.

First, they use **single path** sampling where they collect a sequence of initial states sampled from $\rho_0$ and simulating the policy on it for some number of time steps to generate a trajectory. These trajectories are then used for approximation.

Additionally, they use a **vine** strategy where the policy is used to generate many trajectories. Then a subset of $N$ states along these trajectories are used, and then they sample actions from those states from the q-values usually based on the policy.

> The benefit of the vine method over the single path method that is our local estimate of the objective has much lower variance given the same number of Q-value samples in the surrogate objective. That is, the vine method gives much better estimates of the advantage values.

The vine method gives much better estimates of advantage values because it gets more information about the average value at certain states, but it comes with the disadvantage of calling the simulator more frequently.

### Practical Algorithm

> Here, we present two practical policy optimization algorithms based on the ideas above, which use either the _single path_ or _vine_ sampling scheme from the preceding section.

Both of the algorithms are based on the following:

1. Collect state-action pairs and Monte Carlo estimates of their Q-values
2. Average over the samples to get an estimated objective and constraint function as described
3. Approximately solve the constrained optimization problem given by the two equations

The primary relationship between the theory and the practical algorithm:

- The theory justifies using a surrogate objective for optimization with a constraint on KL divergence. In practice, they limit the KL divergence with $\delta$.
- They use a simplified KL divergence constrained for computational efficiency.
- They ignore estimation error in the advantage function for the sake of simplicity.

### Experiments

**1. Simulated Robotic Locomotion**

![Screenshot 2024-11-05 at 6.12.07 PM.png](../images/Screenshot_2024-11-05_at_6.12.07_PM.png)

They test several deep RL algorithms in MuJoCo with robot 3 robot locomotion problems (swimmer, hopper, walker) with established reward functions.

TRPO using single path and vine sampling learned all the problems successfully and yielded the best solutions.

> These results provide empirical evidence that constraining the KL divergence is a more robust way to choose step sizes and make fast, consistent progress, compared to using a fixed penalty.

> Note that TRPO learned all of the gaits with general purpose policies and simple reward functions, using minimal prior knowledge. This is in contrast with most prior methods for learning locomotion, which typically rely on
> hand-architected policy classes that explicitly encode notions of balance and stepping.

TRPO learned all correct gaits without strong priors encoded into the reward functions by humans, indicating a far superior learning method.

**2. Playing Games from Images**

![Screenshot 2024-11-05 at 6.15.59 PM.png](../images/Screenshot_2024-11-05_at_6.15.59_PM.png)

> The 500 iterations of our algorithm took about 30 hours (with slight variation between games) on a 16-core computer.

> Unlike the prior methods, our approach was not designed specifically for this task.

TRPO was not designed for these tasks but still generalized well.

> The ability to apply the same policy search method to methods as diverse as robotic locomotion and image-based game playing demonstrates the generality of TRPO.

### Discussion

> We proved monotonic improvement for an algorithm that repeatedly optimizes a local approximation to the expected return of the policy with a KL divergence penalty, and we showed that an approximation to this method that incorporates a KL divergence constraint achieves good empirical results on a range of challenging policy learning tasks, outperforming prior methods.

> To our knowledge, no prior work has learned controllers from scratch for all
> of these tasks, using a generic policy search method and non-engineered, general-purpose policy representations.

> Since the method we proposed is scalable and has strong theoretical foundations, we hope that it will serve as a jumping-off point for future work on training large, rich function approximators for a range of challenging problems.

> At the intersection of the two experimental domains we explored, there is the possibility of learning robotic control policies that use vision and raw sensory data as input, providing a unified scheme for training robotic controllers that perform both perception and control.

They predict here that given the utility of this policy in the case of gait control in the MuJoCo simulation, and interpreting image data in the Atari task, this algorithm may be useful for robotic control policies. This became acurrate especially with PPO.

# Curiosity

<aside>
📜

[Large-Scale Study of Curiosity Driven Learning](https://arxiv.org/pdf/1808.04355)

</aside>

Many reinforcement learning environments rely on well-defined extrinsic rewards. In the absence of extrinsic rewards, models often use intrinsic rewards.

In scenarios where there are no extrinsic rewards, models can use intrinsic rewards alone like curiosity. Humans seem to use curiosity as their primary optimization function during early development.

> Indeed, there is evidence that pre-training an agent on a given environment using only intrinsic rewards allows it to learn much faster when fine-tuned to a novel task in a novel environment.

Training with purely curiosity as the objective in an environment shows evidence of being a good pre-training objective with value for fine-tuning to specific tasks later on.

> The central idea is to represent intrinsic reward as the error in predicting the consequence of the agent’s action given its current state, i.e., the prediction error of learned forward-dynamics of the agent.

This is just like the free energy principle. Minimizing free energy is about minimizing prediction error with an active element, which is exactly what curiosity driven learning is doing.

> To ensure stable online training of dynamics, we argue that the desired embedding space should: (a) be compact in terms of dimensionality, (b) preserve sufficient information about the observation, and (c) be a stationary function of the observations.

In this paper, they accomplish:

1. A large scale study of the efficacy of curiosity based learning on different tasks/environments.
2. The different feature spaces used for learning dynamics-based curiosity.
3. Exploring the limitations of prediction-error based curiosity formulation.

### Dynamics-Based Curiosity Driven Learning

> We want to incentivize the agent with a reward $r_t$ relating to how informative the state transition was.

This is accomplished with a network to embed observations into representations $\phi(x)$, and a forward dynamics network that predicts the representation of the next state from the previous state and action $p(\phi(x_{t+1})|x_t, a_t)$.

Then, the exploration reward is given by the **surprisal** of the model given a transition tuple: $r_t = -\log p(\phi(x_{t+1})|x_t, a_t)$.

This means that the model will favor maximally exploring areas of the environment that it least understands, which may be areas that are unexplored or areas with complex dynamics.

This determines the reward function for the RL environment, which can be used in any of the variety of deep RL training methods.

**1. Feature spaces for forward dynamics**

> A good choice of feature space can make the prediction task more tractable and filter out irrelevant aspects of the observation space.

A good feature space should be compact (real features should be able to be modeled in lower dimensional features), sufficient (the representations contain all required information), and stable (the model features change over time, and old information becomes boring - this needs to be addressed).

They evaluate a few different models for the feature space:

1. **Pixels** - the identity transformation where $\phi(x) = x$. This is stable but makes it hard to learn from the environment
2. **Random Features** - the network is fixed after initialization. stable and more complex than the identity, but still insufficient.
3. **VAEs** - fit latent variable generative models $\rho(x, z)$ for observed data $x$ and latent variable $x$ with prior $p(z)$ using variational inference. The mapping to mean can be used as the embedding network $\phi$. This filters out more noise, but features update as the VAE trains.
4. **Inverse Dynamics Features (IDFs)** - given $(s_t, s_{t+1}, a_t)$, the inverse dynamics task is to predict the action $a_t$ given the previous and next states $s_t$ and $s_{t+1}$.

**2. Practical considerations in training an agent driven purely by curiosity**

- They use PPO for all their experiments
- They normalize the scale of rewards so the value function can learn quickly
- They normalize the advantages in PPO
- They normalize the observations when training
- They use many parallel actors

**3. “Death is not the end:” discounted curiosity with infinite horizon**

The done signal actually leaks information about the external environment to the agent because it indicates that a reward point has been reached or death has been achieved, so they instead make the game loop back to the beginning on done so there is no bias except curiosity.

The agent then does learn to avoid dying just because it puts it back to a point of low information again at the beginning.

### Experiments

**1. Curiosity-driven learning without extrinsic rewards**

![Screenshot 2024-11-05 at 5.18.15 PM.png](../images/Screenshot_2024-11-05_at_5.18.15_PM.png)

> A pure curiosity-driven agent can learn to obtain external rewards even without using any extrinsic rewards during training.

RF and IDF models performed best.

> We found that an IDF-curious agent collects more game reward than a random agent in 75% of the Atari games, an RF-curious agent does better in 70%.

![Screenshot 2024-11-05 at 5.20.29 PM.png](../images/Screenshot_2024-11-05_at_5.20.29_PM.png)

> This result suggests that the performance of a purely curiosity-driven agent would improve as the training of base RL algorithm (which is PPO in our case) gets better.

> We see from the episode length that the agent learns to have more and longer rallies over time, learning to play pong without any teacher – purely by curiosity on both sides.

In pong with just two curiosity driven agents, both agents learn to play longer rallies over time, driven purely by curiosity.

**2. Generalization across novel levels in Super Mario Bros.**

![Screenshot 2024-11-05 at 5.25.19 PM.png](../images/Screenshot_2024-11-05_at_5.25.19_PM.png)

> These results might suggest that while random features perform well on training environments, learned features appear to generalize better to novel levels.

> Overall, we find some promising evidence showing that skills learned by curiosity help our agent explore efficiently in novel environments.

**3. Curiosity with Sparse External Reward**

Combining curiosity with an intrinsic reward can help to maximize reward faster (though they use a contrived example in this paper).

This is likely what humans use.

However, the combination of extrinsic and intrinsic rewards isn’t new; achieving this balance is the goal of all dual descent entropy maximization training methods, and the purpose of introduce $\epsilon$ in q-learning.

### Discussion

> There are some Atari games where exploring the environment does not correspond to extrinsic reward.

> More generally, these results suggest that, in environments designed by humans, the extrinsic reward is perhaps often aligned with the objective of seeking novelty.

This makes sense especially in video games, since curiosity is one of the main functions driving humans to interact in their environments.

> A more serious potential limitation is the handling of stochastic dynamics. If the transitions in the environment are random, then even with a perfect dynamics model, the expected reward will be the entropy of the transition, and the agent will seek out transitions with the highest entropy.

Seeking out entropy is only useful to a point. The model may converge on sources of noise, which is not a valuable source of learning. It would be interesting to see a curiosity driven learning that isn’t just focused on unpredicted data but is focused on new signal that has structure.

![Screenshot 2024-11-05 at 5.33.14 PM.png](../images/Screenshot_2024-11-05_at_5.33.14_PM.png)

When they add a noisy TV to training environment, the model just gets stuck there forever and learning halts. This needs to be addressed.

---

# ALVINN

<aside>
📜

[ALVINN: An Autonomous Land Vehicle in A Neural Network](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)

</aside>

> Specifically, ALVINN is an artificial neural network designed to control the NAVLAB, the Carnegie Mellon autonomous navigation test vehicle.

### Network Architecture

![Screenshot 2024-11-06 at 12.57.22 PM.png](../images/Screenshot_2024-11-06_at_12.57.22_PM.png)

ALVINN takes input in the form of the blue channel of a camera (because this channel has the highest contrast between road and non-road) and a range measurement input, as well as a road intensity feedback unit pulling from the previous inference run that measures if the road is getting higher or lower contrast from it’s environment.

It uses these values to predict a linear representation of the curvature of the road, which it uses to determine the direction to turn in to move closer to the center of the road.

### Training and Performance

They need data from a variety of road conditions including different lighting and noise, which is hard to collect directly.

Instead, they have a human drive around in a car and capture the input road images and driving commands.

Then, they use a simulator to augment this dataset by generating new road images with more noise using transformations of images from this data.

Since they trained on human driving data and learned a control policy to mimic human driving behavior, this is often considered the first behavior cloning paper.

They set the road intensity unit input to a random activation level during early training to prevent the model from just learning to copy the input to the output (since real road intensity will be the same between images).

By doing this, they basically allow the model to independently form a road intensity prediction and only factor in the previous intensity once it has learned to do this.

> After 40 epochs of training on the 1200 simulated road snapshots, the network correctly dictates a tum curvature within two units of the correct answer approximately 90% of the time on novel simulated road images.

### Network Representation

> The representation developed by the network to perform the road following task depends dramatically on the characteristics of the training set.

When the roads in the training set are all fixed with, the network features become overlapping road filters.

When the roads are of varying lengths, the units start to be independent feature detectors, like one unit detecting the left edge of the road.

### Discussion and Extensions

> The distinct representations developed for different circumstances illustrate a key advantage provided by neural networks for autonomous navigation. Namely, in this paradigm the data, not the programmer, determines the salient image features crucial to accurate road navigation.

An early demonstration feeling the fact that neural networks tune parameters far faster than humans can. They compare the 1 hour training time of ALVINN to the months spent by the CMU team to make features using traditional image processing techniques.

> By interactively training the network on real road images taken as a human drives the NAVLAB, we hope to develop a system that adapts its processing to accommodate current circumstances.

They further suggest behavior cloning as a way for the neural network to learn from humans.

The network has to presented with enough variability in training to generalize properly.

> In addition, the network must not solely be shown examples of accurate driving, but also how to recover (i.e. return to the road center) once a mistake has been made.

When training with behavior cloning, the network needs to know how to operate in failure cases. If it isn’t explicitly trained in these environments, it won’t know what to do.

> Another important advantage gained through the use of neural networks for autonomous navigation is the ease with which they assimilate data from independent sensors.

Appreciating that neural networks can easily integrate any correlated data with a task to make predictions.

> In the area of planning, interesting extensions include stopping for, or planning a path around, obstacles.

> Beyond dealing with individual intersections, we would eventually like to integrate a map into the system to enable global point-to-point path planning.

These really highlight why Tesla has always been a robotics company. Making autonomous vehicles is completely a robotics problem involving perception, planning, and control.

### Conclusion

> We are optimistic concerning the eventual contributions neural networks will make to the area of autonomous navigation.

> We certainly believe it is important to begin researching and evaluating neural networks in real world situations, and we think autonomous navigation is an interesting application for such an approach.

# DAgger

<aside>
📜

A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning [[Video 1](https://www.youtube.com/watch?v=V00npNnWzSU), [Video 2](https://www.youtube.com/watch?v=anOI0xZ3kGM)]

</aside>

Training robotics controllers using imitation learning has been a very effective approach in robotics problems where robots need to learn sequence prediction.

Imitation learning typically involves training a model to predict an expert’s behavior given encountered observations.

> Since the learner’s prediction affects future input observations/states during execution of the learned policy, this violate the crucial i.i.d. assumption made by most statistical learning approaches.

Mistakes in learner predictions can affect future observations, bringing the agent further out of the distribution of data it was trained on (because the dataset is not i.i.d).

A classifier that makes mistakes with probability $\epsilon$ may make $T^2 \epsilon$ mistakes in $T$ time-steps due to the compounding errors made from unfamiliar observations compared with expert data. This was discussed at the end of the ALVINN paper.

> We propose a new meta-algorithm for imitation learning which learns a stationary deterministic policy guaranteed to perform well under its induced distribution of states.

They have an algorithm where the policy error grows linearly with $T$ and $\epsilon$ instead of the quadratic $T^2 \epsilon$ we saw with typical imitation learning approaches.

### Preliminaries

We deal with some scenario where a model is trying to mimic the behavior of an expert.

We denote the expected immediate cost of performing some action $a$ in state $s$ as $C(s, a)$, representing the incorrectness of the action compared to the expert demonstration.

Then we can define $C_\pi(s) = \mathbb{E}_{a \sim \pi(s)}[C(s, a)]$ representing the expected immediate cost of using policy $\pi$ in $s$ by averaging the cost across it’s distribution of actions.

Then the total cost of executing policy $\pi$ for $T$ steps is given by (with $d_\pi^t$ representing the distribution of states at time $t$ under policy $\pi$):

$$
J(\pi) = \sum_{t=1}^{T} \mathbb{E}_{s \sim d_\pi^t}[C_\pi(s)]
$$

We can simplify this by using the following average visitation frequency of each state across the entire episode $d_\pi = \frac{1}{T} \sum_{t=1}^T d_\pi^t$.

Then we get the following reward function:

$$
J(\pi) = T \mathbb{E}_{s \sim d_\pi}[C_\pi(s)]
$$

Ideally, we want to optimize our policy by driving this cost $J(\pi)$ down to 0.

However, we don’t have direct access to $C(s, a)$, we only have access to the observed behavior of the expert in demonstrations.

This gives us access to the observed surrogate loss between $\pi$ and $\pi^*$ given by $\ell(s, \pi)$. In some cases, this loss may be exactly equal to $C$, for example, in the case where the model has to predict the expert action directly.

We want to find a policy $\hat{\pi}$ to minimize the surrogate loss:

$$
\hat{\pi} = \underset{\pi \in \Pi}{\arg \min} \mathbb{E}_{s \sim d_\pi} [\ell(s, \pi)]
$$

**1. Supervised Approach to Imitation Learning**

Given an error $\epsilon$ in $\pi$, we get that $J(\pi) \leq J(\pi^*) + T^2 \epsilon$.

This quadratic term gives a poor error guarantee. The policy $\pi$ will tend to perform well in the distribution of states $d_{\pi^*}$ encountered by the expert.

**2. Forward Training**

In forward training, they learn an individual policy for each time step $\pi_1, \pi_2, \pi_3, …$. By doing this, $\pi_t$ mimics $\pi^*$ on the distribution of states provided by the prior policy at time $t$.

> Hence the forward algorithm guarantees that the expected loss under the distribution of states induced by the learned policy matches the average loss during training, and hence improves performance.

This algorithm is impractical because it requires $T$ policies for $T$ time steps which is infeasible.

> Hence it can not be applied to most real-world applications.

**3. Stochastic Mixing Iterative Learning**

Another approach that solves the problem with forward training (increasing valuing it’s own policy over the expert) and achieves near-linear cost of $T\epsilon$.

### Dataset Aggregation

DAgger uses the expert policy to gather a dataset of trajectories $\mathcal{D}$ to train a policy $\hat{\pi}_2$. It then continues, adding newly collected trajectories to $\mathcal{D}$ and training a new policy $\hat{\pi}_{n+1}$ on all the trajectories collected by $\hat{\pi}_n$ and before.

![Screenshot 2024-11-06 at 3.00.44 PM.png](../images/Screenshot_2024-11-06_at_3.00.44_PM.png)

> The intuition behind this algorithm is that over the iterations, we are building up the set of inputs that the learned policy is likely to encounter during its execution based on previous experience (training iterations).

Using DAgger, the dataset doesn’t just contain states encountered by the expert, but is trained on the entire distribution of states based on what the policy actually interacts with.

We can optionally allow querying the expert an any iteration with the following policy update specification:

$$
\pi_i = \beta_i \pi^* + (1 - \beta_i)\hat{\pi}_i
$$

They set $\beta = 1$ initially since the model shouldn’t learn from the randomly initialized expert. Then they can choose a decay function like $\beta_i = p^{i-1}$ to set an exponentially decaying probability of using the expert.

DAgger than constantly queries the expert which is one of its limitations. It essentially uses the policy to find states that the expert has yet to demonstrate. It’s main utility is in making the expert demonstration dataset more robust.

They they theoretically prove that for DAgger, the worst case overall error is linear with $T$ and $\epsilon$

### Theoretical Analysis

**1. Online Learning**

Online learning is a scenario where an algorithm first gives a policy $\pi_n$ with loss $\ell_n(\pi_n)$, and then uses this observed loss to provide a new policy $\pi_{n+1}$ with a new loss $\ell_{n+1}(\pi_{n+1})$.

This process repeats iteratively, and the model constantly integrates new data observed from a better policy.

A **no-regret algorithm** produces a sequence of polices $\pi_1, \pi_2, …, \pi_N$ so the average regret goes to 0 as the policies change.

**2. No Regret Algorithm Guarantees**

They go through a long proof to show that a no-regret algorithm gives some theoretical guarantees for the error and number of trajectories necessary for DAgger.

### Experiments

**1. Super Tux Kart**

![Screenshot 2024-11-06 at 2.40.30 PM.png](../images/Screenshot_2024-11-06_at_2.40.30_PM.png)

> A human expert is used to provide demonstrations of the correct steering (analog joystick value in [-1,1]) for each of the observed game images.

They measure performance based on the number of falls per lap.

![Screenshot 2024-11-06 at 2.41.41 PM.png](../images/Screenshot_2024-11-06_at_2.41.41_PM.png)

> We first observe that with the baseline supervised approach where training always occurs under the expert’s trajectories that performance does not improve as more data is collected.

More data collected from the expert doing well doesn’t help the model learn from mistakes it will inevitably make, so more data collection using the default supervised learning approach doesn’t help it.

> For DAgger, we were able to obtain a policy that never falls off
> the track after 15 iterations of training.

**2. Super Mario Bros**

> Our expert in this scenario is a near-optimal planning algorithm that has full access to the game’s internal state and can simulate exactly the consequence of future actions.

The expert can perform perfectly.

They measure performance based on the distance traveled by Mario per stage before dying, running of time, or finishing.

![Screenshot 2024-11-06 at 2.46.44 PM.png](../images/Screenshot_2024-11-06_at_2.46.44_PM.png)

> A reason the supervised approach gets such a low score is that under the learned controller, Mario is often stuck at some location against an obstacle instead of jumping over it.

Since the expert always jumps over obstacles at a significant distance away, the controller did not learn how to get unstuck in situations where it is right next to an obstacle.

>

A cool specific example of a scenario where the model leaves the state distribution that the expert encountered an no longer knows what to do.

### Conclusion

> We show that by batching over iterations of interaction with a system, no-regret methods, including the presented DAGGER approach can provide a learning reduction with strong performance guarantees in both imitation learning and structured prediction.

# IRL

<aside>
📜

[Apprenticeship Learning via Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)

</aside>

> [Observing an expert demonstration] is useful in applications (such as the task of driving) where it may be difficult to write down an explicit reward function specifying exactly how different desiderata should be traded off.

Imitation learning is especially useful when the reward function is unclear or is a complex linear combination of many factors that are hard to explicitly quantify.

> The MDP formalism is useful for many problems because it is often easier to specify the reward function than to directly specify the value function (or optimal policy).

The whole reason we optimize against the reward function is that we believe we can construct the reward function much more easily than we can the policy.

> However, we believe that even the reward function is frequently difficult to specify manually.

In real environments, the reward function itself can be intractable.

> We believe that the difficulty of manually specifying a reward function represents a significant barrier to the broader applicability of reinforcement learning and optimal control algorithms.

> The entire field of reinforcement learning is founded on the presupposition that the reward function […] is the most succinct, robust, and transferable definition of the task.

This is the core assumption behind many reinforcement learning approaches. The reward function and the environment contain all the information necessary to complete the task.

IRL instead deals with scenarios where the reward function is intractable because it isn’t directly accessible, whereas the policy can be observed and imitated to simulate the reward function it’s optimize for.

> The problem of deriving a reward function from observed behavior is referred to as **inverse reinforcement learning.**

### Preliminaries

We define MDP\R to be an MDP with no explicit reward function of the form $(S, A, T, \gamma, D)$.

We assume there some vector of features $\phi: S \rarr [0, 1]^k$ being factored into a “true” reward function $R^*(s) = w^* \cdot \phi(s)$, with $w^*$ representing the relative weighting of importance of the different features in $\phi$.

This represents the assumption that the true reward function represents a linear combination of some of the features learnable from $s$.

With this reward function, we then have the expected value of a policy $\pi$ over the infinite trajectories with starting state $s_0$ sampled from $D$ as:

$$
\mathbb{E}_{s_0 \sim D}[V^\pi(s_0)] = \mathbb{E}[\sum_{t=0}^\infty \gamma^t R(s_t)|\pi]
$$

We can fill in our reward function to get the expected value determined by the features $\phi$ and their relative importances:

$$
\mathbb{E}_{s_0 \sim D}[V^\pi(s_0)] = w \cdot \mathbb{E}[\sum_{t=0}^\infty \gamma^t \phi(s_t)|\pi]
$$

Then we can define the **feature expectations** to be:

$$
\mu(\pi) = \mathbb{E}\left[  \sum_{t=0}^\infty \gamma^t \phi(s_t)|\pi \right] \in \mathbb{R}^k
$$

And the reward function can be redefined as $\mathbb{E}_{s_0 \sim D}[V^\pi(s_0)] = w \cdot \mu(\pi)$.

We also assume access to demonstrations by an expert policy $\pi_E$. Given a set of trajectories sampled from $\pi_E$ (expert demonstrations), we can approximate the experts feature expectations $\mu_E$:

$$
\hat{\mu}_E = \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^\infty \gamma^t \phi(s_t^{(i)})
$$

Which we have access to because we can compute the features $\phi$ on the states $s_t$ visited by the expert in each trajectory.

This should approximately show us the features that the expert most values, giving us an approximation of $w$.

### Algorithm

Given an MDP\R, features $\phi$, and the expert feature expectations $\mu_E$ approximated from demonstrations, we have to find a policy that has similar performance to the expert on the unknown underlying reward function $R^* = w^{*T}\phi$.

Then we have to find a policy $\hat{\pi}$ such that $|| \mu(\hat{\pi}) - \mu_E ||_2 \leq \epsilon$, where $\epsilon$ specifies the error threshold between the feature expectations. This is the optimization because the actual reward and performance of the policy follow from the feature expectations.

To find this policy $\hat{\pi}$, we take the following steps:

1. Pick a random policy $\pi^{(0)}$ and compute $\mu^{(0)}$ for it
2. Select the value for $w$ that maximizes the difference between $w^T\mu_E$ and $w^T \mu^{(i)}$ of the best policy you have so far. This gives you access to the underlying reward function $R^*$ in which your policy would perform the worst compared with the expert policy. In this scenario, you get the error:

   $$
   t^{(i)} = \max_{w: ||w||_2 \leq 1} \min_{j \in \{ 0..(i-1) \}} w^T(\mu_E - \mu^{(j)})
   $$

3. If the worst case error $t^{(i)} \leq \epsilon$ then we’ve found an algorithm that’s good enough and can terminate.
4. Otherwise, use a traditional RL algorithm to get the best policy $\pi^{(i)}$ for the MDP with $R = (w^{(i)})^T \phi$. Here, we use the inferred reward function to train another RL algorithm.
5. Then we compute $\mu^{(i)}$ with this new policy, increment $i$ and repeat this process until we converge.

Step 2 is similar to finding the maximum margin hyperplane separating two sets of points from SVM.

The above method is called the **max-margin** method since it finds the reward function with the maximum margin of error between the policies in step 2.

There is also the **projection method** where we replace step 2 with a method that projects the last 2 policies and $\mu_E$ on a line and moves the policies toward $\mu_E$.

They also show that the algorithm will have performance that isn’t significantly worse than the experts so it will eventually terminate.

### Experiments

**1. Gridworld**

There are 64 macrocells, each of which have a feature $\phi_i(s)$ indicating if state $s$ is in macrocell $i$.

The true reward function specifies $R^* = (w^*)^T \phi$ with the weights $w^*$ generated randomly to give random and sparse rewards.

This is a bit of a contrived scenario because they’ve designed it so that the features perfectly match the reward function, which is only possible when you do in fact have the reward function already.

They also compare the algorithm to a bunch of random RL algorithms that aren’t actually good.

![Screenshot 2024-11-06 at 3.55.50 PM.png](../images/Screenshot_2024-11-06_at_3.55.50_PM.png)

> Thus, by learning a compact representation of the reward function, our algorithm significantly outperforms the other methods.

> We also observe that when the algorithm is told in advance which features have non-zero weight in the true reward function, it is able to learn using fewer expert trajectories.

Even when they provide the reward function in the inverse reinforcement learning format (providing weights on features), their method works better.

**2. Car Driving Simulation**

> For our second experiment, we implemented a car driving simulation, and applied apprenticeship learning to try to learn different “driving styles.”

They create features indicating the cars current lane ( including off-road) and distance to other drivers.

They collected expert demonstrations of 5 different driving styles varying in which lane (in-lane vs. off-road right) and how nice to drive (nice vs. crash).

The IRL algorithm was able to learn all the different driving styles successfully.

This suggests that if the axes for features are chosen correctly, IRL allows the model to tune itself to learn the right reward function and then build a policy around it.

### Conclusions

> Our algorithm assumed the reward function is expressible as a linear function of known features. If the set of features is sufficiently rich, this assumption is fairly unrestrictive.

The selected feature set needs to be rich enough to express all practical reward functions. This also cannot learn reward functions that are non-linear combinations of the features.

# GAIL

<aside>
📜

[Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)

</aside>

They are specifically concerned with imitation learning cases where the model has to learn from expert demonstrations, can’t query the expert after initial data collection, and doesn’t have any other form of reinforcement learning signal.

> Behavioral cloning, while appealingly simple, only tends to succeed with large amounts of data, due to compounding error caused by covariate shift.

IRL gets over this problem by learning a cost function from the expert behavior, which allows the model to infer what the expert would do in out of distribution states.

> Unfortunately, many IRL algorithms are extremely expensive to run, requiring reinforcement learning in an inner loop. Scaling IRL methods to large environments has thus been the focus of much recent work.

Since RL training is part of each iteration of IRL against new inferred cost functions, it can be computationally expensive.

> Our characterization introduces a framework for directly learning policies from data, bypassing any intermediate IRL step.

> We find that [our algorithm] outperforms competing methods by a wide margin in training policies for complex, high-dimensional physics-based control tasks over various amounts of expert data.

### Background

This paper is built on maximum causal entropy IRL with the following optimization problem:

$$
\underset{c \in C}{\textrm{maximize}} \left( \underset{\pi\in\Pi}{\min} -H(\pi) + \mathbb{E}_\pi[c(s,a)]  \right) - \mathbb{E}_{\pi_E}[c(s, a)]
$$

This optimization finds the cost function in the space of viable cost functions $c \in C$ that maximizes the cost difference between the expert policy and the best known high-entropy policy.

The inner section finds the policy $\pi \in \Pi$ with the best tradeoff of minimizing cost and maximizing entropy (for the sake of exploration) given a cost function $c$. The outer section then finds the cost function where the best policy has the largest difference from the expert policy.

### Characterizing the induced optimal policy

Their goal is to find an imitation learning algorithm that doesn’t need the IRL step of inferring a cost function and training an RL policy many times, and that works well in large environments.

Using expressive cost functions is important to make IRL work properly (cost function needs to be able to express all the necessary information). Neural networks are often a choice here.

They consider the most brought possible set of learned cost function $\mathcal{C}: \mathcal{S} \times \mathcal{A} \rarr \mathbb{R}$.

Given how large this space is, they need a regularizer $\psi: \mathbb{R}^{\mathcal{S} \times \mathcal{A}} \rarr \overline{\mathbb{R}}$ that can regularize the cost:

$$
\textrm{IRL}_\psi(\pi_E) = \underset{c \in \mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{\arg \max} -\psi(c) + \left( \underset{\pi\in\Pi}{\min} -H(\pi) + \mathbb{E}_\pi[c(s,a)]  \right) - \mathbb{E}_{\pi_E}[c(s, a)]
$$

Then we can look at the characteristics of the specific output policy given by the RL algorithm.

For any policy $\pi \in \Pi$, it’s occupancy measure $\rho_\pi: \mathcal{S} \times \mathcal{A} \rarr \mathbb{R}$ is defined by $\rho_\pi(s, a) = \pi(a|s) \sum_{t=0}^\infty \gamma^t P(s_t = s|\pi)$.

This gives the distribution of state-action pairs that an agent would encounter when using policy $\pi$ to navigate the environment over infinite trajectories.

Importantly, there’s a one-to-one mapping from an occupancy measure to a policy. Policy $\pi_\rho$ is the only policy that has occupancy measure $\rho$.

They then show that the RL to imitation learning pipeline can be described by the following optimization:

$$
\arg \min_{\pi \in \Pi} -H(\pi) + \psi^*(\rho_\pi - \rho_{\pi_E})
$$

This optimization can be viewed as trying to bring the occupancy measure of the learned policy to be as close as possible to the expert policy while maintaining entropy.

If the convex regularization function were constant (no regularization), the model would learn the exact observed occupancy measure, which is nowhere close to the real occupancy measure in a sufficiently complex environment.

This is why the regularizer is necessary.

Then, we can switch our framing of IRL.

> IRL is traditionally defined as the act of finding a cost function such that the expert policy is uniquely optimal, but now, we can alternatively view IRL as a procedure that tries to induce a policy that matches the expert’s occupancy measure.

### Practical occupancy measure matching

In practice, constant regularizer functions are impractical since the model has to learn from a finite set of expert samples.

This requires a relaxation on the occupancy measure matching:

$$
\textrm{minimize}_\pi d_\psi(\rho_\pi, \rho_E) - H(\pi)
$$

Such that $d_\psi$ smoothly penalizes violations in difference between occupancy measures.

**1. Entropy-regularized apprenticeship learning**

> It turns out that with certain settings of $\psi$, the above equation takes on the form of regularized variants of existing apprenticeship learning algorithms, which indeed do scale to large environments with parameterized policies.

Since many IRL approaches use a cost function that’s limited to a linear combination of feature vectors, they show that these algorithms are equivalent to running an optimization with the above equation using a linear cost function $\psi = \delta_\mathcal{C}$.

This feature matching approach often doesn’t allow expert policies to be accurately recovered.

> We can understand exactly why apprenticeship learning may fail to imitate: it forces $\pi_E$ to be encoded as an element of $\mathcal{C}$. If $\mathcal{C}$ does not include a cost function that explains expert behavior well, then attempting to recover a policy from such an encoding will not succeed.

By forcing the cost function to fit into a linear combination of the features, FEM limits recovery of the actual occupancy measures.

> Use of these linear cost function classes, however, limits their approach to settings in which expert behavior is well-described by such classes.

### Generative adversarial imitation learning

They select the following regularizer which has the property that it makes the regularization difference between the expert and predicted distribution equal to the Jensen-Shannon divergence between them.

![Screenshot 2024-11-06 at 6.38.42 PM.png](../images/Screenshot_2024-11-06_at_6.38.42_PM.png)

$$
\underset{\pi}{\textrm{minimize}} \: \psi^*_{GA}(\rho_\pi - \rho_{\pi_E} - \lambda H(\pi)) = D_{JS}(\rho_\pi, \rho_{\pi_E}) - \lambda H(\pi)
$$

> [This optimization] finds a policy whose occupancy measure minimizes Jensen-Shannon divergence to the expert’s. It minimizes a true metric between occupancy measures, so, unlike linear apprenticeship learning algorithms, it can imitate expert policies exactly.

The Jensen-Shannon divergence is a squared metric between distributions.

This equation then resembles a GAN. The purpose of the discriminator part $D_{JS}$ is to determine the difference between a generated distribution and target distribution, which resembles the role of the discriminator in a GAN.

Then, they want to find the saddle point where $D$ optimizes to detect between predicted and expert distributions:

$$
\mathbb{E}_\pi[\log(D(s, a))] + \mathbb{E}[\log(1-D(s,a))] - \lambda H(\pi)
$$

They use a parameterized policy $\pi_\theta$ and a discriminator network $D_w: \mathcal{S} \times \mathcal{A} \rarr (0,1)$. Then they alternate between an Adam optimized gradient step on $w$ and a TRPO step on $\theta$ to improve the policy until the model converges.

![Screenshot 2024-11-06 at 6.43.00 PM.png](../images/Screenshot_2024-11-06_at_6.43.00_PM.png)

### Experiments

They test on low-level control tasks from traditional RL and difficult high-dimensional tasks like 3D humanoid locomotion.

![Screenshot 2024-11-06 at 4.55.37 PM.png](../images/Screenshot_2024-11-06_at_4.55.37_PM.png)

They tested their algorithm against behavioral cloning, feature expectation matching (FEM), and game-theoretic apprenticeship learning (GTAL).

> Our algorithm almost always achieved at least 70% of expert performance for all dataset, nearly always dominating all the baselines.

### Discussion

> As we demonstrated, our method is generally quite sample efficient in terms of expert data.

They are sample efficient with expert data, but not environment interaction during training.

> Our approach does not interact with the expert during training

> We believe that we could significantly improve learning speed for our algorithm by initializing policy parameters with behavioral cloning, which requires no environment interaction at all.

Cool concept. They suggest using a far less effective but less computationally expensive algorithm for weight initialization to accelerate early training for their more effective algorithm.

# MAML

<aside>
📜

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400)

</aside>

> The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples.

> In effect, our method trains the model to be easy to fine-tune.

> The primary contribution of this work is a simple model and task-agnostic algorithm for meta-learning that trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task.

### Model-Agnostic Meta-Learning

**1. Meta-Learning Problem Set-Up**

> The model or learner is trained during a meta-learning phase on a set
> of tasks, such that the trained model can quickly adapt to new tasks using only a small number of examples or trials.

> In effect, the meta-learning problem treats entire tasks as training examples.

We specifically deal with a model $f$ that maps inputs $x$ to outputs $a$. We consider an individual task:

$$
T = \{ \mathcal{L}(x_1, a_1, …, x_H, a_H), q(x_1), q(x_{t+1}|x_t, a_t), H \}
$$

With the loss function $\mathcal{L}$, distribution over initial observations $q(x_1)$, transition distribution $q(x_{t+1}|x_t,a_t)$, and episode length $H$.

Given a distribution of tasks $p(\mathcal{T})$, we want our model to be able to learn a new task $\mathcal{T_i}$ drawn from $p(\mathcal{T})$ with $K$ samples.

For each training step, a new task is drawn and trained on using gradient descent. Then the test set is validated, and the loss from the test set is used to improve $f$.

> In effect, the test error on sample tasks $\mathcal{T}_i$ serves as the training error of the meta-learning process.

**2. A Model-Agnostic Meta-Learning Algorithm**

> The intuition behind this approach is that some internal representations are more transferrable than others.

Some internal representations are easier to fine-tune to new tasks. We want to push the model to learn representations that are most sensitive to new task loss. In other words, small changes to the model parameters given the loss of a new task should result in big changes to the network outputs.

![Screenshot 2024-11-06 at 10.03.33 PM.png](../images/Screenshot_2024-11-06_at_10.03.33_PM.png)

The algorithm is basically performing gradient descent on a number of tasks to update parameters, then using the sum of validation errors across tasks for these new parameters to further update parameters with the meta step-size $\beta$

![Screenshot 2024-11-06 at 10.05.40 PM.png](../images/Screenshot_2024-11-06_at_10.05.40_PM.png)

### Experimental Evaluation

**1. Regression**

The first task is training a model to fit to a new sine curve of variable amplitude and wave-length.

They test the model by fine-tuning it on $K= \{ 5, 10, 25 \}$ samples from a new task.

![Screenshot 2024-11-06 at 10.12.15 PM.png](../images/Screenshot_2024-11-06_at_10.12.15_PM.png)

MAML is way better than standard pre-training for updating to this task.

**2. Classification**

They also test MAML against N-way classification of the Omniglot and MiniImagenet datasets for classifying among $N$ classes randomly selected by the task.

> MAML compares well to the state-of-the-art results on this task, narrowly outperforming the prior methods.

> A significant computational expense in MAML comes from the use of second derivatives when back-propagating the meta-gradient through the gradient operator in the meta-objective.

**3. Reinforcement Learning**

They use REINFORCE as the main RL algorithm and TRPO as the meta-optimizer.

They first test MAML out on a symbol 2D target navigation problem.

> The results show that MAML can learn a model that adapts much more quickly in a single gradient update, and furthermore continues to improve with additional updates

It also performs well on learning locomotion.

### Discussion

> Our approach has a number of benefits. It is simple and does not introduce any learned parameters for meta-learning.

> It can be combined with any model representation that is amenable to gradient-based training, and any differentiable objective, including classification, regression, and reinforcement learning.

MAML is highly flexible.

> Reusing knowledge from past tasks may be a crucial ingredient in making high-capacity scalable models, such as deep neural networks, amenable to fast training with small datasets.

# One Shot

<aside>
📜

[One-Shot Imitation Learning](https://arxiv.org/pdf/1703.07326) [[Video](https://www.youtube.com/watch?v=oMZwkIjZzCM)]

</aside>

> We are interested in robotic systems that are able to perform a variety of complex useful task. The robot should be able to learn new tasks without long system interaction time.

We need to be able to effectively communicate the task to the robot, and they need to have the dexterity to accomplish it.

> Demonstrations are an extremely convenient form of information we can use to teach robots to overcome these two challenges.

> Ideally, we hope to demonstrate a certain task only once or a few times to the robot, and have it instantly generalize to new situations of the same task, without long system interaction time or domain knowledge about individual tasks.

> The use of soft attention over both types of inputs made strong generalization possible.

### One Shot Imitation Learning

> For each task, the goal is to control a 7-DOF Fetch robotic arm to stack various numbers of cube-shaped blocks into a specific configuration specified by the user.

> Furthermore, in each episode the starting positions of the blocks may vary, which requires the learned policy to generalize even within the training tasks.

![Screenshot 2024-11-06 at 10.47.41 PM.png](../images/Screenshot_2024-11-06_at_10.47.41_PM.png)

They use imitation learning with DAgger where they collect demonstrations from each task, then sample a set of tasks during training to perform meta-learning on.

### Architecture

> Our proposed architecture consists of three modules: the demonstration network, the context network, and the manipulation network.

**1. Demonstration Network**

> The demonstration network receives a demonstration trajectory as input, and produces an embedding of the demonstration to be used by the policy.

Because training sequences are long, they randomly remove a subset of the time steps, known as **temporal dropout**.

> Since our neural network needs to handle demonstrations with variable numbers of blocks, it must have modules that can process variable-dimensional inputs.

We need a way to map variable inputs to variable outputs. To accomplish this, each block has it’s own attention head that takes in relevant context across all vectors, has it’s own query, and has an explicit knowledge of the block coordinates and input embedding.

We have the following output of each attention head as:

$$
\textrm{output}_i \larr \textrm{Linear}(\textrm{concat}(h_i^{in}, \textrm{result}_i, (x_i, y_i, z_i), s_\textrm{robot}))
$$

**2. Context Network**

> The context network is the crux of our model. It processes both the current state and the embedding produced by the demonstration network, and outputs a context embedding.

This network provides context on the relevant demonstrations and scenario important for the task.

> For the block stacking environment specifically, the robot should only need to pay attention to the position of the block it is trying to pick up (the source block), as well as the position of the block it is trying to place on top of (the target block).

> Therefore, a properly trained network can learn to match the current
> state with the corresponding stage in the demonstration, and infer the identities of the source and target blocks expressed as soft attention weights over different blocks, which are then used to extract the corresponding positions to be passed to the manipulation network.

**3. Manipulation Network**

Once the source and target block are selected, the manipulation network just has a simple MLP that can be used to stack the block.

> This division of labor opens up the possibility of modular training: the manipulation network may be trained to complete this simple procedure.

This modularity allows the simple control network for the task. This only really works because the task is contrived. End-to-end training is likely preferable in most cases.

### Experiments

![Screenshot 2024-11-06 at 11.10.31 PM.png](../images/Screenshot_2024-11-06_at_11.10.31_PM.png)

> As the difficulty (number of stages) increases, however, conditioning on the entire demonstration starts to outperform conditioning on the final state.

> More surprisingly, conditioning on the entire demonstration also seems to outperform conditioning on the snapshot, which we originally expected to perform the best.

### Conclusions

> In this work, we presented a simple model that maps a single successful demonstration of a task to an effective policy that solves said task in a new situation.

---

# MuJoCo

<aside>
📜

[MuJoCo: A physics engine for model-based control](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

</aside>

> Existing physics engines can be used to test controllers that are
> already designed. However they lack the speed, accuracy and overall feature sets needed to automate the controller design process itself.

Current physics engines aren’t fast enough for controller design. Tools that are used for controller design don’t have physics simulation capabilities.

They suggest that the absence of good simulation tools to design controllers may be one reason modern robots perform poorly.

> We believe that numerical optimization is the most powerful and generally applicable tool for automating processes that would otherwise require human intelligence.

This is the design philosophy behind MuJoCo. They’re also right. Numerical optimization underlies ML, and probably the human brain. What does this suggest about numbers and information.

> The essence of control optimization is to automatically construct many candidate controllers, evaluate their performance in simulation, and use the data to construct better controllers.

The design process that motivated the creation of MuJoCo.

> Either way, optimizing a controller requires a vast number of dynamics evaluations for different states and controls.

In a recent work, they needed 200,000,000 evaluations, which took 10 minutes using their software, and 1 month on the previous standard software (OpenDynamics Engine [ODE]). This is a 3 order-of-magnitude increase.

This increase comes from better compute utilization, parallelization, and higher accuracy/stability allowing large time steps per calculation.

> In the context of control optimization, however, the controller is being
> "tuned" to the engine and not the other way around.

If the physics engine allows cheating, the controller will exploit this cheat. So the engine has to be accurate.

Prior physics engines were limited by either enforcing joint constraints numerically, or ignoring contact dynamics, neither of which is sufficient for robotics.

> These observations indicated that we need a new engine, representing the state in joint coordinates and simulating contacts in ways that are related to LCP but better.

So they made MuJoCo - **Mu**lti-**Jo**int Dynamics with **Co**ntact.

Contact dynamics simulation is still an area of active development, unlike smooth multi-joint dynamics which is solved.

MuJoCo is also built with several added benefits on top of a traditional simulator, like evaluating systems in parallel (useful for ML), inverse dynamics, a convenient language/compatibility, etc.

### Algorithmic Foundations

**1. Equations of Motion and Smooth Dynamics**

They use the following quantities:

| **Symbol**       | **Value**                                              | **Meaning**                                                                                                                        |
| ---------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| $\textrm{q}$     | position in generalized coordinates                    | The momentary state of the entire system. The end goal of simulation is just to render accurate positions over time.               |
| $\textrm{v}$     | velocity in generalized coordinates                    | The momentary velocities of the entire system (changes in $\textrm{q}$).                                                           |
| $M$              | inertia matrix in generalized coordinates              | Specifies how mass is distributed throughout the system to resist change in motion.                                                |
| $\textrm{b}$     | “bias” forces: Coriolis, centrifugal, gravity, springs | Forces external to the system. Ex: the forces on Earth                                                                             |
| $\tau$           | external/applied forces                                | Forces applied on the system in simulation. Ex: resisting forces applied to an actuator.                                           |
| $\phi$           | equality constraints: $\phi(\textrm{q}) = 0$           | The constraints for what can’t happen, like rigid-body overlap, and contact force applied only when touching.                      |
| $J_E$            | Jacobian of equality constraints                       | How changes to the environment would change equality constraints                                                                   |
| $\textrm{v}^*_E$ | desired velocity in equality constraint coordinates    | Defines how quickly the system will readjust to fix itself when an equality constraint is violated                                 |
| $\textrm{f}_E$   | impulse caused by equality constraints                 | The forces caused by maintaining the equality constraints (like those implied by a stationary object).                             |
| $J_C$            | Jacobian of active contacts                            | Maps how changes in generalized coordinates of link positions/joints change the position/velocity of the system and contact points |
| $\textrm{v}_C$   | velocity in contact coordinates                        | How the contact coordinates are moving over time. Useful for modeling contact behavior, like friction.                             |
| $\textrm{f}_C$   | impulse caused by contacts                             | The forces caused by maintaining contact equality constraints; objects don’t penetrate each other so they create forces instead.   |
| $\textrm{h}$     | time step                                              | Shorter time step means more accuracy but requires more computational resources.                                                   |

The first calculation is the standard motion and smooth dynamics calculations in continuous time, representing the end calculation of how all the bodies move.

They calculate this with the following steps:

1. Compute the positions and orientations of all rigid bodies (forward kinematics); detect potential collisions; construct Jacobians $J_E$, $J_C$
2. Compute the inertia matrix $M$ and the bias forces $\textrm{b}$
3. Express the equality constraint impulse $f_E$ as a function of the (unknown) $f_C$ contact impulses, calculated later. Apply constraint stabilization.
4. Solve for $f_C$ and $v_C$
5. Integrate everything numerically to get the next state.

Steps 3, 4, and 5 involved complex calculations of contact impulses that MuJoCo has implemented their own algorithms for

**2. Solving for the Contact Impulse**

Then they have to solve for the contact impulses which determine the forces of all the different rigid bodies on each other.

Instead of using the standard approach, MuJoCo uses 3 of their own algorithms for this step.

**3. Implicit Complementarity Solver**

The most accurate MuJoCo solver computes an exact solution for steps 3, 4, 5 using the complementarity constraint (2 rigid bodies either have a force and are in contact, or have no force and are not in contact).

**4. Convex Solver**

A trade-off for the prior solver, which is slightly less accurate but can be computed far more efficiently.

**5. Diagonal Solver**

The least accurate but fastest contact solver.

**6. Computational Complexity**

> The bottleneck now is in memory access. Thus the performance of physics engines such as MuJoCo tends to be dominated by cache misses more than traditional computational complexity considerations, and the only way to assess performance reliably is to run extensive timing tests.

The speed of simulation is constrained by compute.

**7. Inverse Dynamics**

> We now describe the computation of inverse dynamics, which is a unique feature of MuJoCo.

Most physics simulators don’t have inverse dynamics capabilities like MuJoCo.

This is useful for computing torques that could be used to make a robot follow a specific trajectory.

### Modeling

**1. Different ways to construct a MuJoCo model**

There are 3 different formats to make a MuJoCo model, which all contain the same information:

1. XML in MJCF file
2. C++ API calls for model construction
3. C generated by the compiler

They XML file just defines a structure to define the C++ API, which is all eventually compiled into the C.

![Screenshot 2024-11-05 at 10.40.07 AM.png](../images/Screenshot_2024-11-05_at_10.40.07_AM.png)

Missing information for the simulation is filled in to defaults.

**2. Elements of a MuJoCo model**

1. **Bodies** - Elements used to build kinematic trees
2. **Joints** - Define degrees of freedom between a body and it’s parents
3. **DOF** - Degree of freedom available
4. **Geom** - Massless geometric objects used for collisions
5. **Site** - Points of interest
6. **Constraint** - Impose any kinematic equality constraints like 3D position constraints, joint angle constraints, etc.
7. **Tendon** - Spatial paths that can be used for actuation
8. **Actuator** - Have control inputs, activation states (for pneumatics), and gains.

### Timing Tests

MuJoCo has comparable speed to SD/FAST.

> On a single desktop machine, we are able to run nearly 400,000 evaluations per second including contact dynamics.

### Summary

> In terms of smooth multi-joint dynamics, single-threaded MuJoCo is comparable to SD/FAST

> MuJoCo was developed to enable our research in model based control.

> The experience so far indicates that it is a very useful and widely applicable tool, that can accelerate progress in robotic control. Thus we have decided to make it publicly available. It will be free for non-profit research.

# Domain Randomization

<aside>
📜

[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/pdf/1703.06907)

</aside>

> Performing robotic learning in a physics simulator could accelerate the impact of machine learning on robotics by allowing faster, more scalable, and lower-cost data collection than is possible with physical robots.

Data collection with physical robots is slow and expensive. Data collection in simulation would be orders of magnitude faster and cheaper and allow much larger scale.

It could also benefit from advancements in deep reinforcement learning, where random exploration would be dangerous in the real world but could work safely and quickly in simulation.

> Unfortunately, discrepancies between physics simulators
> and the real world make transferring behaviors from simulation challenging.

Differences between simulation and the real world make up the **reality gap**.

System identification (tuning the simulation parameters to match the real system) is time consuming.

Un-modeled physical effects can make the actual model perform poorly.

Simulation often lacks the richness and noise of the real-world.

> Instead of training a model on a single simulated environment, we randomize the simulator to expose the model to a wide range of environments at training time.

Domain randomization trains the model on many random initial tuning conditions, allowing the model to generalize to the real world as just another variant of the simulation.

This paper uses domain randomization for object localization.

> To our knowledge, this is the first successful transfer of a deep neural network trained only on simulated RGB images to the real world for the purpose of robotic control.

This is the first successful robotics manipulation model successfully trained only on simulation data alone.

### Related Work

Object detection and pose estimation (detecting the position and orientation of an object) is a studied problem in robotics.

Traditional approach detect the full 3D pose of objects. Their approach avoids 3D reconstruction by using deep learning.

The **domain adaptation** problem deals with adapting vision-based models trained in a source domain to an unseen target domain. There are many domain adaptation approaches. Domain randomization eliminates the need for domain adaptation, or can be used together with it.

There have been many approaches to bridging the reality gap.

**Iterative learning control** uses a loop to train a model in simulation, then use the error in reality to further improve the simulated environment.

Domain randomization requires no further training on real world data. It doesn’t require any supervised learning or labeling that other approaches require.

The approach in this paper doesn’t rely on precise camera information or specific textures. It instead randomly generates the conditions of the simulation environment.

### Method

They want to train an object detector model that takes a single camera frame and maps it to the coordinates of a set of objects.

**1. Domain Randomization**

![Screenshot 2024-11-04 at 1.14.58 PM.png](../images/Screenshot_2024-11-04_at_1.14.58_PM.png)

> The purpose of domain randomization is to provide enough simulated variability at training time such that at test time the model is able to generalize to real-world data.

They randomize the number of distractor objects, position and texture of all objects and environment, position of camera, lighting, noise, etc.

Everything is rendered randomly in MuJoCo.

The camera is randomly places in a box around where the real camera for the robot is in reality. This lets them avoid precise camera calibration.

**2. Model Architecture and Training**

They used a VGG-16 architecture convolutional neural net, pre-trained on ImageNet to transfer learning to this case.

### Experiments

**1. Experimental Setup**

They feed their model an image of one of 8 target objects (trained on a 3D mesh of the object in simulation), along with many distractors. The model has to localize the target object by giving it’s Cartesian coordinate in 3D within an allowable error threshold.

**2. Localization Accuracy**

![Screenshot 2024-11-04 at 1.17.50 PM.png](../images/Screenshot_2024-11-04_at_1.17.50_PM.png)

> Even with over-fitting, the accuracy is comparable at a similar distance to the translation error in traditional techniques for pose estimation in clutter from a single monocular camera frame [5] that use higher-resolution images.

**3. Ablation Study**

![Screenshot 2024-11-04 at 1.20.05 PM.png](../images/Screenshot_2024-11-04_at_1.20.05_PM.png)

> Our hypothesis that pre-training would be essential to generalizing to the real world proved to be false.

Pre-training on ImageNet was unnecessary to achieve good results with sufficient training samples. This means that with sufficient samples, the models trained entirely in simulation with random weight initialization still performed well.

![Screenshot 2024-11-04 at 1.21.46 PM.png](../images/Screenshot_2024-11-04_at_1.21.46_PM.png)

> For our experiments, using a large number of random textures (in addition to random distractors and object positions) is necessary to achieving transfer.

**4. Robotics Experiments**

> We evaluated the use of our object detection networks for localizing an object in clutter and performing a prescribed grasp.

To demonstrate the utility of this sim2real transfer for robotics, they used the object localization model for grasping.

![Screenshot 2024-11-04 at 1.24.52 PM.png](../images/Screenshot_2024-11-04_at_1.24.52_PM.png)

> We deployed the pipeline on a Fetch robot [49], and found it was able to successfully detect and pick up the target object in 38 out of 40 trials, including in highly cluttered scenes with significant occlusion of the target object.

The object detection model trained purely in simulation worked successfully for grasping.

### Conclusion

> We demonstrated that an object detector trained only in simulation can achieve high enough accuracy in the real world to perform grasping in clutter.

# Dynamics Randomization

<aside>
📜

[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/pdf/1710.06537) [[Video](https://www.youtube.com/watch?v=XUW0cnvqbwM)]

</aside>

> By randomizing the dynamics of the simulator during training, we are able to develop policies that are capable of adapting to very different dynamics, including ones that differ significantly from the dynamics on which the policies were trained.

They use domain randomization on the dynamics of the simulator to allow the model to generalize to work for any set of dynamics.

> This adaptivity enables the policies to generalize to the dynamics of the real world without any training on the physical system.

This allows the robot to learn to operate in a huge number of realities, where our reality is just one of them.

This is a method to bridge the reality gap for robotics control.

> Transferring policies from simulation to the real world entails challenges in bridging the ”reality gap”, the mismatch between the
> simulated and real world environments.

> Narrowing this gap has been a subject of intense interest in robotics, as it offers the potential of applying powerful algorithms that have so far been relegated to simulated domains.

Bringing the reality gap is highly valuable because deep reinforcement learning algorithms have been so powerful in simulation.

> The effectiveness of our approach is demonstrated on an object pushing task, where a policy trained exclusively in simulation is able to successfully perform the task with a real robot without additional training on the physical system.

### Related Work

> [Deep reinforcement learning] has enabled simulated agents to develop highly dynamic motor skills.

> But due to the high sample complexity of RL algorithms and other physical limitations, many of the capabilities demonstrated in simulation have yet to be replicated in the physical world.

**1. Domain Adaptation**

Transferring learning from simulation to reality is a subset of the domain adaptation problem.

The key assumption of domain adaptation is that different domains share sufficient characteristics so that representations learned in 1 domain would be useful in the others.

> While promising, these methods nonetheless still require data from the target domain during training.

There are many useful domain adaptation approaches like learning invariant features and progressive networks.

However, all these approaches require data from the real world during training, which significantly reduces the utility of training in simulation.

**2. Domain Randomization**

> With domain randomization, discrepancies between the source and target domains are modeled as variability in the source domain.

Domain randomization uses variability to make the model generalize to differences between the source and target domain.

This method has been successfully used for sim2real transfer for training robotics vision-based policies.

There have been previous approaches to transfer learning for control policies but they have been entirely for simulation to simulation transfer.

> We show that memory-based policies are able to cope with greater variability during training and also better generalize to the dynamics of the real world.

The prior approach to sim2real transfer for robotics control using domain randomization used memoryless feedforward networks for their motor policies.

This limited their ability to adapt to the error between the simulation and real environment.

In this paper, they use memory-based policies which they claim adapt better to the mismatch between environments.

**3. Non-prehensile Manipulation**

Pushing is hard for robots to adopt because of the complexity of modeling contacts between surfaces and friction.

Deep learning has been used to make successful pushing models, but requires large datasets that take long to collect.

> In contrast, we will show that adaptive policies can be trained exclusively in simulation and using only sparse rewards.

### Background

The state of the environment at time $t$ is $s_t \in S$. The robot policy is $\pi(a|s)$ at each moment.

The reward at each instant is given by $r_t = r(s_t, a_t)$.

The agent has to maximize the multi-step reward function over the entire duration of the episode with $T$ time-steps, with discount factor $\gamma \in [0, 1]$.

$$
R_t = \sum_{t'=1}^{T} \gamma^{t'-t}r_{t'}
$$

The learning objective is to optimize the policy $\pi^*$ to maximize the return of the agent $J(\pi)$, $\pi^* = \textrm{arg max}_\pi J(\pi)$, with the following

$$
J(\pi) = \mathbb{E}[R_0|\pi] = \mathbb{E}_{r \sim p(\tau | \pi)}[\sum_{t=0}^{T-1}r(s_t|a_t)]
$$

With $p(\tau|\pi)$ the likelihood of a trajectory $\tau = (s_0, a_0, s_1, …, a_{T-1}, s_T)$ under the policy $\pi$.

$$
p(\tau|\pi) = p(s_0) \prod_{t=0}^{T-1} p(s_{t+1}|s_t, a_t)\pi(s_t, a_t)
$$

Most importantly, $p(s_{t+1}|s_t, a_t)$ is completely defined by the dynamics of the environment, so this is the place where this factor influences the policy learned.

**1. Policy Gradient Methods**

> Policy gradient methods are a popular class of algorithms for learning parametric policies where an estimate of the gradient of the objective $\nabla_\theta J(\pi_\theta)$ is used to perform gradient ascent to maximize the expected return.

This can be generalized to tasks where the agent has a different goal every episode with a universal policy.

> A universal policy is a simply extension where the goal $g \in G$ is provided as an additional input to the policy $\pi(a|s, g)$.

In this case, the goal is the location to push to and is set at the beginning of the episode and stays consistent throughout.

**2. Hindsight Experience Replay**

Instead of designing complex reward functions for reinforcement learning, it’s easier to use a binary reward that only indicates if a goal is satisfied in a given state $r(s, g)$ that yields 1 if $g$ is satisfied in $s$ and -1 otherwise.

This will yield -1 reward for all time steps in most initial episodes, preventing learning.

Instead, they use Hindsight Experience Relay (HER) to choose a goal such that it is satisfied in the final state of an episode, and use that to learn.

### Method

The goal is to train a policy that can perform a task under the dynamics of the real world $p^*(s_{t+1}|s_t, a_t)$ by training using an approximate dynamics model $\hat{p}(s_{t+1}|s_t, a_t) \approx p^*(s_{t+1}|s_t, a_t)$.

> It has been observed that DeepRL policies are prone to exploiting idiosyncrasies of the simulator to realize behaviors that are infeasible in the real world.

> Instead of training a policy under one particular dynamics model, we train a policy that can perform a task under a variety of different dynamics models.

They use domain randomization on the actual dynamics of the environment.

> By training policies to adapt to variability in the dynamics of the environment, the resulting policy might then better generalize to the dynamics of real world.

**1. Task**

Uses a binary reward with a puck that has to be pushed to a randomly selected goal position $g$ by a 7-DoF robotic arm.

![Screenshot 2024-11-04 at 2.14.29 PM.png](../images/Screenshot_2024-11-04_at_2.14.29_PM.png)

**2. State and Action**

The state is the joint positions, gripper position, velocities, and puck position, orientation, and velocities.

Actions from the policy specify target joint angles.

**3. Dynamics Randomization**

> At the start of each episode, a random set of dynamics parameters $\mu$ are sampled according to to $\rho_\mu$ and held fixed for the duration of the episode.

Mass of links, damping of joints, friction/mass/damping of puck, height of table, noise, time-steps between actions, etc. are all randomized.

**4. Adaptive Policy**

> In the absence of direct knowledge of the parameters, the dynamics can be inferred from a history of past states and actions.

The dynamics of a system are not directly accessible in the real world but can be inferred from the response to actions.

Previous systems can use a decomposition to identify system dynamics $\phi(s_t, h_t) = \hat{\mu}$ which uses the history of past states and actions $h_t$ to infer the specific dynamics properties $\hat{\mu}$, which can then be passed to the policy $\pi(a_t|s_t, \hat{\mu})$.

However, this requires specification of the exact dynamics properties you want the model to predict, which is hard in real world scenarios where dynamics can differ in unpredictable ways.

Instead, they use a memory system $\pi(a_t|s_t, z_t, g)$ where $z_t = z(h_t)$ acts as a summary of past states and actions.

> This model can then be trained end-to-end and the representation of the internal memory can be learned without requiring manual identification of a set of dynamics parameters to be inferred at runtime.

By giving the model access to the dynamics history, it can learn to infer dynamics properties on it’s own.

**5. Recurrent Deterministic Policy Gradient**

They use HER to augment the training data by turning original failed episodes into replayed goals. This requires off-policy learning.

DDPG is useful for off-policy learning for continuous control. RDPG is a form of DDPG for recurrent cases (like this one where past pushes influence future trajectory).

**6. Network Architecture**

![Screenshot 2024-11-04 at 2.27.55 PM.png](../images/Screenshot_2024-11-04_at_2.27.55_PM.png)

The model uses a policy and value network which each have a feedforward branch and recurrent branch.

The recurrent branch is tasked with inferring the dynamics of the system given past observations.

LSTM units are used for internal memory.

### Experiments

All simulations are performed in MuJoCo.

They randomize the dynamics of the environment, and they also simulate sensor noise by applying gaussian noise to the observed state features.

> Little calibration was performed to ensure that the behavior of the simulation closely conforms to that of the real robot.

> The success rate is determined as the portion of episodes where the goal is fulfilled at the end of the episode.

![Screenshot 2024-11-04 at 2.34.23 PM.png](../images/Screenshot_2024-11-04_at_2.34.23_PM.png)

> The LSTM learns faster while also converging to a higher success rate than the feedforward models.

![Screenshot 2024-11-04 at 2.35.43 PM.png](../images/Screenshot_2024-11-04_at_2.35.43_PM.png)

> The feedforward network trained without randomization is unable
> to perform the task under the real world dynamics.

![Screenshot 2024-11-04 at 2.36.49 PM.png](../images/Screenshot_2024-11-04_at_2.36.49_PM.png)

> Policies trained without randomizing the action time-step and observation noise show particularly noticeable drops in performance. This suggests that coping with the latency of the controller and sensor noise are important factors in adapting to the physical system.

### Conclusions

> Training policies with randomized dynamics in simulation enables the resulting policies to be deployed directly on a physical robot despite
> poor calibrations.

> By training exclusively in simulation, we are able to leverage simulators to generate a large volume of training data, thereby enabling us to use powerful RL techniques that are not yet feasible to apply directly on a physical system.

# Simulated Manipulation

<aside>
📜

[Learning Dexterous In-Hand Manipulation](https://arxiv.org/pdf/1808.00177) [[Video](https://www.youtube.com/watch?v=jwSbzNHGflM)]

</aside>

> Modern-day robots are typically designed for specific tasks in constrained settings and are largely unable to utilize complex end-effectors.

> The Shadow Dexterous Hand [58] is an example of a robotic hand designed for human-level dexterity.

> The hand has been commercially available since 2005; however it still has not seen widespread adoption, which can be attributed to the daunting difficulty of controlling systems of such complexity.

> The state-of-the-art in controlling five-fingered hands is severely limited.

The difficulty of dexterous manipulation is so high that after 20 years, we still haven’t made enough progress to make good robotic hands popular.

> Some prior methods have shown promising in-hand manipulation results
> in simulation but do not attempt to transfer to a real world robot.

There have been good dexterous manipulation results in simulation but they have failed to transfer to reality.

> The resulting policy exhibits unprecedented levels of dexterity and
> naturally discovers grasp types found in humans, such as the tripod, prismatic, and tip pinch grasps, and displays contact-rich, dynamic behaviors such as finger gaiting, multi-finger coordination, the controlled use of gravity, and coordinated application of translational and torsional forces to the object.

The method in this paper creates never before seen dexterous manipulation abilities using sim2real transfer.

![Screenshot 2024-11-04 at 2.49.42 PM.png](../images/Screenshot_2024-11-04_at_2.49.42_PM.png)

The success in their paper can be attributed to:

[1] Extensive randomization used in the simulated environment
[2] Control policies with memory to infer environmental dynamics
[3] Large scale distributed reinforcement learning.

### Task and System Overview

The object under consideration is placed into the palm of a humanoid robot. The robot then has to reorient the object to a desired target configuration.

Once a configuration is achieved, the robot gets a new goal. The process repeats until the robot drops the object.

**1. Hardware**

The robot arm hardware has joint sensing and 24 DoF. It’s trained on both PhaseSpace markets and 3 RGB cameras (better to match the real world scenario).

**2. Simulation**

They use MuJoCo with a simulated version of the robotic arm. The simulation has a reality gap.

### Transferable Simulations

> We cannot train on the physical robot because deep reinforcement learning algorithms require millions of samples; conversely, training only in simulation results in policies that do no transfer well due to the gap between the simulated and real environments.

This is the dilemma of using deep reinforcement learning for robotics training, though it is the best method.

To solve this, they adjust their environment to a **distribution over many simulations** to foster transfer, using the principles from domain randomization and dynamics randomization.

**1. Observations**

They omit usage of sensor values that would be inaccurate in simulation compared with reality, like the fingertip tactile sensors in the hand that depend on too many confounding variables.

**2. Randomizations**

They use domain randomization to randomize most aspects of the simulated environment so the policy generalizes.

They randomize observation noise and physics parameters.

They use a model of motor backlash to introduce action delays and action noise to model imperfect actuation.

They try to explicitly model all imperfections in reality in the simulator, and then account for other un-modeled dynamics by applying small random forces on the object in simulation.

They also randomize the visual properties of the simulator scene.

This extensive randomization can be thought of as creating noise in every dimensions possible aside from the purely signal dimensions, allowing the model to generalize to use the necessary signal and treat the noise in reality as just another noise variable that it’s used to.

### Learning Control Policies from State

**1. Policy Architecture**

They use the same policy learning architecture as the dynamics randomization paper where they use memory to infer the dynamics of the environment, using LSTM cells for memory.

They use Proximal Policy Optimization (PPO) for learning with a policy network and a value network, and use Asymmetric Actor-Critic.

![Screenshot 2024-11-04 at 3.10.59 PM.png](../images/Screenshot_2024-11-04_at_3.10.59_PM.png)

**2. Actions and Rewards**

Policy actions correspond to the desired joint angles.

They use PPO with discrete action spaces because they notice it performs better empirically.

The reward at a timestep $t$ is given by $r_t = d_t - d_{d+1}$ where $d_t$ and $d_{t+1}$ are the rotation angles between the objects orientation and the goal orientation before and after the action. The robot is rewarded 5 when the target orientation is reached and -20 when the object is dropped.

**3. Distributed Training with Rapid**

> We use the same distributed implementation of PPO that was used to train OpenAI Five without any modifications.

They train the arm with the same distribution PPO implementation used for the OpenAI Dota 2 player.

> Overall, we found that PPO scales up easily and requires little hyper-parameter tuning.

PPO is very practical for scaling up.

![Screenshot 2024-11-04 at 3.16.58 PM.png](../images/Screenshot_2024-11-04_at_3.16.58_PM.png)

> In our implementation, a pool of 384 worker machines, each with 16 CPU cores, generate experience by rolling out the current version of the policy in a sample from distribution of randomized simulations.

> This setup allows us to generate about 2 years of simulated experience per hour.

Such a cool way of quantifying the amount of learning happening!

### State Estimation from Vision

> The policy that we describe in the previous section takes the object’s position as input and requires a motion capture system for tracking the object on the physical robot.

> In this work, we therefore wish to infer the object’s pose from vision alone.

To match the real world environment, they want to be able to operate the full system with just vision rather than using an object tracking system which couldn’t be used outside the lab.

They train a network with convolutional layers to use the 3 cameras to predict the position and rotation of the object.

### Results

**1. Qualitative Results**

> During deployment on the robot as well as in simulation, we notice that our policies naturally exhibit many of the grasps found in humans.

> Furthermore, the policy also naturally discovers many strategies for dexterous in-hand manipulation described by the robotics community such as finger pivoting, finger gaiting, multi-finger coordination, the controlled use of gravity, and coordinated application of translational and torsional forces to the object.

So cool to see. The robot is rediscovering real world grasping behaviors using reinforcement learning.

Usually we’ve seen reinforcement learning infer good moves from the principles of a game, or in a simulated environment.

Very cool to see this working in real life with sufficient scale and sim2real transfer problem solved.

They also found that locking the wrist-pitch joint resulted in more intentional manipulation of the joints.

**2. Quantitative Results**

![Screenshot 2024-11-04 at 3.26.02 PM.png](../images/Screenshot_2024-11-04_at_3.26.02_PM.png)

> When using vision for pose estimation, we achieve slightly worse results both in simulation and on the real robot.

> In general, we found that problems with hardware breakage were
> one of the key challenges we had to overcome in this work.

**3. Ablation of Randomization**

![Screenshot 2024-11-04 at 3.28.56 PM.png](../images/Screenshot_2024-11-04_at_3.28.56_PM.png)

Adding randomizations does come at a cost. Policies with more randomizations requires more compute and takes longer time in simulation.

![Screenshot 2024-11-04 at 3.29.52 PM.png](../images/Screenshot_2024-11-04_at_3.29.52_PM.png)

Each of the different types of randomizations adds its own portion of quality to the end model by allowing the model to detect different forms of signal in any environment, creating a tradeoff of training time to model quality.

**4. Effect of Memory in Policies**

![Screenshot 2024-11-04 at 3.31.14 PM.png](../images/Screenshot_2024-11-04_at_3.31.14_PM.png)

Having memory to infer environmental conditions is important for functionality.

![Screenshot 2024-11-04 at 3.32.10 PM.png](../images/Screenshot_2024-11-04_at_3.32.10_PM.png)

**5. Sample Complexity & Scale**

![Screenshot 2024-11-04 at 3.33.44 PM.png](../images/Screenshot_2024-11-04_at_3.33.44_PM.png)

More years of experience and more GPUs improves model quality.

### Conclusion

> In this work, we demonstrate that in-hand manipulation skills learned with RL in a simulator can achieve an unprecedented level of dexterity on a physical five-fingered hand.

> This is possible due to extensive randomizations of the simulator, large-scale distributed training infrastructure, policies with memory, and a choice of sensing modalities which can be modeled in the simulator.

# SimOpt

<aside>
📜

[Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://arxiv.org/pdf/1810.05687)

</aside>

> However, design of the appropriate simulation parameter distributions remains a tedious task and often requires a substantial expert knowledge.

Design of the simulator distribution used in domain randomization is important.

The signal vs. noise that the model learns is based on the randomization parameters, and this determines the models ability to generalize to the real environment from simulation.

> In this work, we apply a data-driven approach and use real world data to adapt simulation randomization such that the behavior of the policies trained in simulation better matches their behavior in the real world.

They provide a better approach to creating the simulation distribution for domain randomization that’s grounded in real world data.

> Our system uses partial observations of the real world and only needs to compute rewards in simulation, therefore lifting the requirement for full state knowledge or reward instrumentation in the real world.

### Closing the Sim-to-Real Loop

![Screenshot 2024-11-04 at 4.28.39 PM.png](../images/Screenshot_2024-11-04_at_4.28.39_PM.png)

**1. Simulation Randomization**

The system dynamics for the reinforcement learning system are either induced by the real world $p(s_{t+1}|s_t, a_t)$ or by an approximate simulation $\hat{p}(s_{t+1}|s_t, a_t)$.

We can define the distribution of simulation parameters as $\xi \sim p_\phi(\xi)$ parameterized by $\phi$.

The resulting simulation engine dynamics are defined by $p_{\xi \sim p_\phi} = p(s_{t+1}|s_t, a_t, \xi)$.

> It’s possible to design a distribution of simulation parameters $p_\phi(\xi)$ such that a policy trained on $p_{\xi \sim p_\phi}$ would perform well on a real world dynamics distribution.

This is exactly what domain randomization is.

> It is often disadvantageous to use overly wide distributions of simulation parameters as they can include scenarios with infeasible solutions that hinder successful policy learning, or lead to exceedingly conservative policies.

Using wide simulation distributions works but leads to inefficiencies in training and overly conservative policies that generalize to too much outside of reality.

So tuning of the simulation distribution is important in order to prevent these inefficiencies.

**2. Learning Simulation Randomization**

> The goal of our framework is to find a distribution of simulation parameters that brings observations or partial observations induced by the policy trained under this distribution closer to the observations of the real world.

The goal of optimizing the simulation distribution is to minimize the following objective, which involves minimizing the divergence between the observations generated from the simulation and the observations generated by reality.

$$
\textrm{min}_\phi \mathbb{E}_{p_{\xi \sim p_\phi}} [\mathbb{E}_{\pi_{\theta, p_\phi}} [D(\tau_\xi^{ob}, \tau_{real}^{ob})]]
$$

Instead of doing this over the entire space of observations generated by reality and the simulation (which would be intractable), they instead use the robot policy $\pi_{\theta, p_\phi}$ to optimize the simulation parameters and to sample real world observations corresponding with the actions taken by the policy.

They use a KL-divergence step $\epsilon$ as they update the distributions to avoid going out of the trust region of the policy.

$$
D_{KL}(p_{\phi_i + 1}||p_{\phi_i}) \leq \epsilon
$$

**3. Implementation**

They use PPO on a multi-GPU cluster to run the RL training.

They parameterize the simulation parameter distribution as a Gaussian $p_\phi(\xi) \sim \mathcal{N}(\mu, \Sigma)$ with $\phi = (\mu, \Sigma)$. They also use weights for the importance of each observation dimension which can be tuned.

The simulator is non-differentiable so they use a sampling-based gradient-free algorithm based on relative entropy policy search for optimizing the objective (they don’t use a gradient based algorithm for optimizing the simulator but instead use random sampling search within the parameter space).

> We choose weighted $\ell_1$ and $\ell_2$ norms between simulation and real world observations for our observation discrepancy function $D$.

$$
D(\tau_\xi^{ob}, \tau_{rea}^{ob}) = \\ w_{\ell_1}\sum_{i=1}^T |W(o_{i, \xi} - o_{i,real})| + w_{\ell_2}\sum_{i=1}^T ||W(o_{i, \xi} - o_{i,real})||_2^2
$$

> Sampling of simulation parameters and the corresponding policy roll-outs is highly parallelizable, which we use in our experiments to evaluate large amounts of simulation parameter samples.

Given the simulation and real-world data from a single batch, they can test many different sets of simulation parameters very quickly to optimize parameters further to minimize discrepancy between simulation and real observations.

### Experiments

> As we observe, training on very wide parameter distributions is significantly more difficult and prone to fail compared to initializing with a conservative parameter distribution and updating it using _SimOpt_ afterwards.

Instead of starting with domain randomization that randomizes everything aggressively, it’s more efficient to start with a parameter distribution that’s very narrow, and expanding the size of the distribution using _SimOpt_.

> Next, we show that we can successfully transfer policies to real robots […] for complex articulated tasks such as cabinet drawer opening, and tasks with non-rigid bodies and complex dynamics, such as swing-peg-in-hole task with the peg swinging on a soft rope.

**1. Tasks**

Swing peg in hole and drawer opening.

> We would like to emphasize that our method does not require the full state information of the real world

**2. Simulation Engine**

> We use NVIDIA Flex as a high-fidelity GPU based physics simulator.

**3. Comparison to Standard Domain Randomization**

> Moreover, learning performance of standard domain randomization depends strongly on the variance of the parameter distribution.

![Screenshot 2024-11-04 at 4.19.34 PM.png](../images/Screenshot_2024-11-04_at_4.19.34_PM.png)

> Increasing variance further, in an attempt to cover a wider operating
> range, can often lead to simulating unrealistic scenarios and catastrophic breakdown of the physics simulation with various joints of the robot reaching their limits.

![Screenshot 2024-11-04 at 4.13.31 PM.png](../images/Screenshot_2024-11-04_at_4.13.31_PM.png)

> As we can observe in Fig 4, a large part of the randomized instances does not have a feasible solution

![Screenshot 2024-11-04 at 4.22.59 PM.png](../images/Screenshot_2024-11-04_at_4.22.59_PM.png)

![Screenshot 2024-11-04 at 4.22.50 PM.png](../images/Screenshot_2024-11-04_at_4.22.50_PM.png)

> Fig. 6 shows how the source distribution variance adapts to the target distribution variance for this experiment and Fig. 7 shows that our method starts with a conservative guess of the initial distribution of the
> parameters and changes it using target scene roll-outs until policy behavior in target and source scenes starts to match.

**4. Real Robot Experiments**

![Screenshot 2024-11-04 at 4.26.43 PM.png](../images/Screenshot_2024-11-04_at_4.26.43_PM.png)

> At each iteration, we perform 100 iterations of RL in approximately 7 minutes and 3 roll-outs on the real robot using the currently trained policy to collect real world observations. Then, we run 3 update steps of the simulation parameter distribution with 9600 simulation samples per update.

Both the swing-peg-in-hole and drawer opening scenarios improve with multiple SimOpt updates.

### Conclusion

> In this work, we demonstrated that adapting simulation randomization using real world data can help in learning simulation parameter distributions that are particularly suited for a successful policy transfer without the need for exact replication of the real world environment.

> We showed that updating simulation distributions is possible using partial observations of the real world while the full state still can be used for the reward computation in simulation.

---

# E2E

<aside>
📜

[End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/pdf/1504.00702)

</aside>

> Designing the perception and control software for autonomous operation remains a major challenge, even for basic tasks.

> In this article, we aim to answer the following question: can we acquire more effective policies for sensorimotor control if the perception system is trained jointly with the control policy, rather than separately?

Separate policy and perception training requires the model to perform policy search with hand engineered features which are often prone to errors.

> Successful applications of deep neural networks typically rely on large amounts of data and direct supervision of the output, neither of which is available in robotic control.

> From the control perspective, a further complication is that observations from the robot’s sensors do not provide us with the full state of the system.

> We address these challenges by developing a guided policy search algorithm for sensorimotor deep learning, as well as a novel CNN architecture designed for robotic control.

Their CNN architecture has a spatial feature point transformation that improves spatial reasoning.

> This allows us to train our policies with relatively modest amounts of data and only tens of minutes of real-world interaction time.

### Related Work

> Applications of deep learning in robotic control have been less prevalent in recent years than in visual recognition.

> Pioneering early work on neural network control used small, simple networks, and has largely been supplanted by methods that use carefully designed policies that can be learned efficiently with reinforcement learning.

Early control work using neural networks was replaced with hand designed robotic control policies.

> CNNs have also been trained to play video games
>
> However, such methods have only been demonstrated on synthetic domains that lack the visual complexity of the real world, and require an impractical number of samples for real-world robotic learning.

> Our method is sample efficient, requiring only minutes of interaction
> time. To the best of our knowledge, this is the first method that can train deep visuomotor policies for complex, high-dimensional manipulation skills with direct torque control.

Their method can be trained on only a few minutes of data, compared with reinforcement learning that requires huge amounts of samples in simulation.

> Learning visuomotor policies on a real robot requires handling complex observations and high dimensional policy representations. We tackle these challenges using guided policy search.

> In guided policy search, the policy is optimized using supervised learning, which scales gracefully with the dimensionality of the policy.

### Background

> The core component of our approach is a guided policy search algorithm that separates the problem of learning visuomotor policies into separate supervised learning and trajectory learning phases, each of which is easier than optimizing the policy directly.

**1. Problem Formulation**

The goal of policy search is to learn a policy $\pi_\theta(u_t|o_t)$ that allows the robot to perform actions $u_t$ based on observations $o_t$. The policy implicitly learns about the dynamics of the environment through observations. The goal is to minimize the task loss $\ell(x_t, u_t)$.

**2. Approach Summary**

The system has two components. The first is a supervised learning algorithm that trains policies $\pi_\theta(u_t| o_t) = \mathcal{N}(\mu^\pi(o_t), \Sigma^\pi(o_t))$ with $\mu^\pi(o_t)$ as a deep CNN and $\Sigma^\pi(o_t)$ as an observation independent learned covariance.

The second is a trajectory centric RL algorithm that generates guiding distributions $p_i(u_t|x_t)$ for supervision.

Supervised learning for policies doesn’t produce good long-horizon policies because the policy doesn’t know how to act out of distribution. The training data has to come from the policy’s state distribution to address this.

> We achieve this by alternating between trajectory-centric RL and supervised learning.

The RL stage provides supervision at states visited by the policy.

![Screenshot 2024-11-08 at 10.16.47 AM.png](../images/Screenshot_2024-11-08_at_10.16.47_AM.png)

They use pre-training for the CNN to reduce the amount of necessary data.

> The intuition behind our pre-training is that, although we ultimately seek to obtain sensorimotor policies that combine both vision and control, low-level aspects of vision can be initialized independently.

They initialize the vision part of their network by training it to predict real elements of the image $x_t$ given the observations $o_t$ to bootstrap it with necessary skills.

### Guided Policy Search with BADMM

> Guided policy search transforms policy search into a supervised learning problem, where the training set is generated by a simple trajectory-centric RL algorithm.

The end goal is to train the network that can operate on $\pi_\theta(u_t|o_t)$. To bootstrap the learning, they first collect ground truth data about the system and train the network with initialization $\pi_\theta(u_t|x_t)$ on more fine-grained control policies.

Then, they use this pre-training to train the network on $\pi_\theta(u_t|x_t)$ that makes it more accurate. The downside is they have to collect real world data about the system.

### End-to-End Visuomotor Policies

> Guided policy search allows us to optimize complex, high-dimensional policies with raw observations, such as when the input to the policy consists of images from a robot’s onboard camera.

> The guided policy search trajectory optimization phase uses the full state of the system, though the final policy only uses the observations.

They train with full data on the true state. The network then learns to correlate observations with true state probably.

> To speed up learning, we initialize both the vision layers in the policy and the trajectory distributions for guided policy search by leveraging the fully observed training setup.

### Experimental Evaluation

> We simulated 2D and 3D peg insertion, octopus arm control, and
> planar swimming and walking.

![Screenshot 2024-11-08 at 10.46.52 AM.png](../images/Screenshot_2024-11-08_at_10.46.52_AM.png)

![Screenshot 2024-11-08 at 10.47.14 AM.png](../images/Screenshot_2024-11-08_at_10.47.14_AM.png)

> These comparisons show that training even medium-sized neural network policies for continuous control tasks with a limited number of samples is very difficult for many prior policy search algorithms.

> The visual processing layers of our architecture automatically learn features points using the spatial softmax and expectation operators. These feature points encapsulate all of the visual information received by the motor layers of the policy.

### Discussion

> In this paper, we presented a method for learning robotic control policies that use raw input from a monocular camera.

> These policies are represented by a novel convolutional neural network architecture, and can be trained end-to-end using our guided policy search algorithm, which decomposes the policy search problem in a trajectory optimization phase that uses full state information and a supervised learning phase that only uses the observations.

> Our experimental results show that our method can execute complex manipulation skills, and that end-to-end training produces significant improvements in policy performance compared to using fixed vision layers trained for pose prediction.

> The success of CNNs on exceedingly challenging vision tasks suggests that this class of models is capable of learning invariance to irrelevant distractor features.

> Our method takes advantage of a known, fully observed state space during training. This is both a weakness and a strength.

---

# BC-Z

<aside>
📜

[BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning](https://arxiv.org/pdf/2202.02005)

</aside>

> End-to-end learning from pixels is a flexible choice for modeling the behavior of such generalist robots, as it has minimal assumptions about the state representation of the world.

End-to-end learning based on vision data is a generalized approach to robotics that allows functionality in all environment.

> With sufficient real-world data, these methods should in principle enable robots to generalize across new tasks, objects, and scenes without requiring hand-coded, task-specific representations.

These systems could generalize given sufficient data, though this scale of data doesn’t exist.

> In this paper, we study the problem of enabling a robot to generalize zero-shot or few-shot to new vision-based manipulation tasks.

> Achieving such generalization depends on solving challenges relating to scaling up data collection and learning algorithms for diverse data.

> First, our system incorporates shared autonomy into teleoperation to allow us to collect both raw demonstration data and human interventions to correct the robot’s current policy.

> Second, our system flexibly conditions the policy on different forms of task specification, including a language instruction or a video of a person performing the task.

The system allows robots to train on videos and allow humans to improve the robots policy over time.

> Our main contribution is an empirical study of a large-scale interactive imitation learning system that solves a breadth of tasks, including zero-shot and few-shot generalization to tasks not seen during training.

### Related Work

Imitation learning using deep learning has allowed robots to learn grasping and pick-place tasks from raw image observations.

Priori imitation learning work has achieved different forms of generalization.

Demonstrations are often collected from teleoperation data and use methods like DAgger to address distribution shift.

### Method Overview

The goal is to train a conditional policy that can take RGB images $s \in \mathcal{S}$ with task command $w \in \mathcal{W}$ in the form of a language instruction or video and accomplish the intended objective.

The policy can be written as $\mu : \mathcal{S} \times \mathcal{W} \rarr \mathcal{A}$ where $\mathcal{A}$ corresponds with the action space consisting of the 6-DoF pose of the end effector as well as the 7th degree of freedom for continuous control of the parallel jaw gripper.

The policy is trained with data collected with VR-based teleoperation using demonstration and human-in-the-loop shared autonomy, resembling HG-DAgger.

The model architecture has an encoder $q(z|w)$ that predicts an embededding $z$ from the instruction $w$ and a control layer $\pi : \mathcal{S} \times \mathcal{Z} \rarr \mathcal{A}$ that predicts an action $a$ from $w$ and the image $s$.

### Learning Algorithm

**1. Language and Video Encoders**

The encoder takes a language command or video of a human as input and produces a task embedding $w$. A lingual sentence encoder is used for language, and a ResNet-18 is used for video.

Then given examples of a human video $w_h^i$ and a demonstration demo $\{ (s, a) \}^i$, the human video is encoded $z^i \sim q(\cdot | w_h^i)$ and the embedding is passed to the control layer $\pi(a|s, z^i)$, and then the gradient of the behavior cloning loss is backpropagated through the policy and encoder.

![Screenshot 2024-11-01 at 6.15.04 PM.png](../images/Screenshot_2024-11-01_at_6.15.04_PM.png)

The loss function uses a **language regression loss** that makes the embedding space smoother by pushing the corresponding language and video embeddings closer to each other.

**2. Policy Training**

They train $\pi(a|s, z)$ using Huber loss to control the 6-DoF of the robot and gripper angle.

Images are cropped/downsampled to improve generalization during training.

At inference time, the robot predicts the next action in closed-loop fashion, but in training the model also predicts the next 10 steps in open-loop to provide another auxiliary training objective.

**3. Network Architecture**

The architecture uses a ResNet-18 body to process the images with many action heads (each a 2 hidden layer MLP). The task is conditioned on the embedding $z$ using FiLM layers.

### Experimental Results

![Screenshot 2024-11-01 at 6.25.28 PM.png](../images/Screenshot_2024-11-01_at_6.25.28_PM.png)

![Screenshot 2024-11-01 at 6.25.35 PM.png](../images/Screenshot_2024-11-01_at_6.25.35_PM.png)

BC-Z is able to generalize to new objects that weren’t operated on with the same task during training, although all the skills were still learned.

![Screenshot 2024-11-01 at 6.27.28 PM.png](../images/Screenshot_2024-11-01_at_6.27.28_PM.png)

Ablation studies show that training on multi-tasks was essential for generalization as well as HG-DAgger augmentation with human intervention rather than just expert demonstrations alone.

### Discussion

> We presented a multi-task imitation learning system that combines flexible task embeddings with large-scale training on a 100-task demonstration dataset, enabling it to generalize to entirely new tasks that were not seen in training based on user-provided language or video commands.

> The performance on novel tasks varies significantly. However, even for tasks that are less successful, the robot often exhibits behavior suggesting that it understands at least part of the task, reaching for the right object or performing a semantically related motion.

# SayCan

<aside>
📜

[Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/pdf/2204.01691)

</aside>

> A significant weakness of language models is that they lack real-world
> experience, which makes it difficult to leverage them for decision making within a given embodiment.

LLMs can provide high level semantic knowledge for complex task planning, but they can come up with narratives that aren’t grounded in the reality of the embodiment or the environment of a robot.

> In this way, the LLM describes the probability that each skill contributes to completing the instruction, and the affordance function describes the probability that each skill will succeed – combining the two provides the probability that each skill will perform the instruction successfully.

The LLM part (Say) provides knowledge on the best skills to perform the task, and the affordance function (Can) grounds it in what skills can actually work in the current environment.

> Grounding the LLM in the real-world via affordances nearly doubles the performance over the non-grounded baselines.

> Additionally, by evaluating the performance of the system with different LLMs, we show that a robot’s performance can be improved simply by enhancing the underlying language model.

The intelligence of the robot increases by just switching out the LLM, as it’s probably providing higher skill likelihoods.

### Preliminaries

**1. Large Language Models**

> In this work, we utilize the vast semantic knowledge contained in LLMs to determine useful tasks for solving high-level instructions.

**2. Value Functions and RL**

> Our goal is to be able to accurately predict whether a skill (given by a
> language command) is feasible at a current state. We use temporal-difference-based (TD) reinforcement learning to accomplish this goal.

They use Q-learning in the environment to learn whether an action is possible or not.

The robot receives a reward of 1 or 0 at the end of the episode if it succeeded or failed, so the RL function is trained to be an affordance function that specifies whether a skill is possible or not.

### SayCan

![Screenshot 2024-11-01 at 2.08.45 PM.png](../images/Screenshot_2024-11-01_at_2.08.45_PM.png)

We are given an instruction $i$ and a set of valid skills where each skill $\pi \in \Pi$ has a short language description $\ell_\pi$.

To execute the instruction, we need to learn the function that models the probability that a given skill $\pi$ will make progress toward the completion of the instruction $i$ in the current state $s$: $p(c_i | i, s, \ell_\pi)$ where $c_i$ is the probability of completion.

We can factorize this as $p(c_i | i, s, \ell_\pi) \propto p(c_\pi|s, \ell_\pi)p(\ell_\pi|i)$.

Then the LLM gives us $p(\ell_\pi|i)$, the probability that a skill $\pi$ corresponding with language description $\ell_\pi$ will make progress toward $i$ (where we sample from the LLM scoring function).

$p(c_\pi|s, \ell_\pi)$ corresponds to the affordance function of whether the skill is possible given the current state, called the world-grounding.

> The optimal skill according to the language model is computed via $\ell_\pi = \mathrm{arg max}_{\ell_\pi \in \ell_\Pi} p(\ell_\pi|i)$. Once selected, the process proceeds by iteratively selecting a skill and appending it to the instruction.

The prompt is structured as a series of human robot conversations to given an idea of the task structure.

> The key idea of SayCan is to ground large language models through value functions – affordance functions that capture the log likelihood that a particular skill will be able to succeed in the current state.

The affordance function models the likelihood that a skill can be executed successfully given the current state (with an image and other sensors).

The combination of the LLM and the value function are used to select the final skill that gets used, and then the language instruction $\ell_\pi$ gets added to the prompt and the process continues until it arrives at a termination skill.

$$
\pi = \textrm{argmax}_{\pi \in \Pi} \: p(c_\pi|s, \ell_\pi)p(\ell_\pi|i)
$$

### Implementing SayCan

> To instantiate SayCan, we must provide it with a set of skills, each of which has a policy, a value function, and a short language description.

> In our implementation, we train the individual skills either with image-based behavioral cloning, following the BC-Z method, or reinforcement learning, following MT-Opt.

SayCan can use both imitation learning using behavior cloning (from tele-op examples) or reinforcement learning from simulation.

> Regardless of how the skill’s policy is obtained, we utilize value functions trained via TD backups as the affordance model for that skill.

The value functions (affordance functions) use Q-learning in either case.

> While we find that the BC policies achieve higher success rates at the current stage of our data collection process, the value functions provided by the RL policies are crucial as an abstraction to translate control capabilities to a semantic understanding of the scene.

SayCan also conditions the BC/RL models on language using large sentence encoders to make the policies conditioned on language.

> We utilize both BC and RL policy training procedures to obtain the
> language-conditioned policies and value functions, respectively.

### Experimental Evaluation

> We test across 101 instructions from 7 instruction families.

![Screenshot 2024-11-01 at 2.23.18 PM.png](../images/Screenshot_2024-11-01_at_2.23.18_PM.png)

They measure **plan success rate**, based on if the skills selected by the model are correct for a given instruction, and **execution success rate**, based if the robot actually successfully executed the task.

> In the mock kitchen, PaLM-SayCan achieved a planning success rate of 84% and an execution rate of 74%.

![Screenshot 2024-11-01 at 2.26.16 PM.png](../images/Screenshot_2024-11-01_at_2.26.16_PM.png)

> We also find that PaLM-SayCan struggles with negation (e.g., “bring me a snack that isn’t an apple”) and ambiguous references (e.g. asking for drinks with caffeine), which is a known issue inherited from underlying language models.

These robotics models that use LLMs inherit issues from the LLMs.

**New Capabilities**

> SayCan is capable of integrating new skills by simply adding the new skills as options for the LLM and providing accompanying value functions and add an example in the prompt with that skill.

> SayCan can be integrated with recent work improving LLM reasoning, such as Chain of Thought.

> While not explicitly designed to work with multilingual queries, PaLM-SayCan is able to handle them.

### Conclusion

> We presented SayCan, a method that enables leveraging and grounding the rich knowledge in large language models to complete embodied tasks.

SayCan grounds an LLM in real-world skills by pre-training skills in a robot and then using these skills to ground the LLM.

> More specifically, we use reinforcement learning as a way to learn value functions for the individual skills that provide affordances of what is possible in the world, and then use textual labels for these skills as potential responses that are scored by a language model.

The primary limitation on SayCan is in the range of skills that it has access to.

Natural language provides semantic understanding but may not be the most descriptive medium for certain tasks.

# RT-1

<aside>
📜

[RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/pdf/2212.06817)

</aside>

Modern frontier models use transfer learning from large internet-scale task-agnostic datasets to solve a variety of downstream tasks with zero-shot or smaller task datasets.

End-to-end learning for robotics, using imitation/reinforcement, usually depends on collecting narrow task-specific.

> We therefore ask: can we train a single, capable, large multi-task backbone model on data consisting of a wide variety of robotic tasks?

And does such a model enjoy the benefits observed in other domains, exhibiting zero-shot generalization to new tasks, environments, and objects?

>

This is about applying the same learning from training LLMs and other successful frontier generative models using internet-scale data and generalization + transfer learning.

Building such a generalized robot is hard. The two challenges are in creting a good dataset and designing a good model.

Good data is especially critical and rare in robotics. Datasets have to be robot-specific and gathered manually.

Good generalization requires sufficient breadth and scale in the datasets, with many tasks and settings that are connected enough for the robot to generalize.

Designing the model for the robot is also a challenge. Multi-task learning for robots requires a high capacity model, which the transformer is perfect for.

However, robot controllers have to run in real time.

Robotics Transformer 1 (RT-1) encodes inputs/outputs (images, instructions, motor commands) into token representations that can be used by the transformer, allowing real-time inference.

> Our results show that RT-1 can perform over 700 training instructions at 97% success rate, and can generalize to new tasks, distractors, and backgrounds 25%, 36% and 18% better than the next best baseline, respectively.

> We further show that RT-1 can incorporate data from simulation or even other robot types, retaining performance on the original tasks and improving generalization to new scenarios.

### **Related Work**

> Our work takes the application of Transformers a step further and treats the mapping of language and vision observations to robot actions as a sequence modeling problem, using a Transformer to learn this mapping.

Prior attempts at using transformers for robotics use transformers just for the language modeling and planning side of the problem. RT-1 uses transformers for the entire language + vision → action sequence.

> On the technical side, our work examines how Transformer-based policies can be built so as to combine high capacity and generalization with the computational efficiency necessary for real-time control.

The design of RT-1 is meant to make real time use of transformers for robotics feasible.

> Our work adds further evidence in support of the power of multi-task, language-conditioned robotic learning.

### **Preliminaries**

The robot needs to learn a policy $\pi(\cdot | i, \{x_j\}_{j=0}^t)$ that can be used to sample a set of actions $\{a_0..a_t\}$ that ends in a terminating step $T$, which ends in a reward 1 from the binary reward function $r \in \{0, 1\}$.

RT-1 uses a transformer to parameterize $\pi$, where the transformer takes in tokens from the instruction $i$ and the images $\{x_j\}_{j=0}^t$ as inputs, and outputs action tokens $a_j$ as outputs.

RT-1 uses imitation learning with behavior cloning to learn from a dataset with many demonstrations (episodes) concluding in $r= 1$:

$$
\mathcal{D} = \{ (i^{(n)}, \{ x_t^{(n)}, a_t^{(n)} \})_{n=0}^N \}
$$

### System Overview

> The goal of this work is to build and demonstrate a general robot learning system that can absorb large amounts of data and generalize effectively.

> Our training data consists of human-provided demonstrations, and we annotate each episode with a textual description of the instruction that the robot just performed.

> The instructions usually contain a verb and one or more nouns describing the target objects. To group these instructions together, we split them into a number of skills (e.g., verbs such as “pick”, “open” or “place upright”) and objects (e.g., nouns such as “coke can”, “apple”, or “drawer”).

The data collection strategy uses manual task labeling. They group tasks into equivalent categories for generalization by using similar verbs and nouns.

![Screenshot 2024-10-31 at 2.55.00 PM.png](../images/Screenshot_2024-10-31_at_2.55.00_PM.png)

### Robotics Transformer

**1. Model**

> The RT-1 architecture relies on a data-efficient and compact tokenization of images and language instruction.

RT-1 has to be very token efficient to enable real-time inference.

It takes in a history of 6 images with $300 \times 300$ resolution, and passes it through an **EfficientNet-B3** pre-trained on ImageNet, taking the output of the final convolution layer with $9 \times 9 \times 512$.

Instead of patchifying the images before feeding them to the transformer, it uses these as 81 tokens of dimension 512 to pass to later layers.

They add a [FiLM layer](https://arxiv.org/abs/1709.07871) to the EfficientNet which takes in an embedding of the instruction from Universal Sentence Encoder to further extract information from the images into tokens based on information relevant to the instructions. The **FiLM layer** is initialized to the identity to not disrupt the EfficientNet’s pre-training.

> To further compress the number of tokens that RT-1 needs to attend over and thus speed up inference, RT-1 uses **TokenLearner**

TokenLearner uses attention to select image tokens based on their information, and only passes important tokens farther.

The TokenLearner reduces the 81 image tokens to just 8 final tokens per image passed to the transformer layer.

The transformer itself takes 8 tokens per image for a total of 48 image tokens in its input. The transformer itself is only 19M parameters.

> To tokenize actions, each action dimension in RT-1 is discretized into
> 256 bins.

> Action tokenization uses 7 variables for arm movement ($x$, $y$, $z$, roll, pitch, yaw, opening of gripper), three variables for base movement ($x$, $y$, yaw), and a discrete variable to switch between three modes: controlling arm, base, or terminating the episode.

The robot has to match the humans speed of task execution. Humans executed tasks at 2-4s / task. This allowed 3Hz control frequency, so inference has to be <100ms.

This is why using the EfficientNet + TokenLearner to reduce the total number of input tokens is important.

**2. Data**

![Screenshot 2024-10-31 at 3.18.51 PM.png](../images/Screenshot_2024-10-31_at_3.18.51_PM.png)

> Our primary dataset consists of ∼130k robot demonstrations, collected with a fleet of 13 robots over the course of 17 months.

> RT-1 is able to perform over 700 language instructions in multiple realistic office kitchen environments that we evaluate and describe in detail in the experiments.

### Experiments

**1. Experimental Setup**

RT-1 is compared to Gato and BC-Z, which both use different architectures. Both Gato and BC-Z are retrained on the RT-1 dataset and are much smaller than the papers to be able to run in real-time.

> The policies are evaluated for performance on training tasks as well as generalization to new tasks, robustness to unseen environments, and performance when chained together for long-horizon tasks.

They evaluate **seen task performance** by using tasks in the dataset but alter the configurations/placement of objects.

They evaluate **unseen task generalization** by testing new instructions with known skills/objects combined in ways that were unseen int eh dataset.

They evaluate **robustness** they change environment (for background robustness) and add unknown objects (for distractor robustness)

To evaluate **long-horizon scenarios** they test 15 long-horizon instructions which require execution of many distinct steps [using SayCan to come up with these tasks].

**2. Generalization**

> Can an RT-1 learn to perform a large number of instructions, as well as to generalize in zero shot to new tasks, objects and environments?

![Screenshot 2024-10-31 at 3.32.37 PM.png](../images/Screenshot_2024-10-31_at_3.32.37_PM.png)

RT-1 performs better on generalization across the board, doing well on unseen tasks and robustness.

**3. Simulation & External Data**

> Can we push the resulting model even further by incorporating heterogeneous data sources, such as simulated data or data from different robots?

> We demonstrate how RT1 can incorporate and learn from vastly different data sources and improve from such data.

RT-1 can use data from both real/simulation as well as data from other robots.

![Screenshot 2024-10-31 at 3.38.45 PM.png](../images/Screenshot_2024-10-31_at_3.38.45_PM.png)

RT-1 trained on the original dataset and a new simulation dataset doesn’t make the real tasks much worse but also adds the ability to transfer learning from simulation well.

![Screenshot 2024-10-31 at 3.41.23 PM.png](../images/Screenshot_2024-10-31_at_3.41.23_PM.png)

RT-1 data combined with a separate Kuka bin-picking dataset mostly maintains RT-1 ability to perform the original tasks while also improving its ability at bin picking, indicating it’s ability to generalize and combine different datasets.

This is impressive given how different the RT-1 and Kuka datasets are.

**4. Long-Horizon Generalization**

![Screenshot 2024-10-31 at 3.44.23 PM.png](../images/Screenshot_2024-10-31_at_3.44.23_PM.png)

Using SayCan, the RT-1 can do long horizon tasks and generalizes to a new kitchen far better than previous models.

**5. Data Quantity**

> In many robotics works the model size is often not the primary bottleneck, and the maximum size is limited by the latency requirement for running such models on real robots.

> Since data collection is particularly expensive for real robots, it is important to quantify what kind of data our models need to achieve a certain performance and generalization.

![Screenshot 2024-10-31 at 3.48.11 PM.png](../images/Screenshot_2024-10-31_at_3.48.11_PM.png)

> Our key takeaway is thus that data diversity is more essential than data quantity.

### Conclusion

RT-1 comes with a few challenges:

It uses imitation learning so it may be capped by the performance of demonstrators.

Generalization to new instructions is limited to previously seen concepts and can’t currently generalize to new motions.

The task set is very low in dextrous manipulation requirements.

# ACT

<aside>
📜

[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/pdf/2304.13705)

</aside>

> Existing systems for fine manipulation use expensive robots
> and high-end sensors for precise state estimation. In this work, we seek to develop a low-cost system for fine manipulation that is, in contrast, accessible and reproducible.

Dexterous manipulation systems are too expensive. ALOHA & ACT are meant to create a simpler and more cost effective dexterous manipulation system.

> However, low-cost hardware is inevitably less precise than high-end platforms, making the sensing and planning challenge more pronounced. One promising direction to resolve this is to incorporate learning into the system.

Need to mitigate the fact that cheaper hardware has less precision in sensing.

> Humans also do not have industrial-grade proprioception, and yet we are able to perform delicate tasks by learning from closed-loop visual feedback and actively compensating for errors.

Humans don’t use LiDAR or other expensive sensing setups, which means that the manipulation problem is solvable without them.

They use a cheap teleoperation setup with 2 low-cost robotic arms and 3D printed components, leading to a teleoperation system that costs <$20k.

> Small errors in the predicted action can incur large differences in the state, exacerbating the “compounding error” problem of imitation learning.

If imitation learning actions are taken one at a time, errors in one earlier part of the action sequence compound into larger errors throughout the sequence.

> To tackle this, we take inspiration from action chunking, a concept in psychology that describes how sequences of actions are grouped together as a chunk, and executed as one unit.

They make a policy that predicts the action sequence for the next $k$ time steps instead of just 1 timestep, which reduces compounding errors.

They use a transformer trained as a conditional VAE (CVAE) to implement action chunking. This is called an **action chunking transformer** (ACT).

> The key contribution of this paper is a low-cost system for learning fine manipulation, comprising a teleoperation system and a novel imitation learning algorithm

> The synergy between these two parts allows learning of 6 fine manipulation skills directly in the real-world, such as opening a translucent condiment cup and slotting a battery with 80-90% success, from only 10 minutes or 50 demonstration trajectories.

ACT allows very fast and efficient fine manipulation training.

### Related Work

Imitation-learning allows robots to learn from experts. It’s commonly done with **behavior cloning**, where imitation learning is treated as a supervised learning problem.

Behavior cloning suffers from compounding errors where errors from previous time steps build up, especially in fine manipulation.

Previous solutions like DAgger (annotation is expensive), noise injection (reduces execution quality), and synthetic correction data (uses low dimensional visual data) all have issues.

> We propose to reduce the effective horizon of tasks through action chunking, i.e., predicting an action sequence instead of a single action, and then ensemble across overlapping action chunks to produce trajectories that are both accurate and smooth.

Previous bi-manual manipulation efforts originally used classical control using environment dynamics, then used learning like reinforcement learning and imitation learning.

The ACT teleoperation setup uses joint-space mapping between the leader and follower robots, and has a setup of 3D printed parts that can be assembled in 2 hours.

### ALOHA

![Screenshot 2024-11-01 at 3.45.20 PM.png](../images/Screenshot_2024-11-01_at_3.45.20_PM.png)

> A Low-cost Open-source Hardware System for Bimanual Teleoperation

The system should be [1] low-cost, [2] versatile, [3] user-friendly, [4] repairable, [5] easy-to-build.

They use two ViperX 6-DoF robot arms which each cost $5600 as the robot arms, and replace the fingers to be better for fine manipulation.

For the controllers, they noticed that joint-space manipulation is better than task-space manipulation (better to use a physical control system then VR controllers), so they use another set of WidowX arms as the “leader” arms, which each cost $3300.

They also use 4 Logitech C922 webcams with 480 x 640 RGB image resolution streaming.

ALOHA is good at precise tasks (like threading zip ties), contact-rich tasks (like inserting RAM into a motherboard/turning book pages), and dynamic tasks (like juggling a ping pong ball with a paddle).

> Skills such as threading a zip tie, inserting RAM, and juggling ping pong ball, to our knowledge, are not available for existing teleoperation systems with 5-10x the budget.

ALOHA is more effective than way more expensive systems.

### Action Chunking with Transformers

> Existing imitation learning algorithms perform poorly on fine-grained tasks that require high-frequency control and closed-loop feedback.

To train ACT, they use human demonstrations using ALOHA.

They use the joint positions of the _leader_ as the actions.

A PID controller is used to cause the follower arm movement based on the leader movement.

They use the joint positions of the _follower_ and the image feed from the 4 cameras as the observations.

Then, they train ACT to predict the sequence of future actions given the observations.

**1. Action Chunking and Temporal Ensemble**

> We are inspired by action chunking, a neuroscience concept where individual actions are grouped together and executed as one unit, making them more efficient to store and execute.

The model policy predicts the next $k$ time steps of actions, effectively predicting a chunk of $k$ actions, which results in a $k$-fold reduction in the effective horizon of the task.

The policy models $\pi_\theta(a_{t:t+k}|s_t)$ instead of $\pi_\theta(a_t|s_t)$.

A single-step model would also struggle with temporal confusion like pauses, whereas pauses in an individual action chunk wouldn’t be an issue.

> To improve smoothness and a void discrete switching between, executing and observing, we query the policy at every time step.

Instead of running inference for $k$ actions every $k$ time steps which would be clunky, they run inference every time-step and take a **temporal ensemble** of all the action predictions at that time step with a weight average $w_i = \textrm{exp}(-m * i)$.

They are aggregating action predictions all for the same time step.

**2. Modeling Human Data**

> Another challenge that arises is learning from noisy human demonstrations. Given the same observation, a human can use different trajectories to solve the task. Humans will also be more stochastic in regions where precision matters less.

Human demonstrations can have high variance in the how demonstrators execute tasks. The model has to learn to be precise when it matters but learn a general distribution of approaches when more freedom is permissible. This is the perfect structure for VAE representations (where the model can learn the distributions to model the signal and noise in different action sequences).

> Thus, it is important for the policy to focus on regions where high precision matters.

The policy uses a conditional variational autoencoder (CVAE) to generate an action sequence based on observations.

The encoder is only used for training and predicts the mean and variance of the internal variable $z$’s distribution based on the current action sequence and observations (uses just proprioceptive data instead of images for simplicity).

The decoder uses both $z$ and current observations to predict the action sequence.

**3. Implementing ACT**

> We implement the CVAE encoder and decoder with transformers, as transformers are designed for both synthesizing information across a sequence and generating new sequences.

The CVAE encoder takes in the $k$ next target actions from the demonstration dataset with the [CLS] token and generates the $z$ style variable.

The decoder then takes the $z$ style variable and the current observations and predicts the next $k$ actions.

The CVAE decoder uses ResNet image encoders and a transformer encoder to synthesize information from different camera view points, joint positions, and the style variable and a transformer decoder to generate a coherent action sequence.

The transformer output dimensions is $k \times 512$ which is then projected down into $k \times 14$ where each value corresponds with the predicted joint position for each action time step.

![Screenshot 2024-11-01 at 6.21.54 PM.png](../images/Screenshot_2024-11-01_at_6.21.54_PM.png)

### Experiments

They use 6 real-world tasks and 2 fine manipulation tasks in MuJoCo which they use for simulation.

![Screenshot 2024-11-01 at 4.26.41 PM.png](../images/Screenshot_2024-11-01_at_4.26.41_PM.png)

They collect 50 episodes for each of the tasks, where each episode is 8-14s which corresponds to 400-700 time steps given the 50Hz control frequency.

All the different demonstrations are stochastic.

ACT performs far better than other state of the art models on all of these tasks.

![Screenshot 2024-11-01 at 4.29.28 PM.png](../images/Screenshot_2024-11-01_at_4.29.28_PM.png)

Compared with BeT and RT-1 which discretize the action space [output is a categorical distribution over discrete bins], ACT directly predicts continuous actions which is necessary for fine manipulation.

The model also performs better with action chunking and temporal ensembling.

![Screenshot 2024-11-01 at 4.45.01 PM.png](../images/Screenshot_2024-11-01_at_4.45.01_PM.png)

They also find that higher control frequency leads to faster task completion.

### Conclusion

> We present a low-cost system for fine manipulation, comprising a teleoperation system ALOHA and a novel imitation learning algorithm ACT.

> The synergy between these two parts allows us to learn fine manipulation skills directly in the real world, such as opening a translucent condiment cup and slotting a battery with a 80-90% success rate and around 10 min of demonstrations.

# RT-2 [VLA]

<aside>
📜

[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/assets/rt2.pdf)

</aside>

The semantic reasoning, problem solving, and visual interpretation capabilities of large vision-language trained on web-scale internet data would be highly valuable to general purpose robots.

However, it’s challenging for robotics to independently acquire these capabilities.

> While a brute force approach might entail collecting millions of robotic interaction trials, the most capable language and vision-language models are trained on billions of tokens and images from the web - an amount unlikely to be matched with robot data in the near future.

Very important point. The scale of useful data for LLMs provided by the internet is unlikely to be reached through collecting robotic training data directly through trials.

This makes it attractive to integrate vision-language models (VLMs) with robotics.

Current approaches have attempted to use VLMs for high-level robotic planning, taking control of a state-machine that selects individual task primitives executed by lower-level controllers.

This feels like the same mistake as trying to have deep learning controlling symbols/features determined by humans. It doesn’t allow the deep learning systems to generalize even to the low-level parts of the task.

> Therefore, in this paper we ask: can large pre-trained vision-language models be integrated directly into low-level robotic control to boost generalization and enable emergent semantic reasoning?

RT-2 combines existing vision-language models with large compute costs already dumped into them with actions to build **vision-language-action models**.

> We instantiate VLA models by building on the protocol proposed for RT-1, using a similar dataset, but expanding the model to use a large vision-language backbone.

> Besides the expected benefit of dramatically improving generalization to novel objects and semantically varied instructions, we observe a number of emergent capabilities.

VLA models use the generalization from VLMs to increase generalization present in RT-1, though the abilities are still limited to the same set of tasks/actions.

> We show that RT-2 enable significant improvements to generalization
> over objects, scenes, and instructions, and exhibit a breadth of emergent capabilities inherited from web-scale vision-language pre-training.

### Related Work

Most vision-language models use representation-learning [CLIP] or use the vision + text → text pattern.

Building generalizing robot controllers is challenging.

Prior methods have achieved generalization across object instances, new skills, new goals/language instructions, new tasks, and new environments.

> We aim to develop and study a single model that can generalize to unseen conditions along all of these axes.

The RT-2 architecture differs from prior attempts to integrate VLMs with robotics in that it doesn’t use a separate action-only model layers but all action and language are integrated into a single model.

### Vision-Language-Action Models

**1. Pre-Trained Vision-Language Models**

RT-2 uses PaLI-X and PaLM-E to make RT-2-PaLI-X and RT-2-PaLM-E.

**2. Robot-Action Fine-Tuning**

VLM has to be fine-tuned in order to convert it into a useful VLA model. RT-2 uses the same action space as RT-1.

RT-2 uses 256 tokens to be used as action tokens, which can either directly use tokens from the VLM or override the 256 least used tokens in the VLM as action tokens.

RT-2 uses co-fine-tuning to train the model with both vision-language-action pairs and standard web text-image pairs so the model retains its original abilities.

The model has output constraints in robot-task inference to only sample action tokens, whereas it can sample any token in normal inference.

**3. Real-Time Inference**

RT-2 uses a cloud multi-TPU setup instead of on-robot GPUs to meet 1-3 Hz requirements by running robot inference over the cloud.

### Experiments

The model is trained on original web-scale datasets used in other papers and the robotic action dataset used in RT-1.

**1. Generalization**

![Screenshot 2024-10-31 at 4.19.27 PM.png](../images/Screenshot_2024-10-31_at_4.19.27_PM.png)

RT-2 performs similarly to RT-1 on seen tasks, but performs far better on unknown tasks, suggesting that the usage of the VLM mainly increases generalization to new image-text concepts.

**2. Emergent Capabilities**

RT-2 is able to accomplish tasks like “put strawberry in the correct bowl” or “pick up the bag about to fall off the table.”

![Screenshot 2024-10-31 at 4.23.17 PM.png](../images/Screenshot_2024-10-31_at_4.23.17_PM.png)

RT-2 improves performance across symbol understanding, reasoning, and human recognition tasks.

**4. Chain-of-Thought**

![Screenshot 2024-10-31 at 4.24.59 PM.png](../images/Screenshot_2024-10-31_at_4.24.59_PM.png)

They augment the data to include a plan step and fine-tune on this data. RT-2 can then carry out more complex tasks.

### Limitations

Generalization with web-scale data doesn’t give RT-2 the ability to perform new motions/skills. It learns to deploy skills in new ways but still has the same seen skills.

VLA models in real time is computationally expensive. Quantization and distillation may be needed to make the models cheaper.

![Screenshot 2024-10-31 at 4.27.51 PM.png](../images/Screenshot_2024-10-31_at_4.27.51_PM.png)

# Physical Intelligence

![Screenshot 2024-11-08 at 11.15.46 AM.png](../images/Screenshot_2024-11-08_at_11.15.46_AM.png)

> We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge.

> Similarly, we may find that for effective specialized robot systems, it is more effective to first pre-train on highly diverse robot data, and then fine-tune or prompt for the desired task.

This is one of their key insights. Training at scale on diverse robot data may help to generalize to the level necessary for learning new tasks.

> This can resolve the data scarcity challenge, because many more sources of data are available to a generalist model — including data from other tasks, other robots, or even non-robot sources — and it may resolve robustness and generalization challenges.

More data allows generalization to more scenarios, robot types, etc. This is the purpose of training a foundational model on robotics. They suggest that this actually allows them to use more broad data.

> Thus, adopting a large-scale pre-training approach to robot learning
> has the potential to address many of the field’s challenges.

Developing generalist robots requires large scale, correct architecture, and the right training approach.

They use a VLA model, with an ACT, and flow-matching to represent complex continuous actions.

They combine high and low quality data so the model can perform tasks well and learn to recover from mistakes. They use large-scale pre-training to learn more general knowledge and the post-training to acquire dexterity, robustness, and efficiency for specific tasks.

They train on >10,000 hours of data.

### Related Work

> Incorporating these concepts into a VLA model, we introduce what to our knowledge is the first flow matching VLA that produces high-frequency action chunks for dexterous control.

VLA + action chunking + diffusion.

> Since one of our aims is to study complex and dexterous behaviors, we utilize a much larger dataset, with about 10,000 hours of demonstrations, complemented by the open-source OXE dataset. To our knowledge, this represents by far the largest robot learning experiment in terms of the amount of robot data.

> The complexity of the tasks we illustrate goes significantly beyond prior work. […] We show that our framework can learn very long tasks, sometimes tens of minutes in length, for behaviors that combine both physical dexterity and combinatorial complexity.

### Overview

![Screenshot 2024-11-08 at 11.52.18 AM.png](../images/Screenshot_2024-11-08_at_11.52.18_AM.png)

> We first assemble a pre-training mixture consisting of a weighted combination of our own dexterous manipulation datasets, collected on 7 different robot configurations for 68 different tasks, and the entire OXE dataset, which contains data from 22 robots.

> The purpose of the pre-training phase is to train a base model that exhibits broad capabilities and generalization, but is not necessarily specialized for high performance on any one task.

The base model has abilities for basic tasks and can follow language commands. It’s then fine-tuned on more complex tasks.

### The $\pi_0$ Model

They train a VLA model with a pre-trained VLM that they use ViT (vision transformers) to pass the robots images as standard tokens as well.

They use a flow matching to model the continuous distribution of actions. This makes the model have higher precision and better modeling capability. This is very useful for high frequency dexterous tasks.

The use Transfusion where a single transformer is trained on a flow matching loss (to improve smoothness) and a supervised cross-entropy loss (to learn the actual correct actions).

They also use a MoE model where the first expert takes in vision and language data, and the second expert takes in robotics configuration data and outputs the actions.

They pre-train the full architecture on additional robotics data from their own dataset as well as OXE.

They want to model the distribution $p(A_t|o_t)$ with the action chunk of length 50 $A_t = [a_t, a_{t+1},…, a_{t+H-1}]$. The observation $o_t$ consists of the images from multiple RGB cameras, the language command, and the robots configuration state.

They feed predicted action tokens through the action expert that uses a conditional flow matching loss.

### Data Collection and Training Recipe

**1. Pre-training and Post-training**

> The pre-training dataset should cover as many tasks as possible, and within each of those tasks should cover a diversity of behaviors. The post-training dataset should instead cover behaviors that are conducive to effective task execution, which should exhibit a consistent and fluent
> strategy.

> To learn dexterous and more complex tasks, we also use 903M time steps of data from our own datasets, where 106M steps are from single-arm robots and 797M are from dual-arm robots. This data has 68 tasks, where each task is composed of complex behaviors.

**2. Language and High-Level Policies**

They use a SayCan like high-level planning framework

**3. Robot System Details**

> Our dexterous manipulation datasets include 7 different robot configurations and 68 tasks.

### Experimental Evaluation

**1. Evaluating the base model**

> In our first set of experiments, we evaluate the model after pre-training on our full mixture, without any post-training, to evaluate how well our base model can perform a variety of tasks.

They test on a variety of in-distribution tasks like shirt folding, bussing, grocery bagging, toast out of toaster.

![Screenshot 2024-11-08 at 12.18.12 PM.png](../images/Screenshot_2024-11-08_at_12.18.12_PM.png)

> $\pi_0$ attains by far the best results across the board on all the zero-shot tasks, with near perfect success rates on shirt folding and the easier bussing tasks, and large improvements over all baselines.

**2. Following Language Commands**

![Screenshot 2024-11-08 at 12.22.23 PM.png](../images/Screenshot_2024-11-08_at_12.22.23_PM.png)

> The language accuracy of $\pi_0$ is significantly better than that of $\pi_0$ small. This suggests a significant improvement from the larger pre-trained VLM initialization.

**3. Learning New Dexterous Tasks**

![Screenshot 2024-11-08 at 12.24.32 PM.png](../images/Screenshot_2024-11-08_at_12.24.32_PM.png)

> In the next set of experiments, we evaluate our model on new tasks that differ significantly from the pre-training data, requiring entirely new behaviors.

**4. Mastering Complex Multi-Stage Tasks**

![Screenshot 2024-11-08 at 12.27.45 PM.png](../images/Screenshot_2024-11-08_at_12.27.45_PM.png)

> In our final set of experiments, we tackle a range of challenging multi-stage tasks via a combination of fine-tuning and language. For some of these tasks, data is present in pre-training, but fine-tuning is required to attain mastery.

### Discussion

> We presented a framework for training a robot foundation model, which we refer to as $\pi_0$, that consists of pre-training on highly diverse data, followed by either zero-shot evaluation or fine-tuning to complex downstream tasks.

> To our knowledge, this represents the largest pre-training mixture ever used for a robot manipulation model.

Large pre-training acquires most of the knowledge, and then fine-tuning on a task has more robust learning.

> We hope that our results will serve as a stepping stone toward general and broadly applicable robot foundation models

It’s unclear what data composition works, some of their tasks don’t work, and it’s unclear how much transfer there is from a large number of tasks.
