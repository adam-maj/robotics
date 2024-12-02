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
