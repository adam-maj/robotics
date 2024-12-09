# robotics [WIP]

A deep-dive on the entire history of robotics, highlighting the series of innovations that have enabled the development of general purpose humanoids like Optimus and Figure.

For each key milestone, I've included the critical papers in this repository, along with my notes and high-level explanations where relevant.

The rest of this page is my breakdown of everything we can learn from this history about the future of robotics and how the humanoid arms race will play out.

We will see that fully autonomous humanoids may be much farther away than expected.

> [!IMPORTANT]
>
> This project is designed so that everyone can get most of the value by just reading my overview on the rest of this page.
>
> Then, people curious to learn about the technical details of each innovation can explore the rest of the repository via the links in the [resources](#resources) section.
>
> **For those who don't care about any technical details and are only concerned with implications about the future of robotics, skip to the [future](#3-future) section.**

> [!NOTE]
>
> For more context, checkout the [original twitter thread](https://x.com/majmudaradam)

## Table of Contents

- [Overview](#o)
- [1. Fundamentals](#)
- [2. Progress](#)
  - [2.1 Hardware](#)
  - [2.2 Software](#)
    - [2.2.1 Perception](#)
    - [2.2.2 Planning](#)
    - [2.2.3 Control](#)
  - [2.3 Generalization](#)
- [3. Future](#)
  - [3.1. Constraints](#)
  - [3.2. Data](#)
  - [3.3. The Winning Strategy](#)
  - [3.4. Who Wins?](#)
  - [3.5. Timelines](#)
- [4. Reflections](#)
  - [4.1. Secrecy](#)
  - [4.2. Nature's Engineering](#)
  - [4.3. Carbon vs. Silicon](#)
- [Resources](#)

<br />

# Overview

Riding the tailwinds of recent progress in deep learning, robotics has again regained the spotlight, with companies like [Optimus](https://www.tesla.com/we-robot), [Figure](https://www.figure.ai/), and [1x](https://www.1x.tech/) deploying hundreds of millions of dollars (see: [Figure raises $675M](https://www.prnewswire.com/news-releases/figure-raises-675m-at-2-6b-valuation-and-signs-collaboration-agreement-with-openai-302074897.html), [1x raises $100M](https://www.1x.tech/discover/1x-secures-100m-in-series-b-funding)) to develop general-purpose humanoids.

Given recent hype, twitter sentiment, venture narratives, and recent demos (see: [Tesla Optimus](https://www.youtube.com/watch?v=cpraXaw7dyc), [Figure 02](https://www.youtube.com/watch?v=0SRVJaOg9Co), [Figure 02 + OpenAI](https://www.youtube.com/watch?v=Sq1QZB5baNw), [1x NEO](https://www.youtube.com/watch?v=bUrLuUxv9gE)), it may seem like fully autonomous humanoids are right around the corner. I originally anticipated that they might arrive within the next 2-3 years.

However, as I went farther into my deep dive, I noticed that the technical realities of current robotics progress point to a very different future than what current narratives suggest.

To see this realistic future of the robotics industry, we'll first need to understand the series of innovations that have gotten the industry to where it is today.

Then, we'll use this to explore the answers to the following questions:

- What are state-of-the-art robots currently capable of?
- What are the constraints limiting progress toward fully-autonomous generally-intelligent robotics?
- What is the path to successfully build generally-intelligent robots?
- How long will it take to create generally-intelligent robots?
- Who will win the humanoid arms race?
- What does this mean for investment and company building in the robotics industry?

Let's start by understanding the fundamentals of robotics from first principles.

<br />

# 1. Fundamentals

![Fundamentals](./images/placeholder.png)

Robotics is about building systems that can alter the physical world to accomplish arbitrary goals. Practically, we're interested in creating robots capable of automating the majority of economically valuable physical labor.

At the simplest level, robots convert ideas into actions.

> [!IMPORTANT]
>
> In order to accomplish this, robotic systems need to:
>
> 1. Observe and understand the state of their environment
> 2. Plan what actions they need to take to accomplish their goals
> 3. Know how to physically execute these actions
>
> These requirements cover the 3 essential functions of all robotic systems:
>
> 1. **Perception**
> 2. **Planning**
> 3. **Control**

We may initially expect that planning is the hardest of these problems, since it depends on complex reasoning. However, we will see that the opposite is the case - planning is the easiest of these problems and is largely solved today.

Meanwhile, the biggest barrier to progress in robotics today is in developing reliable control systems.

The end goal of robotics is to achieve full **autonomy** and broad **generalization**.

We don't want a robot that's specialized for just a single goal, task, object, or environment; we want a robot that can accomplish any goal, perform any task, on any object, in any environment, without the help of any human.

With such general purpose robotic systems available, we would have what [Eric Jang](https://x.com/ericjang11) calls a "[read/write API to physical reality](https://evjang.com/2024/03/03/all-roads-robots.html)," where we could make all desired changes to the physical world by issuing commands to robots using software alone.

This is the holy grail of robotics; such a system would be so economically valuable that the prospect of it has motivated the billions of dollars flowing into the industry today.

From here on, I'll refer to these fully autonomous, generally intelligent, and broadly capable robotic systems as general-purpose robotics.

Before we can understand how close we are to the goal of general-purpose robotics, we first need to look at the series of innovations that have gotten us to the current state of robotics.

<br />

# 2. Progress

![Placeholder](./images/placeholder.png)

The challenge of developing general-purpose robotics is both a hardware and a software problem.

Since a robot's software is entirely dependent on its hardware for sensory inputs and control outputs, we'll briefly cover robotics hardware first.

Then, we'll turn to understanding the series of software innovations over the past decade that are largely responsible for the recent interest in robotics.

<br />

## 2.1 Hardware

A robot is made of a group of **rigid bodies**, connected by **joints**, driven by **actuators**, with collocated **sensors** and **compute**.

Each of these parts corresponds with one of the 3 critical functions of a robot:

1. Cameras, LiDAR, IMUs, and other sensors allow the robot to perceive its environment.
2. Actuators let the robot move at its joints, allowing it to move itself relative to its environment, or to move objects within its environment.
3. Compute is used to process sensory information, convert it into action plans, and execute these action plans by controlling actuators.

> [!NOTE]
>
> Though there are a number of hardware considerations that will have an important impact on the scalability and functionality of general-purpose robotics, **hardware has not been the primary constraint limiting robotics progress for a few decades.**
>
> For example, [here's a video](https://www.youtube.com/watch?v=o7JH3UWO6I0) of the PR-1 robot from 2008.
>
> We can see that even 15 years ago, it was capable of doing pick and place tasks. Additionally, its hardware resembles that used in many modern robotics research papers like [SayCan](./4-generalization/3-say-can/1-saycan.pdf).

<br />

### Considerations

Designing general-purpose robotic hardware involves several trade-offs that have to be balanced:

1. **Degrees of Freedom** - The robot needs enough degrees of freedom of movement for it to perform a diversity of tasks, like climbing stairs, traversing uneven terrain, opening doors, and manipulating various objects.
2. **Configuration Complexity** - While we want robots with sufficient degrees of freedom, more complex configurations also mean more difficulty training robotic control systems. The hardware must strike a balance between flexibility and simplicity.
3. **Torque vs. Weight** - Robotic actuators need to have a high torque to weight ratio, so they can lift objects and manipulate the environment without weighing down the robot and impeding its movement.
4. **Safety** - Robots that are meant to operate in environments with humans must pay attention to their safety. This requires actuators with low rotational speeds to prevent injury to nearby humans (check out [this blog post on motor physics and safety](https://evjang.com/2024/08/31/motors.html) for more depth).
5. **Cost** - If general-purpose robots are to be deployed at scale, they need to be cheap enough to mass produce, and eventually, to be purchased by consumers. This means costly sensors like LiDAR and other expensive hardware have a high opportunity cost.

It's worth noting that modern motors are often expensive, high rotational velocity, low torque, and heavy, which is suboptimal for many of these considerations.

This is why developing cheaper, lighter, safer actuators is an important focus for companies like [Clone](https://clonerobotics.com/) (developing [artificial muscle actuators](https://x.com/clonerobotics/status/1849181515022053845)).

<br />

### Form Factor

In addition to these trade-offs, selecting a specific robotic form factor has important downstream consequences on future improvements.

We will see that robotic software is heavily dependent on data collected from exactly the same robot that is meant to be deployed on. Robotic software learns to take actions based on the exact joints, sensors, and actuators it is trained with data from.

Significantly changing the robot's hardware often means prior software becomes obsolete.

> [!IMPORTANT]
> Companies that are able to maintain the same hardware over time will benefit from the **compounding advantages** of deploying robots in the world, collecting diverse real-world datasets, creating improved models for their robots, and then using these improved models to motivate more deployments and revenue to further fuel this process.
>
> For this reason, **it's important that robotics companies design hardware systems that are sufficiently general**, so they can keep reaping the rewards of this data flywheel without having to alter their hardware.

This is why so many companies have now opted to develop humanoids.

Their argument is that the world is designed for humans, so humanoids will be generally capable of performing most tasks in our world.

In other words, they believe that the humanoid form factor is sufficiently general such that they will be able to focus on collecting data and improving their software over time without having to alter their hardware too much.

> [!NOTE]
>
> There has also been significant progress developing quadruped robots over the past decade from companies like [Unitree](https://www.unitree.com/) and [Boston Dynamics](https://bostondynamics.com/products/spot/), though this form factor is far less generally capable, so I won't focus on it in this deep dive.

<br />

### Humanoids

Since most large robotics companies have now bet on the humanoid form factor, let's briefly look at the hardware capabilities of modern humanoid systems.

We can get a cursory overview by looking at the demo for [Optimus Gen 2](https://www.youtube.com/watch?v=cpraXaw7dyc&t=3s) or [Figure 02](https://www.youtube.com/watch?v=0SRVJaOg9Co).

They don't give much information, but we can pick up on a few important considerations:

- They have high degree-of-freedom hands with dexterous manipulation capabilities. These will enable much more complex motor control in the future, though they are much more difficult to train than simple graspers.
- They appear to use only cameras for vision, opting against LiDAR systems that are often used on quadrupeds (due to cost optimization for mass production).
- They have AI compute on board, which can be used for running inference on vision-language-models. We'll see that modern robotics software has adapted these models to be an essential part of their control systems.
- Figure mentions their battery life. These robots may initially require frequent charging or need to be plugged in to operate, which is a hardware limitation that will improve over time.

Most importantly, though these robots are expensive and have limited compute and battery life, their basic functionality should be sufficient to accomplish most physical labor.

With this context in mind, we can turn to understanding robotic software systems, which are currently the real bottleneck in the way of developing general-purpose robotics.

<br />

## 2.2 Software

![Placeholder](./images/placeholder.png)

Software is where most of the progress in robotics has occurred over the past decade, and is the place we must look to understand where the future of robotics is headed.

In this section, we'll focus on the series of innovations that have led us to the current frontier of robotic software. Then, we'll use this to understand the limitations of current capabilities and what we must accomplish to achieve general-purpose robotics.

Robotic software defines the "brain" of the robot; its responsible for using sensor data and actuators to process the robots' **perception**, **plan** actions, and issue **control** commands.

We may initially expect that planning is the most difficult of these functions - it often requires high-level reasoning abilities, understanding of environmental context, natural language, and more, whereas controlling limbs to grab and manipulate object seems comparatively simple.

In reality, the opposite is the case. Planning is the easiest of these functions and is now largely solved with models like [SayCan](./4-generalization/3-say-can/1-saycan.pdf) and [RT2](./4-generalization/6-vla/1-vla.pdf) (which we will cover soon), whereas creating effective motor control policies is the main constraint limiting progress today.

> [!NOTE]
>
> This counter-intuitive difficulty of robotic control is captured in **Moravec's paradox**:
>
> "Moravec's paradox is the observation in the fields of artificial intelligence and robotics that, contrary to traditional assumptions, reasoning requires very little computation, but sensorimotor and perception skills require enormous computational resources." \- [Wikipedia](https://en.wikipedia.org/wiki/Moravec%27s_paradox)
>
> We can see the truth in this in the fact that modern AI systems have long been able to accomplish complex reasoning tasks like beating the best human chess player, Go player, passing the Turing test, and now being more intelligent than the average human, all while robots consistently fail to perform basic sensorimotor tasks that a 1-year-old human could, like grasping objects and crawling.

Moravec's paradox is not really paradox, it is instead a direct result of the complexity of the real world.

Tasks that seem simple to us often actually require complex multi-step motor routines, an intuitive understanding of real world kinematics and dynamics, calibration against variable material frictions, resistance against external disruptive forces, and more. Meanwhile, symbol manipulation is a relatively lower-dimensional and less complex problem.

To get a sense for this complexity that we often fail to appreciate, check out [this video where Eric Jang annotates all the motor routines required to open a package of dates](https://www.youtube.com/watch?v=b1lysnGFpqI).

The truth of Moravec's paradox is also reflected in the human brain, which has far more computational resources allocated toward controlling our hands and fingers than the rest of our body. This may also be why motor control feels so easy to us compared to high-level reasoning.

With this context, let's first look at the innovations that have changed perception and planning systems before we dive into the far more complex challenge of robotic control.

<br />

## 2.2.1 Perception

Robotic perception is concerned with processing sensory data about the robot's environment to understand:

1. The structure of the environment
2. The presence and location of objects in the environment
3. Its own position and orientation within the environment

All of these necessities require the robot to construct an internal representation of its environment that it can constantly update as it moves and reference in its decision-making.

This is exactly the goal of SLAM systems.

> [!NOTE]
>
> Sensory perception is also a significant part of robotic control since control heavily depends on sensorimotor policies, but we will cover that separately in the control section.

<br />

### Breakthrough #1: Early SLAM

**Simultaneous Localization and Mapping (SLAM)** systems use robotic sensor data to construct a consistent internal representation of the environment (mapping) and understand the robot's position in it (localization).

Early SLAM systems often used LiDAR sensors, sometimes combined with cameras, IMUs, and other sensors, and then used sensor fusion to synthesize all this data into a single map.

If sensors were perfectly accurate, we would have no need for SLAM - the robot would easily be able to understand its exact trajectory as it moved through the environment and with a LiDAR sensor, it could perfectly construct a map of its environment with point-wise depth data.

The challenge with SLAM comes in the fact that sensors have some error. As the robot navigates the environment, this error slowly accumulates, causing the robot to miscalculate where it has moved (due to slightly inaccurate IMU readings) which then distorts it's understanding of the environment since this shifts the relative position of points in its environment.

SLAM solutions all solve this problem in the same way:

1. As the robot navigates through the environment, store the relative positions of points of interest around it
2. Detect when the robot sees the same point of interest from multiple different perspectives
3. Triangulate the locations of all the different points of interest to reduce errors in localization and mapping

<p align="center">
  <img src="/images/readme/slam-correlations.png" alt="slam correlations" width="50%" />
</p>
<p align="center">
  <i>The robot detects points of interest with correlations. As the correlations between points of interest grow, estimated locations become more accurate.</i>
</p>

Early SLAM solutions like [EKF-SLAM and FastSLAM](./1-perception/1-slam/1-slam.pdf) used purely algorithmic methods like particle filters to construct a map of the environment.

However, these solutions often depended on expensive LiDAR sensors. Due to hardware cost concerns, this dependence was infeasible for affordable mass-produced robotics, so the industry turned to SLAM solutions that only required visual data from cameras.

<br />

### Breakthrough #2: Monocular SLAM

[ORB-SLAM](./1-perception/4-orb-slam/1-orb-slam.pdf) represented a major breakthrough by providing a SLAM solution that only depended on a single camera, with no dependence on LiDAR.

Because monocular SLAM systems don't have access to point-wise depth data from LiDAR that makes SLAM much easier, they have to estimate the relative positions of the camera and points of interest in the environment purely from visual data.

Earlier monocular SLAM solutions like [ORB-SLAM](./1-perception/4-orb-slam/1-orb-slam.pdf) accomplish this by detecting some form of image features (like corners, in this case [ORB](./1-perception/3-orb/orb.pdf) features), and then triangulating these image features across key-frames using strategies like **bundle adjustment** and **pose graph optimization**.

These solutions also started to integrate **loop closures** where a robot could perform a massive map readjustment and error correction every time it returned to the same location (since errors in relative positions between points of interest become obvious).

<br />

### Breakthrough #3: SLAM with Deep Learning

It's worth noting that modern SLAM solutions like [DROID-SLAM](./1-perception/5-droid-slam/1-droid-slam.pdf) and [NeRF-SLAM](https://arxiv.org/pdf/2210.13641) (among many others) have started to integrate deep learning into their systems to varying degrees.

However, these deep learning systems don't look like modern internet scale models where they have few priors and rely on massive amounts of data to refine their weights. Instead, they are still usually primarily algorithm solutions with heavy priors built into their architecture, with deep learning integrated into just a few places.

Notably, [ORB-SLAM3](https://arxiv.org/abs/2007.11898) is a purely algorithmic SLAM solution built after ORB-SLAM that still has nearly state-of-the-art performance, indicating that deep learning has yet to offer a significant advantage in robotic perception.

This suggests that the robotic perception problem is structured with a complexity such that a purely deep learning based solution is unrecoverable given the current scale of data we have, and that significant inductive bias is required.

> [!IMPORTANT]
>
> The bottom-line on robotic perception is that functional monocular SLAM solutions currently exist with loop-closing and the ability to recover from errors. These solutions are still far from the quality of state-of-the-art LiDAR based solutions and have a lot of room for improvement, but are not currently the blocker for deploying humanoid robotics in the world.

<p align="center">
  <img src="/images/readme/figure-slam.jpeg" alt="Figure SLAM" width="50%" />
</p>
<p align="center">
  <i>A SLAM solution created by a Figure humanoid robot, from <a href="https://x.com/adcock_brett/status/1864420719138099391" target="_blank">Brett Adcock's twitter</a>.</i>
</p>

<br />

## 2.2.2 Planning

Robotic planning is about using an understanding of the environment to convert the robot's goals into concrete action plans.

Specifically, this consists of **path planning**, **task planning**, and **motion planning**. We will focus on path planning and task planning here, as low-level motion planning is really the job of robotic control.

<br />

### Path Planning

The challenge of robotic path planning is primarily concerned with safety; the robot needs to navigate its environment to a target position without colliding with humans and objects in the environment.

Traditional path-finding algorithms like [A\*](/2-planning/1-path-planning/1-a-star/) work to find optimal paths in discrete and relatively simple environments, but robots operate in complex environments with continuous configuration spaces (the number of specific trajectories a robot could take from one location to another is near infinite).

To address this challenge, robots have to use random sampling based path planning algorithms like [Probabilistic Roadmaps (PRM)](./2-planning/1-path-planning/2-prm/prm.pdf) and [Rapidly-exploring Random Trees](./2-planning/1-path-planning/3-rrt/1-rrt.pdf) to create best-effort trajectory plans that avoid collisions. Then, they can use optimization algorithms like [CHOMP](./2-planning/1-path-planning/4-chomp/1-chomp.pdf) to ensure that selected trajectories optimize smoothness in addition to just avoiding collisions.

> [!IMPORTANT]
>
> **Capabilities & Limitations: Path Planning**
>
> - Modern path planning systems can effectively generate best-effort trajectories in complex continuous environments
> - These algorithms are capable of optimizing to avoid collisions and maximize smoothness
> - Modern algorithms still struggle with path planning in the presence of dynamic objects in the environment (like walking humans)

<br />

### Task Planning

Robotic task planning involves converting the high-level goal of the robot into sub-tasks and eventually individual motor routines to accomplish the goal.

This requires an understanding of the robot's environment and the objects within it, the capabilities of the robot, and high-level reasoning abilities to plan within these constraints.

Until a few years ago, task planning systems all used hierarchical symbolic approaches to task planning like hierarchical task networks (HTN), [STRIPS](./2-planning/2-task-planning/1-strips/strips.pdf) and [Planning Domain Definition Language (PDDL)](./2-planning/2-task-planning/3-pddl/pddl.pdf) which allow roboticists to manually define the domain of valid concepts to reason about.

This worked for simple environments where robots had a limited set of problems to consider (like in industrial cases where robots have a very limited task space) but is infeasible for any general-purpose robotics system where the complexity of environments quickly explodes.

This problem remained unsolved until the recent success of multi-modal LLMs provided access to models with advanced visual and semantic reasoning capabilities.

Recent robotic systems like [SayCan](./4-generalization/3-say-can/1-saycan.pdf) and [RT2](./4-generalization/6-vla/1-vla.pdf) use these pre-trained VLMs for their reasoning abilities and fine-tune them to understand the capabilities afforded by robotic control systems to create effective task planning systems that can direct the robot to accomplish long-horizon tasks and solve reasoning problems that were previously intractable.

> [!IMPORTANT]
>
> **Capabilities & Limitations: Task Planning**
>
> - Modern task planning systems have advanced reasoning abilities and are grounded in the realities of actions that the robot can actually perform
> - These systems have effectively integrated high-level task planning with low-level robotic control to successfully accomplish goal-oriented behavior in complex environments
> - Robotic task planning can now be considered a relatively solved problem

<br />

## 2.2.3 Control

### Breakthrough #1: Classical Control

### Breakthrough #2: Simulation

### Breakthrough #3: Deep Reinforcement Learning

### Breakthrough #4: End-to-end Learning

### Breakthrough #5: Tele-operation

### Breakthrough #6: Robotic Transformers

### Breakthrough #7: Pre-training + Fine-tuning

### Breakthrough #8: Cross-embodiment

<br />

## 2.3 Generalization

<br />

# 3. Future

### 3.1 Constraints

### 3.2 Data

### 3.3 The Winning Strategy

### 3.4 Who Wins?

### 3.5 Timelines

<br />

# 4. Reflections

### 4.1 Secrecy

### 4.2 Nature's Engineering

### 4.3 Carbon vs. Silicon

<br />

# Resources
