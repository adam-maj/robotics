# robotics [WIP]

A deep-dive on the entire history of robotics, highlighting the series of innovations that have enabled the development of general purpose humanoids like Optimus and Figure.

For each key milestone, I've included the critical papers in this repository, along with my notes and high-level explanations where relevant.

The rest of this page is my breakdown of everything we can learn from this history about the future of robotics and how the humanoid arms race will play out.

We will see that fully autonomous humanoids may be much farther away than expected.

> [!IMPORTANT]
>
> **This project is designed so everyone can get most of the value by just reading my overview on the rest of this page.**
>
> Then, people curious to learn about the technical details of each innovation can explore the rest of the repository via the links in the [resources](#resources) section.

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

# 2. Hardware

The challenge of developing general-purpose robotics is both a hardware and a software problem.

Since a robot's software is entirely dependent on its hardware for sensory inputs and control outputs, we'll briefly cover robotics hardware first.

A robot is made of a group of **rigid bodies**, connected by **joints**, driven by **actuators**, with collocated **sensors** and **compute**.

Each of these parts corresponds with one of the 3 critical functions of a robot:

1. Cameras, LiDAR, IMUs, and other sensors allow the robot to perceive its environment.
2. Actuators let the robot move at its joints, allowing it to move itself relative to its environment, or to move objects within its environment.
3. Compute is used to process sensory information, convert it into action plans, and execute these action plans by controlling actuators.

> [!NOTE]
>
> Though there are a number of important considerations for improving these hardware systems which we will discuss, it's important to note that robotic hardware hasn't been the primary constraint on progress for a few decades.
>
> For example, [here's a video](https://www.youtube.com/watch?v=o7JH3UWO6I0) of the PR-1 robot from 2008. We can see that even 15 years ago, it was capable of doing pick and place tasks, and its hardware resembles that used in many modern robotics research papers like [SayCan](./4-generalization/3-say-can/1-saycan.pdf).

Designing general-purpose robotic hardware involves several trade-offs that have to be balanced:

1. **Degrees of Freedom** - The robot needs enough degrees of freedom of movement for it to perform a diversity of tasks, like climbing stairs, traversing uneven terrain, opening doors, and manipulating various objects.
2. **Configuration Complexity** - While we want robots with sufficient degrees of freedom, more complex configurations also mean more difficulty training robotic control systems. The hardware must strike a balance between flexibility and simplicity.
3. **Torque vs. Weight** - Robotic actuators need to have a high torque to weight ratio, so they can lift objects and manipulate the environment without weighing down the robot and impeding its movement.
4. **Safety** - Robots that are meant to operate in environments with humans must pay attention to their safety. This requires actuators with low rotational speeds to prevent injury to nearby humans (check out [this blog post on motor physics and safety](https://evjang.com/2024/08/31/motors.html) for more depth).
5. **Cost** - If general-purpose robots are to be deployed at scale, they need to be cheap enough to mass produce, and eventually, to be purchased by consumers. This means costly sensors like LiDAR and other expensive hardware have a high opportunity cost.

Additionally, once a particular hardware configuration is selected and robots of that form-factor are deployed, they will be able to collect data specifically useful for improving the generalization of that robotic hardware.

For this reason, it's important that robotics companies design hardware systems that are sufficiently general that they will be able to reap compounding benefits from the increasing availability of data post-deployment.

This is why so many companies have now opted to develop humanoid robotics. Their argument is that the world is designed for humans, so the humanoid is the robotic form factor that will be most generally capable of performing tasks in our world.

We will soon explore what it would take for this bet on humanoids to be justified.

It's also worth noting that there has been significant progress on quadruped robots from companies like [Unitree](https://www.unitree.com/) and [Boston Dynamics](https://bostondynamics.com/products/spot/), though this form factor is far less generally capable so we will not focus on it here.

The bottom-line on robotic hardware is that there are a number of considerations for improving cost, safety, and flexibility that will become more important over time, but it is not currently the bottleneck in developing general-purpose humanoids.

For that, we'll have to turn to understanding progress in robotics software.

# 3. Software

![Placeholder](./images/placeholder.png)

## 3.1 Perception

### Breakthrough #1: SLAM

### Breakthrough #2: ORB-SLAM

### Breakthrough #3: SLAM with Deep Learning

## 3.2 Planning

### Breakthrough #1: Hierarchical Task Planning

### Breakthrough #2: Reasoning with LLMs

## 3.3 Control

### Breakthrough #1: Classical Control

### Breakthrough #2: Simulation

### Breakthrough #3: Deep Reinforcement Learning

### Breakthrough #4: End-to-end Learning

### Breakthrough #5: Tele-operation

### Breakthrough #6: Robotic Transformers

### Breakthrough #7: Pre-training + Fine-tuning
