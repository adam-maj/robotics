# robotics [WIP]

A deep-dive on the entire history of robotics and what it tells us about the future of general purpose humanoids.

For each key milestone, I've included the critical papers in this repository, along with my notes and high-level explanations where relevant.

<br />

The rest of this page is my breakdown of everything we can learn from this history. We will see that:

1. Progress in robotics is now primarily bottlenecked by data
2. All humanoid robotics startups must design their strategy around this constraint
3. The inherent difficulty of this challenge may imply drastically longer timelines than the public expects today

<br />

We will cover each of the above concerns in depth and focus on the constraints to understand **exactly what we must accomplish** to create general purpose humanoids, and **how long it may take** to accomplish this.

Finally, we will explore how our development of humanoid robotics gives us grounds to appreciate the engineering of the human body and brain, and how these carbon vs. silicon intelligence stacks compare to each other.

<br />

> [!IMPORTANT]
>
> **This project is designed so everyone can get most of the value by just reading my overview on the rest of this page.**
>
> Then, people curious to learn about the technical details of each innovation can explore the rest of the repository via the links in the [resources](#resources) section.

> [!NOTE]
>
> For more context, checkout the [original twitter thread](https://x.com/majmudaradam)

<br />

### Table of Contents

- [Overview](#overview)
  - [1. Fundamentals](#1-fundamentals)
  - [2. History](#2-history)
    - [2.1. Classical Control](#21-classical-control)
    - [2.2. Deep Reinforcement Learning](#22-deep-reinforcement-learning)
    - [2.3. Robotic Transformers](#23-robotic-transformers)
  - [3. Progress](#3-progress)
    - [3.1. Generalization](#31-generalization)
    - [3.2. Constraints](#32-constraints)
  - [4. Data](#2-data)
    - [4.1. Internet Data](#41-internet-data)
    - [4.2. Simulation](#42-simulation)
    - [4.3. Imitation](#43-imitation)
  - [5. Future](#5-future)
    - [5.1. The Winning Strategy](#51-the-winning-strategy)
    - [5.2. Who Wins?](#52-who-wins)
    - [5.3. Timelines](#53-timelines)
  - [6. Reflections](#6-reflections)
    - [6.1. Nature's Engineering](#61-natures-engineering)
    - [6.2. Carbon vs. Silicon](#62-carbon-vs-silicon)
- [Resources](#resources)
  - [Topics](#topics)
  - [Papers](#papers)

# 1. Fundamentals
