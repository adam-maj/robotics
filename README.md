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
  - [6.1. Secrecy](#61-secrecy)
  - [6.3. Nature's Engineering](#63-natures-engineering)
  - [6.4. Carbon vs. Silicon](#64-carbon-vs-silicon)
- [Resources](#resources)

<br />

# Overview

Riding the tailwinds of recent progress in deep learning, robotics has again become a central technology focus area, with companies like [Optimus](https://www.tesla.com/we-robot), [Figure](https://www.figure.ai/), and [1x](https://www.1x.tech/) deploying hundreds of millions of dollars (see: [Figure raises $675M](https://www.prnewswire.com/news-releases/figure-raises-675m-at-2-6b-valuation-and-signs-collaboration-agreement-with-openai-302074897.html), [1x raises $100M](https://www.1x.tech/discover/1x-secures-100m-in-series-b-funding)) to develop general purpose humanoids.

Given recent hype, twitter sentiment, venture narratives, and recent demos (see: [Tesla Optimus](https://www.youtube.com/watch?v=cpraXaw7dyc), [Figure 02](https://www.youtube.com/watch?v=0SRVJaOg9Co), [Figure 02 + OpenAI](https://www.youtube.com/watch?v=Sq1QZB5baNw), [1x NEO](https://www.youtube.com/watch?v=bUrLuUxv9gE)), it may seem like fully autonomous humanoids are right around the corner. Before my deep dive, my timelines were around 2-3 years.

However, as I went farther into my deep dive, I noticed that the technical realities of the robotics industry point to a very different story than what current narratives suggest.

To understand this true story of robotics, we must first look to the past to understand the series of innovations that have gotten the industry to where it is today. Then, we can understand what this tells us about what it will take to reach the goal of fully autonomous generally intelligent humanoids.

Through this process, we'll explore the answers to the following questions:

- What are current humanoid robotics systems truly capable of?
- What are the constraints limiting progress toward general purpose humanoids?
- What is the path to successfully building fully autonomous humanoids and deploying them at scale?
- Who will win the humanoid arms race?
- How long will it take to achieve general purpose humanoids?

Let's start by understanding the fundamentals of robotics from first principles.

<br />

# 1. Fundamentals

![Fundamentals](./images/placeholder.png)

Robotics is about building systems that can alter the physical world to accomplish arbitrary goals.

We are particularly concerned with creating robots capable of automating the majority of economically valuable physical labor.

With such general purpose robotic systems available, we would have what [Eric Jang](https://x.com/ericjang11) calls a "[read/write API to physical reality](https://evjang.com/2024/03/03/all-roads-robots.html)," where we could make all desired changes to the physical world by issuing commands to robots using software alone.

> [!IMPORTANT]
>
> To convert goals into actions, these systems need to observe the state of their environment, understand what actions to take to accomplish their goals, and know how to act to physically execute their plans.
>
> These requirements cover the 3 essential functions of all robotic systems:
>
> 1. **Perception**
> 2. **Planning**
> 3. **Control**

We may be initially inclined to believe that planning is the hardest of these problems, since it often requires complex reasoning.

However, we will see that the opposite is the case - planning is the easiest of these problems and is largely solved, whereas perception and control are far harder. We will see that the biggest barrier to progress in robotics today is in developing reliable control systems.

This counter-intuitive difficulty of robotic control is reflected in Moravec's paradox:

> [!NOTE]
>
> "Moravec's paradox is the observation in the fields of artificial intelligence and robotics that, contrary to traditional assumptions, reasoning requires very little computation, but sensorimotor and perception skills require enormous computational resources." \- [Wikipedia](https://en.wikipedia.org/wiki/Moravec%27s_paradox)

This observation is reflected in the fact that modern AI systems have long been able to accomplish complex reasoning tasks like beating the world's best Go player, passing the Turing test, and now being more intelligent than the average human, while failing to perform sensorimotor tasks that a 1-year-old human could, like grasping objects and crawling.

This paradox is a direct result of the complexity of the real world. Seemingly simple tasks often require [complex multi-step motor routines](https://www.youtube.com/watch?v=b1lysnGFpqI), an intuitive understanding of real world kinematics and dynamics, calibration against variable material frictions, and more. Dexterous manipulation is really hard.

This truth is also reflected in the fact that the human brain has far more computational resources allocated toward controlling our hands and fingers than anything else, which is also why motor control feels much easier to us than high-level reasoning.

The end goal of robotics is to achieve full **autonomy** and broad **generalization**.

We don't want a robot that's specialized for just a single goal, task, object, or environment; we want a robot that can accomplish any goal, perform any task, on any object, in any environment, without the help of any human.

With this context, let's look to the series of innovations that have gotten us to the current state of robotics to understand how far we have come toward full autonomy and generalization.

<br />

# 2. History

![Plaeholder](./images/placeholder.png)
