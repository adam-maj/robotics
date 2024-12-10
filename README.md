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
- [2. Progress](#2-progress)
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
> In order to accomplish this, robotics systems need to:
>
> 1. Observe and understand the state of their environment
> 2. Plan what actions they need to take to accomplish their goals
> 3. Know how to physically execute these actions
>
> These requirements cover the 3 essential functions of all robotic systems: [1] **perception**, [2] **planning**, and [3] **control**.

We may initially expect that planning is the hardest of these problems, since it depends on complex reasoning. However, we will see that the opposite is the case - planning is the easiest of these problems and is largely solved today.

Meanwhile, the biggest barrier to progress in robotics today is in developing reliable control systems.

The end goal of robotics is to achieve full **autonomy** and broad **generalization**.

We don't want a robot that's specialized for just a single goal, task, object, or environment; we want a robot that can accomplish any goal, perform any task, on any object, in any environment, without the help of any human.

With such general purpose robotic systems available, we would have what [Eric Jang](https://x.com/ericjang11) calls a "[read/write API to physical reality](https://evjang.com/2024/03/03/all-roads-robots.html)," where we could make all desired changes to the physical world by issuing commands to robots using software alone.

This is the holy grail of robotics; such a system would be so economically valuable that the prospect of it has motivated the billions of dollars flowing into the industry today.

Before we can understand how close we are to this goal, we first need to look at the series of innovations that have gotten us to the current state of robotics.

<br />

# 2. History

![Placeholder](./images/placeholder.png)
