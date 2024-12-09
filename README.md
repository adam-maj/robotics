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
  - [6.2. Nature's Engineering](#62-natures-engineering)
  - [6.3. Carbon vs. Silicon](#63-carbon-vs-silicon)
- [Resources](#resources)

<br />

# Overview

Riding the tailwinds of recent progress in deep learning, robotics has again become a central technology focus area, with companies like [Optimus](https://www.tesla.com/we-robot), [Figure](https://www.figure.ai/), and [1x](https://www.1x.tech/) deploying hundreds of millions of dollars [see: [Figure raises $675M](https://www.prnewswire.com/news-releases/figure-raises-675m-at-2-6b-valuation-and-signs-collaboration-agreement-with-openai-302074897.html), [1x raises $100M](https://www.1x.tech/discover/1x-secures-100m-in-series-b-funding)] to develop general purpose humanoids.

Given recent hype, twitter sentiment, venture narratives, and recent demos (see: [Tesla Optimus](https://www.youtube.com/watch?v=cpraXaw7dyc), [Figure 02](https://www.youtube.com/watch?v=0SRVJaOg9Co), [Figure 02 + OpenAI](https://www.youtube.com/watch?v=Sq1QZB5baNw), [1x NEO](https://www.youtube.com/watch?v=bUrLuUxv9gE)), it may seem like fully autonomous humanoids are right around the corner. Before my deep dive, my timelines were around 2-3 years.

However, as I went farther into my deep dive, I noticed that the technical realities of the robotics industry point to a very different story than what current narratives suggest.

To understand this true story of robotics, we must first look to the past to understand the series of innovations that have gotten the industry to where it is today. Then, we can understand what this tells us about what it will take to reach the goal of fully autonomous generally intelligent humanoids.

Through this process, we'll explore the answers to the following questions:

- How much progress have we made toward humanoids today? What are current robotics systems truly capable of?
- What are the constraints limiting progress toward general purpose humanoids?
- What is the path to successfully building fully autonomous humanoids and deploying them at scale?
- Who will win the humanoid arms race?
- How long will it take to achieve general purpose humanoids?

Before we look into the series of innovations that have led us to modern robotics, let's first understand the fundamentals of what we are trying to accomplish with robotics from first principles.

# 1. Fundamentals
