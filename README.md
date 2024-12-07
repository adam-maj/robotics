# robotics [WIP]

A deep-dive on the entire history of robotics and what it tells us about the future of general purpose humanoids.

For each key milestone, I've included the critical papers in this repository, along with my notes and high-level explanations where relevant.

The rest of this page is my breakdown of everything we can learn from this history. We will see that:

1. Progress in the robotics is now primarily constrained by data
2. All humanoid robotics startups must design their strategy around this constraint
3. The inherent difficulty of this challenge may imply drastically longer timelines than the public expects today

Finally, we will explore how our development of advanced robotics has given us more grounds for appreciating the human brain, and we will see the relative abilities of the carbon vs. silicon intelligence stack.

> [!IMPORTANT]
>
> **This project is designed so everyone can get most of the value by just reading my overview on the rest of this page.**
>
> Then, people curious to learn about the technical details of each innovation can explore the rest of the repository via the links in the [resources](#resources) section.

> [!NOTE]
>
> For more context, checkout the [original twitter thread](https://x.com/majmudaradam)

### Table of Contents

# Overview

- robotics is about building systems that can alter the physical world given arbitrary goals
- they convert goals into physical action
- this requires [1] observing the state of the environment (perception) [2] understanding what to do to accomplish a goal (planning) [3] and the ability to alter the environment to accomplish plans (control)
- we have been trying to build systems to accomplish this for decades. It seems much easier than the harder problem of symbolic intelligence (thinking).
- This is very wrong - embodied intelligence in the physical world is way harder (Moravec's paradox). humans like to think our thinking is harder. this is because we have way more compute dedicated toward physical manipulation (homunculus)
- The complexity of the problem is determined by the complexity of the state space/model. Clearly, the model of the world is more complex. In order to accomplish control, we need to understand the dynamics of the world. This is really hard.
- We have been trying to solve this problem for 30 years, and have made slow progress until recently. Like AI, we started with feature engineering, then moved more toward deep learning with RL, and now finally large multi-modal models.
- Unliked LLMs, we haven't gotten lucky with access to a huge network effect generate internet scale dataset to train on. Data is the barrier to robotics, we will see how this problem is being addressed, and how timelines may be farther off then they seem.
- The ultimate goal of robotics is generalization to new tasks, skills, objects, and environments, like humans. As systems improve in these dimensions, they get closer in architecture to the human brain. This challenge helps us to appreciate the incredible engineering of the human brain and body.
- We are currently far from generalization. We have gotten to full task/object level generalization, and some environment level generalization, and are extremely far from skill level generalization.

**Humanoids/AGI**:

- Does this actually make sense right now?
- The goal is to create a single company that will use the compounding capital and talent from successful deployment to build more and more generalized robotics.
- In order to do this, the hardware form factor is critical
- Robotics training data requires camera and joint data for a specific hardware. If the hardware improves and then needs to change to do another task, you need a new robot, which destroys all the advantages of compounding. To mitigate this, you have to choose a single hardware which can remain mostly unchanged for a long time (so data doesn't become obsolete - not exactly accurate with Pi0).
- The humanoid form factor works everywhere; the world is built for humans, so humanoids should be able to operate well in it.
- We are far from generalization, so teleoperation will be needed for a long time. It might be too far away to make this work.

**What is currently the bottleneck**

- Robotics is now just deep learning so we can look at the 7 constraints of deep learning to determine bottlenecks. We have compute / etc. necessary for everything. The bottlenecks are just architecture, and mostly data.
- How much data do we need?
- It took the entire internet for good LLMs. The internet was generated by network effects. Moravec's paradox tells us that robotics / real world is even more challenging of a problem than LLMs. How do we get this scale of data?
- What is the exact data scale necessary?
- What had to align to create the internet scale of data in the first place? Can VLA allow us to channel diffusion data about real world physics into robots.

**How can we get this scale of data and train a model**

- There are only 2 ways to get the necessary data scale. We can either train with imitation learning from real world data/demonstrations operated by a human, or we can do simulation transfer.
- Simulation is the ideal RL environment because we can run infinite loops in parallel. This is the ideal deep learning scale of data. The challenge is it's not actually reality. Domain randomization is very interesting - generalize beyond what we perceive, generalizing to any physics. This is ok for some control policies, but we need the robots to also learn the rich complexity of the real world for everything else. This would require us recreating that complexity in simulation, which is intractable. So at best, simulation can help with low level control, but is insufficient for complete generalization.
- This is why everyone has chosen to go with imitation learning. We are going to need huge datasets of robots operating in the real world to have any hope of doing this. There is no other path.
- What is the scale of deployments we need? 100,000 humanoids in the real world collecting data for how long? How large of a dataset would this give us?
- What are the adoption constraints. Cost, who wants it, who's willing to use tele-operated robots. It's all about number of paid deployments + correct hardware.
- The issue is that generalization to new skills is unlikely without massive data scale. But ideally simpler tasks can be automated over time slowly.

**What is the current state of the art**

- Autonomous is pretty bad at anything general / out of distribution
- pi0 is by far state of the art for non-humanoids. It shows that we can create sufficient generalization on objects/environments and can fine-tune training directly on specific tasks/skills, but that task/skill level generalization is very far away.
- dexterous manipulation edge cases, locomotion, etc. are all still good but not good enough for many real world scenarios. They have to operate in controlled environments.
- the constraint of safety is a big one with humans.

**Replicating the brain**

- As we move closer to generalization purely through empiricism, we end up getting closer to replicating the brain
- We have moved toward end-to-end models, neural approaches (DL), reinforcement learning (brain with environment), even curiosity, generalization + fine-tuning (brain doing fine-tuning in real time), an ensemble of architectures for specific tasks all integrated with one cost function (like different brain regions specializing)
- This gives us a basis for the complexity of the brains engineering, and we also see that it's far less arbitrary than it seems. Given the landscape of the problem (the real world) and the goal similar to humans (generalized manipulation and goal-oriented action), the human trial-and-error process is converging more and more close toward the solution created by natural selection.

**Carbon vs. silicon intelligence**

- We are approaching embodied intelligent systems built with silicon semiconductors instead of carbon-based organic molecules.
- These systems are far superior in compute capacity, energy usage (direct and simple), scale, life span.
- We are currently superior in energy efficiency, out of distribution generalization. It's unlikely that we will stay ahead for long. Everything is moving toward silicon intelligence surpassing the capabilities of carbon based intelligence. Neuralink is the only hope at mitigating this.

# Resources
