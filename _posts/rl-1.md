---
title: 'Reinforcement Learning - An Intro'
date: 2021-05-19
permalink: /posts/2021/05/rl-1/
tags:
  - Reinforcement Learning
---

The blog will discuss what reinforcement learning (RL) is and what its key elements are.

# Reinforcement Learning

The problem RL addresses is how an **agent** can maximize its **reward** in a complex and uncertain **environment**. The interaction between agent and environment involves

- agent obtaining the current **state** from the environment;
- agent outputting an **action** (aka a decision) based on the states;
- environment returning the next state and reward which directly results from the previous action taken.

Examples include game play, robot control, etc. Many hold high expectation towards the field since it has the potential to exceed a mortal's ability in terms of decision making. 

## Comparison with Supervised Learning

Supervised learning methods are based on the hypotheses:

- Input data are independent, i.e., they are i.i.d. (independent & identically distributed). 
- A label is provided with each datum.

Both are necessary to train a classifier (e.g., a neural network) via backpropagation of error between the true value (i.e., the label) and the predicted one.

Apparently, none of the hypotheses can apply to the problem described above, for its inputs are time-series data, resulting in a delayed reward; no "correct value" can be fed into the learner, i.e., agent has to explore via trial-and-error to discover the best action to take under a given state. Hence, one needs to strike a balance between *exploration* and *exploitation*, where the former takes some "risky" moves and the later maintains the old decision-making policy to obtain the same reward. 