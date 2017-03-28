---
layout: post
title: A simple policy gradient implementation with keras (part 1)
---

In this post I'll show how to set up a standard keras network so that it optimizes
a reinforcement learning objective using policy gradients, 
following [Karpathy's excellent explanation](http://karpathy.github.io/2016/05/31/rl/).

First, as a way to figure this stuff out myself, I'll try my own explanation of reinforcement learning and policy gradients, 
with a bit more attention on the loss function and how it can be implemented in
frameworks with automatic differentiation. In the next post I'll apply that to a pong example.

## Reinforcement learning
(Please skip this section if you already know the RL setting)

One way to understand reinforcement learning is as a technique that tries to learn an optimal *policy*.
A policy is a function that maps states to actions.
So on every (discrete) time-step, an *agent* observes the world and then decides what to do. A reinforcement learning
algorithm tries to learn this function to maximize *rewards* (and minimize punishment, or negative reward).

To do this the agent is released into a world and tries out different actions and sees what happens - sometimes
it is rewarded. For now let's talk about some type of simulated world like a computer game - for example pong. 
(I know this example is maybe overdone but I think it helps to have an easily recognizable example - like the
iris data set for classification or the sprinkler example for causality)

In pong the player/agent observes the world by looking at the screen and takes actions by pushing the up or down buttons.

Since neural networks can represent (almost) arbitrary functions, let's use a neural network to implement
the policy function. The input to the network is a `I x J` vector of pixel values and the output is a `2 x 1` 
vector that represents the two actions - up or down.

## Policy gradient
Policy gradients (PG) is a way to learn a neural network to maximize the total expected future reward
that the agent will receive.

Reinforcement learning is of course more difficult than normal supervised learning because we don't
have training examples - we don't know what the best action is for different inputs. The agent only
indirectly knows that some of the preceding actions were good when it receives a reward.

Policy gradients interprets the policy function as a probability distribution over actions 
\\( q(a|s;\theta) \\) - so the probability of an action given the input, parameterized by \\( \theta \\).

We are then interested in adjusting the parameters so that the expected reward is maximized when we sample actions from 
this distribution.
So the loss function is:

\\[
E_{a\sim q(a|s;\theta)}[f(a, s)] = \sum_a q(a|s;\theta) f(a, s).
\\]

Here, \\( r = f(a, s) \\) is the function that returns the reward after an action is taken. This function represents 
the world - in other words we don't directly know what this function is but we can evaluate it by letting
the agent actually perform the action and then seeing what the reward was.

The thing that the agent can update is \\( \theta \\). So, to perform hill climbing on this expected value
it would be useful to have the loss function's gradient with respect to \\( \theta \\). This is done with
some algebra - putting the gradient inside the expectation; multiply and divide by the distribution; and
using the derivative of a log. The result is that the gradient of the loss can be written in terms of
the gradient of the log of the model:

\\[
\nabla E_a[f(a, s)] = E_a[f(a, s)\nabla \log q(a|s;\theta)]
\\]

## Automatic differentiation
With an automatic differentiation system (like keras) we cannot easily set the starting 
gradient that must be back-propagated. One way to get around this is to design an alternative loss function 
that has the correct gradient. 

Suppose we have a gazillion example data points (actions, observations, and rewards) - \\( (a_i, s_i, r_i) \\).
Then the gradient of the loss is estimated as

\\[
\nabla E_a[f(a, s)] \approx \sum_i r_i \nabla \log q(a_i|s_i;\theta),
\\]

and a dummy loss with that derivative is 

\\[
L = \sum_i r_i \log q(a_i|s_i;\theta).
\\]

## Cross entropy loss
Now that \\( q(a_i|s_i;\theta) \\) looks suspiciously like a likelihood. So, minimizing \\( L \\) is
the same as maximizing a weighted negative log likelihood. 

When the distribution is over discrete actions, like our example, then the *categorical crossentropy*
can be interpreted as the likelihood. To see this suppose the observed actions \\( a_i \\) and model 
action \\( a \\) are one-hot encoded vectors \\( a = [a^1 a^2 \dots a^M]^T \\), where \\( M \\) is the
number of possible actions. Then,

\begin{align}
\sum_i H(a_i, a') &= \sum_a p(a) \log q(a_i|s_i;\theta) \\\
                  &= \sum_{m=1}^M a^m \log q(a_i^m|s_i;\theta) \\\
                  &= \mathcal{L}
\end{align}

## Conclusion
So our weighted likelihood \\( L \\) can be implemented with a neural network with cross-entropy loss
and sample weights equal to the reward.

In the next post I'll see whether these speculations are true by trying an example implementation.
