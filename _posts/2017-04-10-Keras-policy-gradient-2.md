---
layout: post
title: A simple policy gradient implementation with keras (part 2)
---

This post describes how to set up a simple policy gradient network with Keras and pong.

tl;dr---it works but easily gets stuck. 

## Pong with OpenAI gym

I used [OpenAI's gym](https://gym.openai.com) to set up the experiment---it is amazingly easy to 
install and the interface is as easy as they come:

```python
import gym

env = gym.make('Pong-v0')
observation = env.reset()

while (True):
    action = sample_action(observation)  # This function is where the model will live.
    observation, reward, done, info = env.step(action)
```

## Keras model

Recall from [the last post](/Keras-policy-gradients/) that we want to maximize the
weighted log likelihood

\\[
L = \sum_i r_i \log q(a_i|s_i;\theta).
\\]

A training step now consists of three steps:

1. Sample \\( (s, a, r) \\) tuples from the replay history.
2. Subtract the mean from the rewards.
3. Train the network for an epoch with \\( s \\) as the inputs, \\( a \\) as the labels, 
and \\( r \\) as the sample weights.

```python
model = Sequential()
model.add(Convolution2D(2, 3, 3, input_shape=(1, I, J), border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(env.action_space.n, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def train(model, replay_history):
    x_train, y_train, discounted_rewards = sample_from_history(replay_history)
    advantage = discounted_rewards - discounted_rewards.mean()  # "Centre" the points.
    model.fit(x_train, y_train, sample_weight=advantage, nb_epoch=1)
```

First a model is compiled---it takes an `1xIxJ` array of pixel values and
predicts one of the actions in the action space. Then we define a training
function that can periodically be called to adjust the model.

## Tricks
(The following is my unscientific impression after trying out a few things---I haven't run 
proper experiments yet because it takes a long time to evaluate one setup and I just wanted to
get something going first.)

### Number of convolutional layers
I experimented with more complicated models but they usually get stuck.
For example, when adding more layers, after a few iterations the paddle usually goes straight 
to the top or bottom and stays there. 

### Subtract mean discounted reward
I also found that subtracting the mean from the discounted rewards is essential. 
Without this one weird trick (that I think is quite popular) the net outputs NaNs after
a few iterations.

### Re-train frequency and number of training points
It made sense to me that you would want to run an iteration of training after every game.
This is about once every 50 to 100 time-steps. I also tried once every 20 games (when
the game state is 'done'), but training took longer.

I think that the number of training points you sample from the replay memory
defines the learning rate, given the training frequency. So for this train frequency
I found that sampling 1000s of points is way too much---the model quickly gets stuck.
About 30 works well. Training for more than one Keras epoch has the same effect. 

### Replay history
I also oversampled the positive rewards---it seemed to help.

Initially I oversampled only recent games, but the paddle usually got 
stuck near one of the two sides. I think it is because it sometimes randomly wins 
by smashing the ball back from one of the corners, and then because that win
is recently in memory it just stays close to that corner, reinforcing that
behaviour by occasionally winning again.

### Pre-processing
Karpathy pre-processed the inputs by subtracting the previous frame to give 
an indication of the direction of the ball. I rather added half of the previous frame.
This gives a bit more history and might help by activating more inputs at a time, thus sharing
information with other runs. 

### Discount factor
Initially I had the discount factor at 0.95---this results in discounted rewards of 1.0 immediately
after a reward, and 0.95 ** 50 ~ 0.08 after 50 time steps (about the length of a game if the RL player does nothing). 
This didn't work too well because positive rewards occurred too late after the RL agent's action,
so I increased the discount factor to 0.99. This results in a reward of 0.61 after 50 frames. Initially I
thought this is workable but later I tried 0.98 (with a result of 0.36 after 50 frames) which worked much better.

I think this last one gives a stronger signal to avoid losing a point by concentrating just on the last action or two,
while also rewarding wins.

It is interesting to me how sensitive the performance is to this parameter and how much it depends on
the specific game mechanics. 

### Adding noise
It was helpful to add some randomness to actions early in a run.
So for the first 20 games no learning takes place. After that I added 0.5 to all the predicted action weights and
then normalized to one again before sampling an action. After another 20 games that decreased to 0.1 and later to 0.01 
and so forth. This helps to build up a representative history of wins and losses which helps the model not get stuck.
(By just doing random moves the agent wins about 5% of the games.)

This setup is maybe similar to \\(\epsilon\\)-learning in a way, where random actions are selected some fraction
of the time. AFAIK policy gradient methods theoretically don't need this but here it seemed to help.

### Conclusion
I was surprised by how difficult it is to get this implementation going. Next I'll try other methods
like actor-critic methods that address some of the issues with vanilla policy gradients 
(See [minpy's docs](http://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html)
for example). 







