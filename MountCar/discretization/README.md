## Introduction  

A Reinforcement Learning project running on OpenAI gym.  

Using discretization to generalize the Q-Learning algorithm to continuous space. 

## Environment  

#### MountainCarContinuous v0

An underpowered car must climb a one-dimensional hill to reach a target. Unlike MountainCar v0, the action (engine force applied) is allowed to be a continuous value.

The target is on top of a hill on the right-hand side of the car. If the car reaches it or goes beyond, the episode terminates.

On the left-hand side, there is another hill. Climbing this hill can be used to gain potential energy and accelerate towards the target. On top of this second hill, the car cannot go further than a position equal to -1, as if there was a wall. Hitting this limit does not generate a penalty (it might in a more challenging version).

The other detail information about this environment in on

 <https://github.com/openai/gym/wiki/MountainCarContinuous-v0>

