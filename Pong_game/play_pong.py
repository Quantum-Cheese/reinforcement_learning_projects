import gym
import time
from Pong_game import pong_utils
import matplotlib
import matplotlib.pyplot as plt


env = gym.make('PongDeterministic-v4')
# show what a preprocessed image looks like
env.reset()
_, _, _, _ = env.step(0)
# get a frame after 20 steps
for _ in range(20):
    frame, _, _, _ = env.step(1)

plt.subplot(1,2,1)
plt.imshow(frame)
plt.title('original image')

plt.subplot(1,2,2)
plt.title('preprocessed image')

# 80 x 80 black and white image
plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
plt.show()

print("List of available actions: ", env.unwrapped.get_action_meanings())