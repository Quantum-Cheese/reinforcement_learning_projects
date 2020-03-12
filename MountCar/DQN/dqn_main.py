import gym
from collections import deque


def train_dqn(env,agent,n_episode,max_t):
    scores=[]
    scores_window=deque(maxlen=100)
    for i_episode in range(n_episode):
        state=env.reset()
        total_reward=0
        for t in range(max_t):
            action=agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # agent store exp or learn
            agent.step(state,action,reward,next_state,done)
            total_reward+=reward
            state=next_state
            if done:
                break
        scores.append(total_reward)
        scores_window.append(total_reward)



if __name__=="__main__":
    env=gym.make('MountainCar-v0')

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)