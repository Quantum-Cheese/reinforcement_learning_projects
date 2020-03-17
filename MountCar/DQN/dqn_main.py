import gym
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from MountCar.DQN.dqn_agent import DQN


def train_dqn(env,agent,n_episode,model_file,eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores=[]
    scores_window=deque(maxlen=100)
    epsilon=eps_start

    for i_episode in range(1,n_episode+1):
        state=env.reset()
        total_reward=0
        episode_loss=[]
        while True:
            action=agent.act(state,epsilon)
            next_state, reward, done, _ = env.step(action)
            # agent store exp or learn
            loss=agent.step(state,action,reward,next_state,done)
            if loss is not None:
                episode_loss.append(loss)

            total_reward+=reward
            state=next_state
            if done:
                break
        scores.append(total_reward)
        scores_window.append(total_reward)
        epsilon = max(eps_end, eps_decay * epsilon)  # decrease epsilon

        print('\rEpisode {}\t Loss: {} \tAverage Score: {:.2f}'.format(i_episode,np.mean(episode_loss),np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t Loss: {} \t Average Score: {:.2f}'.format(i_episode,np.mean(episode_loss),np.mean(scores_window)))
        if np.mean(scores_window)>=-110.0:
            print('\r Environment solved in {} episode. Average score: {:.2f}'.format(i_episode,np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_file)
            break

    return scores



def plot_scores(scores,save_name):
    plt.plot(np.arange(len(scores)), scores)
    # rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(save_name)
    plt.show()


if __name__=="__main__":
    env=gym.make('MountainCar-v0')

    # print("State space:", env.observation_space)
    # print("Action space:", env.action_space)
    # print(env.observation_space.shape[0])

    agent=DQN(state_size=2,action_size=3,seed=0)
    train_scores=train_dqn(env,agent,2000,"dqn_1.pth")
    plot_scores(train_scores,"dqn_1.png")
