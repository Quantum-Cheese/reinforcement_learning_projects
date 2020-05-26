import gym


if __name__=="__main__":
    env = gym.make('Pendulum-v0')

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    # 观察一个未经训练的随机智能体
    state = env.reset()
    print(state)

    done = False
    for _ in range(1000):
        env.render()
        if not done:
            next_state, reward, done, _ = env.step(env.action_space.sample())
            print(next_state, reward)
        else:
            break

    env.close()
