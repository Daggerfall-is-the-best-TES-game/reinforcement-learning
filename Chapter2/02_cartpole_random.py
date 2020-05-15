import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1

    print(f"Episode done in {total_steps:d}, total reward {total_reward:.2f}")
