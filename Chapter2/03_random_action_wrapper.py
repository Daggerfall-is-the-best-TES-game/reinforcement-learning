from random import random

from gym import ActionWrapper, make


class RandomActionWrapper(ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(make("CartPole-v0"), epsilon=0.5)

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break
    print(f"Reward got: {total_reward:.2f}")
