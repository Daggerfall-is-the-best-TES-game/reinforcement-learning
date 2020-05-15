from collections import defaultdict

from gym import make
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = make(ENV_NAME)
        self.state = self.env.reset()
        self.values = defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        actions = range(self.env.action_space.n)

        def evaluator(action): return self.values[state, action]

        best_action = max(actions, key=evaluator)
        return evaluator(best_action), best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[s, a] = old_val * (1 - ALPHA) + new_val * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = make(ENV_NAME)
    agent = Agent()
    with SummaryWriter(comment="tabular-q-learning") as writer:
        iter_no = 0
        best_reward = 0.0
        while True:
            iter_no += 1
            s, a, r, next_s = agent.sample_env()
            agent.value_update(s, a, r, next_s)
            reward = 0.0
            for _ in range(TEST_EPISODES):
                reward += agent.play_episode(test_env)
            reward /= TEST_EPISODES
            writer.add_scalar("reward", reward, iter_no)
            if reward > best_reward:
                print(f"Best reward updated {best_reward:.3f} -> {reward:.3f}")
                best_reward = reward
            if reward > 0.80:
                print(f"solved in {iter_no} iterations")
                break
