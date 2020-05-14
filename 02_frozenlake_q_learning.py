from collections import defaultdict, Counter

from gym import make
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = defaultdict(float)
        self.transits = defaultdict(Counter)
        self.values = defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        actions = range(self.env.action_space.n)

        def evaluator(action): return self.values[state, action]

        return max(actions, key=evaluator)

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = make(ENV_NAME)
    agent = Agent()
    with SummaryWriter(comment="-q-learning") as writer:
        iter_no = 0
        best_reward = 0.0
        while True:
            iter_no += 1
            agent.play_n_random_steps(100)
            agent.value_iteration()
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
