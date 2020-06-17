import numpy as np
import torch
from gym import make
from ptan.agent import PolicyAgent, float32_preprocessor
from ptan.experience import ExperienceSourceFirstLast
from torch import tensor
from torch.nn import Module, Sequential, Linear, ReLU, functional as F

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


class PGN(Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = Sequential(
            Linear(input_size, 128),
            ReLU(),
            Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = make("CartPole-v0")
    with SummaryWriter(comment="-cartpole-reinforce") as writer:
        net = PGN(env.observation_space.shape[0], env.action_space.n)
        print(net)
        agent = PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)
        exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
        optimizer = Adam(net.parameters(), lr=LEARNING_RATE)

        total_rewards = []
        step_idx = 0
        done_episodes = 0

        batch_episodes = 0

        cur_rewards = []
        batch_states, batch_actions, batch_qvals = [], [], []

        for step_idx, exp in enumerate(exp_source):
            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            cur_rewards.append(exp.reward)

            if exp.last_state is None:
                batch_qvals.extend(calc_qvals(cur_rewards))
                cur_rewards.clear()
                batch_episodes += 1

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                done_episodes += 1
                reward = new_rewards[0]
                total_rewards.append(reward)
                mean_rewards = float(np.mean(total_rewards[-100:]))
                print(f"{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, epsiodes: {done_episodes}")

                writer.add_scalar("reward", reward, step_idx)
                writer.add_scalar("reward_100", mean_rewards, step_idx)
                writer.add_scalar("episodes", done_episodes, step_idx)
                if mean_rewards > 195:
                    print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                    break
            if batch_episodes < EPISODES_TO_TRAIN:
                continue

            optimizer.zero_grad()
            states_v = tensor(batch_states, dtype=torch.float32)
            batch_actions_t = tensor(batch_actions, dtype=torch.long)
            batch_qvals_v = tensor(batch_qvals, dtype=torch.float32)

            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
            loss_v = -log_prob_actions_v.mean()
            loss_v.backward()
            optimizer.step()
            batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_qvals.clear()
