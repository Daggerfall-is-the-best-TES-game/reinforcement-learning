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
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_STEPS = 10


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

if __name__ == "__main__":
    env = make("CartPole-v0")
    with SummaryWriter(comment="-cartpole-pg") as writer:
        net = PGN(env.observation_space.shape[0], env.action_space.n)
        print(net)
        agent = PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)
        exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
        optimizer = Adam(net.parameters(), lr=LEARNING_RATE)

        total_rewards = []
        step_rewards = []
        step_idx = 0
        done_episodes = 0
        reward_sum = 0.0
        bs_smoothed = entropy = l_entropy = l_policy = l_total = None

        batch_states, batch_actions, batch_scales = [], [], []

        for step_idx, exp in enumerate(exp_source):
            reward_sum += exp.reward
            baseline = reward_sum / (step_idx + 1)
            writer.add_scalar("baseline", baseline, step_idx)

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_scales.append(exp.reward - baseline)

            #handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                done_episodes += 1
                reward = new_rewards[0]
                total_rewards.append(reward)
                mean_rewards = float(np.mean(total_rewards[-100:]))
                print(
                    f"{step_idx}: reward: {reward:6.2f}, mean_100: {mean_rewards:6.2f}, episodes: {done_episodes}")

                writer.add_scalar("reward", reward, step_idx)
                writer.add_scalar("reward_100", mean_rewards, step_idx)
                writer.add_scalar("episodes", done_episodes, step_idx)
                if mean_rewards > 195:
                    print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                    break
            if len(batch_states) < BATCH_SIZE:
                continue

            optimizer.zero_grad()
            states_v = tensor(batch_states, dtype=torch.float32)
            batch_actions_t = tensor(batch_actions, dtype=torch.long)
            batch_scale_v = tensor(batch_scales, dtype=torch.float32)

            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            

