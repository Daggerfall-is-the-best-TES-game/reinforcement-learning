from argparse import ArgumentParser
from collections import namedtuple, deque
from functools import partial
from random import sample
from time import time

import numpy as np
import torch
from gym import make
from numpy.random import random
from torch import tensor, save, no_grad
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Chapter6.lib import wrappers, dqn_model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = namedtuple("Experience", field_names="state action reward done new_state")


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        entries = zip(*sample(self.buffer, k=batch_size))
        array_data_types = (None, np.int64, np.float32, np.bool, None)
        return tuple(map(np.array, entries, array_data_types))


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @no_grad
    def play_step(self, net, epsilon=0.0, device="cuda"):
        done_reward = None
        if random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = tensor(self.state).unsqueeze(0).to(device)
            q_vals_v = net(state_v)
            _, act_v = q_vals_v.max(dim=1)
            action = int(act_v.item())
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch, net, tgt_net, device="cuda"):
        cuda_tensor = partial(tensor, device=device)
        states, actions, rewards, dones, next_states = map(cuda_tensor, batch)
        state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with no_grad():
            next_state_values = tgt_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * GAMMA + rewards
        return MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment")
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help=f"Mean reward boundary for stop of training, default={MEAN_REWARD_BOUND:.2f}")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.wrap_deepmind(make(args.env), episode_life=1, clip_rewards=1, frame_stack=1, scale=1)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    with SummaryWriter(comment="-" + args.env) as writer:
        print(net)
        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Agent(env, buffer)

        optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time()
        best_mean_reward = None
        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward = agent.play_step(net, epsilon, device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time() - ts)
                ts_frame = frame_idx
                ts = time()
                mean_reward = np.mean(total_rewards[-100:])
                print(f"{frame_idx} done {len(total_rewards)} games, mean reward {mean_reward:.3f},"
                      f" eps {epsilon:.2f}, speed {speed:.2f}")
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    save(net.state_dict(), args.env + "-best.dat")
                    if best_mean_reward is not None:
                        print(f"Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}, model saved")
                    best_mean_reward = mean_reward
                if mean_reward > args.reward:
                    print(f"solved in {frame_idx} frames!")
                    break
            if len(buffer) < REPLAY_START_SIZE:
                continue
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = agent.calc_loss(batch, net, tgt_net, device)
            loss_t.backward()
            optimizer.step()
