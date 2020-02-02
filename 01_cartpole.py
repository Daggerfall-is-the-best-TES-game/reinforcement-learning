from numpy.random import default_rng
from torch.nn import Module, Sequential, Linear, ReLU, CrossEntropyLoss, Softmax
from torch import tensor, float32, long
from collections import namedtuple
import numpy as np


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
rng = default_rng()

class Net(Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = Sequential(
            Linear(obs_size, hidden_size),
            ReLU(),
            Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple("Episode", field_names="reward steps")
EpisodeStep = namedtuple("EpisodeStep", field_names="observation action")


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = Softmax(dim=1)
    while True:
        obs_v = tensor([obs], dtype=float32)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = rng.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(obs, action))
        if is_done:
            batch.append(Episode(episode_reward, episode_steps))
            episode_reward = 0.0
            episode_steps.clear()
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch.clear()
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    train_obs, train_act = zip(*(example for example in batch if example.reward < reward_bound))
    train_obs_v = tensor(train_obs, dtype=long)
    train_act_v = tensor(train_act, dtype=long)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
