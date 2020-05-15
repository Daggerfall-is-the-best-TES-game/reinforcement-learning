from collections import namedtuple

import numpy as np
from gym import ObservationWrapper
from gym import make
from gym.spaces import Discrete, Box
from torch import tensor, float32, long
from torch.distributions.categorical import Categorical
from torch.nn import Module, Sequential, Linear, ReLU, CrossEntropyLoss, Softmax
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class DiscreteOneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Discrete)
        self.observation_space = Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


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
        action = Categorical(act_probs_v).sample().item()
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(obs, action))
        if is_done:
            batch.append(Episode(episode_reward, episode_steps.copy()))
            episode_reward = 0.0
            episode_steps.clear()

            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch.clear()

        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(example.reward for example in batch)
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs, train_act = zip(*(step for example in batch for step in example.steps if example.reward >= reward_bound))

    train_obs_v = tensor(train_obs, dtype=float32)
    train_act_v = tensor(train_act, dtype=long)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = DiscreteOneHotWrapper(make("FrozenLake-v0"))
    #env = Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = CrossEntropyLoss()
    optimizer = Adam(params=net.parameters(), lr=0.01)

    with SummaryWriter(comment="-frozenlake-naive") as writer:

        for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
            optimizer.zero_grad()
            action_scores_v = net(obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            optimizer.step()
            print(f"{iter_no:d}: loss={loss_v.item():.3f}, reward_mean={reward_m:.1f}, reward_bound={reward_b:.1f}")
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
            if reward_m > 0.8:
                print("Solved!")
                break

