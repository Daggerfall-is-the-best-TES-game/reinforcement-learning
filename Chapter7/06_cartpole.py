from gym import make
from ptan.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from torch import no_grad, tensor
from torch.nn import Module, Sequential, ReLU, Linear
from torch.nn.functional import mse_loss
from torch.optim import Adam

HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_SIZE = 1000
LR = 1e-3
EPS_DECAY = 0.99


class Net(Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = Sequential(
            Linear(obs_size, hidden_size),
            ReLU(),
            Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


@no_grad()
def unpack_batch(batch, net, gamma):
    states, actions, rewards, done_masks, last_states = \
        map(list, zip(*((state, action, reward, last_state is None, state if last_state is None else last_state)
                        for state, action, reward, last_state in batch)))

    states_v, actions_v, rewards_v, last_states_v = map(tensor, (states, actions, rewards, last_states))
    last_states_q_v = net(last_states_v)
    best_last_q_v = last_states_q_v.max(dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


if __name__ == "__main__":
    env = make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    tgt_net = TargetNet(net)
    selector = ArgmaxActionSelector()
    selector = EpsilonGreedyActionSelector(epsilon=1, selector=selector)
    agent = DQNAgent(net, selector)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = Adam(net.parameters(), LR)

    step = episode = 0
    solved = False
    while True:
        step += 1
        buffer.populate(1)
        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            print(f"{step}: episode {episode} done, {reward=:.3f}, epsilon={selector.epsilon:.2f}")
            solved = reward > 150
        if solved:
            print("YAY!")
            break
        if len(buffer) < 2 * BATCH_SIZE:
            continue
        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, tgt_q = unpack_batch(batch, tgt_net.target_model, GAMMA)
        optimizer.zero_grad()
        q = net(states_v)
        q = q.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss = mse_loss(q, tgt_q)
        loss.backward()
        optimizer.step()
        selector.epsilon *= EPS_DECAY
        if step % TGT_NET_SYNC == 0:
            tgt_net.sync()
