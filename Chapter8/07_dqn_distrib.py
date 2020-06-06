from argparse import ArgumentParser

from gym import make
from ignite.engine import Engine
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from torch import device, tensor
from torch.nn.functional import log_softmax
from torch.optim import Adam

from Chapter8.lib import common, dqn_extra

NAME = "07_distrib"


def calc_loss(batch, net, tgt_net, gamma, device="cuda"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = tensor(states, device=device)
    actions_v = tensor(actions, device=device)
    next_states_v = tensor(next_states, device=device)

    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_acts = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v)
    next_distr = next_distr.data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_acts]
    proj_distr = dqn_extra.distr_projection(next_best_distr, rewards, dones, gamma)

    distr_v = net(states_v)
    sa_vals = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = log_softmax(sa_vals, dim=1)
    proj_distr_v = tensor(proj_distr, device=device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


if __name__ == "__main__":

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_extra.DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = DQNAgent(lambda x: net.qvals(x), selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss = calc_loss(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item(), "epsilon": selector.epsilon}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
