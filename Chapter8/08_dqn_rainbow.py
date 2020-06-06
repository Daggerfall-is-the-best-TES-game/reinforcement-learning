from argparse import ArgumentParser
from functools import partial

from gym import make
from ignite.engine import Engine
from ptan.actions import ArgmaxActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast
from torch import device, tensor, no_grad
from torch.optim import Adam

from Chapter8.lib import common, dqn_extra

NAME = "08_rainbow"
N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


def calc_loss_double_dqn(batch, batch_weights, net, tgt_net, gamma, device="cuda", double=True):
    cuda_tensor = partial(tensor, device=device)
    states, actions, rewards, dones, next_states = map(cuda_tensor, common.unpack_batch(batch))
    batch_weights_v = tensor(batch_weights, device=device)
    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with no_grad():
        next_states_v = next_states.detach()
        if double:
            next_state_acts = net(next_states_v).max(1)[1].unsqueeze(-1)
            next_state_vals = tgt_net(next_states_v).gather(1, next_state_acts).squeeze(-1)
        else:
            next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[dones] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards

    losses = (state_action_values - exp_sa_vals) ** 2
    losses *= batch_weights_v
    return losses.mean(), (losses + 1e-5).data.cpu().numpy()


def calc_loss_prio(batch, batch_weights, net, tgt_net, gamma, device="cuda"):
    cuda_tensor = partial(tensor, device=device)
    states, actions, rewards, dones, next_states = map(cuda_tensor, common.unpack_batch(batch))
    batch_weights_v = tensor(batch_weights, device=device)
    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with no_grad():
        next_state_values = tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards
    l = (state_action_values - expected_state_action_values) ** 2
    losses = batch_weights_v * l
    return losses.mean(), (losses + 1e-5).data.cpu().numpy()


if __name__ == "__main__":

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_extra.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = dqn_extra.PrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss, sample_prios = calc_loss_prio(batch, batch_weights, net, tgt_net.target_model,
                                            gamma=params.gamma ** N_STEPS)
        loss.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item(), "beta": buffer.update_beta(engine.state.iteration)}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
