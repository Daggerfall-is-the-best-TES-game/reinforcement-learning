from argparse import ArgumentParser
from functools import partial

from gym import make
from ignite.engine import Engine
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast
from torch import tensor, no_grad, device
from torch.optim import Adam

from Chapter8.lib import dqn_model
from Chapter8.lib.common import unpack_batch, HYPERPARAMS, EpsilonTracker, setup_ignite, batch_generator
from Chapter8.lib.dqn_extra import PrioReplayBuffer

NAME = "05_prio_replay"
PRIO_REPLAY_ALPHA = 0.6


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cuda"):
    cuda_tensor = partial(tensor, device=device)
    states, actions, rewards, dones, next_states = map(cuda_tensor, unpack_batch(batch))
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

    params = HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = EpsilonTracker(selector, params)
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = PrioReplayBuffer(exp_source, buffer_size=params.replay_size, prob_alpha=PRIO_REPLAY_ALPHA)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss, sample_prios = calc_loss(batch, batch_weights, net, tgt_net.target_model, gamma=params.gamma)
        loss.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item(), "epsilon": selector.epsilon, "beta": buffer.update_beta(engine.state.iteration)}


    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, NAME)
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))
