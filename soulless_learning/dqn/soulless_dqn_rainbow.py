from argparse import ArgumentParser
from functools import partial

from gym import make
from ignite.engine import Engine
from ptan.actions import ArgmaxActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.experience import ExperienceSourceFirstLast
from torch import device, tensor, no_grad
from torch.optim import RMSprop

from soulless_learning.dqn.lib import common, dqn_extra, wrappers

NAME = "soulless_rainbow"
N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


def calc_loss_double_dqn(batch, batch_weights, net, tgt_net, gamma, t_device, double=True):
    cuda_tensor = partial(tensor, device=t_device)
    states, actions, rewards, dones, next_states = map(cuda_tensor, common.unpack_batch(batch))
    batch_weights_v = tensor(batch_weights, device=t_device)
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
    return (losses * batch_weights_v).mean(), (losses + 1e-5).data.cpu().numpy()


if __name__ == "__main__":

    params = common.HYPERPARAMS["soulless"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--initial", type=int, help="the iteration the last checkpoint left off at")
    parser.add_argument("--start", action="store_true", help="true if new run, false is resume from checkpoint")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrappers.wrap_dqn(env)
    net = dqn_extra.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = common.StatePrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = RMSprop(net.parameters(), lr=params.learning_rate, momentum=0.95, eps=0.01)


    def process_batch(engine, batch):
        batch, batch_indices, batch_weights = batch
        optimizer.zero_grad()
        loss, prios = calc_loss_double_dqn(batch, batch_weights, net, tgt_net.target_model, gamma=params.gamma, t_device=device)
        loss.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item()}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME, net, optimizer, buffer, tgt_net)
    engine.run(common.batch_generator(buffer, args.initial, params.batch_size, start=args.start), max_epochs=100000,
               epoch_length=1000)
