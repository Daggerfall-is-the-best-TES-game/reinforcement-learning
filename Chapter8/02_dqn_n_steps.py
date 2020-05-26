from argparse import ArgumentParser

from gym import make
from ignite.engine import Engine
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from torch import device
from torch.optim import Adam

from Chapter8.lib import common, dqn_model

NAME = "02_n_steps"
DEFAULT_N_STEPS = 4

if __name__ == "__main__":

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", type=int, default=DEFAULT_N_STEPS, help="steps to do on Bellman unroll")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma, steps_count=args.n)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma ** args.n, device=device)
        loss.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item(), "epsilon": selector.epsilon}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, f"{NAME}={args.n}")
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
