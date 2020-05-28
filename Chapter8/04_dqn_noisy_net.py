from argparse import ArgumentParser

from gym import make
from ignite.engine import Engine
from ptan.actions import ArgmaxActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from torch import device
from torch.optim import Adam

from Chapter8.lib import common, dqn_extra

NAME = "04_noisy"
NOISY_SNR_EVERY_ITERS = 100

if __name__ == "__main__":

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_extra.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = ArgmaxActionSelector()
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma=params.gamma)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        if engine.state.iteration % NOISY_SNR_EVERY_ITERS == 0:
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                engine.state.metrics[f"snr_{layer_idx + 1}"] = sigma_l2
        return {"loss": loss.item()}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME, extra_metrics=("snr_1", "snr_2"))
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
