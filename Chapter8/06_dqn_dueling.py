from argparse import ArgumentParser

import numpy as np
from gym import make
from ignite.engine import Engine
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from torch import device, no_grad, tensor
from torch.optim import Adam

from Chapter8.lib import common, dqn_extra

NAME = "06_dueling"
STATES_TO_EVALUATE = 1000
EVAL_EVER_FRAME = 100


@no_grad()
def evaluate_states(states, net, device, engine):
    s_v = tensor(states, device=device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics["adv"] = adv.mean().item()
    engine.state.metrics["val"] = val.mean().item()


if __name__ == "__main__":

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--double", default=True, action="store_true", help="Enable double dqn")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    env = make(params.env_name)
    env = wrap_dqn(env)
    env.seed(123)
    net = dqn_extra.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
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
        if engine.state.iteration % EVAL_EVER_FRAME == 0:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
                engine.state.eval_states = eval_states
            evaluate_states(eval_states, net, device, engine)
        return {"loss": loss.item(), "epsilon": selector.epsilon}


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME, extra_metrics=("adv", "val"))
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
