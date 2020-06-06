from argparse import ArgumentParser
from datetime import timedelta, datetime
from warnings import simplefilter

from gym import make
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ptan.actions import EpsilonGreedyActionSelector
from ptan.agent import TargetNet, DQNAgent
from ptan.common.wrappers import wrap_dqn
from ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer
from ptan.ignite import EndOfEpisodeHandler, EpisodeFPSHandler, EpisodeEvents, PeriodicEvents, PeriodEvents
from torch import device
from torch.optim import Adam

from Chapter8.lib import common, dqn_model

NAME = "02_n_envs"


def batch_generator(buffer, initial: int, batch_size: int, steps: int):
    buffer.populate(initial)
    while True:
        buffer.populate(steps)
        yield buffer.sample(batch_size)


if __name__ == "__main__":
    simplefilter("ignore", category=UserWarning)

    params = common.HYPERPARAMS["pong"]

    parser = ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--envs", type=int, default=3, help="Amount of environments to run in parallel")
    args = parser.parse_args()
    device = device("cuda" if args.cuda else "cpu")

    envs = []
    for _ in range(args.envs):
        env = make(params.env_name)
        env = wrap_dqn(env)
        env.seed(123)
        envs.append(env)
    params.batch_size *= args.envs
    net = dqn_model.DQN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = DQNAgent(net, selector, device=device)
    exp_source = ExperienceSourceFirstLast(envs, agent, gamma=params.gamma)
    buffer = ExperienceReplayBuffer(exp_source, buffer_size=params.replay_size)
    optimizer = Adam(net.parameters(), lr=params.learning_rate)


    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration * args.envs)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {"loss": loss.item(), "epsilon": selector.epsilon}


    engine = Engine(process_batch)

    EndOfEpisodeHandler(exp_source, bound_avg_reward=17.0).attach(engine)
    EpisodeFPSHandler(fps_mul=args.envs).attach(engine)


    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))


    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True


    now = datetime.now().isoformat(timespec="minutes").replace(":", "-")
    logdir = f"runs/{now}-{params.run_name}-{NAME}={args.envs}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=PeriodEvents.ITERS_100_COMPLETED)

    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size, args.envs))
