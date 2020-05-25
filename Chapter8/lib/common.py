from datetime import timedelta, datetime
from functools import partial
from types import SimpleNamespace
from typing import List, Iterable
from warnings import simplefilter

import numpy as np
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceFirstLast, ExperienceReplayBuffer
from ptan.ignite import EndOfEpisodeHandler, EpisodeFPSHandler, EpisodeEvents, PeriodicEvents, PeriodEvents
from torch import tensor, no_grad
from torch.nn import MSELoss

HYPERPARAMS = {"pong": SimpleNamespace(env_name="PongNoFrameskip-v4",
                                       stop_reward=18.0,
                                       run_name="pong",
                                       replay_size=100000,
                                       replay_initial=10000,
                                       target_net_sync=1000,
                                       epsilon_frames=10 ** 5,
                                       epsilon_start=1.0,
                                       epsilon_final=0.02,
                                       learning_rate=1e-4,
                                       gamma=0.99,
                                       batch_size=32)}


def unpack_batch(batch: List[ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.bool), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cuda"):
    cuda_tensor = partial(tensor, device=device)
    states, actions, rewards, dones, next_states = map(cuda_tensor, unpack_batch(batch))
    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with no_grad():
        next_state_values = tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards
    return MSELoss()(state_action_values, expected_state_action_values)


class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


def batch_generator(buffer: ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, extra_metrics: Iterable[str] = ()):
    simplefilter("ignore", category=UserWarning)
    handler = EndOfEpisodeHandler(exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get("time_passed", 0)
        print("Episode {}: reward={:.0f}, steps={}, speed={:.1f} f/s, elapsed={}".format(
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get("avg_fps", 0), timedelta(seconds=int(passed))))

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics["time_passed"]
        print(f"Game solved in {timedelta(seconds=int(passed))} after {trainer.state.episode}"
              f" episodes and {trainer.state.iteration} iterations!")
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec="minutes").replace(":", "-")
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v["loss"])
    run_avg.attach(engine, "avg_loss")
    metrics = ["reward", "steps", "avg_reward"]
    handler = OutputHandler(tag="episodes", metric_names=metrics)
    event = EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    metrics = ["avg_loss", "avg_fps"]
    metrics.extend(extra_metrics)
    handler = OutputHandler(tag="train", metric_names=metrics, output_transform=lambda a: a)
    event = PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
