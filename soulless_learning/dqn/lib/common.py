import os
from datetime import timedelta, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import List, Iterable
from warnings import simplefilter

import joblib
import numpy as np
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, Checkpoint
from ignite.metrics import RunningAverage
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceFirstLast, PrioritizedReplayBuffer
from ptan.ignite import EndOfEpisodeHandler, EpisodeFPSHandler, EpisodeEvents, PeriodicEvents, PeriodEvents
from torch import tensor, no_grad
from torch.nn import MSELoss

SEED = 123
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
                                       batch_size=32),
               "soulless": SimpleNamespace(env_name="soulless-v0",
                                           stop_reward=10000.0,
                                           run_name="soulless",
                                           replay_size=100000,
                                           replay_initial=50000,
                                           target_net_sync=10000,
                                           epsilon_frames=10 ** 6,
                                           epsilon_start=1.0,
                                           epsilon_final=0.02,
                                           learning_rate=0.00025,
                                           gamma=0.99,
                                           batch_size=32)
               }


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


def calc_values_of_states(states, net, device="cuda"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = tensor(batch, device=device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
    mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


class StatePrioReplayBuffer(PrioritizedReplayBuffer):
    """A version of PrioritizedReplayBuffer that can interface with ignite checkpointing via the state_dict methods"""

    def __init__(self, experience_source, buffer_size, alpha, beta_start, beta_frames):
        super().__init__(experience_source, buffer_size, alpha)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames

    def state_dict(self):
        # generators must be excluded from the state_dict or it won't be serializable
        dictionary = self.__dict__.copy()
        del dictionary["experience_source_iter"]
        return dictionary

    def load_state_dict(self, dictionary):
        exp_source = self.experience_source_iter
        self.__dict__ = dictionary
        # the generators excluded from the state_dict in self.state_dict()
        # must be added back for the replay buffer to function
        self.__dict__["experience_source_iter"] = exp_source

    def populate(self, samples):
        super().populate(samples)
        self.beta += (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, self.beta)

    def sample(self, batch_size, beta):
        return super().sample(batch_size, self.beta)


class LightDiskSaver(DiskSaver):
    """a version of DiskSaver that uses joblib instead of pickle for serialization to save memory usage"""
    def __call__(self, checkpoint, filename, metadata=None):
        path = os.path.join(self.dirname, filename)
        joblib.dump(checkpoint, path)


def batch_generator(buffer, initial: int, batch_size: int, start=False):
    if start:
        buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size, None)


def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, model, optimizer, buffer, target_net,
                 extra_metrics: Iterable[str] = (),
                 ):
    simplefilter("ignore", category=UserWarning)
    handler = EndOfEpisodeHandler(exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    objects_to_checkpoint = {'model': model, 'optimizer': optimizer, 'trainer': engine, "buffer": buffer,
                             "target_net": target_net}
    checkpoint_dir = Path("models backup")
    saver = LightDiskSaver(str(checkpoint_dir), create_dir=True, require_empty=False)
    handler = Checkpoint(objects_to_checkpoint, saver, n_saved=1)
    engine.add_event_handler(Events.ITERATION_COMPLETED(every=30000), handler)


    checkpoints_paths = list(checkpoint_dir.iterdir())
    if checkpoints_paths:
        checkpoint = joblib.load(checkpoints_paths[-1])
        print(f"Loading checkpoint {checkpoints_paths[-1].name}")
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

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
