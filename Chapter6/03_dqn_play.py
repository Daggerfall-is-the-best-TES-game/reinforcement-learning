from argparse import ArgumentParser
from time import time, sleep

from gym import make
from gym.wrappers import Monitor
from torch import load, tensor, device

from Chapter6.lib import wrappers, dqn_model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help=f"Environment name to use, default={DEFAULT_ENV_NAME}")
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    args = parser.parse_args()

    device = device("cuda")
    env = wrappers.wrap_deepmind(make(args.env), episode_life=1, clip_rewards=1, frame_stack=1, scale=1)
    if args.record:
        env = Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(load(args.model))

    state = env.reset()
    total_reward = 0.0
    while True:
        start_ts = time()
        env.render()

        state_v = tensor(state).unsqueeze(0).to(device)
        q_vals_v = net(state_v)
        _, action = q_vals_v.max(dim=1)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1 / FPS - (time() - start_ts)
        if delta > 0:
            sleep(delta)
    print(f"Total reward: {total_reward: .2f}")
