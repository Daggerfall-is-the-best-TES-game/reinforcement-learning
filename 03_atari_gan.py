from gym import make
from gym.core import ObservationWrapper
from gym.spaces import Box
import numpy as np
from cv2 import resize
from random import choice
from torch import FloatTensor, device, ones, zeros
from torch.nn import BCELoss
from torch.optim import Adam
from argparse import ArgumentParser
from tensorboardX import SummaryWriter

LATENT_VECTOR_SIZE = 100
IMAGE_SIZE = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(ObservationWrapper):
    def __init__(self, *args):
        super().__init__(*args)
        assert isinstance(self.observation_space, Box)
        old_space = self.observation_space
        self.observation_space = Box(self.observation(old_space.low), self.observation(old_space.high),
                                     dtype=np.float32)
    def observation(self, observation):
        #resize image
        new_obs = resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32) / 255.0

    def iterate_batches(self, envs, batch_size=BATCH_SIZE):
        batch = [e.reset() for e in envs]

        while True:
            e = choice(envs)
            obs, reward, is_done, _ = e.step(e.action_space.sample())
            # The check for the nonzero mean of the observation is required due to a bug in one of the games
            # to prevent the flickering of an image.
            if np.mean(obs) > 0.01:
                batch.append(obs)
            if len(batch) == batch_size:
                yield FloatTensor(batch)
                batch.clear()
            if is_done:
                e.reset()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true")
    args = parser.parse_args()
    device_ = device("cuda" if args.cuda else "cpu")

    env_names = ("Breakout-v0", "AirRaid-v0", "Pong-v0")
    envs = [InputWrapper(make(name)) for name in env_names]
    input_shape = envs[0].observation_space.shape

    Writer = SummaryWriter()
    net_discr = Discrimintor(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = BCELoss()
    gen_optimizer = Adam(params=net_gener.parameters(), lr=LEARNING_RATE)
    dis_optimizer = Adam(params=net_discr.parameters(), lr=LEARNING_RATE)

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = ones(BATCH_SIZE, dtype=torch.float32, device=device_)
    fake_labels_v = zeros(BATCH_SIZE, dtype=torch.float32, device=device_)

    for batch_v in iterate_batches(envs):
        # generate extra fake samples, input is 4D: batch, filters, x, y
        gen_input_v = FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device_)
        batch_v = batch_v.to(device_)
        gen_output_v = net_gener(gen_input_v)

        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64]), iter_no)
            writer.add_image("real", vutils.make_grid(batch_v.data[:64]), iter_no)
