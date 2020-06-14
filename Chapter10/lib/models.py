from math import prod

from torch import zeros
from torch.nn import Module, Sequential, Conv1d, ReLU, Linear


class SimpleFFDQN(Module):
    def __init__(self, obs_len, n_actions):
        super().__init__()

        self.fc_val = Sequential(
            Linear(obs_len, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 1)
        )
        self.fc_adv = Sequential(
            Linear(obs_len, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, n_actions)
        )

    def forward(self, x):
        adv, val = self.fc_adv(x), self.fc_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1D(Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = Sequential(
            Conv1d(input_shape[0], 128, 5),
            ReLU(),
            Conv1d(128, 128, 5),
            ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = Sequential(
            Linear(conv_out_size, 512),
            ReLU(),
            Linear(521, 1)
        )
        self.fc_adv = Sequential(
            Linear(conv_out_size, 512),
            ReLU(),
            Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return prod(o.size())

    def forward(self, x):
        conv_out = self.conv(x).flatten(start_dim=1)
        adv, val = self.fc_adv(conv_out), self.fc_val(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))
