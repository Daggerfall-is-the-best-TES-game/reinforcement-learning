from math import prod

from torch import zeros
from torch.nn import Module, Sequential, Conv2d, ReLU, Linear


class DQN(Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = Sequential(
            Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = Sequential(
            Linear(conv_out_size, 512),
            ReLU(),
            Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return prod(o.size())

    def forward(self, x):
        conv_out = self.conv(x).flatten(start_dim=1)
        return self.fc(conv_out)
