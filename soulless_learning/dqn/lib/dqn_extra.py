from math import sqrt

import numpy as np
import torch
from numpy import prod
from torch import full, zeros, Tensor
from torch.nn import Linear, Parameter, Module, Sequential, Conv2d, ReLU, Softmax
from torch.nn.functional import linear

N_ATOMS = 51
Vmin = -10
Vmax = 10
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class NoisyLinear(Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = Parameter(w)
        z = zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = full((out_features,), sigma_init)
            self.sigma_bias = Parameter(w)
            z = zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input: Tensor) -> Tensor:
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return linear(input, v, bias)


class NoisyFactorizedLinear(Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / sqrt(in_features)
        w = full((out_features, in_features), sigma_init)
        self.sigma_weight = Parameter(w)
        z1 = zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = full((out_features,), sigma_init)
            self.sigma_bias = Parameter(w)

    def forward(self, input: Tensor) -> Tensor:
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        def func(x): return torch.sign(x) * torch.sqrt(torch.abs(x))

        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return linear(input, v, bias)


class NoisyDQN(Module):
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
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = Sequential(
            self.noisy_layers[0],
            ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return prod(o.size())

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class DuelingDQN(Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = Sequential(
            Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1)
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = Sequential(
            Linear(conv_out_size, 256),
            ReLU(),
            Linear(256, n_actions)
        )
        self.fc_val = Sequential(
            Linear(conv_out_size, 256),
            ReLU(),
            Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return int(prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


class DistributionalDQN(Module):
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
            Linear(512, n_actions * N_ATOMS)
        )
        sups = torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
        self.register_buffer("supports", sups)
        self.softmax = Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return prod(o.size())

    def forward(self, x):
        fx = x.float() / 256
        batch_size = x.size()[0]
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc(conv_out).view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


def distr_projection(next_distr, rewards, dones, gamma):
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    def discretize(values):
        tz_j = np.clip(v, a_min=Vmin, a_max=Vmax)
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        ne_mask = u != l
        return tz_j, b_j, l, u, eq_mask, ne_mask

    for atom in range(N_ATOMS):
        v = rewards + (Vmin + atom * DELTA_Z) * gamma
        tz_j, b_j, l, u, eq_mask, ne_mask = discretize(v)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(
            Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = \
                (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = \
                (b_j - l)[ne_mask]
    return proj_distr


class RainbowDQN(Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = Sequential(
            Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1)
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = Sequential(
            NoisyLinear(conv_out_size, 256),
            ReLU(),
            NoisyLinear(256, n_actions)
        )
        self.fc_val = Sequential(
            Linear(conv_out_size, 256),
            ReLU(),
            Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(zeros(1, *shape))  # 1 is the batch size
        return int(prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)
