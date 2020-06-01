from math import sqrt, prod

import numpy as np
import torch
from torch import full, zeros, Tensor
from torch.nn import Linear, Parameter, Module, Sequential, Conv2d, ReLU
from torch.nn.functional import linear

BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:
    def __init__(self, exp_source, buffer_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx):
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


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
        return prod(o.size())

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)
