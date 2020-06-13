from enum import Enum

import numpy as np
from gym import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Box
from gym.utils import seeding

from . import data


class Actions(Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.have_position = None
        self.open_price = None
        self._prices = None
        self._offset = None

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit
        return (3 + self.volumes) * self.bars_count + 1 + 1,

    def encode(self):
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self):
        open_price = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open_price * (1.0 + rel_close)

    def step(self, action):
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()

        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0
        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0] - 1
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1)
        return reward, done


class State1D(State):
    @property
    def shape(self):
        return 5 + self.volumes, self.bars_count

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset - (self.bars_count - 1)
        stop = self._offset + 1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
        dst = 3 + self.volumes
        if self.have_position:
            res[dst] = 1.0
            res[dst + 1] = self._cur_close() / self.open_price - 1.0
        return res


class StockEnv(Env):
    metadata = {"render.modes": ["human"]}
    spec = EnvSpec("StocksEnv-v0")

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StockEnv(prices, **kwargs)

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT, commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, conv_1d=False, random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices

        conv_args = (bars_count, commission, reset_on_close)
        conv_kwargs = {"reward_on_close": reward_on_close, "volumes": volumes}
        self._state = State1D(*conv_args, **conv_kwargs) if conv_1d else State(*conv_args, **conv_kwargs)
        self.action_space = Discrete(n=len(Actions))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()
        self._instrument = None
        self.np_random = None

    def reset(self):
        self._instrument = self.np_random.choice(list(self._prices))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars if self.random_ofs_on_reset else bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state.offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
