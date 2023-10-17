import numpy as np

import gymnasium
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box

from rl_crazyflie.envs.BalanceAviary import BalanceAviary

class MOBalanceAviary(BalanceAviary, EzPickle):
    def __init__(self, **kwargs):
        # discard the attr
        if kwargs.get("render_mode"):
            self.metadata["render_modes"] = [kwargs.pop("render_mode")]
            # kwargs.pop("render_mode")

        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)

        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

        # redefine spaces, use "gymnasium" API instead of "gym"
        # (required for mo-gymnasium)
        self.observation_space = gymnasium.spaces.Box(low=self.observation_space.low, high=self.observation_space.high, shape=self.observation_space.shape)
        self.action_space = gymnasium.spaces.Box(low=self.action_space.low, high=self.action_space.high, shape=self.action_space.shape)


    def step(self, action):
        observation, reward, done, info = super().step(action)
        vec_reward = np.array([info["bal_rew"], info["nav_rew"]])

        return observation, vec_reward, done, done, info
    
    def reset(self, **kwargs):
        obs = super().reset()
        print(obs)

        # obs, info
        return obs, {}