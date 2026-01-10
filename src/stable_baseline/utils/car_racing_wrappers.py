import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecFrameStack

from src.stable_baseline.utils.discrete_actions_wrapper import DiscreteActionsWrapper

import numpy as np
import gymnasium as gym

class CarRacingActionRescale(gym.ActionWrapper):
    """
    Mapuje akcje z Box([-1,-1,-1],[1,1,1]) na:
      steering in [-1,1]
      gas in [0,1]
      brake in [0,1]
    """
    def __init__(self, env):
        super().__init__(env)

    def action(self, a):
        a = np.asarray(a, dtype=np.float32)
        steer = np.clip(a[0], -1.0, 1.0)
        gas   = np.clip((a[1] + 1.0) / 2.0, 0.0, 1.0)
        brake = np.clip((a[2] + 1.0) / 2.0, 0.0, 1.0)
        return np.array([steer, gas, brake], dtype=np.float32)


def _resize_nearest(obs: np.ndarray, size=(84, 84)) -> np.ndarray:
    h, w = obs.shape[:2]
    new_h, new_w = size
    row_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    return obs[row_idx][:, col_idx]

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size=(84, 84)):
        super().__init__(env)
        self.size = size
        shape = (self.observation_space.shape[-1],) + self.size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = _resize_nearest(obs, self.size)
        obs = obs.astype(np.uint8)
        return np.transpose(obs, (2, 0, 1))

def _apply_image_preprocess(env: gym.Env) -> gym.Env:
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, size=(84, 84))
    return env

def build_car_racing_wrapper(use_discrete_actions: bool):
    def _wrapper(env: gym.Env) -> gym.Env:
        if use_discrete_actions:
            env = DiscreteActionsWrapper(env)
        else:   
            env = CarRacingActionRescale(env)
            
        return _apply_image_preprocess(env)
    
    return _wrapper

def apply_frame_stack(vec_env, n_stack: int = 4):
    return VecFrameStack(vec_env, n_stack=n_stack, channels_order="first")


