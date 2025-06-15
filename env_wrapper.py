import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym import ObservationWrapper, spaces
import cv2, numpy as np
from gym_tetris.actions import SIMPLE_MOVEMENT
import gym

# Allowed discrete actions - using predefined actions from gym_tetris
ACTIONS = SIMPLE_MOVEMENT

class GrayResize(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[..., None]
    
    def reset(self, **kwargs):
        # Handle both old and new gym API
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self.observation(obs), info
        else:
            # Old gym API - just return the observation
            return self.observation(result)
    
    def step(self, action):
        # Handle both old and new gym API
        result = self.env.step(action)
        if len(result) == 4:
            # Old gym API: (obs, reward, done, info)
            obs, reward, done, info = result
            return self.observation(obs), reward, done, info
        else:
            # New gym API: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            return self.observation(obs), reward, terminated, truncated, info

def make_env():
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, ACTIONS)
    env = GrayResize(env)
    return env
