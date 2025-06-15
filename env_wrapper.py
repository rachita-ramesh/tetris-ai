import gym_tetris
from nes_py.wrappers import JoypadSpace
from gymnasium import ObservationWrapper
import cv2, numpy as np

# Allowed discrete actions
ACTIONS = [["NOOP"], ["Left"], ["Right"], ["A"], ["Down"]]

class GrayResize(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[..., None]

def make_env():
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, ACTIONS)
    env = GrayResize(env)
    return env
