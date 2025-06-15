import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym import ObservationWrapper, spaces
import cv2, numpy as np
from gym_tetris.actions import SIMPLE_MOVEMENT

class GrayResize(ObservationWrapper):
    """Convert RGB frames to grayscale and resize to 84x84"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, frame):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Add channel dimension
        return resized[..., None]

def make_env():
    """Create Tetris environment with preprocessing"""
    # Create base environment
    env = gym_tetris.make("TetrisA-v3")
    
    # Apply action space wrapper
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply observation preprocessing
    env = GrayResize(env)
    
    return env
