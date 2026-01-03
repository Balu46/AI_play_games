import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteActionsWrapper(gym.ActionWrapper):
    """
    Wraps continuous action space into discrete actions.
    Designed for CarRacing but can work with other continuous environments.
    
    Converts Box action space to Discrete with predefined action mappings.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define discrete action mapping for CarRacing
        # CarRacing actions: [steering, gas, brake]
        # steering: -1.0 (full left) to +1.0 (full right)
        # gas: 0.0 to 1.0
        # brake: 0.0 to 1.0
        self.actions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),   # 0: Do nothing / Coast
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 1: Turn left
            np.array([1.0, 0.0, 0.0], dtype=np.float32),   # 2: Turn right
            np.array([0.0, 1.0, 0.0], dtype=np.float32),   # 3: Gas (accelerate)
            np.array([0.0, 0.0, 1.0], dtype=np.float32),   # 4: Brake
        ]
        
        # Override action space to discrete
        self.action_space = spaces.Discrete(len(self.actions))
    
    def action(self, act):
        """
        Convert discrete action index to continuous action vector.
        
        Args:
            act: Discrete action index (0-4)
            
        Returns:
            Continuous action array [steering, gas, brake]
        """
        return self.actions[act]
