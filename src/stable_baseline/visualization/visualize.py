import gymnasium as gym
import os
import torch as T
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.stable_baseline.utils.discrete_actions_wrapper import DiscreteActionsWrapper

# Using the same maps as train.py for consistency
# In a larger project, these should be in a shared config file
ALGO_MAP = {
    "dqn": DQN,
    "a2c": A2C,
    "ppo": PPO,
}

ENV_MAP = {
    "lunar_lander": "LunarLander-v3",
    "car_racing": "CarRacing-v3",
    "cart_pole": "CartPole-v1",
}

def visualize(algo_name: str = None, env_name: str = "lunar_lander", episodes: int = 5):
    """
    Load a trained model and render it in the environment.
    """
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_MAP.keys())}")

    gym_env_id = ENV_MAP[env_name]
    
    # Determine which algorithms to visualize
    if algo_name is None or algo_name == "all":
        algos_to_run = list(ALGO_MAP.keys())
    else:
        if algo_name not in ALGO_MAP:
            raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(ALGO_MAP.keys())}")
        algos_to_run = [algo_name]

    print(f"Visualizing: {algos_to_run} on {env_name}")

    for algo in algos_to_run:
        model_path = f"{env_name}/models/{algo}/best_model.zip"
        
        if not os.path.exists(model_path):
            print(f"Skipping {algo}: Model file not found at {model_path}")
            continue

        print(f"\n--- Running {algo.upper()} ---")
        

        # Initialize environment with render_mode='human'
        env = gym.make(gym_env_id, render_mode="human")
        if "CarRacing" in gym_env_id and algo == "dqn":
            env = DiscreteActionsWrapper(env)

        
        # Load Model
        AlgoClass = ALGO_MAP[algo]
        try:
            model = AlgoClass.load(model_path, env=env)
            
            for ep in range(episodes):
                obs, info = env.reset()
                done = False
                score = 0.0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    score += reward
                
                print(f"[{algo.upper()}] Episode {ep+1}: Score {score:.2f}")

        except Exception as e:
            print(f"Error running {algo}: {e}")
        except KeyboardInterrupt:
            print("\nVisualization interrupted.")
            env.close()
            return
        finally:
            env.close()
