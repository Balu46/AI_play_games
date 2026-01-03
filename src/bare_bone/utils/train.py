import gymnasium as gym
import os
import torch as T
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback, CallbackList

from src.utils.discrete_actions_wrapper import DiscreteActionsWrapper

from src.algorithms.DNQ.DNQ_baseline import DQN
from src.algorithms.A2C.A2C import A2C
from src.algorithms.PPO.PPO import PPO

# Map algorithm names to classes
ALGO_MAP = {
    "dqn": DQN,
    "a2c": A2C,
    "ppo": PPO,
}

# Map simple environment names to Gym IDs
ENV_MAP = {
    "lunar_lander": "LunarLander-v3",
    "car_racing": "CarRacing-v3",
    "cart_pole": "CartPole-v1",
}

def train(algo_name: str, env_name: str, total_timesteps: int = None, total_episodes: int = None):
    """
    Unified training function for PPO, A2C, and DQN.
    """
    if algo_name not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: {algo_name}. Available: {list(ALGO_MAP.keys())}")
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_MAP.keys())}")

    gym_env_id = ENV_MAP[env_name]
    print(f"Starting training with {algo_name.upper()} on {gym_env_id}...")

    # Create directories for saving models and logs
    model_dir = f"{env_name}/models/{algo_name}"
    log_dir = f"{env_name}/logs/{algo_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    # Wrap continuous action spaces for DQN (requires discrete actions)
    if algo_name == "dqn" and gym_env_id == "CarRacing-v3":
        env = make_vec_env(gym_env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv,
                          wrapper_class=DiscreteActionsWrapper)
        eval_env = make_vec_env(gym_env_id, n_envs=1, seed=42, vec_env_cls=DummyVecEnv,
                               wrapper_class=DiscreteActionsWrapper)
    else:
        env = make_vec_env(gym_env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)
        eval_env = make_vec_env(gym_env_id, n_envs=1, seed=42, vec_env_cls=DummyVecEnv)

    # Select Policy Type
    if "CarRacing" in gym_env_id:
        policy_type = "CnnPolicy"
    else:
        policy_type = "MlpPolicy"
    
    print(f"Using policy: {policy_type}")

    # Initialize Agent
    AlgoClass = ALGO_MAP[algo_name]
    
    # DQN-specific parameters
    agent_kwargs = {
        "policy": policy_type,
        "env": env,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": "auto"
    }
    
    # Reduce buffer size for image-based envs (saves memory)
    if algo_name == "dqn" and policy_type == "CnnPolicy":
        agent_kwargs["buffer_size"] = 50_000  # Default is 1M, too large for images
    
    agent = AlgoClass(**agent_kwargs)

    # Callbacks
    callbacks = []
    
    # Eval Callback - Saves the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    # Episode Limit Callback
    if total_episodes is not None:
        print(f"Training for {total_episodes} episodes (ignoring timesteps limit)...")
        stop_train_callback = StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1)
        callbacks.append(stop_train_callback)
        
        if total_timesteps is None:
            total_timesteps = 10_000_000 # High number to ensure episodes limit is hit
    else:
        if total_timesteps is None:
             total_timesteps = 100000
        print(f"Training for {total_timesteps} timesteps...")
    
    # Combine callbacks
    callback = CallbackList(callbacks)

    agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Ensure existence of best_model.zip
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        print(f"Warning: Best model not found (eval callback might not have triggered). Saving final model as best model.")
        agent.save(best_model_path)
    else:
        print(f"Best model already saved at {best_model_path}")

    # Removed explicit saving of final_model.zip to satisfy request for single model only.
    
    env.close()
    eval_env.close()

        