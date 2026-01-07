import gymnasium as gym
import os
import torch as T
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback, CallbackList, StopTrainingOnNoModelImprovement

from src.stable_baseline.utils.discrete_actions_wrapper import DiscreteActionsWrapper

from stable_baselines3.dqn import DQN
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO

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

def train(algo_name: str, env_name: str, total_timesteps: int = None, total_episodes: int = None, hyperparams: dict = None, patience: int = 5):
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
    wrapper_class = None
    if algo_name == "dqn" and gym_env_id == "CarRacing-v3":
        wrapper_class = DiscreteActionsWrapper

    env = make_vec_env(gym_env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv, wrapper_class=wrapper_class)
    eval_env = make_vec_env(gym_env_id, n_envs=1, seed=42, vec_env_cls=DummyVecEnv, wrapper_class=wrapper_class)

    # Select Policy Type
    if "CarRacing" in gym_env_id:
        policy_type = "CnnPolicy"
    else:
        policy_type = "MlpPolicy"

    print(f"Using policy: {policy_type}")

    # Initialize Agent
    AlgoClass = ALGO_MAP[algo_name]
    
    # Default parameters
    learning_rate = 3e-4
    net_arch = [256, 256]
    gamma = 0.99
    
    # New hyperparams defaults
    activation_fn_name = "ReLU"
    optimizer_class_name = "AdamW"
    weight_decay = 1e-5
    
    # Mappings
    ACTIVATION_FN_MAP = {
        "ReLU": T.nn.ReLU,
        "Tanh": T.nn.Tanh,
        "ELU": T.nn.ELU,
    }
    
    OPTIMIZER_MAP = {
        "Adam": T.optim.Adam,
        "AdamW": T.optim.AdamW,
        "RMSprop": T.optim.RMSprop,
    }

    # General arguments for the agent constructor
    kwargs = {}

    # Override with hyperparams if provided
    if hyperparams:
        print(f"Overriding defaults with: {hyperparams}")
        learning_rate = hyperparams.get("learning_rate", learning_rate)
        gamma = hyperparams.get("gamma", gamma)
        
        if "net_arch" in hyperparams:
             arch_type = hyperparams.get("net_arch")
             if arch_type == "tiny": net_arch = [64, 64]
             elif arch_type == "small": net_arch = [128, 128]
             elif arch_type == "medium": net_arch = [256, 256]
             elif arch_type == "large": net_arch = [512, 512]
             elif isinstance(arch_type, list): net_arch = arch_type
        
        if "activation_fn" in hyperparams:
            activation_fn_name = hyperparams["activation_fn"]
        
        if "optimizer_class" in hyperparams:
            optimizer_class_name = hyperparams["optimizer_class"]

        if "weight_decay" in hyperparams:
            weight_decay = hyperparams["weight_decay"]

        # Extract common params
        if "batch_size" in hyperparams and algo_name != "a2c":
             kwargs["batch_size"] = hyperparams["batch_size"]
        
        # DQN specific
        if algo_name == "dqn":
            if "target_update_interval" in hyperparams: kwargs["target_update_interval"] = hyperparams["target_update_interval"]
            if "train_freq" in hyperparams: kwargs["train_freq"] = hyperparams["train_freq"]
            if "gradient_steps" in hyperparams: kwargs["gradient_steps"] = hyperparams["gradient_steps"]
            if "exploration_fraction" in hyperparams: kwargs["exploration_fraction"] = hyperparams["exploration_fraction"]
            if "exploration_final_eps" in hyperparams: kwargs["exploration_final_eps"] = hyperparams["exploration_final_eps"]
            
        # PPO / A2C specific
        if algo_name in ["ppo", "a2c"]:
            if "ent_coef" in hyperparams: kwargs["ent_coef"] = hyperparams["ent_coef"]
            if "vf_coef" in hyperparams: kwargs["vf_coef"] = hyperparams["vf_coef"]
            if "max_grad_norm" in hyperparams: kwargs["max_grad_norm"] = hyperparams["max_grad_norm"]
            if "n_steps" in hyperparams: kwargs["n_steps"] = hyperparams["n_steps"]
            if "gae_lambda" in hyperparams: kwargs["gae_lambda"] = hyperparams["gae_lambda"]

    # Get classes from maps
    activation_fn = ACTIVATION_FN_MAP.get(activation_fn_name, T.nn.ReLU)
    optimizer_class = OPTIMIZER_MAP.get(optimizer_class_name, T.optim.AdamW)

    if algo_name == "dqn":
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=net_arch,
            optimizer_class=optimizer_class,
            optimizer_kwargs=dict(weight_decay=weight_decay),
            normalize_images=True
        )
    else:
        # PPO / A2C
        policy_kwargs = dict(
            activation_fn=activation_fn,
            net_arch=[dict(pi=net_arch, vf=net_arch)], # Shared architecture structure
            optimizer_class=optimizer_class,
            optimizer_kwargs=dict(weight_decay=weight_decay),
            normalize_images=True
        )
    
    agent_kwargs = {
        "policy": policy_type,
        "env": env,
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": "auto",
        "policy_kwargs": policy_kwargs,
        "learning_rate": learning_rate,
        "gamma": gamma,
    }
    
    # Merge additional kwargs
    agent_kwargs.update(kwargs)
    
    # Reduce buffer size for image-based envs due to memory constraints
    if algo_name == "dqn" and policy_type == "CnnPolicy":
        # Default might be 1M, we set it to 50k for safety unless overridden?
        # If user overrides buffer_size in future we'd handle it here, but for now strict safety
        if "buffer_size" not in agent_kwargs:
            agent_kwargs["buffer_size"] = 50_000

    agent = AlgoClass(**agent_kwargs)

    # Callbacks
    callbacks = []
    
    # Early Stopping Callback
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, verbose=1)

    # Eval Callback - Saves the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=10000 if not hyperparams else 2000, # More frequent eval during optimization
        callback_after_eval=stop_callback, # Add early stopping here
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    # Episode or Timestep Limit
    if total_episodes is not None:
        print(f"Training for {total_episodes} episodes...")
        callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1))
        if total_timesteps is None:
            total_timesteps = 10_000_000
    else:
        if total_timesteps is None:
            total_timesteps = 100000
        print(f"Training for {total_timesteps} timesteps...")
    
    callback = CallbackList(callbacks)

    agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Ensure existence of best_model.zip
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        print(f"Warning: Best model not found. Saving final model as best model.")
        agent.save(best_model_path)
    
    env.close()
    eval_env.close()
    
    return eval_callback.best_mean_reward

        