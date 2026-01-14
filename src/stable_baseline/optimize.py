import optuna
import argparse
import json
import os
import logging
from src.stable_baseline.utils.train import train
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def objective(trial, args):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    
    hyperparams = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "net_arch": net_arch_type
    }
    
    # Optimizer settings
    optimizer_class = trial.suggest_categorical("optimizer_class", ["Adam", "AdamW", "RMSprop"])
    hyperparams["optimizer_class"] = optimizer_class
    hyperparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    hyperparams["activation_fn"] = trial.suggest_categorical("activation_fn", ["ReLU", "Tanh", "ELU"])
    
    if args.algo == "dqn":
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
        hyperparams["batch_size"] = batch_size
        hyperparams["target_update_interval"] = trial.suggest_categorical("target_update_interval", [1000, 5000, 10000, 20000])
        hyperparams["train_freq"] = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
        hyperparams["gradient_steps"] = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8])
        hyperparams["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        hyperparams["exploration_final_eps"] = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    elif args.algo in ["ppo", "a2c"]:
        hyperparams["vf_coef"] = trial.suggest_float("vf_coef", 0.2, 0.8)
        hyperparams["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        
        if args.algo == "ppo":
            hyperparams["ent_coef"] = trial.suggest_float("ent_coef", 1e-5, 0.05, log=True)
            hyperparams["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.98)
            hyperparams["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
            hyperparams["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
            hyperparams["batch_size"] = batch_size
            
        if args.algo == "a2c":
            hyperparams["ent_coef"] = trial.suggest_float("ent_coef", 1e-5,  5e-4, log=True)
            hyperparams["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.98)
            hyperparams["n_steps"] = trial.suggest_categorical("n_steps", [256, 512, 1024])

    try:
        # Run training
        # Use fewer timesteps/episodes for optimization trials to speed up
        # But enough to get a signal.
        # For CartPole, it trains fast. For CarRacing, it's slow.
        # We'll rely on eval_callback to give us the best score.
        
        # We can use pruning? 
        # Integration with Optuna pruning would require a custom callback in SB3, 
        # skipping for now to keep it simple as per plan.
        
        # Determine strict limits for optimization trials to avoid infinite loops
        # or extremely long training sessions
        timesteps = args.timesteps if args.timesteps else 100000
        
        best_reward = train(
            algo_name=args.algo,
            env_name=args.env,
            total_timesteps=timesteps,
            total_episodes=None,
            hyperparams=hyperparams,
        )
        
        return best_reward
        
    except Exception as e:
        logger.error("Trial failed", exc_info=True)
        raise

def run_optimization(args):
    """
    Main entry point for optimization from CLI
    """
    log_file = os.path.join(args.env, "logs", "optimize.log")
    setup_logging(log_file, __name__)

    # Create output directory
    output_dir = os.path.join(args.env, "optimization", args.algo)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use absolute path for sqlite to avoid relative path issues if CWD changes
    db_path = os.path.abspath(os.path.join(output_dir, "optuna.db"))
    storage_url = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    
    logger.info("Starting optimization for %s on %s with %s trials...", args.algo, args.env, args.n_trials)
    logger.info("Results will be saved to %s", output_dir)
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    logger.info("Number of finished trials: %s", len(study.trials))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: %s", trial.value)
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info("    %s: %s", key, value)
        
    # Save best params
    best_params_file = os.path.join(output_dir, "best_params.json")
    with open(best_params_file, "w") as f:
        json.dump(trial.params, f, indent=4)
        
    logger.info("Best parameters saved to %s", best_params_file)
