import json
import os
import argparse
import logging
from src.stable_baseline.utils.train import train
from src.stable_baseline.visualization.visualize import visualize
from src.stable_baseline.visualization.plot import plot_training
from src.stable_baseline.optimize import run_optimization
import torch
from src.logging_utils import setup_logging

CONFIG_PATH = "config.json"
LOG_PATH = os.path.join("logs", "app.log")

logger = logging.getLogger(__name__)

def load_config(path):
    if not os.path.exists(path):
        logger.error("Config file not found at %s", path)
        return None
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    setup_logging(LOG_PATH, __name__)
    logger.info("device : %s", "cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="RL Agent Manager")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "visualize", "plot", "optimize"], help="Operation mode")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to configuration file")
    
    # Optional overrides
    parser.add_argument("--algo", type=str, help="Algorithm override (e.g. ppo)")
    parser.add_argument("--env", type=str, help="Environment override (e.g. lunar_lander)")
    parser.add_argument("--no-optimized-params", action="store_true", help="Skip loading optimized hyperparameters")
    
    # Optimization specific
    parser.add_argument("--n-trials", type=int, default=40, help="Number of trials for optimization")
    parser.add_argument("--timesteps", type=int, default=150000, help="Timesteps per trial for optimization")
    parser.add_argument("--study-name", type=str, default="rl_optimization", help="Study name for optimization")
    parser.add_argument("--plot-tag", type=str, default=None, help="Optional tag to separate plot outputs")

    args = parser.parse_args()

    # Load defaults from config
    config = load_config(args.config)
    
    # Determine settings (CLI args take precedence)
    games = args.env if args.env else (config.get("game") if config else None)
    if games == "all" : 
        games = ["car_racing", "lunar_lander", "cart_pole"]
    else:
        games = [games]
    
    if args.mode == "train":
        if not config:
            logger.error("Config is required for training mode (unless fully overridden, but config is safer)")
            exit(1)
        for game in games:

            algorithms = config.get("algorithms", [])
            # Default to 100k steps if neither is provided in config
            timesteps = config.get("total_timesteps")
            episodes = config.get("total_episodes")
            
            target = f"{episodes} episodes" if episodes else f"{timesteps or 100000} steps"
            logger.info("Training on %s for %s...", game, target)
            
            for algo in algorithms:
                logger.info("--- Starting %s ---", algo.upper())
                
                # Check for optimized hyperparameters
                hyperparams = None  
                optimization_path = os.path.join(game, "optimization", algo, "best_params.json")
                if not args.no_optimized_params and os.path.exists(optimization_path):
                    logger.info("Found optimized hyperparameters at %s", optimization_path)
                    try:
                        with open(optimization_path, "r") as f:
                            hyperparams = json.load(f)
                    except Exception as e:
                        logger.error("Error loading optimized hyperparameters: %s", e)
                elif args.no_optimized_params:
                    logger.info("Skipping optimized hyperparameters (flag enabled).")
                
                train(algo, game, total_timesteps=timesteps, total_episodes=episodes, hyperparams=hyperparams)

    elif args.mode == "visualize":
        
        json_path = os.path.join("videos", "episode_rewards.json")
        
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                pass
        
        for game in games:
            if not game:
                logger.error("Visualization requires --env (or config game)")
                exit(1)
            visualize(args.algo, game)

    elif args.mode == "plot":
        for game in games:
            if not game:
                logger.error("Plotting requires --env (or config game)")
                exit(1)
            plot_training(game, plot_tag=args.plot_tag)

    elif args.mode == "optimize":
        for game in games:
            if not game or not args.algo:
                logger.error("Optimization requires --env and --algo")
                exit(1)
            run_optimization(args)
