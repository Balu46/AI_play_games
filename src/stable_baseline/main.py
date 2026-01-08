import json
import os
import argparse
from src.stable_baseline.utils.train import train
from src.stable_baseline.visualization.visualize import visualize
from src.stable_baseline.visualization.plot import plot_training
from src.stable_baseline.optimize import run_optimization
import torch

CONFIG_PATH = "config.json"

def load_config(path):
    if not os.path.exists(path):
        print(f"Config file not found at {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    print(f"device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    parser = argparse.ArgumentParser(description="RL Agent Manager")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "visualize", "plot", "optimize"], help="Operation mode")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to configuration file")
    
    # Optional overrides
    parser.add_argument("--algo", type=str, help="Algorithm override (e.g. ppo)")
    parser.add_argument("--env", type=str, help="Environment override (e.g. lunar_lander)")
    parser.add_argument("--no-optimized-params", action="store_true", help="Skip loading optimized hyperparameters")
    
    # Optimization specific
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for optimization")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per trial for optimization")
    parser.add_argument("--study-name", type=str, default="rl_optimization", help="Study name for optimization")

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
            print("Config is required for training mode (unless fully overridden, but config is safer)")
            exit(1)
        for game in games:

            algorithms = config.get("algorithms", [])
            # Default to 100k steps if neither is provided in config
            timesteps = config.get("total_timesteps")
            episodes = config.get("total_episodes")
            
            target = f"{episodes} episodes" if episodes else f"{timesteps or 100000} steps"
            print(f"Training on {game} for {target}...")
            
            for algo in algorithms:
                print(f"\n--- Starting {algo.upper()} ---")
                
                # Check for optimized hyperparameters
                hyperparams = None  
                optimization_path = os.path.join(game, "optimization", algo, "best_params.json")
                if not args.no_optimized_params and os.path.exists(optimization_path):
                    print(f"Found optimized hyperparameters at {optimization_path}")
                    try:
                        with open(optimization_path, "r") as f:
                            hyperparams = json.load(f)
                    except Exception as e:
                        print(f"Error loading optimized hyperparameters: {e}")
                elif args.no_optimized_params:
                    print("Skipping optimized hyperparameters (flag enabled).")
                
                train(algo, game, total_timesteps=timesteps, total_episodes=episodes, hyperparams=hyperparams, patience=10)

    elif args.mode == "visualize":
        for game in games:
            if not game:
                print("Visualization requires --env (or config game)")
                exit(1)
            visualize(args.algo, game)

    elif args.mode == "plot":
        for game in games:
            if not game:
                print("Plotting requires --env (or config game)")
                exit(1)
            plot_training(game)

    elif args.mode == "optimize":
        for game in games:
            if not game or not args.algo:
                print("Optimization requires --env and --algo")
                exit(1)
            run_optimization(args)
