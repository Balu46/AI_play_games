import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

def extract_scalar_from_event(event_file, tag="rollout/ep_rew_mean"):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return None
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({"step": steps, "value": values})

def plot_training(env_name: str, metric: str = "rollout/ep_rew_mean"):
    log_root = f"{env_name}/logs"
    if not os.path.exists(log_root):
        print(f"No logs found for {env_name} at {log_root}")
        return

    data = []
    
    # Traverse directory to find event files
    for algo in os.listdir(log_root):
        algo_path = os.path.join(log_root, algo)
        if not os.path.isdir(algo_path):
            continue
            
        # Find latest run
        runs = [os.path.join(algo_path, d) for d in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, d))]
        if not runs:
            continue
            
        latest_run = max(runs, key=os.path.getmtime)
        run_name = os.path.basename(latest_run)
        run_path = latest_run
        
        # Find event file
        event_file = None
        for f in os.listdir(run_path):
            if f.startswith("events.out.tfevents"):
                event_file = os.path.join(run_path, f)
                break
        
        if event_file:
            print(f"Reading {algo} - {run_name} (Latest)...")
            df = extract_scalar_from_event(event_file, tag=metric)
            if df is not None:
                df["algorithm"] = algo
                df["run"] = run_name
                data.append(df)

    if not data:
        print(f"No data found for metric {metric}")
        return

    full_df = pd.concat(data, ignore_index=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=full_df, x="step", y="value", hue="algorithm")
    plt.title(f"Training Progress: {env_name} ({metric})")
    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    output_dir = f"{env_name}/debug_out"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_{metric.replace('/', '_')}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")

    # Generate detailed plots per algorithm
    plot_algorithm_metrics(env_name, log_root, output_dir)

def plot_algorithm_metrics(env_name, log_root, output_dir):
    """
    Generate detailed plots (Loss, Reward, Epsilon) for each algorithm.
    """
    metrics = {
        "Reward": "rollout/ep_rew_mean",
        "Loss": "train/loss",
        "Epsilon": "rollout/exploration_rate" 
    }

    for algo in os.listdir(log_root):
        algo_path = os.path.join(log_root, algo)
        if not os.path.isdir(algo_path):
            continue
            
        print(f"Generating detailed plots for {algo}...")
        
        # Collect data for this algo (only latest run)
        runs = [os.path.join(algo_path, d) for d in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, d))]
        if not runs:
            continue

        latest_run = max(runs, key=os.path.getmtime)
        run_name = os.path.basename(latest_run)
        run_path = latest_run
        
        algo_data = []

        # Find event file
        event_file = None
        for f in os.listdir(run_path):
            if f.startswith("events.out.tfevents"):
                event_file = os.path.join(run_path, f)
                break
        
        if event_file:
            for metric_name, tag in metrics.items():
                df = extract_scalar_from_event(event_file, tag=tag)
                if df is not None:
                    df["run"] = run_name
                    df["metric"] = metric_name
                    algo_data.append(df)

        if not algo_data:
            continue

        full_df = pd.concat(algo_data, ignore_index=True)

        # Plot each metric
        for metric_name in metrics.keys():
            metric_df = full_df[full_df["metric"] == metric_name]
            if metric_df.empty:
                continue

            plt.figure(figsize=(8, 5))
            sns.lineplot(data=metric_df, x="step", y="value", style="run", markers=True)
            plt.title(f"{env_name} - {algo.upper()} - {metric_name}")
            plt.xlabel("Timesteps")
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)
            
            filename = f"{algo}_{metric_name.lower().replace(' ', '_')}.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved {filename}")
