import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
import logging
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def extract_scalar_from_event(event_file, tag="rollout/ep_rew_mean"):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({"step": steps, "value": values})

def _smooth_series(df: pd.DataFrame, window: int):
    if df is None or df.empty:
        return df
    if window <= 1:
        return df
    df = df.copy()
    df["value"] = df["value"].rolling(window=window, min_periods=1).mean()
    return df

def _normalize_steps(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["step"] = df["step"] - df["step"].iloc[0]
    return df

def _event_files_for_run(run_path: str):
    event_files = [
        os.path.join(run_path, f)
        for f in os.listdir(run_path)
        if f.startswith("events.out.tfevents")
    ]
    return sorted(event_files, key=os.path.getmtime, reverse=True)

def _latest_event_for_run(run_path: str):
    event_files = _event_files_for_run(run_path)
    if not event_files:
        return None
    return event_files[0]

def _pick_metric_from_event(event_file: str, preferred_tags):
    ea = EventAccumulator(event_file)
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    for tag in preferred_tags:
        if tag in scalar_tags:
            df = extract_scalar_from_event(event_file, tag=tag)
            if df is not None:
                return df, tag
    if scalar_tags:
        tag = scalar_tags[0]
        df = extract_scalar_from_event(event_file, tag=tag)
        if df is not None:
            return df, tag
    return None, None

def _run_index(run_path: str) -> int:
    name = os.path.basename(run_path)
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return -1

def _latest_run_with_metric(runs, preferred_tags):
    latest_run = None
    latest_event = None
    latest_index = None
    latest_mtime = None
    for run_path in runs:
        event_file = _latest_event_for_run(run_path)
        if event_file is None:
            continue
        run_idx = _run_index(run_path)
        mtime = os.path.getmtime(event_file)
        if latest_index is None or run_idx > latest_index:
            latest_index = run_idx
            latest_mtime = mtime
            latest_run = run_path
            latest_event = event_file
        elif run_idx == latest_index and latest_mtime is not None and mtime > latest_mtime:
            latest_mtime = mtime
            latest_run = run_path
            latest_event = event_file
    if latest_run is None or latest_event is None:
        return None, None, None, None
    df, used_tag = _pick_metric_from_event(latest_event, preferred_tags)
    return latest_run, df, latest_event, used_tag

def plot_training(env_name: str, metric: str = "rollout/ep_rew_mean", smooth_window: int = 6, plot_tag: str = None):
    log_file = os.path.join(env_name, "logs", "plot.log")
    setup_logging(log_file, __name__)
    log_root = f"{env_name}/logs"
    if not os.path.exists(log_root):
        logger.warning("No logs found for %s at %s", env_name, log_root)
        return

    data = []
    
    # Traverse directory to find event files
    for algo in os.listdir(log_root):
        algo_path = os.path.join(log_root, algo)
        if not os.path.isdir(algo_path):
            continue
            
        runs = [os.path.join(algo_path, d) for d in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, d))]
        if not runs:
            continue

        # Pick the latest run (by event file mtime) for the selected metric
        best_run, best_df, best_event, used_tag = _latest_run_with_metric(
            runs,
            preferred_tags=[metric, "eval/mean_reward"],
        )

        if best_df is not None:
            run_name = os.path.basename(best_run)
            tag_label = used_tag if used_tag else metric
            logger.info("Reading %s - %s (Latest run, %s points, tag=%s)...", algo, run_name, len(best_df), tag_label)
            best_df = _smooth_series(best_df, smooth_window)
            best_df = _normalize_steps(best_df)
            best_df["algorithm"] = algo
            best_df["run"] = run_name
            data.append(best_df)

    if not data:
        logger.warning("No data found for metric %s", metric)
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
    if plot_tag:
        output_dir = f"{output_dir}_{plot_tag}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_{metric.replace('/', '_')}.png")
    plt.savefig(output_path, dpi=150)
    logger.info("Comparison plot saved to %s", output_path)

    # Generate detailed plots per algorithm
    plot_algorithm_metrics(env_name, log_root, output_dir, smooth_window=smooth_window)

def plot_algorithm_metrics(env_name, log_root, output_dir, smooth_window: int = 6):
    """
    Generate detailed plots (Loss, Reward, Epsilon) for each algorithm.
    """
    # Base metrics that apply to all algorithms
    base_metrics = {
        "Reward": "rollout/ep_rew_mean",
        "Epsilon": "rollout/exploration_rate" 
    }

    for algo in os.listdir(log_root):
        algo_path = os.path.join(log_root, algo)
        if not os.path.isdir(algo_path):
            continue
            
        logger.info("Generating detailed plots for %s...", algo)
        
        # Collect data for this algo (use latest run with reward points)
        runs = [os.path.join(algo_path, d) for d in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, d))]
        if not runs:
            continue
        best_run, best_reward_df, best_event, used_tag = _latest_run_with_metric(
            runs,
            preferred_tags=["rollout/ep_rew_mean", "eval/mean_reward"],
        )

        if best_run is None or best_event is None:
            continue

        run_name = os.path.basename(best_run)
        tag_source = used_tag or "rollout/ep_rew_mean"
        algo_data = []

        # Algorithm-specific loss metrics
        algo_lower = algo.lower()
        if algo_lower == "dqn":
            loss_tag = "train/loss"
        elif algo_lower in ["a2c", "ppo"]:
            # For Actor-Critic algorithms, use value_loss as the primary loss metric
            loss_tag = "train/value_loss"
        else:
            loss_tag = "train/loss"  # fallback

        # Combine base metrics with algorithm-specific loss
        metrics = {**base_metrics, "Loss": loss_tag}

        for metric_name, tag in metrics.items():
            df = extract_scalar_from_event(best_event, tag=tag)
            if df is None and metric_name == "Reward" and tag_source != tag:
                df = extract_scalar_from_event(best_event, tag=tag_source)
            if df is not None:
                if metric_name in ["Reward", "Loss"]:
                    df = _smooth_series(df, smooth_window)
                df = _normalize_steps(df)
                df["run"] = run_name
                df["metric"] = metric_name
                algo_data.append(df)

        if not algo_data:
            continue

        full_df = pd.concat(algo_data, ignore_index=True)

        # Plot each metric
        for metric_name in full_df["metric"].unique():
            metric_df = full_df[full_df["metric"] == metric_name]
            if metric_df.empty:
                continue

            plt.figure(figsize=(8, 5))
            sns.lineplot(data=metric_df, x="step", y="value", style="run")
            plt.title(f"{env_name} - {algo.upper()} - {metric_name}")
            plt.xlabel("Timesteps")
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)
            
            filename = f"{algo}_{metric_name.lower().replace(' ', '_')}.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info("Saved %s", filename)
