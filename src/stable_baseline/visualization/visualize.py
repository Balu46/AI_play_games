import gymnasium as gym
import os
import torch as T
import logging
import cv2
import imageio
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.stable_baseline.utils.car_racing_wrappers import build_car_racing_wrapper, apply_frame_stack
from src.logging_utils import setup_logging

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

logger = logging.getLogger(__name__)

def visualize(
    algo_name: str = None,
    env_name: str = "lunar_lander",
    episodes: int = 2,
    output_format: str = "mp4",  # "mp4" lub "gif"
    fps: int = 30
):
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}")

    log_file = os.path.join(env_name, "logs", "visualize.log")
    setup_logging(log_file, __name__)

    gym_env_id = ENV_MAP[env_name]

    if algo_name is None or algo_name == "all":
        algos_to_run = list(ALGO_MAP.keys())
    else:
        algos_to_run = [algo_name]

    base_video_dir = f"videos/{env_name}"

    os.makedirs(f"{base_video_dir}", exist_ok=True)

    logger.info("===== Starting visualization for environment: %s =====", env_name)
    for algo in algos_to_run:
        logger.info("===== Algorithm to run: %s =====", algo)
    
        
        model_path = f"{env_name}/models/{algo}/best_model.zip"
        if not os.path.exists(model_path):
            logger.warning("Model not found: %s", model_path)
            continue

        wrapper_class = None
        is_car_racing = "CarRacing" in gym_env_id
        if is_car_racing:
            wrapper_class = build_car_racing_wrapper(use_discrete_actions=(algo == "dqn"))

        
        env = make_vec_env(
            gym_env_id,
            n_envs=1,
            seed=0,
            vec_env_cls=DummyVecEnv,
            wrapper_class=wrapper_class,
            env_kwargs={"render_mode": "rgb_array"},
        )
        if is_car_racing:
            env = apply_frame_stack(env, n_stack=4)

        model = ALGO_MAP[algo].load(model_path, env=env)

        for ep in range(episodes):
            obs = env.reset()
            done = False
            frames = []
            score = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                # frame = env.render()[0]
                
                frame = env.envs[0].render()

                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)

                frames.append(frame)

                score += float(reward[0])
                done = bool(done[0])

            output_path = f"{base_video_dir}/{algo}_episode_{ep+1}_reward_{score}.{output_format}"

            if output_format == "mp4":
                height, width, _ = frames[0].shape
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()

            elif output_format == "gif":
                imageio.mimsave(output_path, frames, fps=fps)

            logger.info(
                "[%s] Episode %d | Score %.2f | Saved: %s",
                algo.upper(), ep + 1, score, output_path
            )

        env.close()
