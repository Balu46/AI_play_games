import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.algorithms.DQN import Agent

N_GAMES_FOR_TRANING = 5000

def traning(agent : Agent, n_games: int = N_GAMES_FOR_TRANING, env : gym.Env = None):
    """
    Function to train the sensor network.
    This function initializes the sensor network and runs a series of training episodes.
    Each episode consists of a series of time steps where the sensors interact with the environment.
    The training process involves choosing actions, receiving rewards, and updating the agent's knowledge.
    """
    # Initialize the network
    env = gym.make("LunarLander-v3")
    # env = gym.make("LunarLander-v3", render_mode="human")

    n_actions = env.action_space.n                  # 4
    input_dims = int(np.prod(env.observation_space.shape))  # 8

    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.0001,
                  input_dims=input_dims, batch_size=64, n_actions=n_actions,
                  eps_end=0.01, eps_dec=1e-5)

    scores, eps_history = [], []
    best_score = -np.inf

    for i in range(n_games):
        t = 0
        score = 0.0
        observation, info = env.reset()     
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated  
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_
            score += reward
            t += 1

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if score > best_score:
            best_score = score
            agent.update_best_model()

        print(f'episode {i} score {score:.2f} average score {avg_score:.2f} epsilon {agent.epsilon:.2f}')

    # trzymaj jedną spójną ścieżkę do modelu
    agent.save_best_model('lunar_lander/models/best_model.pth')
    print('Best model saved')
    
    # --- ZAPIS LOGU ---
    df = pd.DataFrame({
        "episode": np.arange(1, len(scores) + 1),
        "score": scores,
        "avg100": pd.Series(scores).rolling(100).mean(),
        "epsilon": eps_history,
    })
    out_csv = "lunar_lander/debug_out/training_log.csv"
    df.to_csv(out_csv, index=False)
    print(f"Zapisano log: {out_csv}")

    # --- WYKESY ---
    # 1) Wynik per epizod + średnia krocząca 100
    plt.figure()
    plt.plot(df["episode"], df["score"], label="score")
    plt.plot(df["episode"], df["avg100"], label="avg100")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lunar_lander/debug_out/score_avg100.png", dpi=150)

    # 2) Epsilon
    plt.figure()
    plt.plot(df["episode"], df["epsilon"], label="epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lunar_lander/debug_out/epsilon.png", dpi=150)

    print("Wykresy zapisane jako: score_avg100.png oraz epsilon.png")
          
        