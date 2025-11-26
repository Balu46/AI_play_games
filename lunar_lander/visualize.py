import gymnasium as gym
import numpy as np
import torch as T

from train import DeepQNetwork 

MODEL_PATH = "lunar_lander/models/best_model.pth"

def load_policy(model_path, n_actions, input_dims, lr=1e-4):
    net = DeepQNetwork(lr=lr, input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)
    net.load_state_dict(T.load(model_path, map_location=net.device))
    net.eval()
    return net

def greedy_action(net, obs):
    with T.no_grad():
        state = T.tensor(obs, dtype=T.float32, device=net.device).unsqueeze(0)
        q = net(state)
        return int(T.argmax(q, dim=1).item())

def watch_best(episodes=3):
    # UWAGA: dla wizualizacji trzeba ustawić render_mode="human"
    env = gym.make("LunarLander-v3", render_mode="human")

    n_actions = env.action_space.n
    input_dims = int(np.prod(env.observation_space.shape))

    policy = load_policy(MODEL_PATH, n_actions=n_actions, input_dims=input_dims)

    for ep in range(episodes):
        obs, info = env.reset()
        done, score = False, 0.0
        while not done:
            action = greedy_action(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            env.render()  # (często zbędne przy render_mode="human", ale nie szkodzi)
        print(f"Episode {ep}: score {score:.2f}")

    env.close()

if __name__ == "__main__":
    watch_best(episodes=3)
