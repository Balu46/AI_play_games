import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt

SAVE_BEST_MODEL = True
LOAD_BEST_MODEL = True

class DeepQNetwork(nn.Module):
    """Implementacja sieci neuronowej DQN do uczenia przez wzmacnianie.
    
    Atrybuty:
        input_dims (int): Wymiar wejścia sieci
        fc1_dims (int): Rozmiar pierwszej warstwy ukrytej
        fc2_dims (int): Rozmiar drugiej warstwy ukrytej
        n_actions (int): Liczba możliwych akcji
        device (torch.device): Urządzenie do obliczeń (CPU/GPU)
    """
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    """Agent wykorzystujący DQN do podejmowania decyzji w sieci sensorowej.
    
    Atrybuty:
         (gammafloat): Współczynnik dyskontowania
        epsilon (float): Współczynnik eksploracji
        lr (float): Szybkość uczenia
        action_space (list): Przestrzeń możliwych akcji
        mem_size (int): Rozmiar pamięci doświadczeń
        Q_eval (DeepQNetwork): Główna sieć Q
        best_Q_eval (DeepQNetwork): Najlepsza sieć Q
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=1000000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.path = 'best_model.pth'

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, 
                                 input_dims=input_dims, 
                                 fc1_dims=256, fc2_dims=256)
        
        self.best_Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, 
                                 input_dims=input_dims, 
                                 fc1_dims=256, fc2_dims=256)


        if os.path.exists(self.path) and LOAD_BEST_MODEL:
            self.Q_eval.load_state_dict(T.load(self.path, map_location=self.Q_eval.device))
            self.best_Q_eval.to(self.Q_eval.device)
            print(f"Model załadowany z pliku: {self.path}")
        else:
            print("Brak zapisanego modelu – uruchamianie od zera.")

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # def choose_action(self, observation):
    #     """Wybierz akcję na podstawie obserwacji (ε-zachłannie).
    
    #     Args:
    #         observation: Aktualny stan środowiska
        
    #     Returns:
    #         int: Wybrana akcja (indeks sensora do uśpienia)
    #     """
    #     if np.random.random() > self.epsilon:
    #         state = T.tensor([observation]).to(self.Q_eval.device)
    #         actions = self.Q_eval.forward(state)
    #         action = T.argmax(actions).item()
    #     else:
    #         action = np.random.choice(self.action_space)
    #     return action
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float32, device=self.Q_eval.device).unsqueeze(0)
            actions = self.Q_eval(state)
            action = T.argmax(actions, dim=1).item()
        else:
            action = np.random.choice(self.action_space)
        return action


    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                      else self.eps_min
                      
    def update_best_model(self):
        self.best_Q_eval.load_state_dict(self.Q_eval.state_dict())
        
    def save_best_model(self, filename):
        if SAVE_BEST_MODEL:
            T.save(self.best_Q_eval.state_dict(), filename)
