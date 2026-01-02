# Deep Reinforcement Learning Optimization
## Automated Hyperparameter Tuning with Optuna

---

### 1. Project Overview & Objectives
* **Goal**: Create a robust pipeline for training RL agents on Gymnasium environments.
* **Key Challenge**: RL algorithms (DQN, PPO) are hypersensitive to hyperparameters.
* **Solution**: Integrate Stable Baselines 3 with Optuna for automated tuning.
* **Target Environments**:
  - CartPole-v1 (Classic Control, Discrete Action)
  - CarRacing-v3 (Pixel-based, Continuous Action)

---

### 2. Methodology: Algorithms
* **DQN (Deep Q-Network)**:
  - Off-policy, value-based method.
  - Adapted for continuous environments using `DiscreteActionsWrapper`.
  - Optimized for sample efficiency with Replay Buffer.
* **PPO (Proximal Policy Optimization)**:
  - On-policy, policy gradient method.
  - More stable, generally requires less tuning but higher sample count.
  - Uses Advantage Estimation (GAE).
* **A2C (Advantage Actor-Critic)**:
  - Synchronous, deterministic variant of Asynchronous Advantage Actor-Critic.

---

### 3. Methodology: Optuna Integration
* **Optimization Engine**: Optuna
* **Sampler**: Tree-structured Parzen Estimator (TPE).
  - Bayesian optimization method that models p(x|y) and p(y).
  - Focuses search on promising regions of hyperparameter space.
* **Storage**: SQLite database for persistence and analysis.
* **Evaluation Metric**: Mean Reward over evaluation episodes (EvalCallback).

---

### 4. Experiment Setup: Search Space
* **Common Parameters**:
  - Learning Rate: [1e-5, 1e-2] (Log scale)
  - Gamma (Discount Factor): [0.9, 0.9999]
  - Network Architecture: [Tiny, Small, Medium]
  - Batch Size: [16, 32, 64, 128, 256, 512]
* **DQN Specific**:
  - Target Update Interval: [1000, 5000, 10000, 20000]
  - Train Freq & Gradient Steps: [1, 4, 8, 16]
  - Exploration Fraction & Final Epsilon
* **PPO/A2C Specific**:
  - Entropy Coefficient, Value Function Coefficient, Max Grad Norm

---

### 5. Implementation: Optimization Loop

```python
def objective(trial, args):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    ...
    # Run training
    best_reward = train(..., hyperparams=hyperparams)
    return best_reward
    
# Storage
storage_url = f"sqlite:///{args.env}_optuna.db"
study = optuna.create_study(storage=storage_url, direction="maximize")
```

---

### 6. Results: CartPole-v1 (DQN)
* **Experiment Status**: Successful Convergence.
* **Best Parameters Found**:
  - Learning Rate: 2.33e-05
  - Gamma: 0.988
  - Batch Size: 512
  - Target Update Interval: 1000
* **Validation**:
  - Agent consistently achieves maximum reward (500) per episode.
  - Training stability significantly improved over baseline.

*(See generated PDF for Learning Curve plots)*

---

### 7. Implementation Highlights
* **Artifacts Created**:
  - `optimize.py`: General purpose optimization script.
  - `train.py`: Unified training entry point refactored for injection.
* **Reproducibility**:
  - Best parameters saved to JSON.
  - Training seeds fixed for comparison.
  - `run.sh` script automates environment activation and path setup.

---

### 8. Future Directions
* **Scale to CarRacing-v3**:
  - Apply finding from CartPole to pixel-based environment.
  - Use CNN Policy with frame stacking.
* **Parallelization**:
  - Use Optuna's distributed optimization capabilities.
* **Advanced Architectures**:
  - Experiment with Double DQN, Dueling DQN, or SAC.
* **Deployment**:
  - Export ONNX models for web-based inference.
