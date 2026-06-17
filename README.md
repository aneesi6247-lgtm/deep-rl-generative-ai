# Deep Reinforcement Learning & Generative AI

Two from-scratch PyTorch projects: a Deep Q-Network (DQN) agent trained to play Flappy Bird, and a Generative Adversarial Network (GAN) trained to synthesize faces from the CelebA dataset.

## 🎮 DQN Agent — Flappy Bird

A Deep Q-Network implemented from scratch with the core ingredients of stable DQN training:

- **Epsilon-greedy exploration** with configurable decay schedule
- **Experience replay buffer** (FIFO, via `collections.deque`) for decorrelated minibatch sampling
- **Target network synchronization** to stabilize Q-value targets
- **Bellman equation optimization** with MSE loss between predicted and target Q-values

Trained on [`flappy_bird_gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium) (OpenAI Gymnasium-compatible environment). The agent surpassed human-level scores within 500 training episodes.

**Files:**
- `dqn.py` — the Q-network architecture (3-layer MLP)
- `Experience_replay.py` — replay buffer implementation
- `agent_2.py` — training/inference loop, hyperparameter loading from YAML, CLI entry point

**Run training:**
```bash
python agent_2.py <hyperparameter_set> --train
```

**Run a trained policy:**
```bash
python agent_2.py <hyperparameter_set>
```

Hyperparameters (learning rate, epsilon schedule, replay buffer size, batch size, sync rate, gamma, reward threshold) are configured per named set in `parameters.yaml`.

## 🎨 GAN — Face Synthesis on CelebA

A generator/discriminator pair trained adversarially on 30,000+ images from CelebA to produce synthetic face images.

- Discriminator: fully-connected network with LeakyReLU activations, sigmoid output (real/fake probability)
- Adversarial training with binary cross-entropy (BCE) loss
- Trained over 20 epochs with Adam optimizers (separate for generator and discriminator)

**File:** `GAN.ipynb`

## Stack

PyTorch · OpenAI Gymnasium · NumPy · Jupyter
