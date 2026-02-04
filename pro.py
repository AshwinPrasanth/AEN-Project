import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Adversarial Multi-Regime Gauntlet
# =================================================================
class MetaGauntletEnv:
    """
    A complex environment featuring three distinct failure-inducing phases:
    1. Stable (0-400): High reward signal for Arm 0.
    2. The Flicker (400-1000): Rapid 50-step oscillations (Adversarial).
    3. The Fade (1000-1500): Low signal-to-noise ratio, minimal arm separation.
    """
    def __init__(self, steps=1500):
        self.steps = steps

    def get_reward(self, action, t):
        if t < 400: # Phase 1: Stable
            probs = [0.85, 0.30, 0.15]
        elif 400 <= t < 1000: # Phase 2: Adversarial Flicker
            probs = [0.10, 0.80, 0.20] if (t // 50) % 2 == 0 else [0.10, 0.20, 0.80]
        else: # Phase 3: The Fade
            probs = [0.45, 0.42, 0.40] 
        return 1 if np.random.rand() < probs[action] else 0

    def get_optimal_reward(self, t):
        if t < 400: return 0.85
        if 400 <= t < 1000: return 0.80
        return 0.45

# =================================================================
# 2. AGENT DEFINITIONS
# =================================================================

class AENAgent:
    """Proposed: Adaptive Ego Network."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.ego = 0.5
        self.c_drive = 0.5

    def choose_action(self):
        # Exploration probability regulated by C-Drive
        epsilon = np.clip(self.c_drive**2, 0.01, 0.5)
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += 0.25 * error
        
        # Internal Regulation
        self.ego = 0.85 * self.ego + 0.15 * (1.0 - abs(error))
        target_c = 1.0 - self.ego
        if self.ego > 0.85: target_c *= 0.2 # Active Settling
        self.c_drive = 0.90 * self.c_drive + 0.10 * target_c

class EXP3Agent:
    """Exponential Weight algorithm for adversarial settings."""
    def __init__(self, n_actions=3, gamma=0.1):
        self.weights = np.ones(n_actions)
        self.gamma = gamma

    def choose_action(self):
        w_sum = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / w_sum) + (self.gamma / 3)
        return np.random.choice(3, p=probs)

    def learn(self, action, reward):
        w_sum = np.sum(self.weights)
        prob = (1 - self.gamma) * (self.weights[action] / w_sum) + (self.gamma / 3)
        estimated_reward = reward / prob
        self.weights[action] *= np.exp(self.gamma * estimated_reward / 3)

class TS_GE_Agent:
    """Thompson Sampling with Group Exploration (Active Probing)."""
    def __init__(self, n_actions=3):
        self.alphas = np.ones(n_actions)
        self.betas = np.ones(n_actions)

    def choose_action(self, t):
        if t % 25 == 0: return np.random.randint(3) # Periodic probe
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(3)]
        return np.argmax(samples)

    def learn(self, action, reward):
        if reward > 0: self.alphas[action] += 1
        else: self.betas[action] += 1

class SW_UCB_Agent:
    """Sliding Window UCB - Standard for non-stationarity."""
    def __init__(self, n_actions=3, window=100):
        self.window = window
        self.history = []
        self.n_actions = n_actions

    def choose_action(self, t):
        if t < self.n_actions: return t
        # Use only window
        recent = self.history[-self.window:]
        counts = np.zeros(self.n_actions)
        rewards = np.zeros(self.n_actions)
        for act, rew in recent:
            counts[act] += 1
            rewards[act] += rew
        
        ucb_values = []
        for i in range(self.n_actions):
            if counts[i] == 0: ucb_values.append(1e6)
            else:
                avg = rewards[i] / counts[i]
                exploration = np.sqrt(2 * np.log(min(t, self.window)) / counts[i])
                ucb_values.append(avg + exploration)
        return np.argmax(ucb_values)

    def learn(self, action, reward):
        self.history.append((action, reward))

class Pessimistic_fDSW_TS:
    """
    f-dsw Thompson Sampling (Pessimistic - f=min).
    Combines Sliding Window memory with Pessimistic (min) sampling rule.
    """
    def __init__(self, n_actions=3, window=100, f_samples=4):
        self.window = window
        self.f_samples = f_samples # How many samples to take for the 'min' rule
        self.history = []
        self.n_actions = n_actions

    def choose_action(self):
        # Calculate stats from sliding window
        recent = self.history[-self.window:]
        alphas = np.ones(self.n_actions)
        betas = np.ones(self.n_actions)
        for act, rew in recent:
            if rew > 0: alphas[act] += 1
            else: betas[act] += 1
        
        # Pessimistic Sampling: take f_samples and use the minimum for each arm
        final_scores = []
        for i in range(self.n_actions):
            samples = [np.random.beta(alphas[i], betas[i]) for _ in range(self.f_samples)]
            final_scores.append(np.min(samples)) # f=min rule
        return np.argmax(final_scores)

    def learn(self, action, reward):
        self.history.append((action, reward))

# =================================================================
# 3. RUNNING THE TOURNAMENT
# =================================================================
def run_tournament():
    steps = 1500
    env = MetaGauntletEnv(steps)
    agents = {
        "AEN (Proposed)": AENAgent(),
        "EXP3": EXP3Agent(),
        "TS-GE": TS_GE_Agent(),
        "SW-UCB": SW_UCB_Agent(),
        "f-dsw TS (Pessimistic)": Pessimistic_fDSW_TS()
    }
    
    regret = {name: [0] for name in agents}
    accuracy = {name: [] for name in agents}

    for t in range(steps):
        # Best arm for accuracy tracking
        if t < 400: best = 0
        elif 400 <= t < 1000: best = 1 if (t // 50) % 2 == 0 else 2
        else: best = 0

        for name, agent in agents.items():
            if name == "TS-GE": act = agent.choose_action(t)
            elif name == "SW-UCB": act = agent.choose_action(t)
            else: act = agent.choose_action()
            
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            accuracy[name].append(1 if act == best else 0)
            loss = env.get_optimal_reward(t) - (env.get_optimal_reward(t) if act == best else 0.1)
            regret[name].append(regret[name][-1] + max(0, loss))

    # =================================================================
    # 4. PLOTTING RESULTS
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, name in enumerate(agents):
        linewidth = 4 if "AEN" in name else 2
        ax1.plot(pd.Series(accuracy[name]).rolling(50).mean(), label=name, linewidth=linewidth, color=colors[i])
    ax1.set_title("Performance: Policy Accuracy across Adversarial Shifts", fontsize=16)
    ax1.set_ylabel("Rolling Optimal Action Probability")
    ax1.axvline(400, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(1000, color='k', linestyle='--', alpha=0.3)
    ax1.legend()
    ax1.grid(alpha=0.3)

    for i, name in enumerate(agents):
        linewidth = 4 if "AEN" in name else 2
        ax2.plot(regret[name], label=name, linewidth=linewidth, color=colors[i])
    ax2.set_title("Efficiency: Cumulative Regret (Lower is Better)", fontsize=16)
    ax2.set_ylabel("Regret")
    ax2.set_xlabel("Steps")
    ax2.axvline(400, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(1000, color='k', linestyle='--', alpha=0.3)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tournament_results.png')
    print("Tournament complete. Image saved.")

run_tournament()