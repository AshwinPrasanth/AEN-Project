import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Non-Stationary "Shifting Gold" Bandit
# =================================================================
class CompetitiveEnv:
    def __init__(self, shift_point=400):
        self.shift_point = shift_point
        
    def get_reward(self, action, t):
        # Phase 1: Arm 0 is optimal
        if t < self.shift_point:
            probs = [0.85, 0.40, 0.10]
        # Phase 2: Arm 2 is optimal, Arm 0 is a penalty trap
        else:
            probs = [-0.20, 0.20, 0.85]
        return 1 if np.random.rand() < probs[action] else 0

    def get_optimal_reward(self, t):
        return 0.85 # The max possible mean reward in any phase

# =================================================================
# 2. THE SOTA AGENT LINEUP
# =================================================================

class AENAgent:
    """Project AEN: Regulated by Ego-Calibration."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.ego = 0.5
        self.c_drive = 0.5
        self.alpha = 0.08

    def choose_action(self):
        epsilon = np.clip(self.c_drive, 0.05, 0.9)
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * error
        # Ego crashes on surprise; C-drive spikes to compensate
        self.ego = (1-self.alpha)*self.ego + self.alpha*(1.0 - abs(error))
        self.c_drive = (1-self.alpha)*self.c_drive + self.alpha*(1.0 - self.ego)

class ThompsonSampling:
    """The Bayesian SOTA: Uses Beta Distributions."""
    def __init__(self, n_actions=3):
        self.alphas = np.ones(n_actions) # Successes
        self.betas = np.ones(n_actions)  # Failures

    def choose_action(self):
        # Sample from the posterior distribution of each arm
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(3)]
        return np.argmax(samples)

    def learn(self, action, reward):
        if reward > 0: self.alphas[action] += 1
        else: self.betas[action] += 1

class SW_UCBAgent:
    """Sliding-Window UCB: Specifically for non-stationary environments."""
    def __init__(self, n_actions=3, window=100):
        self.window = window
        self.history = []

    def choose_action(self, t):
        if t < 3: return t # Initial exploration
        recent = self.history[-self.window:]
        counts = np.array([sum(1 for a, r in recent if a == i) for i in range(3)])
        values = np.array([sum(r for a, r in recent if a == i)/(c + 1e-5) for i, c in enumerate(counts)])
        # Exploration bonus based only on the window size
        bonus = np.sqrt((2 * np.log(min(t, self.window))) / (counts + 1e-5))
        return np.argmax(values + bonus)

    def learn(self, action, reward):
        self.history.append((action, reward))

# =================================================================
# 3. RESEARCH TOURNAMENT EXECUTION
# =================================================================

def run_sota_tournament():
    steps, shift = 1000, 400
    env = CompetitiveEnv(shift)
    
    agents = {
        "Project AEN": AENAgent(),
        "Thompson Sampling (Bayesian)": ThompsonSampling(),
        "Sliding-Window UCB": SW_UCBAgent(window=120)
    }
    
    # Metrics
    accuracy = {name: [] for name in agents}
    regret = {name: [0] for name in agents}

    for t in range(steps):
        best_arm = 0 if t < shift else 2
        
        for name, agent in agents.items():
            # Act
            if "UCB" in name: act = agent.choose_action(t)
            else: act = agent.choose_action()
            
            # Reward and Learn
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            # Record Accuracy
            accuracy[name].append(1 if act == best_arm else 0)
            
            # Record Cumulative Regret
            loss = env.get_optimal_reward(t) - (0.85 if act == best_arm else 0.2)
            regret[name].append(regret[name][-1] + max(0, loss))

    # =================================================================
    # 4. DATA VISUALIZATION
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Recovery Speed (Accuracy)
    for name in agents:
        ax1.plot(pd.Series(accuracy[name]).rolling(50).mean(), label=name, linewidth=2)
    ax1.axvline(x=shift, color='red', linestyle='--', label='Regime Shift')
    ax1.set_title("SOTA Accuracy Comparison: Identifying the New Optimal Arm", fontsize=14)
    ax1.set_ylabel("Optimal Choice Probability")
    ax1.legend()

    # Plot 2: Cumulative Regret (Efficiency)
    for name in agents:
        ax2.plot(regret[name], label=name, linewidth=2.5)
    ax2.set_title("Cumulative Regret: Total Payout Loss", fontsize=14)
    ax2.set_ylabel("Regret Value")
    ax2.set_xlabel("Steps")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sota_tournament()