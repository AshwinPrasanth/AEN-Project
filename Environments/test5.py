import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE TOUGH ENVIRONMENT: Chaotic Multi-Regime Bandit
# =================================================================
class ChaosEnv:
    """
    A brutal environment featuring rapid switches and low-signal phases.
    """
    def get_reward(self, action, t):
        # Phase 1: Stable (0-300) - Arm 0 is superior
        if t < 300:
            probs = [0.80, 0.35, 0.15]
        # Phase 2: The Flicker (300-700) - Rapid 50-step oscillations
        elif 300 <= t < 700:
            if (t // 50) % 2 == 0:
                probs = [0.15, 0.75, 0.20] # Arm 1 is best
            else:
                probs = [0.15, 0.20, 0.75] # Arm 2 is best
        # Phase 3: The Fade (700-1000) - High noise, low separation
        else:
            probs = [0.45, 0.40, 0.38] # Arm 0 is marginally better
            
        return 1 if np.random.rand() < probs[action] else 0

    def get_optimal_reward(self, t):
        if t < 300: return 0.80
        if 300 <= t < 700: return 0.75
        return 0.45

# =================================================================
# 2. THE AGENT LINEUP
# =================================================================

class AENAgent:
    """AEN with Chaos-Tuning: High sensitivity and rapid settling."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.ego = 0.5
        self.c_drive = 0.5
        self.alpha_ego = 0.15  # Increased for faster detection of 'The Flicker'
        self.alpha_c = 0.10

    def choose_action(self):
        # Non-linear settling logic
        epsilon = np.clip(self.c_drive**2, 0.01, 0.45)
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        # Rapid learning rate to pivot during oscillations
        error = reward - self.q_values[action]
        self.q_values[action] += 0.25 * error
        
        # Ego-Calibration Layer
        surprise = abs(error)
        self.ego = (1 - self.alpha_ego) * self.ego + self.alpha_ego * (1.0 - surprise)
        
        # Regulatory Layer: If ego is low (chaos), search pressure stays high
        target_c = 1.0 - self.ego
        if self.ego > 0.85: # Threshold for 'Confidence'
            target_c *= 0.3 # Deep settling for stable phases
            
        self.c_drive = (1 - self.alpha_c) * self.c_drive + self.alpha_c * target_c

class ThompsonSampling:
    """Bayesian Benchmark: Struggles with rapid non-stationarity."""
    def __init__(self, n_actions=3):
        self.alphas = np.ones(n_actions)
        self.betas = np.ones(n_actions)

    def choose_action(self):
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(3)]
        return np.argmax(samples)

    def learn(self, action, reward):
        if reward > 0: self.alphas[action] += 1
        else: self.betas[action] += 1

class SW_UCBAgent:
    """Sliding-Window UCB: Competitive but jittery."""
    def __init__(self, n_actions=3, window=75):
        self.window = window
        self.history = []

    def choose_action(self, t):
        if t < 3: return t
        recent = self.history[-self.window:]
        counts = np.array([sum(1 for a, r in recent if a == i) for i in range(3)])
        values = np.array([sum(r for a, r in recent if a == i)/(c + 1e-5) for i, c in enumerate(counts)])
        bonus = np.sqrt((2 * np.log(min(t, self.window))) / (counts + 1e-5))
        return np.argmax(values + bonus)

    def learn(self, action, reward):
        self.history.append((action, reward))

# =================================================================
# 3. TOURNAMENT EXECUTION
# =================================================================

def run_chaos_tournament():
    steps = 1000
    env = ChaosEnv()
    agents = {
        "AEN (Chaos Tuned)": AENAgent(),
        "Thompson Sampling": ThompsonSampling(),
        "Sliding-Window UCB": SW_UCBAgent(window=75)
    }
    
    accuracy = {name: [] for name in agents}
    regret = {name: [0] for name in agents}

    for t in range(steps):
        # Tracking current best arm for the Accuracy metric
        if t < 300: best = 0
        elif 300 <= t < 700: best = 1 if (t // 50) % 2 == 0 else 2
        else: best = 0

        for name, agent in agents.items():
            act = agent.choose_action(t) if "UCB" in name else agent.choose_action()
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            accuracy[name].append(1 if act == best else 0)
            loss = env.get_optimal_reward(t) - (0.80 if act == best else 0.1) # Weighted penalty
            regret[name].append(regret[name][-1] + max(0, loss))

    # =================================================================
    # 4. VISUALIZATION
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for name in agents:
        ax1.plot(pd.Series(accuracy[name]).rolling(30).mean(), label=name, linewidth=2)
    ax1.axvline(x=300, color='grey', linestyle='--', alpha=0.5)
    ax1.axvline(x=700, color='grey', linestyle='--', alpha=0.5)
    ax1.set_title("Chaos Tournament: Accuracy in Oscillating/Noisy Conditions", fontsize=14)
    ax1.set_ylabel("Optimal Choice Probability")
    ax1.legend()

    for name in agents:
        ax2.plot(regret[name], label=name, linewidth=2.5)
    ax2.set_title("Cumulative Regret: Survival in Chaotic Environments (Lower is Better)", fontsize=14)
    ax2.set_ylabel("Regret Value")
    ax2.set_xlabel("Steps")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_chaos_tournament()