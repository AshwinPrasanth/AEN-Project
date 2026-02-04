import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Non-Stationary "Shift" Bandit
# =================================================================
class CompetitiveEnv:
    def __init__(self, shift_point=400):
        self.shift_point = shift_point
        
    def get_reward(self, action, t):
        if t < self.shift_point:
            # Phase 1: Arm 0 is the "Golden Arm"
            probs = [0.85, 0.40, 0.10]
        else:
            # Phase 2: Arm 0 is a "Trap", Arm 2 is the new "Golden Arm"
            probs = [-0.20, 0.20, 0.85]
        return 1 if np.random.rand() < probs[action] else 0

# =================================================================
# 2. THE AGENT GAUNTLET (The Challengers)
# =================================================================

class AENAgent:
    """Project AEN: Regulated by Ego and Competition Drive."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.ego = 0.5
        self.c_drive = 0.5
        self.alpha = 0.08 # Smoothing factor

    def choose_action(self):
        epsilon = np.clip(self.c_drive, 0.05, 0.9)
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * error
        
        # Ego reflects calibration (surprise-based)
        surprise = abs(error)
        self.ego = (1 - self.alpha) * self.ego + self.alpha * (1.0 - surprise)
        # Inverse Coupling
        self.c_drive = (1 - self.alpha) * self.c_drive + self.alpha * (1.0 - self.ego)

class UCBAgent:
    """The 'Mathematical Optimal': Uses Upper Confidence Bound."""
    def __init__(self, n_actions=3):
        self.counts = np.zeros(n_actions)
        self.values = np.zeros(n_actions)

    def choose_action(self, t):
        if 0 in self.counts: return np.argmin(self.counts)
        # Sharpness comes from the sqrt bonus
        bonus = np.sqrt((2 * np.log(t + 1)) / (self.counts + 1e-5))
        return np.argmax(self.values + bonus)

    def learn(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

class DecayEpsilonAgent:
    """The 'Standard RL': Epsilon-Greedy with time-decay."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.epsilon = 1.0

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        self.q_values[action] += 0.1 * (reward - self.q_values[action])
        self.epsilon *= 0.992 # Slowly stops exploring

class ReactiveAgent:
    """The 'Fast Learner': High learning rate, no regulation."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)

    def choose_action(self):
        # Fixed low exploration
        if np.random.rand() < 0.1: return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        # Sharpness: Very high learning rate to react fast
        self.q_values[action] += 0.5 * (reward - self.q_values[action])

# =================================================================
# 3. TOURNAMENT EXECUTION
# =================================================================

def run_research_tournament():
    steps = 1000
    shift = 400
    env = CompetitiveEnv(shift_point=shift)
    
    agents = {
        "AEN (Regulated)": AENAgent(),
        "UCB (Optimal Math)": UCBAgent(),
        "Decay-Epsilon (Standard)": DecayEpsilonAgent(),
        "Reactive (High-Alpha)": ReactiveAgent()
    }
    
    # Storage for analysis
    results = {name: [] for name in agents}
    regret = {name: [0] for name in agents}

    for t in range(steps):
        for name, agent in agents.items():
            # Action
            if name == "UCB (Optimal Math)":
                act = agent.choose_action(t)
            else:
                act = agent.choose_action()
            
            # Reward
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            # Record Success (1 if hit best arm, else 0)
            best_arm = 0 if t < shift else 2
            results[name].append(1 if act == best_arm else 0)
            
            # Record Cumulative Regret (Total missed opportunities)
            current_regret = (1.0 if t < shift else 0.85) - (0.85 if act == best_arm else (0.4 if act == 1 else -0.2))
            regret[name].append(regret[name][-1] + max(0, current_regret))

    # =================================================================
    # 4. VISUALIZATION OF FINDINGS
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Policy Stability (Rolling Accuracy)
    for name in agents:
        smooth_acc = pd.Series(results[name]).rolling(50).mean()
        ax1.plot(smooth_acc, label=name, linewidth=2)
    
    ax1.axvline(x=shift, color='red', linestyle='--', alpha=0.7, label='Regime Shift')
    ax1.set_title("Behavioral Stability: Accuracy in Identifying the 'Golden Arm'", fontsize=14)
    ax1.set_ylabel("Probability of Optimal Choice")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Cumulative Regret (The Research Proof)
    for name in agents:
        ax2.plot(regret[name], label=name, linewidth=2)
    
    ax2.set_title("Cumulative Regret: Total Loss over Long-Horizon", fontsize=14)
    ax2.set_ylabel("Total Regret")
    ax2.set_xlabel("Steps")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_research_tournament()