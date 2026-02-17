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
        if t < self.shift_point:
            # Phase 1: Arm 0 is the "Golden Arm"
            probs = [0.85, 0.40, 0.10]
        else:
            # Phase 2: Arm 0 is a "Trap", Arm 2 is the new "Golden Arm"
            probs = [-0.20, 0.20, 0.85]
        return 1 if np.random.rand() < probs[action] else 0

    def get_optimal_reward(self, t):
        return 0.85 

# =================================================================
# 2. THE SOTA AGENT LINEUP (Optimized)
# =================================================================

class AENAgent:
    """Project AEN: Optimized with Behavioral Settling and Non-Linear Dampening."""
    def __init__(self, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.ego = 0.5
        self.c_drive = 0.5
        # Meta-parameters tuned for rapid settling
        self.alpha_ego = 0.12  
        self.alpha_c = 0.08    

    def choose_action(self):
        # NEW: Nonlinear settling. Squaring C-drive significantly reduces 
        # the 'exploration tax' when the agent is confident.
        epsilon = np.clip(self.c_drive**2, 0.01, 0.5)
        
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        # A. Optimization Layer (Faster learning rate to flush old data)
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.2 * prediction_error
        
        # B. Calibration Layer (Ego)
        surprise = abs(prediction_error)
        self.ego = (1 - self.alpha_ego) * self.ego + self.alpha_ego * (1.0 - surprise)
        
        # C. Regulatory Layer (C-drive) with Confidence Threshold
        target_c = 1.0 - self.ego
        # If Ego is high (confident), we actively dampen the adaptation pressure
        if self.ego > 0.8:
            target_c *= 0.5 
            
        self.c_drive = (1 - self.alpha_c) * self.c_drive + self.alpha_c * target_c

class ThompsonSampling:
    """The Bayesian SOTA: Excellent in static environments, high inertia."""
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
    """Sliding-Window UCB: Jittery but fast-forgetting."""
    def __init__(self, n_actions=3, window=100):
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
# 3. RESEARCH TOURNAMENT
# =================================================================

def run_optimized_tournament():
    steps, shift = 1000, 400
    env = CompetitiveEnv(shift)
    
    agents = {
        "Project AEN (Settling)": AENAgent(),
        "Thompson Sampling": ThompsonSampling(),
        "Sliding-Window UCB": SW_UCBAgent(window=120)
    }
    
    accuracy = {name: [] for name in agents}
    regret = {name: [0] for name in agents}

    for t in range(steps):
        best_arm = 0 if t < shift else 2
        
        for name, agent in agents.items():
            if "UCB" in name: act = agent.choose_action(t)
            else: act = agent.choose_action()
            
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            # Accuracy Metric
            accuracy[name].append(1 if act == best_arm else 0)
            
            # Regret Metric (Penalty for not choosing the best arm)
            loss = env.get_optimal_reward(t) - (0.85 if act == best_arm else 0.2)
            regret[name].append(regret[name][-1] + max(0, loss))

    # =================================================================
    # 4. VISUALIZATION
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for name in agents:
        ax1.plot(pd.Series(accuracy[name]).rolling(50).mean(), label=name, linewidth=2)
    ax1.axvline(x=shift, color='red', linestyle='--', label='Regime Shift')
    ax1.set_title("Behavioral Accuracy: Recovery Speed Post-Shift", fontsize=14)
    ax1.set_ylabel("Optimal Choice Probability")
    ax1.legend()

    for name in agents:
        ax2.plot(regret[name], label=name, linewidth=2.5)
    ax2.set_title("Cumulative Regret: Total Payout Loss (Lower is Better)", fontsize=14)
    ax2.set_ylabel("Total Regret Value")
    ax2.set_xlabel("Steps")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_optimized_tournament()