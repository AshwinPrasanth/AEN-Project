import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Non-Stationary Stochastic Game
# =================================================================
class CompetitiveEnv:
    """
    A 3-armed bandit environment that undergoes a 'Regime Shift'.
    Designed to test if agents can detect and regulate after failure.
    """
    def __init__(self, shift_point=250):
        self.shift_point = shift_point
        
    def get_reward(self, action, t):
        if t < self.shift_point:
            # Phase 1: Arm 0 is superior (80% success)
            probs = [0.8, 0.4, 0.2]
        else:
            # Phase 2: Arm 0 becomes a 'Trap', Arm 2 becomes superior
            probs = [-0.2, 0.2, 0.8]
        
        return 1 if np.random.rand() < probs[action] else 0

# =================================================================
# 2. THE AEN AGENT: Adaptive Ego Network
# =================================================================
class AENAgent:
    def __init__(self, n_actions=3, alpha_ego=0.08, alpha_c=0.08):
        self.q_values = np.zeros(n_actions)
        self.n_actions = n_actions
        
        # Internal Regulatory Signals
        self.ego = 0.5         # Calibration: Alignment of expectations
        self.c_drive = 0.5     # Regulation: Adaptation/Exploration pressure
        
        # Meta-parameters (Smoothing constants)
        self.alpha_ego = alpha_ego
        self.alpha_c = alpha_c
        
    def choose_action(self):
        # C-drive modulates the Epsilon-Greedy exploration
        epsilon = np.clip(self.c_drive, 0.05, 0.9)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        # A. Optimization Layer (TD Learning)
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * prediction_error
        
        # B. Calibration Layer (Ego)
        # Ego is high when surprise is low (the agent knows the world)
        surprise = abs(prediction_error)
        target_ego = 1.0 - surprise
        self.ego = (1 - self.alpha_ego) * self.ego + self.alpha_ego * target_ego
        
        # C. Regulatory Layer (C-drive)
        # Inversely coupled to Ego. Low Ego -> High C-drive (Search mode)
        target_c = 1.0 - self.ego
        self.c_drive = (1 - self.alpha_c) * self.c_drive + self.alpha_c * target_c

# =================================================================
# 3. THE BASELINE: Brittle Pattern Matcher (LLM Mock)
# =================================================================
class BrittleAgent:
    """Models the pathology of over-adaptation without regulation."""
    def __init__(self, memory_window=15):
        self.memory = []
        self.window = memory_window
        
    def choose_action(self):
        if len(self.memory) < self.window:
            return np.random.randint(3)
        # Greedily follows the most frequent successful action in memory
        return max(set(self.memory), key=self.memory.count)

    def learn(self, action, reward):
        if reward > 0:
            self.memory.append(action)
        if len(self.memory) > self.window:
            self.memory.pop(0)

# =================================================================
# 4. SIMULATION AND VISUALIZATION
# =================================================================
def run_experiment():
    total_steps = 600
    shift_at = 300
    env = CompetitiveEnv(shift_point=shift_at)
    aen = AENAgent()
    brittle = BrittleAgent()
    
    results = []

    for t in range(total_steps):
        # Agents act and learn
        a_act = aen.choose_action()
        a_rew = env.get_reward(a_act, t)
        aen.learn(a_act, a_rew)
        
        b_act = brittle.choose_action()
        b_rew = env.get_reward(b_act, t)
        brittle.learn(b_act, b_rew)
        
        results.append({
            'step': t,
            'aen_ego': aen.ego,
            'aen_c': aen.c_drive,
            'aen_reward': a_rew,
            'brittle_reward': b_rew
        })

    df = pd.DataFrame(results)
    
    # Plotting the Evidence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Internal AEN Regulation
    ax1.plot(df['step'], df['aen_ego'], label='Ego (Calibration)', color='blue', linewidth=2)
    ax1.plot(df['step'], df['aen_c'], label='C-Drive (Adaptation Pressure)', color='orange', linestyle='--')
    ax1.axvline(x=shift_at, color='red', alpha=0.5, label='Regime Shift')
    ax1.set_title("Internal AEN Regulatory States during Environmental Collapse", fontsize=14)
    ax1.set_ylabel("Signal Amplitude")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Performance Comparison (Rolling Average)
    ax2.plot(df['aen_reward'].rolling(30).mean(), label='AEN (Regulated)', color='green')
    ax2.plot(df['brittle_reward'].rolling(30).mean(), label='Brittle (Unregulated)', color='grey', alpha=0.6)
    ax2.axvline(x=shift_at, color='red', alpha=0.5)
    ax2.set_title("Performance Stability: Regulated vs. Unregulated", fontsize=14)
    ax2.set_ylabel("Mean Reward (Rolling Window)")
    ax2.set_xlabel("Episode Step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()