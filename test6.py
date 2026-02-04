import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE TOUGH ENVIRONMENT: Chaotic Multi-Regime Bandit
# =================================================================
class ChaosEnv:
    def get_reward(self, action, t):
        if t < 300: # Stable Phase
            probs = [0.80, 0.35, 0.15]
        elif 300 <= t < 700: # The Flicker (Fast Oscillation)
            if (t // 50) % 2 == 0:
                probs = [0.15, 0.75, 0.20]
            else:
                probs = [0.15, 0.20, 0.75]
        else: # The Fade (Low Signal)
            probs = [0.45, 0.40, 0.38]
        return 1 if np.random.rand() < probs[action] else 0

    def get_optimal_reward(self, t):
        if t < 300: return 0.80
        if 300 <= t < 700: return 0.75
        return 0.45

# =================================================================
# 2. THE ABLATION AGENT CLASS
# =================================================================
class AblationAgent:
    def __init__(self, mode, n_actions=3):
        self.q_values = np.zeros(n_actions)
        self.mode = mode
        self.action_history = []
        
        configs = {
            "AEN (Original Gem)": (0.5, 0.5),
            "High Ego (Arrogant)": (0.95, 0.05),
            "Low Ego (Paranoid)": (0.10, 0.05),
            "Low C-Drive (Lazy)": (0.5, 0.05),
            "High C-Drive (Chaotic)": (0.5, 0.80),
            "High Ego + High C": (0.95, 0.80),
            "Low Ego + High C": (0.10, 0.80)
        }
        self.ego, self.c_drive = configs[mode]

    def choose_action(self):
        epsilon = np.clip(self.c_drive**2, 0.01, 0.5)
        if np.random.rand() < epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(self.q_values)
        self.action_history.append(action)
        return action

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += 0.25 * error
        
        if self.mode == "AEN (Original Gem)":
            self.ego = 0.85 * self.ego + 0.15 * (1.0 - abs(error))
            target_c = 1.0 - self.ego
            if self.ego > 0.85: target_c *= 0.3 
            self.c_drive = 0.90 * self.c_drive + 0.10 * target_c

# =================================================================
# 3. RUNNING AND PLOTTING STABILITY
# =================================================================
def run_full_analysis():
    steps = 1000
    env = ChaosEnv()
    modes = ["AEN (Original Gem)", "High Ego (Arrogant)", "Low Ego (Paranoid)", 
             "Low C-Drive (Lazy)", "High C-Drive (Chaotic)"]
    
    agents = {m: AblationAgent(m) for m in modes}
    regret = {name: [0] for name in agents}
    stability = {name: [] for name in agents}

    for t in range(steps):
        if t < 300: best = 0
        elif 300 <= t < 700: best = 1 if (t // 50) % 2 == 0 else 2
        else: best = 0

        for name, agent in agents.items():
            act = agent.choose_action()
            rew = env.get_reward(act, t)
            agent.learn(act, rew)
            
            # Record Regret
            loss = env.get_optimal_reward(t) - (env.get_optimal_reward(t) if act == best else 0.1)
            regret[name].append(regret[name][-1] + max(0, loss))
            
            # Calculate Stability: Inverse of action-switching rate over 20-step window
            if len(agent.action_history) > 20:
                recent = agent.action_history[-20:]
                switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
                # Stability = 1.0 (No switches) to 0.0 (Constantly switching)
                stability[name].append(1.0 - (switches / 19.0))
            else:
                stability[name].append(1.0)

    # PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. Cumulative Regret
    for name in modes:
        ax1.plot(regret[name], label=name, linewidth=3 if "Gem" in name else 1.5)
    ax1.set_title("Total Loss Comparison (Regret)")
    ax1.legend()

    # 2. Stability Plot
    for name in modes:
        ax2.plot(pd.Series(stability[name]).rolling(30).mean(), label=name, 
                 linewidth=3 if "Gem" in name else 1.5)
    ax2.set_title("Behavioral Stability Index (Higher = More Composed)")
    ax2.set_ylabel("Stability (1 - Switching Rate)")
    ax2.set_xlabel("Steps")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_analysis()