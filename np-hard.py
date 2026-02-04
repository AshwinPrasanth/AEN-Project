import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Real-World Unpredictability
# =================================================================
class RealWorldUnpredictableEnv:
    """
    Simulates high-risk real-world dynamics:
    - Budget Recharge: Continuous supply of potential change.
    - Black Swan Events: Radical regime shifts at intervals.
    - Continuous Volatility: Background drift that consumes budget.
    """
    def __init__(self, K=5, horizon=10000, seed=None):
        if seed is not None: np.random.seed(seed)
        self.K = K
        self.T = horizon
        self.mu = np.random.uniform(0.3, 0.7, size=K)
        self.v_budget = 1.0
        self.recharge_rate = 0.005 
        self.best_arm = np.argmax(self.mu)

    def _apply_physics(self, t):
        # Environment recovers its 'energy' to change over time
        self.v_budget = min(5.0, self.v_budget + self.recharge_rate)
        
        # Scenario: Sudden Black Swan Event (Every 2500 steps)
        if t > 0 and t % 2500 == 0:
            self.mu = np.random.uniform(0.1, 0.9, size=self.K)
            self.v_budget = 0 # Event consumes current budget
        else:
            # Persistent stochastic drift
            drift = np.random.normal(0, 0.01 * self.v_budget, size=self.K)
            self.mu = np.clip(self.mu + drift, 0.05, 0.95)
            self.v_budget -= np.sum(np.abs(drift))
            
        self.best_arm = np.argmax(self.mu)

    def step(self, action, t):
        self._apply_physics(t)
        # Probabilistic reward based on current arm means
        reward = 1 if np.random.rand() < self.mu[action] else 0
        return reward, self.best_arm, np.max(self.mu)

# =================================================================
# 2. THE AGENT LINEUP
# =================================================================

class FABAgent:
    """Proposed: Adaptive Ego Network (FAB Model)."""
    def __init__(self, n_actions=5):
        self.q_values = np.zeros(n_actions)
        self.phi = 0.5         # Fidelity (Internal sensor)
        self.lambda_p = 0.5    # Adaptation Pressure (Regulator)
        self.alpha = 0.25      # Learning Rate

    def choose_action(self):
        # Balance (B): Automated exploration threshold
        epsilon = np.clip(self.lambda_p**2, 0.01, 0.45)
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.q_values))
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += self.alpha * error
        
        # Fidelity (Phi) Sensing: Measures model reliability
        self.phi = 0.85 * self.phi + 0.15 * (1.0 - abs(error))
        
        # Adaptation Pressure (Lambda) Regulation
        target_lambda = 1.0 - self.phi
        if self.phi > 0.85: target_lambda *= 0.15 # Settling/Balance
        self.lambda_p = 0.9 * self.lambda_p + 0.1 * target_lambda

class ThompsonSamplingAgent:
    """Standard Bayesian Thompson Sampling (TS)."""
    def __init__(self, n_actions=5):
        self.alphas = np.ones(n_actions)
        self.betas = np.ones(n_actions)

    def choose_action(self):
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(len(self.alphas))]
        return np.argmax(samples)

    def learn(self, action, reward):
        if reward: self.alphas[action] += 1
        else: self.betas[action] += 1

class SW_UCB_Agent:
    """Sliding-Window Upper Confidence Bound (SW-UCB)."""
    def __init__(self, n_actions=5, window=250):
        self.n_actions = n_actions
        self.window = window
        self.history = []

    def choose_action(self, t):
        if t < self.n_actions: return t
        recent = self.history[-self.window:]
        counts = np.bincount([h[0] for h in recent], minlength=self.n_actions)
        rewards = np.zeros(self.n_actions)
        for act, rew in recent: rewards[act] += rew
        
        ucb_values = []
        for i in range(self.n_actions):
            if counts[i] == 0: ucb_values.append(1e6)
            else:
                avg = rewards[i] / counts[i]
                explore = np.sqrt(2 * np.log(min(t, self.window)) / counts[i])
                ucb_values.append(avg + explore)
        return np.argmax(ucb_values)

    def learn(self, action, reward):
        self.history.append((action, reward))

class EXP3Agent:
    """Adversarial bandit algorithm (EXP3)."""
    def __init__(self, n_actions=5, gamma=0.1):
        self.w = np.ones(n_actions)
        self.gamma = gamma

    def choose_action(self):
        p = (1-self.gamma)*(self.w/np.sum(self.w)) + (self.gamma/len(self.w))
        return np.random.choice(len(self.w), p=p)

    def learn(self, action, reward):
        p = (1-self.gamma)*(self.w[action]/np.sum(self.w)) + (self.gamma/len(self.w))
        self.w[action] *= np.exp(self.gamma * (reward/p) / len(self.w))

# =================================================================
# 3. TOURNAMENT RUNNER
# =================================================================
def run_simulation(horizon=10000):
    agents_map = {
        "FAB Model (Proposed)": FABAgent(),
        "Thompson Sampling": ThompsonSamplingAgent(),
        "SW-UCB (Window=250)": SW_UCB_Agent(),
        "EXP3": EXP3Agent()
    }
    
    results = {name: {"regret": [0], "acc": []} for name in agents_map}

    for name, agent in agents_map.items():
        env = RealWorldUnpredictableEnv(horizon=horizon, seed=42)
        for t in range(horizon):
            if "UCB" in name: act = agent.choose_action(t)
            else: act = agent.choose_action()
            
            reward, best, max_mu = env.step(act, t)
            agent.learn(act, reward)
            
            results[name]["acc"].append(1 if act == best else 0)
            results[name]["regret"].append(results[name]["regret"][-1] + (max_mu - env.mu[act]))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    for name in agents_map:
        style = {'linewidth': 4 if "FAB" in name else 2, 'alpha': 0.8}
        ax1.plot(pd.Series(results[name]["acc"]).rolling(200).mean(), label=name, **style)
        ax2.plot(results[name]["regret"], label=name, **style)

    ax1.set_title("Strategy Robustness: Accuracy during Black Swan Events", fontsize=16)
    ax1.set_ylabel("Rolling Optimal Action Prob.")
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.2)

    ax2.set_title("Operational Efficiency: Cumulative Regret (10k Step Marathon)", fontsize=16)
    ax2.set_ylabel("Total Loss")
    ax2.set_xlabel("Steps")
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()