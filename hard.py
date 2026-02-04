import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================================================================
# 1. THE ENVIRONMENT: Level 5 Adversarial "Final Boss"
# =================================================================
class Level5AdversarialBandit:
    """
    Worst-case bounded adversarial non-stationary bandit.
    Actively counters agents based on their Dominance and Jitter.
    """
    def __init__(self, K=5, horizon=2000, variation_budget=6.0, seed=42):
        np.random.seed(seed)
        self.K = K
        self.T = horizon
        self.V = variation_budget
        self.min_gap = 0.08
        self.flicker_gap = 0.15
        self.window = 50
        self.t = 0
        self.remaining_V = variation_budget
        self.mu = np.random.uniform(0.4, 0.6, size=K)
        self.best_arm = np.argmax(self.mu)
        self.action_history = []

    def _agent_statistics(self):
        if len(self.action_history) < self.window: return None, None
        recent = self.action_history[-self.window:]
        counts = np.bincount(recent, minlength=self.K)
        dominance = counts.max() / self.window
        jitter = np.mean(np.diff(recent) != 0)
        return dominance, jitter

    def _update_means(self):
        if self.remaining_V <= 0: return
        dominance, jitter = self._agent_statistics()
        if dominance is None: return

        mu_new = self.mu.copy()
        # Adversarial Logic: Target agent habits
        if dominance > 0.85: # Target Thompson Sampling "Stubbornness"
            new_best = np.random.choice([i for i in range(self.K) if i != self.best_arm])
            mu_new[self.best_arm] -= self.flicker_gap
            mu_new[new_best] += self.flicker_gap
        elif jitter > 0.6: # Target EXP3 "Volatility"
            mu_new += np.random.uniform(-0.03, 0.03, size=self.K)
        else: # Subtle Drift
            mu_new += np.random.uniform(-0.015, 0.015, size=self.K)

        delta = np.sum(np.abs(mu_new - self.mu))
        if delta <= self.remaining_V:
            self.remaining_V -= delta
            self.mu = np.clip(mu_new, 0.05, 0.95)
        else:
            scale = self.remaining_V / (delta + 1e-8)
            self.mu += scale * (mu_new - self.mu)
            self.mu = np.clip(self.mu, 0.05, 0.95)
            self.remaining_V = 0
        self.best_arm = np.argmax(self.mu)

    def step(self, action):
        self._update_means()
        reward = 1 if np.random.rand() < self.mu[action] else 0
        self.action_history.append(action)
        self.t += 1
        return reward, self.best_arm, np.max(self.mu)

# =================================================================
# 2. THE AGENT LINEUP
# =================================================================

class FAB_Agent:
    """The FAB Model: Fidelity, Adaptation Pressure, Balance."""
    def __init__(self, n_actions=5):
        self.q_values = np.zeros(n_actions)
        self.phi = 0.5    # Fidelity (F)
        self.lambda_p = 0.5 # Adaptation Pressure (A)
        self.alpha = 0.2

    def choose_action(self):
        # Balance (B): Exploration controlled by Adaptation Pressure
        epsilon = np.clip(self.lambda_p**2, 0.01, 0.5)
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.q_values))
        return np.argmax(self.q_values)

    def learn(self, action, reward):
        error = reward - self.q_values[action]
        self.q_values[action] += self.alpha * error
        # Homeostatic Loop
        self.phi = 0.85 * self.phi + 0.15 * (1.0 - abs(error))
        target_lambda = 1.0 - self.phi
        if self.phi > 0.85: target_lambda *= 0.2 # Balance mechanism
        self.lambda_p = 0.9 * self.lambda_p + 0.1 * target_lambda

class ThompsonSamplingAgent:
    """Standard Bayesian Thompson Sampling."""
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
    """Sliding-Window UCB (Short Memory SOTA)."""
    def __init__(self, n_actions=5, window=100):
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

class EXP3_Agent:
    """Exponential weight algorithm for Adversarial Bandits."""
    def __init__(self, n_actions=5, gamma=0.1):
        self.weights = np.ones(n_actions)
        self.gamma = gamma

    def choose_action(self):
        w_sum = np.sum(self.weights)
        p = (1-self.gamma) * (self.weights/w_sum) + (self.gamma/len(self.weights))
        return np.random.choice(len(self.weights), p=p)

    def learn(self, action, reward):
        w_sum = np.sum(self.weights)
        p = (1-self.gamma) * (self.weights[action]/w_sum) + (self.gamma/len(self.weights))
        estimated_reward = reward / p
        self.weights[action] *= np.exp(self.gamma * estimated_reward / len(self.weights))

# =================================================================
# 3. TOURNAMENT EXECUTION
# =================================================================
def run_tournament():
    horizon = 10000
    agents_map = {
        "FAB Model": FAB_Agent(),
        "Thompson Sampling": ThompsonSamplingAgent(),
        "SW-UCB (Window=100)": SW_UCB_Agent(),
        "EXP3": EXP3_Agent()
    }
    
    metrics = {name: {"regret": [0], "accuracy": []} for name in agents_map}

    for name, agent in agents_map.items():
        env = Level5AdversarialBandit(horizon=horizon)
        for t in range(horizon):
            # Select Action
            if "UCB" in name: act = agent.choose_action(t)
            else: act = agent.choose_action()
            
            # Step Environment
            reward, best_arm, max_mu = env.step(act)
            agent.learn(act, reward)
            
            # Record Data
            metrics[name]["accuracy"].append(1 if act == best_arm else 0)
            regret_inc = max_mu - env.mu[act]
            metrics[name]["regret"].append(metrics[name]["regret"][-1] + regret_inc)

    # =================================================================
    # 4. VISUALIZATION
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    for name in metrics:
        lw = 4 if "FAB" in name else 2
        ax1.plot(pd.Series(metrics[name]["accuracy"]).rolling(100).mean(), label=name, linewidth=lw)
        ax2.plot(metrics[name]["regret"], label=name, linewidth=lw)

    ax1.set_title("Strategy Performance: Optimal Action Probability (Level 5 Adversary)", fontsize=16)
    ax1.set_ylabel("Rolling Accuracy (100 steps)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title("Strategy Efficiency: Cumulative Regret", fontsize=16)
    ax2.set_ylabel("Total Payout Loss")
    ax2.set_xlabel("Steps")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tournament()