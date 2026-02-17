import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# 1. ENVIRONMENT: Exogenous Non-Stationary Bandit
# ================================================================
class ExogenousNonStationaryBandit:
    """
    Neutral non-stationary bandit:
    - Slow stochastic drift
    - Abrupt regime shifts (shocks)
    - Noisy plateaus
    - No agent feedback
    """
    def __init__(
        self,
        K=5,
        horizon=10000,
        drift_std_range=(0.002, 0.02),
        shock_prob=0.002,
        plateau_prob=0.001,
        seed=42
    ):
        np.random.seed(seed)
        self.K = K
        self.T = horizon

        self.drift_std = np.random.uniform(*drift_std_range)
        self.shock_prob = shock_prob
        self.plateau_prob = plateau_prob

        self.mu = np.random.uniform(0.3, 0.7, size=K)
        self.plateau_timer = 0

    def step(self):
        # Plateau: near-stationary but noisy
        if self.plateau_timer > 0:
            self.plateau_timer -= 1
            self.mu = np.clip(
                self.mu + np.random.normal(0, self.drift_std * 0.2, size=self.K),
                0.05, 0.95
            )

        # Abrupt regime shift
        elif np.random.rand() < self.shock_prob:
            self.mu = np.random.uniform(0.1, 0.9, size=self.K)
            self.drift_std = np.random.uniform(0.002, 0.02)

        # Enter plateau
        elif np.random.rand() < self.plateau_prob:
            self.plateau_timer = np.random.randint(200, 600)

        # Normal drift
        else:
            drift = np.random.normal(0, self.drift_std, size=self.K)
            self.mu = np.clip(self.mu + drift, 0.05, 0.95)

        best_arm = np.argmax(self.mu)
        max_mu = np.max(self.mu)
        return self.mu.copy(), best_arm, max_mu


def generate_shared_trajectory(env):
    mus, best_arms, max_mus = [], [], []
    for _ in range(env.T):
        mu, best, max_mu = env.step()
        mus.append(mu)
        best_arms.append(best)
        max_mus.append(max_mu)
    return mus, best_arms, max_mus


# ================================================================
# 2. AGENTS
# ================================================================
class FAB_Agent:
    """Proposed Balance-Controlled Agent"""
    def __init__(self, n_actions=5):
        self.q = np.zeros(n_actions)
        self.phi = 0.5
        self.lmb = 0.5
        self.alpha = 0.2

    def choose_action(self):
        eps = np.clip(self.lmb ** 2, 0.01, 0.5)
        return np.random.randint(len(self.q)) if np.random.rand() < eps else np.argmax(self.q)

    def learn(self, a, r):
        err = r - self.q[a]
        self.q[a] += self.alpha * err
        self.phi = 0.85 * self.phi + 0.15 * (1 - abs(err))
        target = (1 - self.phi)
        if self.phi > 0.85:
            target *= 0.2
        self.lmb = 0.9 * self.lmb + 0.1 * target


class ThompsonSamplingAgent:
    def __init__(self, n_actions=5):
        self.a = np.ones(n_actions)
        self.b = np.ones(n_actions)

    def choose_action(self):
        return np.argmax(np.random.beta(self.a, self.b))

    def learn(self, a, r):
        self.a[a] += r
        self.b[a] += 1 - r


class SW_UCB_Agent:
    def __init__(self, n_actions=5, window=100):
        self.n = n_actions
        self.window = window
        self.history = []

    def choose_action(self, t):
        if t < self.n:
            return t
        recent = self.history[-self.window:]
        counts = np.bincount([x[0] for x in recent], minlength=self.n)
        rewards = np.zeros(self.n)

        for a, r in recent:
            rewards[a] += r

        ucb = np.zeros(self.n)
        for i in range(self.n):
            if counts[i] == 0:
                ucb[i] = 1e6
            else:
                ucb[i] = rewards[i] / counts[i] + np.sqrt(2 * np.log(self.window) / counts[i])
        return np.argmax(ucb)

    def learn(self, a, r):
        self.history.append((a, r))


class EXP3_Agent:
    def __init__(self, n_actions=5, gamma=0.1):
        self.w = np.ones(n_actions)
        self.gamma = gamma

    def choose_action(self):
        p = (1 - self.gamma) * self.w / np.sum(self.w) + self.gamma / len(self.w)
        return np.random.choice(len(self.w), p=p)

    def learn(self, a, r):
        p = (1 - self.gamma) * self.w[a] / np.sum(self.w) + self.gamma / len(self.w)
        self.w[a] *= np.exp(self.gamma * r / (p * len(self.w)))


# ================================================================
# 3. EVALUATION
# ================================================================
def run_experiment():
    horizon = 10000
    env = ExogenousNonStationaryBandit(horizon=horizon)
    mus, best_arms, max_mus = generate_shared_trajectory(env)

    agents = {
        "FAB (Proposed)": FAB_Agent(),
        "Thompson Sampling": ThompsonSamplingAgent(),
        "SW-UCB": SW_UCB_Agent(),
        "EXP3": EXP3_Agent()
    }

    metrics = {k: {"acc": [], "regret": [0]} for k in agents}

    for name, agent in agents.items():
        for t in range(horizon):
            act = agent.choose_action(t) if "UCB" in name else agent.choose_action()
            reward = 1 if np.random.rand() < mus[t][act] else 0
            agent.learn(act, reward)

            metrics[name]["acc"].append(act == best_arms[t])
            metrics[name]["regret"].append(
                metrics[name]["regret"][-1] + (max_mus[t] - mus[t][act])
            )

    # ============================================================
    # 4. VISUALIZATION
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for name in metrics:
        lw = 4 if "FAB" in name else 2
        ax1.plot(pd.Series(metrics[name]["acc"]).rolling(100).mean(), label=name, lw=lw)
        ax2.plot(metrics[name]["regret"], label=name, lw=lw)

    ax1.set_title("Neutral Non-Stationary Bandit: Optimal Action Probability")
    ax1.set_ylabel("Rolling Accuracy (100)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Neutral Non-Stationary Bandit: Cumulative Regret")
    ax2.set_ylabel("Total Regret")
    ax2.set_xlabel("Steps")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
