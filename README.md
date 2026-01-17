# AEN-Project: AI EGO Network

**A psychologically-grounded AI agent framework with dynamic ego, competition drive, and ethical decision-making.**

---

## Overview

Project AEN implements a novel approach to AI agent design by incorporating psychological constructs like ego dynamics, competitive motivation, self-assessment, and hierarchical ethics. The framework is grounded in mathematical formulations inspired by psychology, behavioral economics, and reinforcement learning.

### Key Features

- **Dynamic Ego System**: Three-dimensional ego state (cognitive, motivational, historical) with nonlinear dynamics and memory
- **Multi-Dimensional Competition Drive**: Adaptive competition across multiple performance dimensions with cross-dimensional coupling
- **Bayesian Self-Assessment**: Probabilistic performance beliefs with confidence modulation and multi-source feedback fusion
- **Risk-Aware Decision Engine**: Ego and competition-modulated decision-making with regret aversion and temporal discounting
- **Hierarchical Ethics Framework**: Deontological, consequentialist, and virtue ethics with learnable weights
- **Resilience and Recovery**: Growth from adversity with adaptive ego damping
- **Social Comparison**: Peer influence on ego dynamics with upward/downward bias

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AEN Agent                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Ego State  │  │ Competition  │  │     Self-    │     │
│  │   E(t) ∈ ℝ³  │  │   Drive      │  │  Assessment  │     │
│  │              │  │   C(t) ∈ ℝᵈ  │  │   S(t) ∈ ℝ   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
│                 ┌─────────▼─────────┐                       │
│                 │  Decision Engine   │                       │
│                 │  W_i(t) = f(E,C,S) │                       │
│                 └─────────┬─────────┘                       │
│                           │                                 │
│                 ┌─────────▼─────────┐                       │
│                 │   Ethics Module    │                       │
│                 │  A_i(t) = D·C·V    │                       │
│                 └─────────┬─────────┘                       │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │    Action    │                          │
│                    │   a*(t)      │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Mathematical Framework

### 1. Ego Dynamics

**Ego State Vector:**
```
E(t) = [E_c(t), E_m(t), E_h(t)]ᵀ
```

**Nonlinear Update with Momentum:**
```
E_c(t+1) = E_c(t) + α[S(t) - E_c(t)]tanh(ΔS(t)) - βF(t)e^(-λT_success) + μ∇E_c(t)
```

**Memory-Weighted Historical Ego:**
```
E_h(t) = Σ w_k·E_c(t-k)·e^(-δk)
```

### 2. Competition Drive

**Dimension-Specific Impulse:**
```
I_C^i(t) = C_i(t)·[B_i(t) - P_i(t)]·φ(Rank_i(t))
```

**Total Competition with Coupling:**
```
I_C,total(t) = Σ w_i·I_C^i(t) + ΣΣ θ_ij·I_C^i(t)·I_C^j(t)
```

### 3. Decision Making

**Risk-Adjusted Utility:**
```
U_i(t) = 𝔼[Q_i(t)] - λ_risk·√Var[Q_i(t)] - λ_regret·R_max(Q_i(t))
```

**Ego-Competition Modulation:**
```
W_i(t) = [1 + E_c(t)(1 + ψE_h(t))]·U_i(t) + I_C,total(t)·R_i(t)·e^(-ξ|E_c - E_optimal|)
```

### 4. Ethics

**Hierarchical Alignment:**
```
A_i(t) = A_i,strategic(t)·A_i,tactical(t)·A_i,ethical(t)
```

**Ethical Score:**
```
A_i,ethical(t) = w_d·D_i(t) + w_c·C_i(t) + w_v·V_i(t)
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- (Optional) Gym for extended environments

---

## Quick Start

### 1. Basic Agent Usage

```python
from aen_core import AENConfig
from aen_agent import AENAgent
import torch

# Initialize configuration
config = AENConfig()

# Create agent
agent = AENAgent(
    config=config,
    action_dim=10,
    state_dim=64,
    context_dim=32
)

# Run agent
state = torch.randn(64)
result = agent(
    state=state,
    performance_metrics={
        'performance': 0.7,
        'success': True,
        'failure_magnitude': 0.0
    }
)

print(f"Action: {result['action'].item()}")
print(f"Ego (cognitive): {result['ego_cognitive']:.3f}")
print(f"Competition impulse: {result['competition_impulse']:.3f}")
```

### 2. Multi-Agent Training

```python
from aen_training import CompetitiveEnvironment, AENTrainer
from aen_agent import AENAgent
from aen_core import AENConfig

# Setup
config = AENConfig()
env = CompetitiveEnvironment(
    num_agents=4,
    scenario="resource_allocation"
)

# Create agents
agents = [
    AENAgent(config, action_dim=5, state_dim=32)
    for _ in range(4)
]

# Train
trainer = AENTrainer(agents, env, learning_rate=1e-3)
trainer.train(num_episodes=100)
```

### 3. Running Training Script

```bash
python aen_training.py
```

This will:
- Initialize 4 agents in a competitive environment
- Train for 100 episodes
- Generate performance plots
- Display final agent states

---

## Project Structure

```
project-aen/
├── aen_core.py          # Core ego, competition, self-assessment modules
├── aen_decision.py      # Decision engine, ethics, resilience
├── aen_agent.py         # Complete integrated agent
├── aen_training.py      # Training environment and evaluation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## Core Components

### EgoState
Manages dynamic ego with three dimensions:
- **Cognitive**: Current confidence and assertiveness
- **Motivational**: Intrinsic drive and ambition
- **Historical**: Memory-weighted past states

### CompetitionDrive
Multi-dimensional competition framework:
- Dimension-specific impulses
- Cross-dimensional coupling
- Adaptive weight learning

### SelfAssessment
Bayesian self-assessment:
- Multi-source feedback fusion
- Uncertainty quantification
- Confidence modulation

### DecisionEngine
Risk-aware decision-making:
- Risk and regret aversion
- Ego-competition modulation
- Temporal discounting
- Exploration-exploitation balance

### EthicsModule
Hierarchical ethics:
- Deontological constraints (rules)
- Consequentialist evaluation (outcomes)
- Virtue alignment (character)
- Learnable ethical weights

### ResilienceModule
Recovery dynamics:
- Failure history tracking
- Adaptive ego damping
- Growth from adversity
- Meta-learning from patterns

---

## Configuration

Key parameters in `AENConfig`:

```python
# Ego parameters
alpha: float = 0.1           # Ego adaptation rate
beta_base: float = 0.15      # Humility coefficient
mu_momentum: float = 0.2     # Momentum coefficient

# Competition parameters
kappa_rank: float = 2.0      # Rank sensitivity
eta_weight: float = 0.01     # Weight learning rate

# Decision parameters
lambda_risk: float = 0.5     # Risk aversion
lambda_regret: float = 0.3   # Regret aversion
gamma_discount: float = 0.99 # Temporal discount

# Ethics parameters
w_deontological: float = 0.4
w_consequentialist: float = 0.3
w_virtue: float = 0.3

# Resilience parameters
psi_base: float = 0.5        # Baseline resilience
tau_recovery: float = 5.0    # Recovery time constant
omega_growth: float = 0.2    # Growth from adversity
```

---

## Scenarios

### Resource Allocation
Agents compete for limited resources. Actions represent resource claims (0-4 units). Excess demand triggers proportional allocation.

### Competitive Task
Agents select tasks of varying difficulty. Higher difficulty → higher reward but lower success probability. Tests risk-taking and ego calibration.

---

## Evaluation Metrics

### Ego Dynamics
- Cognitive ego trajectory
- Ego stability (oscillation amplitude)
- Time to recovery after failure

### Competition
- Competition impulse over time
- Dimension-specific focus
- Competitive efficiency (improvement per unit impulse)

### Ethics
- Alignment score distribution
- Ethical constraint violations
- Strategic vs. ethical trade-offs

### Performance
- Cumulative reward
- Win rate (relative to peers)
- Resilience score

---

## Extending the Framework

### Custom Environments

```python
class MyEnvironment:
    def reset(self):
        # Return initial states
        pass
    
    def step(self, actions):
        # Execute actions
        # Return: next_states, rewards, dones, info
        pass
```

### Custom Ethical Rules

```python
# In EthicsModule
agent.ethics.deontological_rules[action_idx] = 0.0  # Forbid action
agent.ethics.deontological_rules[action_idx] = 1.0  # Allow action
```

### Adjust Ego Dynamics

```python
config.alpha = 0.2          # Faster ego adaptation
config.beta_base = 0.3      # Stronger humility
config.mu_momentum = 0.5    # More momentum
```

---

## Research Applications

### Potential Use Cases

1. **Multi-Agent Systems**: Competitive/cooperative scenarios with ego-driven agents
2. **Human-AI Interaction**: Agents with psychologically realistic behaviors
3. **Organizational Simulation**: Model ego dynamics in teams
4. **Game AI**: NPCs with personality and adaptive confidence
5. **Ethical AI**: Study trade-offs between ambition and ethical constraints
