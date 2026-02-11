# Adaptive Fidelity Network (AEN / FAB) Embodied Reinforcement Learning Under Non-Stationarity
## Detailed evaluation in- [evaluation.md](evaluation.md)
## 1. Experimental Philosophy

The goal of these experiments is **not** to demonstrate state-of-the-art performance on standard benchmarks.  
Instead, the focus is on **failure modes, recovery behavior, and robustness** when core assumptions of reinforcement learning break at deployment time.

Across all experiments, we prioritize:
- qualitative behavioral differences,
- regret dynamics under shocks,
- stability vs. plasticity trade-offs,
- and computational simplicity.

Performance under perfect stationarity is intentionally *not* the primary objective.

---

## 2. Agents Evaluated

Across experiments, the following agents are compared:

### Classical Baselines
- **Thompson Sampling (TS)**  
  Bayesian posterior sampling; strong under stationarity, prone to posterior inertia.

- **Sliding-Window UCB (SW-UCB)**  
  Finite-memory optimism-based exploration; sensitive to window size.

- **EXP3**  
  Adversarial bandit algorithm; robust to worst-case rewards but often high variance.

### Adaptive Agent
- **FAB / Adaptive Fidelity Network (AEN)**  
  A lightweight agent with internal regulation of adaptation pressure based on an online reliability signal.

Multiple AEN variants were tested:
- *Fast AEN*: short effective memory, rapid adaptation
- *Slow AEN*: longer memory, greater stability

---

## 3. Stage 1 — Stationary Bandit (Sanity Check)

### Environment
- Fixed reward probabilities
- One arm is optimal throughout
- No drift, no shocks

### Observation
- Thompson Sampling and SW-UCB converge quickly
- FAB/AEN continues mild exploration
- FAB/AEN accumulates higher regret

### Interpretation
This confirms that AEN is **not optimized for stationary exploitation**.  
The exploration cost is intentional and expected.

**Conclusion:**  
FAB/AEN is suboptimal under perfect stationarity.

---

## 4. Stage 2 — Single Regime Shift (Trap Flip)

### Environment
- Long stable phase
- Sudden regime shift
- Previously optimal arm becomes suboptimal
- New optimal arm emerges

### Observation
- Classical agents overcommit to the old arm
- Recovery is slow and costly
- FAB/AEN rapidly reduces confidence and increases exploration
- Faster recovery after the shift

### Interpretation
Posterior and historical inertia dominate classical updates.  
AEN’s internal reliability signal triggers rapid adaptation.

**Conclusion:**  
Explicit regulation improves recovery after confident failure.

---

## 5. Stage 3 — Repeated and Unknown Regime Shifts

### Environment
- Multiple regime shifts
- Unknown timing
- No prolonged stationary phase

### Observation
- Fixed-window methods struggle to choose a suitable window
- Small windows are noisy; large windows adapt too slowly
- FAB/AEN maintains bounded regret without tuning

### Interpretation
Window-based methods require prior knowledge of change scale.  
AEN adapts implicitly via confidence regulation.

**Conclusion:**  
Adaptive time-scale regulation outperforms fixed memory in uncertain environments.

---

## 6. Stage 4 — Bounded Non-Stationarity (Variation Budget)

### Environment
- Gradual drift with bounded total variation
- No adversarial feedback
- Smooth but persistent changes

### Observation
- EXP3 shows high variance
- SW-UCB is stable but inefficient
- FAB/AEN tracks drift with moderate regret

### Interpretation
No single agent dominates.  
FAB/AEN trades optimality for consistency.

**Conclusion:**  
AEN behaves conservatively but avoids collapse.

---

## 7. Stage 5 — Adversarial “Final Boss” Environment

### Environment
- Bounded adversarial bandit
- Environment penalizes:
  - excessive dominance (overcommitment)
  - excessive jitter (instability)
- Targets agent habits explicitly

### Observation
- Fast AEN collapses due to overreaction
- Slow AEN survives with higher regret
- EXP3 oscillates heavily
- No agent achieves stable dominance

### Interpretation
Time-scale choice is critical.  
Overly reactive agents become exploitable.

**Conclusion:**  
AEN must be tuned to environment tempo; no universal setting exists.

---

## 8. Stage 6 — Late-Onset Crisis (Market-Style Scenario)

### Environment
- Long stable period (≈80% of horizon)
- Sudden, irreversible instability
- No return to prior regime

### Observation
- Classical agents dominate early
- Severe collapse after crisis
- FAB/AEN underperforms early
- FAB/AEN limits late regret and dominates cumulatively

### Interpretation
This environment mirrors real-world systems such as:
- financial markets,
- deployed robotics,
- long-running infrastructure.

Survival outweighs short-term optimality.

**Conclusion:**  
FAB/AEN excels when late failure is costly.

---


## 9. Sensitivity to Time-Scale (Fast vs Slow AEN)

### Fast AEN
- Effective memory: ~10–20 steps
- Rapid recovery
- Vulnerable to adversarial exploitation

### Slow AEN
- Effective memory: ~100+ steps
- Stable
- Less responsive to abrupt shocks

**Conclusion:**  
AEN represents a *family* of behaviors, not a single optimal configuration.

---

## 11. Summary of Findings

- Classical RL excels under stability but fails under assumption violation.
- Fixed memory is brittle without prior knowledge.
- Explicit regulation enables graceful degradation.
- FAB/AEN sacrifices optimality for survivability.
- There is no free lunch: robustness has a cost.

---

## 12. Limitations

- No formal regret guarantees
- No optimal parameter selection rule
- Not competitive in stationary benchmarks
- Not a replacement for large-scale RL

---
