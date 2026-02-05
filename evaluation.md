# Experimental Evaluation and Analysis  
## Adaptive Fidelity Network (AEN / FAB)

This document records **all experiments conducted**, explicitly referencing the **exact test files** used.  
The goal of these experiments is to understand **failure modes, recovery behavior, and robustness** under increasingly hostile non-stationary environments.

---

## Experiment Index (By File)

| File | Experiment Type |
|----|----|
| `test2.py` | Single regime shift (trap flip) |
| `test3.py` | Repeated regime shifts |
| `test4.py` | Gradual non-stationary drift |
| `test5.py` | Late-onset non-stationarity |
| `test6.py` | Shared environment tournament |
| `pro.py` | Real-world unpredictable environment |
| `sharedspace.py` | Shared adversary (same env for all agents) |
| `hard.py` | Adversarial non-stationary bandit |
| `np-hard.py` | Worst-case / pathological environment |
| `10.py` | Long-horizon stress test |

---

## 1. Stationary Baseline (Sanity Check)

**Files:** early stages embedded in `test2.py`

### Setup
- Fixed reward probabilities
- One arm optimal throughout
- No drift, no shocks

### Observation
- Thompson Sampling and SW-UCB converge fastest
- FAB/AEN continues mild exploration
- FAB/AEN accumulates higher regret

### Interpretation
This confirms that FAB/AEN is **not designed for stationary exploitation**.

**Conclusion:**  
Under perfect stationarity, classical algorithms dominate.

---

## 2. Single Regime Shift (Trap Flip)

**File:** `test2.py`

### Setup
- Long stable phase
- Sudden flip: optimal arm becomes worst
- New optimal arm appears

### Observation
- TS and UCB collapse temporarily
- Recovery is slow due to accumulated confidence
- FAB/AEN rapidly increases exploration and recovers faster

### Interpretation
Classical methods exhibit **Bayesian inertia**.  
FAB/AEN detects loss of reliability and reduces commitment.

**Conclusion:**  
Explicit regulation improves post-shift recovery.

---

## 3. Repeated Regime Shifts

**File:** `test3.py`

### Setup
- Multiple regime shifts
- Unknown timing
- No prolonged stationary phase

### Observation
- Small-window UCB is noisy
- Large-window UCB adapts too slowly
- FAB/AEN maintains bounded regret without tuning

### Interpretation
Fixed windows require prior knowledge of change scale.  
FAB/AEN adapts implicitly via internal confidence.

**Conclusion:**  
Adaptive time-scale control outperforms fixed memory.

---

## 4. Gradual Drift (Non-Adversarial)

**File:** `test4.py`

### Setup
- Smooth stochastic drift
- No abrupt change points

### Observation
- EXP3 shows high variance
- SW-UCB is stable but inefficient
- FAB/AEN tracks drift conservatively

### Interpretation
No agent dominates.  
FAB/AEN trades optimality for consistency.

**Conclusion:**  
FAB/AEN avoids collapse but does not chase optimality.

---

## 5. Late-Onset Non-Stationarity

**File:** `test5.py`

### Setup
- Long stable phase
- Sudden instability near the end of the horizon
- No recovery to previous regime

### Observation
- Classical agents dominate early
- Severe late collapse
- FAB/AEN underperforms early but limits late regret
- Regret curves cross

### Interpretation
This environment mirrors **real deployment failures**.

**Conclusion:**  
FAB/AEN excels when late failure is costly.

---

## 6. Shared Environment Tournament

**File:** `test6.py`, `sharedspace.py`

### Setup
- All agents interact with the **same evolving environment**
- No agent-specific adaptation by the environment

### Observation
- Removes bias concerns
- Results remain consistent
- FAB/AEN maintains stability across shared conditions

### Interpretation
Confirms behavior is not due to environment-agent coupling.

**Conclusion:**  
Observed robustness is intrinsic to agent design.

---

## 7. Real-World Unpredictability

**File:** `pro.py`

### Setup
- Stochastic drift
- Periodic “Black Swan” resets
- Recharging variation budget

### Observation
- TS and UCB overcommit
- EXP3 oscillates
- FAB/AEN maintains bounded loss

### Interpretation
Matches intuition from real-world systems (markets, infrastructure).

**Conclusion:**  
FAB/AEN favors survivability over short-term efficiency.

---

## 8. Adversarial Non-Stationarity

**File:** `hard.py`

### Setup
- Bounded adversarial changes
- Penalizes predictability and jitter

### Observation
- Fast AEN collapses
- Slow AEN survives
- EXP3 unstable
- No agent dominates

### Interpretation
Time-scale choice is critical.

**Conclusion:**  
FAB/AEN is a *family of behaviors*, not a single solution.

---

## 9. Worst-Case / Pathological Environment

**File:** `np-hard.py`

### Setup
- Worst-case dynamics
- Near-impossible learning conditions

### Observation
- All agents suffer
- FAB/AEN avoids catastrophic divergence

### Interpretation
Some environments are unwinnable.

**Conclusion:**  
Goal shifts from winning to controlled failure.

---

## 10. Long-Horizon Stress Test

**File:** `10.py`

### Setup
- Very long horizon
- Mixed stability and instability

### Observation
- Stationary specialists fail late
- FAB/AEN maintains bounded regret

### Interpretation
Long-horizon behavior reveals hidden fragility.

**Conclusion:**  
Short-term dominance is misleading.

---

## 11. Computational Analysis

| Agent | Memory | Per-Step Cost |
|----|----|----|
| FAB / AEN | O(1) | O(1) |
| SW-UCB | O(W) | O(W) |
| EXP3 | O(K) | O(K) |

FAB/AEN is **cheap and fast**, at the cost of long-term exploitation.

---

## 12. Key Takeaways

- Stationary benchmarks hide failure modes
- Fixed memory is brittle
- Explicit adaptation regulation enables graceful degradation
- FAB/AEN trades optimality for resilience
- There is no universal winner

---

## Final Statement

> The contribution of this work is not higher reward, but **controlled behavior under uncertainty**.

These experiments demonstrate that simple internal regulation can prevent catastrophic collapse when learning assumptions break—a property essential for embodied and deployed systems.

---

