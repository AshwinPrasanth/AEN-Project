# Project FAB

## test-1

Experimental Setup and Results (Clear Explanation)

The environment is a three-armed bandit that changes behavior midway through the run. In the first phase, one action consistently gives high reward. At the shift point, this same action becomes harmful, and a different action becomes optimal. The agents are not told when this change occurs and must infer it from reward feedback.

The regulated agent (AEN) tracks how reliable its predictions are (Ego) and adjusts its exploration level accordingly (C-Drive). When the environment changes, Ego drops, causing C-Drive to increase and the agent to explore. This allows AEN to recover after a short performance drop. In contrast, the unregulated agent continues exploiting outdated patterns, collapses after the shift, and fails to recover. This shows that internal regulation enables stable adaptation when the environment changes unexpectedly.

## test-2

Test 2: Abrupt Regime Shift with Competing Adaptation Strategies

In Test 2, we evaluate agent behavior in a non-stationary bandit with a single abrupt regime shift where the initially optimal action becomes a trap and a different action becomes optimal. Unlike Test 1, this setup directly compares regulated, classical, and reactive learning strategies under the same environment to isolate different failure modes after sudden change.

The results show that while classical UCB achieves high accuracy before the shift, it adapts slowly afterward, leading to increased regret. Reactive learning adapts quickly but exhibits instability and higher long-term loss. Decay-epsilon fails due to premature loss of exploration. The regulated AEN agent does not dominate the stationary phase but recovers reliably after the shift and maintains the most stable long-horizon behavior.

This test demonstrates that internal regulation is not required for optimal performance under stationarity, but becomes critical when prior success turns into failure.
