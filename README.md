# Project FAB

## task-1

Experimental Setup and Results (Clear Explanation)

The environment is a three-armed bandit that changes behavior midway through the run. In the first phase, one action consistently gives high reward. At the shift point, this same action becomes harmful, and a different action becomes optimal. The agents are not told when this change occurs and must infer it from reward feedback.

The regulated agent (AEN) tracks how reliable its predictions are (Ego) and adjusts its exploration level accordingly (C-Drive). When the environment changes, Ego drops, causing C-Drive to increase and the agent to explore. This allows AEN to recover after a short performance drop. In contrast, the unregulated agent continues exploiting outdated patterns, collapses after the shift, and fails to recover. This shows that internal regulation enables stable adaptation when the environment changes unexpectedly.

