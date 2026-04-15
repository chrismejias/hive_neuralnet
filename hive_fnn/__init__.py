"""
hive_fnn -- HiveGo-style feedforward neural network for Hive.

Tiny FNN with hand-crafted board features and successor-state scoring.
Policy is computed by encoding ALL legal successor states and scoring
each against the root embedding. Uses GPU-based Gumbel AlphaZero search.
"""
