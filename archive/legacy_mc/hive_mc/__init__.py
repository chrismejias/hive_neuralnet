"""
hive_mc — Move-conditioned transformer for Hive.

Two-stage architecture: lightweight screening head scores all legal moves,
then top-k successors get full transformer encoding and value comparison.
"""
