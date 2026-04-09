"""
Piece-Relative Space (PRS) Transformer for Hive.

Key modules:
  action_space      — PRS action encoding (6,841 actions)
  prs_encoder       — PRSEncoder and PRSTokenBatch
  prs_transformer   — HivePRSTransformer neural network
  prs_replay_buffer — PRSReplayBuffer for training examples
  prs_orchestrator  — PRSGumbelOrchestrator for self-play
  prs_trainer       — PRSTrainer training loop
  train_prs         — CLI: python -m hive_prs.train_prs
"""
