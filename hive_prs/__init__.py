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

Move-conditioned modules:
  mc_utils         — root/legal-successor batching helpers
  mc_transformer   — HiveMoveTransformer legal-move scorer
  mc_replay_buffer — replay buffer for ragged legal-move targets
  mc_orchestrator  — Gumbel self-play over the legal move list
  mc_trainer       — training loop for the move-conditioned model
  train_mc         — CLI: python -m hive_prs.train_mc
"""
