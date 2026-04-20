"""
Piece-Relative Space (PRS) Transformer for Hive.

Key modules:
  action_space      — PRS helper constants (board geometry / piece ids)
  prs_encoder       — PRSEncoder and PRSTokenBatch
  prs_transformer_v2 — HivePRSTransformerV2 neural network
  prs_replay_buffer_v2 — PRSReplayBufferV2 for training examples
  prs_mcts_orchestrator_v2 — PRS v2 tree-search self-play
  prs_trainer_v2    — PRSTrainerV2 training loop
  train_prs         — default CLI (PRS v2): python -m hive_prs.train_prs

Move-conditioned modules:
  mc_utils         — root/legal-successor batching helpers
  mc_transformer   — HiveMoveTransformer legal-move scorer
  mc_replay_buffer — replay buffer for ragged legal-move targets
  mc_orchestrator  — Gumbel self-play over the legal move list
  mc_trainer       — training loop for the move-conditioned model
  train_mc         — CLI: python -m hive_prs.train_mc

Legacy PRS v1 modules are archived under:
  archive/legacy_prs_v1/
"""
