"""
GPU-accelerated training loop for Hive AI.

Uses GPUMCTSOrchestrator for batched self-play, with the same
train → arena → promote pipeline as hive_engine/trainer.py.

Usage:
    from hive_gpu.gpu_trainer import GPUTrainer, GPUTrainConfig
    from hive_gnn.gnn_net import GNNNetConfig, HiveGNN

    trainer = GPUTrainer(GPUTrainConfig(), GNNNetConfig.small())
    trainer.run()
"""

from __future__ import annotations

import copy
import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from hive_engine.device import get_device
from hive_engine.elo import EloTracker
from hive_engine.game_state import GameState, GameResult
from hive_engine.neural_net import compute_gnn_loss, compute_transformer_loss
from hive_engine.pieces import Color

from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.gnn_replay_buffer import (
    GNNTrainingExample,
    GraphReplayBuffer,
)
from hive_transformer.transformer_replay_buffer import (
    TransformerTrainingExample,
    TokenReplayBuffer,
)

from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig, GPUTrainingExample


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class GPUTrainConfig:
    """Configuration for GPU-accelerated training."""

    # Self-play
    num_iterations: int = 20
    games_per_batch: int = 64       # Number of concurrent games per self-play batch
    batches_per_iteration: int = 1  # Number of self-play batches per iteration
    mcts_simulations: int = 100
    temperature: float = 1.0
    temperature_drop_move: int = 20
    max_game_length: int = 300

    # MCTS
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_max_size: int = 50_000

    # Arena
    arena_games: int = 20
    arena_threshold: float = 0.55
    arena_mcts_simulations: int = 50

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # LR scheduling
    lr_schedule: str = "constant"
    lr_warmup_iterations: int = 0
    lr_min: float = 1e-5

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Device
    device: str | None = None

    # Mixed precision
    use_amp: bool | None = None

    # Encoder type for GPU MCTS
    encoder_type: str = "gnn"  # "gnn" or "transformer"

    # Wave-parallel MCTS
    wave_size: int = 8         # parallel sims per wave (1 = sequential)
    nn_max_batch: int = 0      # max NN batch size per forward pass (0 = no limit)

    # Playout cap randomization (KataGo-style)
    playout_cap_randomize: bool = False
    playout_cap_randomize_prob: float = 0.25   # prob of full playouts; else fast
    playout_cap_fast_sims: int = 0             # 0 = auto (num_simulations // 8)

    # Arena / promotion
    skip_arena: bool = False   # always accept new model, skip arena games

    # Policy softening (KataGo-style)
    policy_softening: float = 0.0   # mix weight toward uniform over legal moves


# ── GPU Trainer ────────────────────────────────────────────────────────


class GPUTrainer:
    """
    AlphaZero-style trainer using GPU-accelerated MCTS.

    Follows the same pipeline as hive_engine/trainer.py:
    self-play → train → arena → promote → checkpoint.
    """

    def __init__(
        self,
        config: GPUTrainConfig | None = None,
        net_config=None,
        net_class=None,
    ) -> None:
        self.config = config or GPUTrainConfig()
        self.net_config = net_config
        self.net_class = net_class
        if self.config.encoder_type == "transformer":
            self.buffer = TokenReplayBuffer(self.config.buffer_max_size)
        else:
            self.buffer = GraphReplayBuffer(self.config.buffer_max_size)

        # Device
        self.device = get_device(self.config.device)

        # Mixed precision
        if self.config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = self.config.use_amp
        self._grad_scaler = None
        if self.use_amp:
            self._grad_scaler = torch.amp.GradScaler("cuda")

        # Create network
        self.best_net = self.net_class(self.net_config).to(self.device)

        # ELO tracking
        self.elo_tracker = EloTracker()
        self._start_iteration = 1

    def _get_learning_rate(self, iteration: int) -> float:
        """Compute LR with optional warmup and cosine schedule."""
        base_lr = self.config.learning_rate

        if (
            self.config.lr_warmup_iterations > 0
            and iteration <= self.config.lr_warmup_iterations
        ):
            return base_lr * (iteration / self.config.lr_warmup_iterations)

        if self.config.lr_schedule == "constant":
            return base_lr

        warmup = self.config.lr_warmup_iterations
        total = self.config.num_iterations - warmup
        progress = (iteration - warmup) / max(total, 1)
        lr_min = self.config.lr_min
        return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))

    def _cuda_cleanup(self) -> None:
        """Synchronize GPU and release cached memory to prevent fragmentation."""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def run(self) -> None:
        """Run the full GPU training loop."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for iteration in range(
            self._start_iteration, self.config.num_iterations + 1
        ):
            # Clean up GPU memory between iterations to prevent fragmentation
            self._cuda_cleanup()

            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")

            # 1. Self-play
            t0 = time.time()
            new_examples, sp_stats = self._self_play_phase()
            self.buffer.add_examples(new_examples)
            sp_time = time.time() - t0
            print(
                f"  Self-play: {len(new_examples)} examples from "
                f"{sp_stats['num_games']} games "
                f"(W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
                f"D:{sp_stats['draws']}, {sp_time:.1f}s)"
            )
            self._cuda_cleanup()

            # 2. Train
            t0 = time.time()
            new_net, train_loss = self._train_phase(iteration)
            train_time = time.time() - t0
            print(f"  Training: loss={train_loss:.4f}, {train_time:.1f}s")
            self._cuda_cleanup()

            # 3. Arena / promotion
            if self.config.skip_arena:
                arena_time = 0.0
                print("  Arena: skipped (auto-accept)")
                self.best_net = new_net
                elo_rating = self.elo_tracker.update(1.0, self.config.arena_games)
            else:
                t0 = time.time()
                win_rate = self._arena_evaluate(new_net)
                arena_time = time.time() - t0
                self._cuda_cleanup()
                print(
                    f"  Arena: win rate = {win_rate:.1%} "
                    f"(threshold: {self.config.arena_threshold:.0%}, "
                    f"{arena_time:.1f}s)"
                )
                elo_rating = self.elo_tracker.update(
                    win_rate, self.config.arena_games
                )
                if win_rate >= self.config.arena_threshold:
                    print("  [+] New model accepted!")
                    self.best_net = new_net
                else:
                    print("  [-] New model rejected.")

            print(f"  ELO: {elo_rating:.0f}")

            # 4. Checkpoint
            self._save_checkpoint(self.best_net, iteration)

            total_time = sp_time + train_time + arena_time
            print(
                f"  Summary: iter={iteration} loss={train_loss:.4f} "
                f"elo={elo_rating:.0f} "
                f"buf={len(self.buffer)} time={total_time:.0f}s"
            )

    # ── Self-play ──────────────────────────────────────────────────────

    def _self_play_phase(
        self,
    ) -> tuple[list, dict]:
        """Run GPU-accelerated self-play, returning training examples."""
        cfg = self.config

        mcts_config = GPUMCTSConfig(
            num_simulations=cfg.mcts_simulations,
            c_puct=cfg.c_puct,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
            temperature=cfg.temperature,
            temperature_drop_move=cfg.temperature_drop_move,
            batch_size=cfg.games_per_batch,
            max_game_length=cfg.max_game_length,
            encoder_type=cfg.encoder_type,
            wave_size=cfg.wave_size,
            nn_max_batch=cfg.nn_max_batch,
            playout_cap_randomize=cfg.playout_cap_randomize,
            playout_cap_randomize_prob=cfg.playout_cap_randomize_prob,
            playout_cap_fast_sims=cfg.playout_cap_fast_sims,
        )

        self.best_net.eval()
        orchestrator = GPUMCTSOrchestrator(self.best_net, mcts_config)
        is_transformer = cfg.encoder_type == "transformer"

        all_examples: list = []
        stats = {
            "num_games": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }

        for batch_idx in range(cfg.batches_per_iteration):
            batch_examples = orchestrator.self_play_batch()

            for game_examples in batch_examples:
                for ex in game_examples:
                    if is_transformer:
                        converted = TransformerTrainingExample(
                            sequence=ex.sequence,
                            policy_target=ex.policy_target,
                            value_target=ex.value_target,
                            mobility_target=ex.mobility_target,
                            queen_surround_target=ex.queen_surround_target,
                            queen_surround_mask=ex.queen_surround_mask,
                            final_mobility_target=ex.final_mobility_target,
                            use_for_value=True,
                        )
                    else:
                        converted = GNNTrainingExample(
                            graph=ex.graph,
                            policy_target=ex.policy_target,
                            value_target=ex.value_target,
                            mobility_target=ex.mobility_target,
                            queen_surround_target=ex.queen_surround_target,
                            queen_surround_mask=ex.queen_surround_mask,
                            final_mobility_target=ex.final_mobility_target,
                            use_for_value=True,
                        )
                    all_examples.append(converted)

                stats["num_games"] += 1
                if game_examples:
                    v = game_examples[0].value_target
                    if v > 0:
                        stats["white_wins"] += 1
                    elif v < 0:
                        stats["black_wins"] += 1
                    else:
                        stats["draws"] += 1
                else:
                    stats["draws"] += 1

        return all_examples, stats

    # ── Training ───────────────────────────────────────────────────────

    def _train_phase(
        self, iteration: int
    ) -> tuple[torch.nn.Module, float]:
        """Train a new network on the replay buffer."""
        new_net = copy.deepcopy(self.best_net)
        new_net.train()

        lr = self._get_learning_rate(iteration)
        optimizer = optim.Adam(
            new_net.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        if len(self.buffer) < self.config.batch_size:
            return new_net, 0.0

        is_transformer = self.config.encoder_type == "transformer"
        total_loss_sum = 0.0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            batches_per_epoch = max(
                1, len(self.buffer) // self.config.batch_size
            )

            for _ in range(batches_per_epoch):
                batch = self.buffer.sample_batch(self.config.batch_size)
                batch = batch.to(self.device)

                optimizer.zero_grad()

                ps = self.config.policy_softening
                if self.use_amp and self._grad_scaler is not None:
                    with torch.amp.autocast("cuda"):
                        if is_transformer:
                            policy_logits, value_pred, aux_outputs = new_net(batch.token_batch)
                            total_loss, loss_dict = compute_transformer_loss(
                                policy_logits, value_pred,
                                batch.policy_targets, batch.value_targets,
                                aux_outputs,
                                batch.mobility_targets,
                                batch.queen_surround_targets,
                                batch.queen_surround_mask,
                                batch.final_mobility_targets,
                                batch.value_mask,
                                batch.board_token_batch,
                                policy_softening=ps,
                            )
                        else:
                            policy_logits, value_pred, aux_outputs = new_net(batch.graph_batch)
                            total_loss, loss_dict = compute_gnn_loss(
                                policy_logits, value_pred,
                                batch.policy_targets, batch.value_targets,
                                aux_outputs,
                                batch.mobility_targets,
                                batch.queen_surround_targets,
                                batch.queen_surround_mask,
                                batch.final_mobility_targets,
                                batch.value_mask,
                                batch.graph_batch.piece_node_batch,
                                policy_softening=ps,
                            )
                    self._grad_scaler.scale(total_loss).backward()
                    if self.config.max_grad_norm > 0:
                        self._grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(), self.config.max_grad_norm
                        )
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    if is_transformer:
                        policy_logits, value_pred, aux_outputs = new_net(batch.token_batch)
                        total_loss, loss_dict = compute_transformer_loss(
                            policy_logits, value_pred,
                            batch.policy_targets, batch.value_targets,
                            aux_outputs,
                            batch.mobility_targets,
                            batch.queen_surround_targets,
                            batch.queen_surround_mask,
                            batch.final_mobility_targets,
                            batch.value_mask,
                            batch.board_token_batch,
                            policy_softening=ps,
                        )
                    else:
                        policy_logits, value_pred, aux_outputs = new_net(batch.graph_batch)
                        total_loss, loss_dict = compute_gnn_loss(
                            policy_logits, value_pred,
                            batch.policy_targets, batch.value_targets,
                            aux_outputs,
                            batch.mobility_targets,
                            batch.queen_surround_targets,
                            batch.queen_surround_mask,
                            batch.final_mobility_targets,
                            batch.value_mask,
                            batch.graph_batch.piece_node_batch,
                            policy_softening=ps,
                        )
                    total_loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(), self.config.max_grad_norm
                        )
                    optimizer.step()

                total_loss_sum += total_loss.item()
                num_batches += 1

        return new_net, total_loss_sum / max(num_batches, 1)

    # ── Arena ──────────────────────────────────────────────────────────

    def _arena_evaluate(self, new_net: torch.nn.Module) -> float:
        """Evaluate new network vs old using CPU MCTS with GNNEncoder."""
        from hive_engine.mcts import MCTS, MCTSConfig

        new_net.eval()
        self.best_net.eval()

        if self.config.encoder_type == "transformer":
            from hive_transformer.transformer_encoder import TransformerEncoder
            encoder = TransformerEncoder()
        else:
            encoder = GNNEncoder()
        mcts_config = MCTSConfig(
            num_simulations=self.config.arena_mcts_simulations,
            temperature=0.0,
        )

        new_wins = 0
        total_games = 0
        half = self.config.arena_games // 2

        for game_idx in range(self.config.arena_games):
            if game_idx < half:
                white_net, black_net = new_net, self.best_net
                new_color = Color.WHITE
            else:
                white_net, black_net = self.best_net, new_net
                new_color = Color.BLACK

            mcts_white = MCTS(white_net, encoder, mcts_config)
            mcts_black = MCTS(black_net, encoder, mcts_config)

            game = GameState()
            move_number = 0

            while (
                game.result == GameResult.IN_PROGRESS
                and move_number < self.config.max_game_length
            ):
                if game.current_player == Color.WHITE:
                    policy = mcts_white.search(game, move_number)
                else:
                    policy = mcts_black.search(game, move_number)

                action = int(np.argmax(policy))
                mask = encoder.get_legal_action_mask(game)

                if mask[action] > 0:
                    move = encoder.decode_action(action, game)
                else:
                    legal_actions = np.where(mask > 0)[0]
                    if len(legal_actions) == 0:
                        break
                    move = encoder.decode_action(int(legal_actions[0]), game)

                game.apply_move(move)
                move_number += 1

            total_games += 1
            result = game.result
            if (
                (result == GameResult.WHITE_WINS and new_color == Color.WHITE)
                or (result == GameResult.BLACK_WINS and new_color == Color.BLACK)
            ):
                new_wins += 1
            elif result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                new_wins += 0.5

        return new_wins / total_games if total_games > 0 else 0.5

    # ── Checkpointing ─────────────────────────────────────────────────

    def _save_checkpoint(self, net: torch.nn.Module, iteration: int) -> str:
        """Save a training checkpoint."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"hive_gpu_checkpoint_{iteration:04d}.pt",
        )
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "net_config": self.net_config,
                "train_config": self.config,
                "iteration": iteration,
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        net_class=None,
        config_overrides: GPUTrainConfig | None = None,
    ) -> GPUTrainer:
        """Restore a trainer from a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        net_config = checkpoint["net_config"]
        train_config = config_overrides or checkpoint.get(
            "train_config", GPUTrainConfig()
        )
        iteration = checkpoint.get("iteration", 0)

        trainer = cls(
            config=train_config,
            net_config=net_config,
            net_class=net_class,
        )
        trainer.best_net.load_state_dict(checkpoint["model_state_dict"])
        trainer.best_net = trainer.best_net.to(trainer.device)
        trainer._start_iteration = iteration + 1

        return trainer
