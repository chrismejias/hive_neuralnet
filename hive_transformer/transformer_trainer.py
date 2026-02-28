"""
Transformer-based AlphaZero trainer for Hive.

Follows the same self-play → train → arena → promote cycle as
hive_engine.trainer.Trainer and hive_gnn.gnn_trainer.GNNTrainer,
but uses transformer token-sequence representations.

Reuses from hive_engine:
    - compute_loss() (same policy/value loss)
    - MCTS (duck-typed with TransformerEncoder/HiveTransformer)
    - MetricsLogger, TBLogger, EloTracker, get_device()
    - GameState, GameResult, Color
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim

from hive_engine.device import get_device, device_summary
from hive_engine.endgame import generate_endgame
from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import compute_loss
from hive_engine.pieces import Color
from hive_engine.trainer import (
    EloTracker,
    IterationMetrics,
    MetricsLogger,
    SelfPlayStats,
    TBLogger,
    TrainStats,
)

from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer
from hive_transformer.transformer_replay_buffer import (
    TransformerTrainingExample,
    TokenReplayBuffer,
)
from hive_transformer.token_types import HiveTokenSequence


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class TransformerTrainConfig:
    """Configuration for Transformer training pipeline."""

    # Training loop
    num_iterations: int = 20
    games_per_iteration: int = 50
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # LR schedule
    lr_schedule: str = "cosine"
    lr_warmup_iterations: int = 2
    lr_min: float = 1e-5

    # MCTS
    mcts_simulations: int = 100
    temperature: float = 1.0
    temperature_drop_move: int = 20

    # Arena
    arena_games: int = 10
    arena_mcts_simulations: int = 50
    arena_threshold: float = 0.55

    # Game limits
    max_game_length: int = 200

    # Endgame bootstrapping
    endgame_ratio: float = 0.0  # fraction of self-play games starting from endgame positions

    # Buffer
    buffer_max_size: int = 50_000

    # Device
    device: str | None = None
    use_amp: bool | None = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints_transformer"
    metrics_file: str | None = None
    tensorboard_dir: str | None = None


# ── Transformer Trainer ───────────────────────────────────────────


class TransformerTrainer:
    """
    AlphaZero-style self-play trainer using Transformer state representation.

    Same pipeline as hive_engine.Trainer / hive_gnn.GNNTrainer:
        1. Self-play → collect (HiveTokenSequence, policy, value) examples
        2. Train new network on replay buffer
        3. Arena evaluation: new net vs best net
        4. Promote if new net wins enough
        5. Save checkpoint
    """

    def __init__(
        self,
        config: TransformerTrainConfig | None = None,
        net_config: TransformerConfig | None = None,
    ) -> None:
        self.config = config or TransformerTrainConfig()
        self.net_config = net_config or TransformerConfig.small()
        self.encoder = TransformerEncoder()
        self.buffer = TokenReplayBuffer(self.config.buffer_max_size)

        # Device
        self.device = get_device(self.config.device)

        # Mixed precision (CUDA only)
        if self.config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = self.config.use_amp
        self._grad_scaler: torch.amp.GradScaler | None = None
        if self.use_amp:
            self._grad_scaler = torch.amp.GradScaler("cuda")

        # Create initial network
        self.best_net = HiveTransformer(self.net_config).to(self.device)

        # Starting iteration (overridden by from_checkpoint)
        self._start_iteration: int = 1

        # ELO tracking
        self.elo_tracker = EloTracker()

    def _get_learning_rate(self, iteration: int) -> float:
        """Compute LR with optional cosine annealing + warmup."""
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
        return lr_min + 0.5 * (base_lr - lr_min) * (
            1 + math.cos(math.pi * progress)
        )

    # ── Main training loop ─────────────────────────────────────────

    def run(self) -> None:
        """Run the full training loop."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        metrics_path = self.config.metrics_file
        if metrics_path is None:
            metrics_path = os.path.join(
                self.config.checkpoint_dir, "metrics.jsonl"
            )
        logger = MetricsLogger(metrics_path)
        logger.open()

        tb_logger = TBLogger(self.config.tensorboard_dir)

        try:
            for iteration in range(
                self._start_iteration, self.config.num_iterations + 1
            ):
                metrics = IterationMetrics(iteration=iteration)

                print(f"\n{'='*60}")
                print(
                    f"Transformer Iteration "
                    f"{iteration}/{self.config.num_iterations}"
                )
                print(f"{'='*60}")

                # 1. Self-play
                t0 = time.time()
                new_examples, sp_stats = self._self_play_phase()
                self.buffer.add_examples(new_examples)
                metrics.selfplay_time_sec = time.time() - t0
                metrics.selfplay_num_games = sp_stats.num_games
                metrics.selfplay_num_examples = sp_stats.num_examples
                metrics.selfplay_avg_game_length = sp_stats.avg_game_length
                metrics.selfplay_white_wins = sp_stats.white_wins
                metrics.selfplay_black_wins = sp_stats.black_wins
                metrics.selfplay_draws = sp_stats.draws
                print(
                    f"  Self-play: {sp_stats.num_examples} examples from "
                    f"{sp_stats.num_games} games "
                    f"(avg {sp_stats.avg_game_length:.0f} moves, "
                    f"W:{sp_stats.white_wins} B:{sp_stats.black_wins} "
                    f"D:{sp_stats.draws}, "
                    f"{metrics.selfplay_time_sec:.1f}s)"
                )

                # 2. Train
                t0 = time.time()
                new_net, train_stats = self._train_phase(iteration)
                metrics.train_time_sec = time.time() - t0
                metrics.train_learning_rate = train_stats.learning_rate
                metrics.train_avg_loss = train_stats.avg_loss
                metrics.train_avg_policy_loss = train_stats.avg_policy_loss
                metrics.train_avg_value_loss = train_stats.avg_value_loss
                metrics.train_num_batches = train_stats.num_batches
                print(
                    f"  Training: loss={train_stats.avg_loss:.4f} "
                    f"(policy={train_stats.avg_policy_loss:.4f}, "
                    f"value={train_stats.avg_value_loss:.4f}) "
                    f"{metrics.train_time_sec:.1f}s"
                )

                # 3. Arena
                t0 = time.time()
                win_rate = self._arena_evaluate(new_net, self.best_net)
                metrics.arena_time_sec = time.time() - t0
                metrics.arena_new_win_rate = win_rate
                print(
                    f"  Arena: new model win rate = {win_rate:.1%} "
                    f"(threshold: {self.config.arena_threshold:.0%}, "
                    f"{metrics.arena_time_sec:.1f}s)"
                )

                # 4. ELO
                elo_rating = self.elo_tracker.update(
                    win_rate, self.config.arena_games
                )
                metrics.elo_rating = elo_rating
                print(f"  ELO: {elo_rating:.0f}")

                # 5. Promote
                if win_rate >= self.config.arena_threshold:
                    print("  [+] New model accepted!")
                    self.best_net = new_net
                    metrics.arena_model_accepted = True
                else:
                    print("  [-] New model rejected, keeping current best.")
                    metrics.arena_model_accepted = False

                # 6. Checkpoint & logging
                metrics.buffer_size = len(self.buffer)
                self._save_checkpoint(self.best_net, iteration)
                logger.log_iteration(metrics)
                tb_logger.log_iteration(metrics)

                total_time = (
                    metrics.selfplay_time_sec
                    + metrics.train_time_sec
                    + metrics.arena_time_sec
                )
                print(
                    f"  Summary: iter={iteration} "
                    f"loss={train_stats.avg_loss:.4f} "
                    f"lr={train_stats.learning_rate:.2e} "
                    f"elo={elo_rating:.0f} "
                    f"arena={win_rate:.1%} "
                    f"{'[+]' if metrics.arena_model_accepted else '[-]'} "
                    f"buf={len(self.buffer)} time={total_time:.0f}s"
                )
        finally:
            logger.close()
            tb_logger.close()

    # ── Self-play ──────────────────────────────────────────────────

    def _self_play_phase(
        self,
    ) -> tuple[list[TransformerTrainingExample], SelfPlayStats]:
        """Generate self-play games and collect training examples."""
        all_examples: list[TransformerTrainingExample] = []
        stats = SelfPlayStats()

        for _ in range(self.config.games_per_iteration):
            use_endgame = (
                self.config.endgame_ratio > 0
                and np.random.random() < self.config.endgame_ratio
            )
            examples, result, game_length = self._self_play_game(
                self.best_net, use_endgame=use_endgame
            )
            all_examples.extend(examples)

            stats.num_games += 1
            stats.num_examples += len(examples)
            stats.total_game_length += game_length

            if result == GameResult.WHITE_WINS:
                stats.white_wins += 1
            elif result == GameResult.BLACK_WINS:
                stats.black_wins += 1
            else:
                stats.draws += 1

        return all_examples, stats

    def _self_play_game(
        self, net: HiveTransformer, use_endgame: bool = False
    ) -> tuple[list[TransformerTrainingExample], GameResult, int]:
        """
        Play one self-play game and return Transformer training examples.

        Each turn:
          1. Encode state as HiveTokenSequence
          2. Run MCTS (uses net.predict(sequence, mask))
          3. Sample/argmax action from policy
          4. Record (sequence, policy, player)

        After game ends, assign value targets.
        """
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            temperature=self.config.temperature,
            temperature_drop_move=self.config.temperature_drop_move,
        )
        mcts = MCTS(net, self.encoder, mcts_config)

        game = generate_endgame() if use_endgame else GameState()
        history: list[tuple[HiveTokenSequence, np.ndarray, Color]] = []

        move_number = 0
        while (
            game.result == GameResult.IN_PROGRESS
            and move_number < self.config.max_game_length
        ):
            sequence = self.encoder.encode_state(game)
            current_player = game.current_player

            policy = mcts.search(game, move_number=move_number)
            history.append((sequence, policy, current_player))

            if move_number >= self.config.temperature_drop_move:
                action = int(np.argmax(policy))
            else:
                action = int(np.random.choice(len(policy), p=policy))

            legal_moves = game.legal_moves()
            mask = self.encoder.get_legal_action_mask(game, legal_moves)

            if mask[action] > 0:
                move = self.encoder.decode_action(action, game)
            else:
                legal_actions = np.where(mask > 0)[0]
                if len(legal_actions) == 0:
                    break
                best = legal_actions[np.argmax(policy[legal_actions])]
                move = self.encoder.decode_action(best, game)

            game.apply_move(move)
            move_number += 1

        # Assign value targets
        result = game.result
        examples: list[TransformerTrainingExample] = []

        for sequence, policy, player in history:
            if result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                value = 0.0
            elif result == GameResult.WHITE_WINS:
                value = 1.0 if player == Color.WHITE else -1.0
            else:
                value = 1.0 if player == Color.BLACK else -1.0

            examples.append(
                TransformerTrainingExample(sequence, policy, value)
            )

        return examples, result, move_number

    # ── Training ───────────────────────────────────────────────────

    def _train_phase(
        self, iteration: int = 1
    ) -> tuple[HiveTransformer, TrainStats]:
        """
        Train a new network on the replay buffer.

        Creates a copy of the best network, trains for num_epochs,
        returns trained copy + stats.
        """
        new_net = copy.deepcopy(self.best_net)
        new_net.train()
        stats = TrainStats()

        lr = self._get_learning_rate(iteration)
        stats.learning_rate = lr

        optimizer = optim.Adam(
            new_net.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        if len(self.buffer) < self.config.batch_size:
            return new_net, stats

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            batches_per_epoch = max(
                1, len(self.buffer) // self.config.batch_size
            )

            for _ in range(batches_per_epoch):
                token_batch, policies, values = self.buffer.sample_batch(
                    self.config.batch_size
                )
                token_batch = token_batch.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                optimizer.zero_grad()

                if self.use_amp and self._grad_scaler is not None:
                    with torch.amp.autocast("cuda"):
                        policy_logits, value_pred = new_net(token_batch)
                        total_loss, p_loss, v_loss = compute_loss(
                            policy_logits, value_pred, policies, values
                        )
                    self._grad_scaler.scale(total_loss).backward()
                    if self.config.max_grad_norm > 0:
                        self._grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(),
                            self.config.max_grad_norm,
                        )
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    policy_logits, value_pred = new_net(token_batch)
                    total_loss, p_loss, v_loss = compute_loss(
                        policy_logits, value_pred, policies, values
                    )
                    total_loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(),
                            self.config.max_grad_norm,
                        )
                    optimizer.step()

                total_loss_sum += total_loss.item()
                policy_loss_sum += p_loss.item()
                value_loss_sum += v_loss.item()
                num_batches += 1

        if num_batches > 0:
            stats.avg_loss = total_loss_sum / num_batches
            stats.avg_policy_loss = policy_loss_sum / num_batches
            stats.avg_value_loss = value_loss_sum / num_batches
            stats.num_batches = num_batches

        return new_net, stats

    # ── Arena ──────────────────────────────────────────────────────

    def _arena_evaluate(
        self, new_net: HiveTransformer, old_net: HiveTransformer
    ) -> float:
        """
        Evaluate new network against old by playing arena games.

        Half with new as white, half as black. Returns win rate.
        """
        new_net.eval()
        old_net.eval()

        new_wins = 0.0
        total_games = 0

        mcts_config = MCTSConfig(
            num_simulations=self.config.arena_mcts_simulations,
            temperature=0.0,
        )

        half = self.config.arena_games // 2

        for game_idx in range(self.config.arena_games):
            if game_idx < half:
                white_net, black_net = new_net, old_net
                new_color = Color.WHITE
            else:
                white_net, black_net = old_net, new_net
                new_color = Color.BLACK

            mcts_white = MCTS(white_net, self.encoder, mcts_config)
            mcts_black = MCTS(black_net, self.encoder, mcts_config)

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
                mask = self.encoder.get_legal_action_mask(game)

                if mask[action] > 0:
                    move = self.encoder.decode_action(action, game)
                else:
                    legal_actions = np.where(mask > 0)[0]
                    if len(legal_actions) == 0:
                        break
                    move = self.encoder.decode_action(
                        int(legal_actions[0]), game
                    )

                game.apply_move(move)
                move_number += 1

            total_games += 1
            result = game.result
            if (
                (result == GameResult.WHITE_WINS and new_color == Color.WHITE)
                or (
                    result == GameResult.BLACK_WINS
                    and new_color == Color.BLACK
                )
            ):
                new_wins += 1
            elif (
                result == GameResult.DRAW
                or result == GameResult.IN_PROGRESS
            ):
                new_wins += 0.5

        return new_wins / total_games if total_games > 0 else 0.5

    # ── Checkpointing ──────────────────────────────────────────────

    def _save_checkpoint(
        self, net: HiveTransformer, iteration: int
    ) -> str:
        """Save a training checkpoint. Returns the path."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"hive_transformer_checkpoint_{iteration:04d}.pt",
        )
        checkpoint = {
            "iteration": iteration,
            "net_state_dict": net.state_dict(),
            "net_config": self.net_config,
            "train_config": self.config,
            "elo_rating": self.elo_tracker.current_rating,
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        return path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_overrides: dict | None = None,
    ) -> TransformerTrainer:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            config_overrides: Optional dict of TransformerTrainConfig overrides.

        Returns:
            A TransformerTrainer ready to continue training.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        train_config = checkpoint["train_config"]
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(train_config, k):
                    setattr(train_config, k, v)

        net_config = checkpoint["net_config"]
        trainer = cls(config=train_config, net_config=net_config)

        trainer.best_net.load_state_dict(checkpoint["net_state_dict"])
        trainer.best_net = trainer.best_net.to(trainer.device)
        trainer._start_iteration = checkpoint["iteration"] + 1

        if "elo_rating" in checkpoint:
            trainer.elo_tracker.ratings.append(checkpoint["elo_rating"])

        return trainer
