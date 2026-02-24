"""
Self-play training loop for Hive AI.

Implements the AlphaZero training pipeline:
  1. Self-play: Generate games using MCTS + current best network
  2. Train: Update network on replay buffer of (state, policy, value) examples
  3. Arena: Evaluate new network against current best
  4. Promote: Replace best network if new one wins enough

Usage:
    from hive_engine.trainer import Trainer, TrainConfig
    from hive_engine.neural_net import NetConfig

    trainer = Trainer(TrainConfig(), NetConfig.small())
    trainer.run()

    # Resume from checkpoint:
    trainer = Trainer.from_checkpoint("checkpoints/hive_checkpoint_0005.pt")
    trainer.run()
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.optim as optim

from hive_engine.augment import augment_example
from hive_engine.device import get_device, device_summary
from hive_engine.elo import EloTracker
from hive_engine.encoder import HiveEncoder
from hive_engine.endgame import generate_endgame
from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.metrics import IterationMetrics, MetricsLogger
from hive_engine.neural_net import HiveNet, NetConfig, compute_loss
from hive_engine.pieces import Color
from hive_engine.tb_logger import TBLogger


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Configuration for the training loop."""

    # Self-play
    num_iterations: int = 20
    games_per_iteration: int = 50
    mcts_simulations: int = 100
    temperature: float = 1.0
    temperature_drop_move: int = 20

    # Training
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_max_size: int = 50_000

    # Arena evaluation
    arena_games: int = 20
    arena_threshold: float = 0.55
    arena_mcts_simulations: int = 50

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Metrics logging
    metrics_file: str | None = None  # Set to path for JSON-lines log

    # Parallel self-play
    num_workers: int = 1  # >1 enables multiprocessing for self-play

    # Endgame bootstrap: fraction of games starting from near-endgame positions
    endgame_ratio: float = 0.0  # 0.0 = all normal, 1.0 = all endgame

    # Max game length to prevent infinite games
    max_game_length: int = 300

    # LR scheduling
    lr_schedule: str = "constant"  # "constant" | "cosine"
    lr_warmup_iterations: int = 0  # linear warmup for first N iterations
    lr_min: float = 1e-5  # minimum LR for cosine schedule

    # Gradient clipping
    max_grad_norm: float = 0.0  # 0 = disabled, >0 = clip grad norm

    # Data augmentation (6-fold hex rotational symmetry)
    augment_symmetry: bool = True

    # TensorBoard logging
    tensorboard_dir: str | None = None  # None = disabled

    # Device selection
    device: str | None = None  # None/"auto" = auto-detect, "cuda", "cuda:1", "mps", "cpu"

    # Mixed precision (float16 on CUDA for ~2x speedup, auto-enabled on CUDA)
    use_amp: bool | None = None  # None = auto (True on CUDA, False otherwise)


# ── Training Example ───────────────────────────────────────────────


class TrainingExample(NamedTuple):
    """A single training example from self-play."""

    state_tensor: np.ndarray  # (26, 13, 13)
    policy_target: np.ndarray  # (29407,) — MCTS visit distribution
    value_target: float  # +1 (won), -1 (lost), 0 (draw)


# ── Replay Buffer ──────────────────────────────────────────────────


class ReplayBuffer:
    """
    Circular replay buffer for training examples.

    Stores training examples from self-play games and provides
    random batch sampling for training.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add_examples(self, examples: list[TrainingExample]) -> None:
        """Add a list of training examples to the buffer."""
        self.buffer.extend(examples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Returns:
            (states, policies, values) as torch tensors:
                states: (batch, 26, 13, 13) float32
                policies: (batch, 29407) float32
                values: (batch, 1) float32
        """
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.array([s.state_tensor for s in samples], dtype=np.float32)
        policies = np.array([s.policy_target for s in samples], dtype=np.float32)
        values = np.array(
            [[s.value_target] for s in samples], dtype=np.float32
        )

        return (
            torch.from_numpy(states),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Self-Play Stats ───────────────────────────────────────────────


@dataclass
class SelfPlayStats:
    """Statistics from one round of self-play games."""

    num_games: int = 0
    num_examples: int = 0
    total_game_length: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0

    @property
    def avg_game_length(self) -> float:
        return self.total_game_length / self.num_games if self.num_games > 0 else 0.0


@dataclass
class TrainStats:
    """Statistics from one training phase."""

    avg_loss: float = 0.0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    num_batches: int = 0
    learning_rate: float = 0.0


# ── Parallel Self-Play Worker ─────────────────────────────────────


def _selfplay_worker(args: tuple) -> list[tuple]:
    """
    Worker function for parallel self-play (must be at module level for pickling).

    Args:
        args: (net_state_dict, net_config, train_config, num_games)

    Returns:
        List of (examples, result_value, game_length) tuples,
        where examples is a list of (state, policy, value) tuples.
    """
    net_state_dict, net_config, train_config, num_games = args

    # Detect device for this worker process
    device = get_device(train_config.device)
    net = HiveNet(net_config).to(device)
    net.load_state_dict(net_state_dict)
    net.eval()

    encoder = HiveEncoder()
    results = []

    endgame_ratio = train_config.endgame_ratio

    for _ in range(num_games):
        mcts_config = MCTSConfig(
            num_simulations=train_config.mcts_simulations,
            temperature=train_config.temperature,
            temperature_drop_move=train_config.temperature_drop_move,
        )
        mcts = MCTS(net, encoder, mcts_config)

        use_endgame = endgame_ratio > 0 and np.random.random() < endgame_ratio
        if use_endgame:
            game = generate_endgame()
        else:
            game = GameState()
        history: list[tuple[np.ndarray, np.ndarray, Color]] = []

        move_number = 0
        while (
            game.result == GameResult.IN_PROGRESS
            and move_number < train_config.max_game_length
        ):
            state_tensor = encoder.encode_state(game)
            current_player = game.current_player
            policy = mcts.search(game, move_number=move_number)
            history.append((state_tensor, policy, current_player))

            if move_number >= train_config.temperature_drop_move:
                action = int(np.argmax(policy))
            else:
                action = int(np.random.choice(len(policy), p=policy))

            legal_moves = game.legal_moves()
            mask = encoder.get_legal_action_mask(game, legal_moves)

            if mask[action] > 0:
                move = encoder.decode_action(action, game)
            else:
                legal_actions = np.where(mask > 0)[0]
                if len(legal_actions) == 0:
                    break
                best = legal_actions[np.argmax(policy[legal_actions])]
                move = encoder.decode_action(best, game)

            game.apply_move(move)
            move_number += 1

        result = game.result
        examples = []
        for state_tensor, policy, player in history:
            if result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                value = 0.0
            elif result == GameResult.WHITE_WINS:
                value = 1.0 if player == Color.WHITE else -1.0
            else:
                value = 1.0 if player == Color.BLACK else -1.0
            examples.append(TrainingExample(state_tensor, policy, value))

        # Apply data augmentation if enabled
        if train_config.augment_symmetry:
            augmented = []
            for ex in examples:
                for s, p, v in augment_example(ex.state_tensor, ex.policy_target, ex.value_target):
                    augmented.append(TrainingExample(s, p, v))
            examples = augmented

        results.append((examples, result, move_number))

    return results


# ── Trainer ────────────────────────────────────────────────────────


class Trainer:
    """
    AlphaZero-style self-play training for Hive.

    Orchestrates the full training pipeline: self-play → train → arena → promote.
    """

    def __init__(
        self,
        config: TrainConfig | None = None,
        net_config: NetConfig | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.net_config = net_config or NetConfig.small()
        self.encoder = HiveEncoder()
        self.buffer = ReplayBuffer(self.config.buffer_max_size)

        # Determine device
        self.device = get_device(self.config.device)

        # Mixed precision: auto-enable on CUDA if not explicitly set
        if self.config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = self.config.use_amp
        self._grad_scaler: torch.amp.GradScaler | None = None
        if self.use_amp:
            self._grad_scaler = torch.amp.GradScaler("cuda")

        # Create the initial network
        self.best_net = HiveNet(self.net_config).to(self.device)

        # Starting iteration (overridden by from_checkpoint)
        self._start_iteration: int = 1

        # ELO tracking
        self.elo_tracker = EloTracker()

    def _get_learning_rate(self, iteration: int) -> float:
        """
        Compute the learning rate for a given iteration.

        Supports constant, cosine annealing, and linear warmup schedules.

        Args:
            iteration: Current training iteration (1-indexed).

        Returns:
            The learning rate to use for this iteration.
        """
        base_lr = self.config.learning_rate

        # Linear warmup phase
        if (
            self.config.lr_warmup_iterations > 0
            and iteration <= self.config.lr_warmup_iterations
        ):
            return base_lr * (iteration / self.config.lr_warmup_iterations)

        if self.config.lr_schedule == "constant":
            return base_lr

        # Cosine annealing (after warmup)
        warmup = self.config.lr_warmup_iterations
        total = self.config.num_iterations - warmup
        progress = (iteration - warmup) / max(total, 1)
        lr_min = self.config.lr_min
        return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))

    def run(self) -> None:
        """
        Run the full training loop.

        For each iteration:
          1. Generate self-play games
          2. Train the network on the replay buffer
          3. Evaluate new network vs current best in an arena
          4. Promote new network if it wins enough
          5. Save checkpoint
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Set up metrics logger
        metrics_path = self.config.metrics_file
        if metrics_path is None:
            metrics_path = os.path.join(
                self.config.checkpoint_dir, "metrics.jsonl"
            )
        logger = MetricsLogger(metrics_path)
        logger.open()

        # Set up TensorBoard logger
        tb_logger = TBLogger(self.config.tensorboard_dir)

        try:
            for iteration in range(
                self._start_iteration, self.config.num_iterations + 1
            ):
                metrics = IterationMetrics(iteration=iteration)

                print(f"\n{'='*60}")
                print(f"Iteration {iteration}/{self.config.num_iterations}")
                print(f"{'='*60}")

                # 1. Self-play
                t0 = time.time()
                new_examples, sp_stats = self._self_play_phase_with_stats()
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
                new_net, train_stats = self._train_phase_with_stats(iteration)
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

                # 4. ELO update
                elo_rating = self.elo_tracker.update(
                    win_rate, self.config.arena_games
                )
                metrics.elo_rating = elo_rating
                print(f"  ELO: {elo_rating:.0f}")

                # 5. Promote
                if win_rate >= self.config.arena_threshold:
                    print("  ✓ New model accepted!")
                    self.best_net = new_net
                    metrics.arena_model_accepted = True
                else:
                    print("  ✗ New model rejected, keeping current best.")
                    metrics.arena_model_accepted = False

                # 6. Checkpoint & logging
                metrics.buffer_size = len(self.buffer)
                self._save_checkpoint(self.best_net, iteration)
                logger.log_iteration(metrics)
                tb_logger.log_iteration(metrics)

                self._print_iteration_summary(metrics)
        finally:
            logger.close()
            tb_logger.close()

    def _print_iteration_summary(self, m: IterationMetrics) -> None:
        """Print a concise summary line for the iteration."""
        total_time = (
            m.selfplay_time_sec + m.train_time_sec + m.arena_time_sec
        )
        accepted_str = "✓" if m.arena_model_accepted else "✗"
        print(
            f"  Summary: iter={m.iteration} loss={m.train_avg_loss:.4f} "
            f"lr={m.train_learning_rate:.2e} elo={m.elo_rating:.0f} "
            f"arena={m.arena_new_win_rate:.1%}{accepted_str} "
            f"buf={m.buffer_size} time={total_time:.0f}s"
        )

    # ── Self-play with stats ──────────────────────────────────────

    def _self_play_phase(self) -> list[TrainingExample]:
        """Generate self-play games and collect training examples."""
        examples, _ = self._self_play_phase_with_stats()
        return examples

    def _self_play_phase_with_stats(
        self,
    ) -> tuple[list[TrainingExample], SelfPlayStats]:
        """Generate self-play games, returning examples and statistics."""
        num_workers = self.config.num_workers
        num_games = self.config.games_per_iteration

        if num_workers > 1 and num_games > 1:
            return self._parallel_self_play(num_workers, num_games)

        return self._sequential_self_play(num_games)

    def _sequential_self_play(
        self, num_games: int
    ) -> tuple[list[TrainingExample], SelfPlayStats]:
        """Run self-play games sequentially."""
        all_examples: list[TrainingExample] = []
        stats = SelfPlayStats()
        endgame_ratio = self.config.endgame_ratio

        for game_idx in range(num_games):
            use_endgame = endgame_ratio > 0 and np.random.random() < endgame_ratio
            examples, result, game_length = self._self_play_game_with_result(
                self.best_net, use_endgame=use_endgame
            )

            # Apply data augmentation (6-fold hex rotational symmetry)
            if self.config.augment_symmetry:
                augmented = []
                for ex in examples:
                    for s, p, v in augment_example(
                        ex.state_tensor, ex.policy_target, ex.value_target
                    ):
                        augmented.append(TrainingExample(s, p, v))
                examples = augmented

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

    def _parallel_self_play(
        self, num_workers: int, num_games: int
    ) -> tuple[list[TrainingExample], SelfPlayStats]:
        """Run self-play games across multiple worker processes."""
        # Move model to CPU for serialization
        cpu_state_dict = {
            k: v.cpu() for k, v in self.best_net.state_dict().items()
        }

        # Divide games among workers
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1

        worker_args = [
            (cpu_state_dict, self.net_config, self.config, n)
            for n in games_per_worker
            if n > 0
        ]

        all_examples: list[TrainingExample] = []
        stats = SelfPlayStats()

        with mp.Pool(processes=len(worker_args)) as pool:
            worker_results = pool.map(_selfplay_worker, worker_args)

        for game_results in worker_results:
            for examples, result, game_length in game_results:
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

    def _self_play_game(self, net: HiveNet) -> list[TrainingExample]:
        """Play one self-play game and return training examples."""
        examples, _, _ = self._self_play_game_with_result(net)
        return examples

    def _self_play_game_with_result(
        self, net: HiveNet, use_endgame: bool = False
    ) -> tuple[list[TrainingExample], GameResult, int]:
        """
        Play one self-play game and return training examples, result, and length.

        Each turn:
          1. Run MCTS to get policy
          2. Sample action from policy (or argmax after temperature drop)
          3. Record (state_tensor, policy, ?) — value filled in later

        After game ends, assign value targets from the game outcome.

        Args:
            net: The neural network to use for MCTS.
            use_endgame: If True, start from a random endgame position.

        Returns:
            (examples, game_result, move_count)
        """
        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            temperature=self.config.temperature,
            temperature_drop_move=self.config.temperature_drop_move,
        )
        mcts = MCTS(net, self.encoder, mcts_config)

        if use_endgame:
            game = generate_endgame()
        else:
            game = GameState()
        history: list[tuple[np.ndarray, np.ndarray, Color]] = []

        move_number = 0
        while (
            game.result == GameResult.IN_PROGRESS
            and move_number < self.config.max_game_length
        ):
            # Encode current state
            state_tensor = self.encoder.encode_state(game)
            current_player = game.current_player

            # Run MCTS
            policy = mcts.search(game, move_number=move_number)

            # Record state and policy (value filled later)
            history.append((state_tensor, policy, current_player))

            # Select action
            if move_number >= self.config.temperature_drop_move:
                # Greedy
                action = int(np.argmax(policy))
            else:
                # Sample from policy
                action = int(np.random.choice(len(policy), p=policy))

            # Decode and apply
            legal_moves = game.legal_moves()
            mask = self.encoder.get_legal_action_mask(game, legal_moves)

            if mask[action] > 0:
                move = self.encoder.decode_action(action, game)
            else:
                # Fallback: pick the legal move with highest policy probability
                legal_actions = np.where(mask > 0)[0]
                if len(legal_actions) == 0:
                    break
                best = legal_actions[np.argmax(policy[legal_actions])]
                move = self.encoder.decode_action(best, game)

            game.apply_move(move)
            move_number += 1

        # Assign value targets
        result = game.result
        examples: list[TrainingExample] = []

        for state_tensor, policy, player in history:
            if result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                value = 0.0
            elif result == GameResult.WHITE_WINS:
                value = 1.0 if player == Color.WHITE else -1.0
            else:  # BLACK_WINS
                value = 1.0 if player == Color.BLACK else -1.0

            examples.append(TrainingExample(state_tensor, policy, value))

        return examples, result, move_number

    # ── Training with stats ───────────────────────────────────────

    def _train_phase(self, iteration: int = 1) -> HiveNet:
        """Train a new network on the replay buffer."""
        net, _ = self._train_phase_with_stats(iteration)
        return net

    def _train_phase_with_stats(
        self, iteration: int = 1
    ) -> tuple[HiveNet, TrainStats]:
        """
        Train a new network on the replay buffer, returning stats.

        Creates a copy of the best network, trains it for num_epochs,
        and returns the trained copy along with training statistics.

        Args:
            iteration: Current iteration (for LR scheduling).
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
            # Number of batches per epoch ≈ buffer_size / batch_size
            batches_per_epoch = max(
                1, len(self.buffer) // self.config.batch_size
            )

            for _ in range(batches_per_epoch):
                states, policies, values = self.buffer.sample_batch(
                    self.config.batch_size
                )
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                optimizer.zero_grad()

                # Mixed precision forward pass (CUDA only)
                if self.use_amp and self._grad_scaler is not None:
                    with torch.amp.autocast("cuda"):
                        policy_logits, value_pred = new_net(states)
                        total_loss, p_loss, v_loss = compute_loss(
                            policy_logits, value_pred, policies, values
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
                    policy_logits, value_pred = new_net(states)
                    total_loss, p_loss, v_loss = compute_loss(
                        policy_logits, value_pred, policies, values
                    )
                    total_loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(), self.config.max_grad_norm
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

    # ── Arena ─────────────────────────────────────────────────────

    def _arena_evaluate(
        self, new_net: HiveNet, old_net: HiveNet
    ) -> float:
        """
        Evaluate new network against old network by playing games.

        Plays arena_games total (half with new as white, half as black).
        Returns the win rate of the new network.
        """
        new_net.eval()
        old_net.eval()

        new_wins = 0
        total_games = 0

        mcts_config = MCTSConfig(
            num_simulations=self.config.arena_mcts_simulations,
            temperature=0.0,  # Greedy in arena
        )

        half = self.config.arena_games // 2

        for game_idx in range(self.config.arena_games):
            # Alternate colors
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
                or (result == GameResult.BLACK_WINS and new_color == Color.BLACK)
            ):
                new_wins += 1
            elif result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                new_wins += 0.5  # Count draws as half wins

        return new_wins / total_games if total_games > 0 else 0.5

    # ── Checkpointing ─────────────────────────────────────────────

    def _save_checkpoint(self, net: HiveNet, iteration: int) -> str:
        """
        Save a full training checkpoint.

        Saves model weights, net config, training config, iteration number,
        and the replay buffer so training can be fully resumed.

        Returns the checkpoint path.
        """
        path = os.path.join(
            self.config.checkpoint_dir,
            f"hive_checkpoint_{iteration:04d}.pt",
        )
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "net_config": self.net_config,
                "train_config": self.config,
                "iteration": iteration,
                "buffer_examples": list(self.buffer.buffer),
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config_overrides: TrainConfig | None = None,
    ) -> Trainer:
        """
        Create a Trainer fully restored from a checkpoint.

        Restores model weights, configurations, replay buffer, and
        sets the starting iteration so training resumes from the
        next iteration.

        Args:
            path: Path to the checkpoint file.
            config_overrides: If provided, overrides the saved TrainConfig.
                Useful for changing num_iterations or learning_rate on resume.

        Returns:
            A Trainer ready to call .run().
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        net_config = checkpoint["net_config"]
        train_config = config_overrides or checkpoint.get(
            "train_config", TrainConfig()
        )
        iteration = checkpoint.get("iteration", 0)

        trainer = cls(config=train_config, net_config=net_config)
        trainer.best_net.load_state_dict(checkpoint["model_state_dict"])
        trainer.best_net = trainer.best_net.to(trainer.device)

        # Restore replay buffer
        buffer_data = checkpoint.get("buffer_examples", [])
        if buffer_data:
            trainer.buffer.add_examples(buffer_data)

        # Resume from the next iteration
        trainer._start_iteration = iteration + 1

        return trainer

    @staticmethod
    def load_checkpoint(
        path: str, device: torch.device | str | None = None
    ) -> HiveNet:
        """
        Load just the model from a checkpoint file (for inference).

        Args:
            path: Path to the checkpoint file.
            device: Device to load onto. Accepts torch.device, string
                ("cuda", "mps", "cpu"), or None for auto-detect.
        """
        if device is None:
            resolved = get_device()
        elif isinstance(device, str):
            resolved = get_device(device)
        else:
            resolved = device

        checkpoint = torch.load(
            path, map_location=resolved, weights_only=False
        )
        net_config = checkpoint["net_config"]
        net = HiveNet(net_config).to(resolved)
        net.load_state_dict(checkpoint["model_state_dict"])
        return net
