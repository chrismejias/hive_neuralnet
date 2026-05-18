"""
Transformer-based AlphaZero trainer for Hive.

Follows the same self-play → train → arena → promote cycle as
hive_engine.trainer.Trainer and hive_gnn.gnn_trainer.GNNTrainer,
but uses transformer token-sequence representations.

Includes KataGo-inspired improvements:
    - Auxiliary mobility head (per-board-token "can this piece move?")
    - Auxiliary queen surround head (per-board-token "adjacent to queen at game end?")
    - Auxiliary final mobility head (per-board-token "mobile at game end?")
    - Playout cap randomization (varied MCTS simulations per game)
    - Policy target pruning (zero out low-visit actions)
    - Continuous updates (skip arena, always promote new model)

Reuses from hive_engine:
    - compute_transformer_loss() (policy/value + aux)
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
from hive_engine.encoder import HiveEncoder
from hive_engine.endgame import generate_endgame
from hive_engine.queen_pressure import (
    find_queen_pressure_wins,
    generate_queen_pressure_state,
)
from hive_engine.game_state import GameState, GameResult
from hive_engine.hex_coord import ALL_DIRECTIONS
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import compute_transformer_loss
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.trainer import (
    EloTracker,
    IterationMetrics,
    MetricsLogger,
    SelfPlayStats,
    TBLogger,
    TrainStats,
)

from archive.legacy_transformer.hive_transformer.token_encoder import TokenEncoder
from archive.legacy_transformer.hive_transformer.transformer_encoder import TransformerEncoder
from archive.legacy_transformer.hive_transformer.transformer_net import TransformerConfig, HiveTransformer
from archive.modules.hive_transformer_cpu.transformer_replay_buffer import (
    TransformerTrainingExample,
    TransformerTrainingBatch,
    TokenReplayBuffer,
)
from hive_common.token_types import HiveTokenSequence


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

    # MCTS policy target pruning
    policy_prune_threshold: float = 0.0  # 0.0 = disabled

    # Arena
    arena_games: int = 10
    arena_mcts_simulations: int = 50
    arena_threshold: float = 0.55
    continuous_updates: bool = False  # skip arena, always promote new model

    # Game limits
    max_game_length: int = 200

    # Endgame bootstrapping
    endgame_ratio: float = 0.0  # fraction of self-play games starting from endgame positions

    # Buffer
    buffer_max_size: int = 50_000

    # Playout cap randomization
    playout_cap_randomization: bool = True
    playout_cap_min_fraction: float = 0.25   # min fraction of full simulations for capped games
    playout_cap_full_fraction: float = 0.25  # fraction of games using full playouts

    # Auxiliary loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    mobility_loss_weight: float = 0.15
    queen_surround_loss_weight: float = 0.15
    final_mobility_loss_weight: float = 0.15

    # Device
    device: str | None = None
    use_amp: bool | None = None

    # Batched inference (parallel self-play)
    num_selfplay_workers: int = 16      # concurrent game threads
    max_inference_batch_size: int = 32  # max states per GPU forward pass
    inference_wait_ms: float = 10.0    # max ms to wait for batch to fill; rule of thumb: num_workers * ~2ms

    # Queen-pressure curriculum learning
    queen_pressure_games: int = 0  # curriculum examples injected per iteration (0 = disabled)

    # Checkpointing
    checkpoint_dir: str = "checkpoints_transformer"
    metrics_file: str | None = None
    tensorboard_dir: str | None = None


# ── Transformer Trainer ───────────────────────────────────────────


class TransformerTrainer:
    """
    AlphaZero-style self-play trainer using Transformer state representation.

    Same pipeline as hive_gnn.GNNTrainer:
        1. Self-play → collect (HiveTokenSequence, policy, value, aux) examples
        2. Train new network on replay buffer
        3. Arena evaluation: new net vs best net (or continuous updates)
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
        self._token_encoder = TokenEncoder()
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

    # ── Auxiliary target computation ──────────────────────────────

    def _get_piece_list(
        self, game_state: GameState, sequence: HiveTokenSequence
    ) -> list[Piece]:
        """
        Get every piece for each board token, in token order.

        Mirrors the iteration order in ``TokenEncoder.encode()``
        (``board.grid.items()``, then bottom-to-top within each stack),
        returning one piece per board token.
        """
        board = game_state.board
        center_q, center_r = self._token_encoder._cached_center(board)
        pieces: list[Piece] = []
        for pos, stack in board.grid.items():
            grid_pos = HiveEncoder._hex_to_grid(
                pos.q, pos.r, center_q, center_r
            )
            if grid_pos is None:
                continue
            for piece in stack:  # bottom to top, all pieces
                pieces.append(piece)
        assert len(pieces) == sequence.num_board_tokens
        return pieces

    def _compute_mobility_target(
        self, game_state: GameState, sequence: HiveTokenSequence
    ) -> np.ndarray:
        """
        Compute per-board-token binary mobility target.

        For each board token, 1.0 if that piece has at least one legal
        move (belongs to current player, queen placed, not pinned,
        has destinations, AND is the top piece in its stack), 0.0 otherwise.
        """
        board = game_state.board
        color = game_state.current_player
        num_board = sequence.num_board_tokens
        mobility = np.zeros(num_board, dtype=np.float32)

        if not game_state._queen_placed[color]:
            return mobility

        articulation_points = board.find_articulation_points()
        center_q, center_r = self._token_encoder._cached_center(board)

        token_idx = 0
        for pos, stack in list(board.grid.items()):
            grid_pos = HiveEncoder._hex_to_grid(
                pos.q, pos.r, center_q, center_r
            )
            if grid_pos is None:
                continue

            stack_height = len(stack)
            for height, piece in enumerate(stack):
                is_top = (height == stack_height - 1)

                if (
                    is_top
                    and piece.color == color
                    and board.is_on_top(piece)
                ):
                    if not (
                        pos in articulation_points
                        and board.stack_height(pos) == 1
                    ):
                        destinations = board.generate_piece_moves(piece)
                        if len(destinations) > 0:
                            mobility[token_idx] = 1.0

                token_idx += 1

        return mobility

    def _get_final_queen_neighbors(
        self, game_state: GameState
    ) -> tuple[set[Piece], set[Piece], np.ndarray]:
        """
        At game end, find which pieces neighbor each queen.

        Returns:
            (white_queen_neighbors, black_queen_neighbors, mask) where
            each set contains Piece objects, and mask is (2,) float32
            with 1.0 if that queen is on the board.
        """
        board = game_state.board
        mask = np.zeros(2, dtype=np.float32)
        white_neighbors: set[Piece] = set()
        black_neighbors: set[Piece] = set()

        for piece, pos in board.piece_positions.items():
            if piece.piece_type != PieceType.QUEEN:
                continue
            color_idx = piece.color.value  # WHITE=0, BLACK=1
            mask[color_idx] = 1.0

            for d in ALL_DIRECTIONS:
                neighbor_pos = pos.neighbor(d)
                top = board.top_piece_at(neighbor_pos)
                if top is not None:
                    if piece.color == Color.WHITE:
                        white_neighbors.add(top)
                    else:
                        black_neighbors.add(top)

        return white_neighbors, black_neighbors, mask

    def _compute_final_mobility(
        self, game_state: GameState
    ) -> set[Piece]:
        """
        At game end, compute which pieces are mobile.

        Returns a set of Piece objects that have at least one legal
        move destination in the final game state. Checks both players.
        """
        board = game_state.board
        mobile_pieces: set[Piece] = set()

        for color in (Color.WHITE, Color.BLACK):
            if not game_state._queen_placed[color]:
                continue

            articulation_points = board.find_articulation_points()

            for piece in board.pieces_of_color(color):
                pos = board.position_of(piece)
                if pos is None:
                    continue
                if not board.is_on_top(piece):
                    continue
                if pos in articulation_points and board.stack_height(pos) == 1:
                    continue
                destinations = board.generate_piece_moves(piece)
                if len(destinations) > 0:
                    mobile_pieces.add(piece)

        return mobile_pieces

    # ── Main training loop ─────────────────────────────────────────

    def run(self) -> None:
        """Run the full training loop."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Print key config at startup for visibility
        cfg = self.config
        print(f"\nTransformer Training Config:")
        print(f"  iterations={cfg.num_iterations}, games={cfg.games_per_iteration}, "
              f"sims={cfg.mcts_simulations}, endgame_ratio={cfg.endgame_ratio}")
        print(f"  workers={cfg.num_selfplay_workers}, max_batch={cfg.max_inference_batch_size}, "
              f"wait_ms={cfg.inference_wait_ms:.0f} "
              f"(rule of thumb: {cfg.num_selfplay_workers * 2}ms for {cfg.num_selfplay_workers} workers)")
        print(f"  device={self.device}, amp={self.use_amp}")
        print(f"  checkpoint_dir={cfg.checkpoint_dir}")

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

                # 1b. Queen-pressure curriculum (optional)
                if self.config.queen_pressure_games > 0:
                    curriculum = self._generate_queen_pressure_examples(
                        self.config.queen_pressure_games
                    )
                    new_examples.extend(curriculum)
                    print(
                        f"  Curriculum: {len(curriculum)} queen-pressure examples"
                    )

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
                    f"value={train_stats.avg_value_loss:.4f}, "
                    f"mobility={train_stats.avg_mobility_loss:4f}, "
                    f"queen_surround={train_stats.avg_queen_surround_loss:4f}, "
                    f"final_mobility={train_stats.avg_final_mobility_loss:4f}) "
                    f"{metrics.train_time_sec:.1f}s"
                )

                # 3. Arena (or continuous updates)
                if self.config.continuous_updates:
                    # Skip arena: always accept the new model
                    metrics.arena_time_sec = 0.0
                    metrics.arena_new_win_rate = 0.0
                    metrics.arena_model_accepted = True
                    self.best_net = new_net
                    elo_rating = (
                        self.elo_tracker.ratings[-1]
                        if self.elo_tracker.ratings
                        else 0.0
                    )
                    metrics.elo_rating = elo_rating
                    print("  Arena: skipped (continuous updates)")
                    print("  [+] New model accepted (continuous updates)")
                else:
                    t0 = time.time()
                    win_rate = self._arena_evaluate(new_net, self.best_net)
                    metrics.arena_time_sec = time.time() - t0
                    metrics.arena_new_win_rate = win_rate
                    print(
                        f"  Arena: new model win rate = {win_rate:.1%} "
                        f"(threshold: {self.config.arena_threshold:.0%}, "
                        f"{metrics.arena_time_sec:.1f}s)"
                    )

                    # ELO
                    elo_rating = self.elo_tracker.update(
                        win_rate, self.config.arena_games
                    )
                    metrics.elo_rating = elo_rating
                    print(f"  ELO: {elo_rating:.0f}")

                    # Promote
                    if win_rate >= self.config.arena_threshold:
                        print("  [+] New model accepted!")
                        self.best_net = new_net
                        metrics.arena_model_accepted = True
                    else:
                        print("  [-] New model rejected, keeping current best.")
                        metrics.arena_model_accepted = False

                # 4. Checkpoint & logging
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
                    f"arena={metrics.arena_new_win_rate:.1%} "
                    f"{'[+]' if metrics.arena_model_accepted else '[-]'} "
                    f"buf={len(self.buffer)} time={total_time:.0f}s"
                )
        finally:
            logger.close()
            tb_logger.close()

    # ── Self-play ──────────────────────────────────────────────────

    def _get_game_params(self) -> tuple[int, bool, bool]:
        """Determine sim count, use_for_value, and use_endgame for one game."""
        if self.config.playout_cap_randomization:
            use_full = (
                np.random.random() < self.config.playout_cap_full_fraction
            )
            if use_full:
                sim_count = self.config.mcts_simulations
                use_for_value = True
            else:
                min_sims = max(
                    1,
                    int(
                        self.config.mcts_simulations
                        * self.config.playout_cap_min_fraction
                    ),
                )
                sim_count = np.random.randint(
                    min_sims, self.config.mcts_simulations + 1
                )
                use_for_value = False
        else:
            sim_count = self.config.mcts_simulations
            use_for_value = True

        use_endgame = (
            self.config.endgame_ratio > 0
            and np.random.random() < self.config.endgame_ratio
        )
        return sim_count, use_for_value, use_endgame

    # ── Queen-pressure curriculum ──────────────────────────────────

    def _generate_queen_pressure_examples(
        self, count: int
    ) -> list[TransformerTrainingExample]:
        """
        Generate *count* supervised curriculum examples from random board states.

        Each example is a single-move "game":
          - Policy target: uniform over winning moves (moves that immobilise /
            newly-adjacent to an already-immobile enemy queen).  If no winning
            move exists, uniform over all legal moves.
          - Value target: +1.0 when winning moves exist, -1.0 otherwise.
          - Auxiliary targets: mobility computed normally; queen_surround and
            final_mobility set to zeros (no end-of-game signal for one-step
            curriculum).

        Returns a (possibly shorter) list if valid states are scarce.
        """
        examples: list[TransformerTrainingExample] = []
        attempts = 0
        max_attempts = count * 20

        while len(examples) < count and attempts < max_attempts:
            attempts += 1
            gs = generate_queen_pressure_state()
            if gs is None or not gs.legal_moves():
                continue

            winning_moves = find_queen_pressure_wins(gs)
            target_moves = winning_moves if winning_moves else gs.legal_moves()
            value = 1.0 if winning_moves else -1.0

            # Policy: uniform over target moves
            policy = np.zeros(self.encoder.ACTION_SPACE_SIZE, dtype=np.float32)
            for move in target_moves:
                idx = self.encoder.encode_action(move, gs)
                if 0 <= idx < self.encoder.ACTION_SPACE_SIZE:
                    policy[idx] = 1.0
            policy_sum = policy.sum()
            if policy_sum == 0:
                continue
            policy /= policy_sum

            # Encode state as token sequence
            sequence = self.encoder.encode_state(gs)

            # Auxiliary targets
            mobility = self._compute_mobility_target(gs, sequence)
            n = sequence.num_board_tokens
            surround = np.zeros((n, 2), dtype=np.float32)
            surround_mask = np.array(
                [
                    float(gs._queen_placed[Color.WHITE]),
                    float(gs._queen_placed[Color.BLACK]),
                ],
                dtype=np.float32,
            )
            final_mob = np.zeros(n, dtype=np.float32)

            examples.append(
                TransformerTrainingExample(
                    sequence=sequence,
                    policy_target=policy,
                    value_target=value,
                    mobility_target=mobility,
                    queen_surround_target=surround,
                    queen_surround_mask=surround_mask,
                    final_mobility_target=final_mob,
                    use_for_value=True,
                )
            )

        return examples

    def _self_play_phase(
        self,
    ) -> tuple[list[TransformerTrainingExample], SelfPlayStats]:
        """Generate self-play games with batched inference across threads."""
        from concurrent.futures import ThreadPoolExecutor

        from hive_engine.batched_inference import (
            BatchedInferenceServer,
            BatchedPredictor,
        )
        from hive_common.token_types import HiveTokenBatch

        # Start batch inference server
        server = BatchedInferenceServer(
            net=self.best_net,
            collate_fn=HiveTokenBatch.collate,
            device=self.device,
            max_batch_size=self.config.max_inference_batch_size,
            max_wait_ms=self.config.inference_wait_ms,
        )
        server.start()
        batched_net = BatchedPredictor(server)

        # Pre-generate game params (uses np.random — not thread-safe)
        game_params = [
            self._get_game_params()
            for _ in range(self.config.games_per_iteration)
        ]

        # Run games in parallel threads
        num_workers = min(
            self.config.num_selfplay_workers,
            self.config.games_per_iteration,
        )

        all_examples: list[TransformerTrainingExample] = []
        stats = SelfPlayStats()

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = []
            for sim_count, use_for_value, use_endgame in game_params:
                futures.append(
                    pool.submit(
                        self._self_play_game,
                        batched_net,
                        use_endgame=use_endgame,
                        mcts_simulations_override=sim_count,
                    )
                )

            for i, future in enumerate(futures):
                examples, result, game_length = future.result()
                _, use_for_value, _ = game_params[i]

                if not use_for_value:
                    examples = [
                        ex._replace(use_for_value=False) for ex in examples
                    ]

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

        server.stop()
        return all_examples, stats

    def _self_play_game(
        self,
        net: HiveTransformer,
        use_endgame: bool = False,
        mcts_simulations_override: int | None = None,
    ) -> tuple[list[TransformerTrainingExample], GameResult, int]:
        """
        Play one self-play game and return Transformer training examples.

        Each turn:
          1. Encode state as HiveTokenSequence
          2. Compute mobility target
          3. Record piece identity list for queen surround backfill
          4. Run MCTS (uses net.predict(sequence, mask))
          5. Sample/argmax action from policy
          6. Record (sequence, policy, player, mobility, piece_list)

        After game ends, compute queen surround and final mobility targets.
        """
        sim_count = mcts_simulations_override or self.config.mcts_simulations
        mcts_config = MCTSConfig(
            num_simulations=sim_count,
            temperature=self.config.temperature,
            temperature_drop_move=self.config.temperature_drop_move,
            policy_prune_threshold=self.config.policy_prune_threshold,
        )
        mcts = MCTS(net, self.encoder, mcts_config)

        game = generate_endgame() if use_endgame else GameState()
        # history: (sequence, policy, player, mobility_target, piece_list)
        history: list[
            tuple[HiveTokenSequence, np.ndarray, Color, np.ndarray, list[Piece]]
        ] = []

        move_number = 0
        while (
            game.result == GameResult.IN_PROGRESS
            and move_number < self.config.max_game_length
        ):
            sequence = self.encoder.encode_state(game)
            current_player = game.current_player

            # Compute auxiliary targets for this position
            mobility = self._compute_mobility_target(game, sequence)
            piece_list = self._get_piece_list(game, sequence)

            policy = mcts.search(game, move_number=move_number)
            history.append(
                (sequence, policy, current_player, mobility, piece_list)
            )

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

        # Compute final-state auxiliary targets
        result = game.result
        white_neighbors, black_neighbors, surround_mask = (
            self._get_final_queen_neighbors(game)
        )
        final_mobile_pieces = self._compute_final_mobility(game)

        # Assign value, queen surround, and final mobility targets
        examples: list[TransformerTrainingExample] = []

        for sequence, policy, player, mobility, piece_list in history:
            # Value target
            if result == GameResult.DRAW or result == GameResult.IN_PROGRESS:
                value = 0.0
            elif result == GameResult.WHITE_WINS:
                value = 1.0 if player == Color.WHITE else -1.0
            else:
                value = 1.0 if player == Color.BLACK else -1.0

            # Queen surround target for this position's pieces
            surround = np.zeros(
                (sequence.num_board_tokens, 2), dtype=np.float32
            )
            # Final mobility target for this position's pieces
            final_mob = np.zeros(
                sequence.num_board_tokens, dtype=np.float32
            )
            for i, piece in enumerate(piece_list):
                if piece in white_neighbors:
                    surround[i, 0] = 1.0
                if piece in black_neighbors:
                    surround[i, 1] = 1.0
                if piece in final_mobile_pieces:
                    final_mob[i] = 1.0

            examples.append(
                TransformerTrainingExample(
                    sequence=sequence,
                    policy_target=policy,
                    value_target=value,
                    mobility_target=mobility,
                    queen_surround_target=surround,
                    queen_surround_mask=surround_mask,
                    final_mobility_target=final_mob,
                    use_for_value=True,  # may be overridden by playout cap
                )
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
        mobility_loss_sum = 0.0
        surround_loss_sum = 0.0
        final_mobility_loss_sum = 0.0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            batches_per_epoch = max(
                1, len(self.buffer) // self.config.batch_size
            )

            for _ in range(batches_per_epoch):
                batch = self.buffer.sample_batch(self.config.batch_size)
                batch = batch.to(self.device)

                optimizer.zero_grad()

                if self.use_amp and self._grad_scaler is not None:
                    with torch.amp.autocast("cuda"):
                        policy_logits, value_pred, aux_outputs = new_net(
                            batch.token_batch
                        )
                        total_loss, loss_dict = compute_transformer_loss(
                            policy_logits,
                            value_pred,
                            batch.policy_targets,
                            batch.value_targets,
                            aux_outputs,
                            batch.mobility_targets,
                            batch.queen_surround_targets,
                            batch.queen_surround_mask,
                            batch.final_mobility_targets,
                            batch.value_mask,
                            batch.board_token_batch,
                            policy_weight=self.config.policy_loss_weight,
                            value_weight=self.config.value_loss_weight,
                            mobility_weight=self.config.mobility_loss_weight,
                            queen_surround_weight=self.config.queen_surround_loss_weight,
                            final_mobility_weight=self.config.final_mobility_loss_weight,
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
                    policy_logits, value_pred, aux_outputs = new_net(
                        batch.token_batch
                    )
                    total_loss, loss_dict = compute_transformer_loss(
                        policy_logits,
                        value_pred,
                        batch.policy_targets,
                        batch.value_targets,
                        aux_outputs,
                        batch.mobility_targets,
                        batch.queen_surround_targets,
                        batch.queen_surround_mask,
                        batch.final_mobility_targets,
                        batch.value_mask,
                        batch.board_token_batch,
                        policy_weight=self.config.policy_loss_weight,
                        value_weight=self.config.value_loss_weight,
                        mobility_weight=self.config.mobility_loss_weight,
                        queen_surround_weight=self.config.queen_surround_loss_weight,
                        final_mobility_weight=self.config.final_mobility_loss_weight,
                    )
                    total_loss.backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            new_net.parameters(),
                            self.config.max_grad_norm,
                        )
                    optimizer.step()

                total_loss_sum += loss_dict["total_loss"].item()
                policy_loss_sum += loss_dict["policy_loss"].item()
                value_loss_sum += loss_dict["value_loss"].item()
                if "mobility_loss" in loss_dict:
                    mobility_loss_sum += loss_dict["mobility_loss"].item()
                if "queen_surround_loss" in loss_dict:
                    surround_loss_sum += loss_dict["queen_surround_loss"].item()
                if "final_mobility_loss" in loss_dict:
                    final_mobility_loss_sum += loss_dict["final_mobility_loss"].item()
                num_batches += 1

        if num_batches > 0:
            stats.avg_loss = total_loss_sum / num_batches
            stats.avg_policy_loss = policy_loss_sum / num_batches
            stats.avg_value_loss = value_loss_sum / num_batches
            stats.avg_mobility_loss = mobility_loss_sum / num_batches
            stats.avg_queen_surround_loss = surround_loss_sum / num_batches
            stats.avg_final_mobility_loss = final_mobility_loss_sum / num_batches
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

        Handles backward compatibility: old checkpoints without
        auxiliary head weights or new config fields load cleanly.

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

        # Add defaults for config fields that may be missing from old checkpoints
        _config_defaults = {
            "playout_cap_randomization": True,
            "playout_cap_min_fraction": 0.25,
            "playout_cap_full_fraction": 0.25,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 1.0,
            "mobility_loss_weight": 0.15,
            "queen_surround_loss_weight": 0.15,
            "final_mobility_loss_weight": 0.15,
            "continuous_updates": False,
            "policy_prune_threshold": 0.0,
            "num_selfplay_workers": 16,
            "max_inference_batch_size": 32,
            "inference_wait_ms": 10.0,
            "queen_pressure_games": 0,
        }
        for attr, default in _config_defaults.items():
            if not hasattr(train_config, attr):
                setattr(train_config, attr, default)

        if config_overrides:
            # num_iterations override is interpreted as "run N more iterations
            # from the checkpoint" rather than "stop at absolute iteration N".
            config_overrides = dict(config_overrides)  # don't mutate caller's dict
            if "num_iterations" in config_overrides:
                config_overrides["num_iterations"] = (
                    checkpoint["iteration"] + config_overrides["num_iterations"]
                )
            for k, v in config_overrides.items():
                if hasattr(train_config, k):
                    setattr(train_config, k, v)

        net_config = checkpoint["net_config"]

        # Add defaults for net config fields from old checkpoints
        _net_config_defaults = {
            "aux_mobility_enabled": True,
            "aux_queen_surround_enabled": True,
            "aux_final_mobility_enabled": True,
            "aux_mobility_hidden": 64,
            "aux_queen_surround_hidden": 64,
            "aux_final_mobility_hidden": 64,
        }
        for attr, default in _net_config_defaults.items():
            if not hasattr(net_config, attr):
                setattr(net_config, attr, default)

        trainer = cls(config=train_config, net_config=net_config)

        # Load with strict=False: old checkpoints won't have aux head weights
        trainer.best_net.load_state_dict(
            checkpoint["net_state_dict"], strict=False
        )
        trainer.best_net = trainer.best_net.to(trainer.device)
        trainer._start_iteration = checkpoint["iteration"] + 1

        if "elo_rating" in checkpoint:
            trainer.elo_tracker.ratings.append(checkpoint["elo_rating"])

        return trainer
