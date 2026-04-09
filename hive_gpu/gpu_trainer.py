"""
GPU-accelerated training loop for Hive AI.

Uses GPUMCTSOrchestrator for batched self-play, with the same
train → arena → promote pipeline as hive_engine/trainer.py.

Usage:
    from hive_gpu.gpu_trainer import GPUTrainer, GPUTrainConfig
    from hive_transformer.transformer_net import TransformerConfig, HiveTransformer

    trainer = GPUTrainer(GPUTrainConfig(), TransformerConfig.small(), HiveTransformer)
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

try:
    from hive_gnn.gnn_encoder import GNNEncoder
    from hive_gnn.gnn_replay_buffer import GNNTrainingExample, GraphReplayBuffer
except ImportError:
    GNNEncoder = None          # type: ignore[assignment,misc]
    GNNTrainingExample = None  # type: ignore[assignment,misc]
    GraphReplayBuffer = None   # type: ignore[assignment,misc]
from archive.modules.hive_transformer_cpu.transformer_replay_buffer import (
    TransformerTrainingExample,
    TokenReplayBuffer,
    GPUTokenReplayBuffer,
)

from archive.modules.hive_gpu_hybrid.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig, GPUTrainingExample
from hive_gpu.gpu_native_mcts import GPUNativeMCTSOrchestrator


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
    batch_size: int = 256  # bumped from 64 — better GPU utilisation on 24 GB+
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4

    # Replay buffer
    buffer_max_size: int = 100_000

    # Arena
    arena_games: int = 20
    arena_threshold: float = 0.55
    arena_mcts_simulations: int = 50

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_keep_every: int = 0  # 0 = keep all; N = keep only multiples of N

    # LR scheduling
    lr_schedule: str = "cosine"
    lr_warmup_iterations: int = 3
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
    skip_arena: bool = True    # always accept new model, skip arena games

    # Policy softening (KataGo-style)
    policy_softening: float = 0.03  # mix weight toward uniform over legal moves

    # Policy surprise weighting (KataGo-style)
    policy_surprise_weight: float = 1.0  # scale for KL-based sample weighting (0=off)

    # Monte Carlo Graph Search (DAG with transposition detection)
    use_mcgs: bool = False  # use MCGS instead of tree MCTS (opt-in via --mcgs)

    # Policy target pruning
    policy_target_pruning: float = 0.02  # zero out visits below this fraction of max

    # Root policy temperature (soften NN prior at root before Dirichlet noise)
    root_policy_temp: float = 1.1  # >1 = softer prior distribution

    # Shaped Dirichlet noise (scale alpha inversely with number of legal moves)
    shaped_dirichlet: bool = True

    # GPU-native MCTS (tree on GPU, CUDA kernels for select/expand/backprop)
    use_gpu_native: bool = False

    # Gumbel AlphaZero search (sequential halving, no MCTS tree)
    use_gumbel: bool = True
    gumbel_max_considered: int = 16  # top-k actions before halving
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0

    # Draw downsampling: keep only this fraction of drawn games (1.0 = keep all)
    draw_keep_rate: float = 1.0

    # Queen pressure value shaping: for drawn games, use queen surround
    # Expansion pieces: 3-bit mask (bit 0=Mosquito, 1=Ladybug, 2=Pillbug)
    # -1 = random each iteration (uniform over 0-7), 0-7 = fixed mask
    expansion_mask: int = 0

    # Endgame curriculum: fraction of games to start from near-endgame positions.
    # 0.0 = disabled (all games start from scratch).
    # 1.0 = all games start from endgame positions.
    # Endgame positions have both queens surrounded by endgame_surround pieces.
    endgame_frac: float = 0.0
    endgame_surround: int = 5   # target neighbor count for both queens


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
        # Device (needed before buffer init to choose GPU buffer)
        self.device = get_device(self.config.device)

        if self.config.encoder_type == "transformer":
            if self.device.type == "cuda":
                self.buffer = GPUTokenReplayBuffer(
                    max_size=self.config.buffer_max_size,
                    device=str(self.device),
                )
            else:
                self.buffer = TokenReplayBuffer(self.config.buffer_max_size)
        else:
            self.buffer = GraphReplayBuffer(self.config.buffer_max_size)

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

    def _cuda_reset(self) -> None:
        """Full CUDA reset after an error — re-initialize the device."""
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()
            # Move model back to device to ensure clean state
            self.best_net = self.best_net.cpu()
            torch.cuda.empty_cache()
            self.best_net = self.best_net.to(self.device)

    def run(self) -> None:
        """Run the full GPU training loop."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        max_cuda_retries = 2

        for iteration in range(
            self._start_iteration, self.config.num_iterations + 1
        ):
            # Clean up GPU memory between iterations to prevent fragmentation
            self._cuda_cleanup()

            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")

            # 1. Self-play (with CUDA error recovery)
            t0 = time.time()
            new_examples = None
            sp_stats = None
            for cuda_attempt in range(max_cuda_retries + 1):
                try:
                    new_examples, sp_stats = self._self_play_phase(iteration)
                    break
                except (RuntimeError, torch.cuda.CudaError) as e:
                    err_msg = str(e).lower()
                    if "cuda" not in err_msg and "illegal" not in err_msg:
                        raise
                    print(
                        f"  [!] CUDA error in self-play (attempt "
                        f"{cuda_attempt + 1}/{max_cuda_retries + 1}): {e}"
                    )
                    self._cuda_reset()
                    if cuda_attempt >= max_cuda_retries:
                        print("  [!] Skipping iteration due to CUDA errors.")
                        break

            if new_examples is None:
                continue

            self.buffer.add_examples(new_examples)
            sp_time = time.time() - t0
            exp_mask = sp_stats.get('expansion_mask', 0)
            exp_str = ""
            if exp_mask < 0:
                exp_str = " [rand]"
            elif exp_mask > 0:
                pieces = []
                if exp_mask & 1: pieces.append("M")
                if exp_mask & 2: pieces.append("L")
                if exp_mask & 4: pieces.append("P")
                exp_str = f" [{'+'.join(pieces)}]"
            sp_line = (
                f"  Self-play: {len(new_examples)} examples from "
                f"{sp_stats['num_games']} games "
                f"(W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
                f"D:{sp_stats['draws']}, {sp_time:.1f}s){exp_str}"
            )
            if "transposition_hits" in sp_stats:
                sp_line += (
                    f"\n  MCGS: {sp_stats['transposition_hits']} transpositions, "
                    f"{sp_stats['nn_evals']} NN evals, "
                    f"{sp_stats['dag_nodes']} DAG nodes"
                )
            print(sp_line)
            self._cuda_cleanup()

            # 2. Train
            t0 = time.time()
            new_net, train_loss, loss_components = self._train_phase(iteration)
            train_time = time.time() - t0
            comp_str = " | ".join(
                f"{k}={v:.4f}" for k, v in loss_components.items()
                if k != "total_loss"
            )
            print(f"  Training: loss={train_loss:.4f}, {train_time:.1f}s")
            if comp_str:
                print(f"    [{comp_str}]")
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
        iteration: int = 1,
    ) -> tuple[list, dict]:
        """Run GPU-accelerated self-play, returning training examples.

        When expansion_mask < 0 (rotating mode), the mask for this iteration
        is chosen as (iteration - 1) % 8, cycling through all 8 expansion
        subsets one per iteration.  When expansion_mask >= 0 all games share
        that fixed mask (original behaviour).
        """
        cfg = self.config
        is_transformer = cfg.encoder_type == "transformer"

        self.best_net.eval()

        # Resolve -1 → a single mask for this iteration (round-robin over 0-7)
        resolved_mask = (iteration - 1) % 8 if cfg.expansion_mask < 0 else cfg.expansion_mask

        run_list = self._build_run_list(
            cfg.games_per_batch,
            cfg.batches_per_iteration,
            resolved_mask,
        )

        all_examples: list = []
        stats = {
            "num_games": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "expansion_mask": resolved_mask if cfg.expansion_mask < 0 else cfg.expansion_mask,
        }
        last_orchestrator = None

        # Pre-generate endgame positions for curriculum learning.
        # We build a pool large enough to cover the whole iteration, then
        # hand out sub_size positions per sub-batch.
        endgame_pool: list[bytes] | None = None
        endgame_pool_idx = 0
        if cfg.endgame_frac > 0.0 and (cfg.use_gpu_native or cfg.use_gumbel):
            from hive_gpu.endgame_generator import (
                generate_endgame_positions, positions_to_tensor, SIZEOF_HIVE_STATE,
            )
            total_games = sum(sz for _, sz in run_list)
            n_endgame = max(1, int(total_games * cfg.endgame_frac))
            t0 = time.time()
            endgame_pool = generate_endgame_positions(
                n_positions=n_endgame,
                expansion_mask=cfg.expansion_mask,
                min_surround=max(1, cfg.endgame_surround - 1),
                max_surround=cfg.endgame_surround,
                verbose=True,
            )
            print(f"  [endgame] pool={len(endgame_pool)} positions "
                  f"in {time.time()-t0:.1f}s")

        for mask, sub_size in run_list:
            mcts_config = GPUMCTSConfig(
                num_simulations=cfg.mcts_simulations,
                c_puct=cfg.c_puct,
                dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_epsilon=cfg.dirichlet_epsilon,
                temperature=cfg.temperature,
                temperature_drop_move=cfg.temperature_drop_move,
                batch_size=sub_size,
                games_per_batch=sub_size,
                max_game_length=cfg.max_game_length,
                encoder_type=cfg.encoder_type,
                wave_size=cfg.wave_size,
                nn_max_batch=cfg.nn_max_batch,
                playout_cap_randomize=cfg.playout_cap_randomize,
                playout_cap_randomize_prob=cfg.playout_cap_randomize_prob,
                playout_cap_fast_sims=cfg.playout_cap_fast_sims,
                policy_target_pruning=cfg.policy_target_pruning,
                root_policy_temp=cfg.root_policy_temp,
                shaped_dirichlet=cfg.shaped_dirichlet,
                expansion_mask=mask,
            )

            if cfg.use_gumbel:
                from hive_gpu.gumbel_mcts import GumbelAlphaZeroOrchestrator, GumbelConfig
                gumbel_config = GumbelConfig(
                    num_simulations=cfg.mcts_simulations,
                    max_num_considered_actions=cfg.gumbel_max_considered,
                    c_visit=cfg.gumbel_c_visit,
                    c_scale=cfg.gumbel_c_scale,
                    temperature=cfg.temperature,
                    temperature_drop_move=cfg.temperature_drop_move,
                    batch_size=sub_size,
                    max_game_length=cfg.max_game_length,
                    encoder_type=cfg.encoder_type,
                    expansion_mask=mask,
                    nn_max_batch=cfg.nn_max_batch,
                )
                orchestrator = GumbelAlphaZeroOrchestrator(self.best_net, gumbel_config)
            elif cfg.use_mcgs:
                from hive_gpu.gpu_mcgs import MCGSOrchestrator
                orchestrator = MCGSOrchestrator(self.best_net, mcts_config)
            elif cfg.use_gpu_native:
                orchestrator = GPUNativeMCTSOrchestrator(self.best_net, mcts_config)
            else:
                orchestrator = GPUMCTSOrchestrator(self.best_net, mcts_config)
            last_orchestrator = orchestrator

            # Build start_states tensor for GPU-native MCTS if endgame pool available.
            start_states_t = None
            if (endgame_pool is not None
                    and (cfg.use_gpu_native or cfg.use_gumbel)
                    and len(endgame_pool) > 0):
                n_eg = int(sub_size * cfg.endgame_frac)
                n_fresh = sub_size - n_eg
                eg_tensors = []
                if n_eg > 0:
                    # Draw from pool (cycling if needed)
                    idxs = [endgame_pool_idx % len(endgame_pool) + i
                            for i in range(n_eg)]
                    endgame_pool_idx += n_eg
                    eg_bytes = [endgame_pool[j % len(endgame_pool)] for j in idxs]
                    eg_tensors.append(positions_to_tensor(eg_bytes, device="cuda"))
                if n_fresh > 0:
                    fresh = orchestrator.ext.create_initial_states(n_fresh, mask)
                    eg_tensors.append(fresh)
                start_states_t = torch.cat(eg_tensors, dim=0) if eg_tensors else None

            if (cfg.use_gpu_native or cfg.use_gumbel) and start_states_t is not None:
                batch_examples = orchestrator.self_play_batch(
                    start_states=start_states_t
                )
            else:
                batch_examples = orchestrator.self_play_batch()

            for game_examples in batch_examples:
                # Draw downsampling: skip drawn games at rate (1 - draw_keep_rate)
                if game_examples and cfg.draw_keep_rate < 1.0:
                    v = game_examples[0].value_target
                    if v == 0.0 and np.random.random() > cfg.draw_keep_rate:
                        stats["num_games"] += 1
                        stats["draws"] += 1
                        continue

                for ex in game_examples:
                    # Compute surprise weight from KL(MCTS || NN_prior)
                    sw = self._compute_surprise_weight(
                        ex.policy_target, ex.nn_prior, cfg.policy_surprise_weight
                    )
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
                            surprise_weight=sw,
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
                            surprise_weight=sw,
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

        # Add MCGS stats if available (from last orchestrator)
        if last_orchestrator is not None and hasattr(last_orchestrator, "total_transposition_hits"):
            stats["transposition_hits"] = last_orchestrator.total_transposition_hits
            stats["nn_evals"] = last_orchestrator.total_nn_evals
            stats["dag_nodes"] = last_orchestrator.total_dag_nodes

        return all_examples, stats

    @staticmethod
    def _build_run_list(
        games_per_batch: int,
        batches_per_iteration: int,
        expansion_mask: int,
    ) -> list[tuple[int, int]]:
        """Build the list of ``(expansion_mask, sub_batch_size)`` self-play runs.

        For random-per-game expansion mode, distribute the exact requested game
        count across all 8 masks instead of dropping the remainder.
        """
        if expansion_mask >= 0:
            return [
                (expansion_mask, games_per_batch)
                for _ in range(batches_per_iteration)
            ]

        total_games = games_per_batch * batches_per_iteration
        num_masks = 8
        base = total_games // num_masks
        remainder = total_games % num_masks
        run_list: list[tuple[int, int]] = []
        for mask in range(num_masks):
            sub_size = base + (1 if mask < remainder else 0)
            if sub_size > 0:
                run_list.append((mask, sub_size))
        return run_list

    @staticmethod
    def _compute_surprise_weight(
        mcts_policy: np.ndarray,
        nn_prior: np.ndarray | None,
        scale: float,
    ) -> float:
        """Compute KL-based surprise weight for a training example.

        Returns max(0.1, 1.0 + scale * KL(mcts || nn_prior)) so that
        positions where MCTS heavily disagrees with the NN get sampled
        more frequently.  A floor of 0.1 ensures no position is fully
        ignored.  If nn_prior is None or scale <= 0, returns 1.0.
        """
        if nn_prior is None or scale <= 0:
            return 1.0

        # Only consider actions with non-zero MCTS probability
        mask = mcts_policy > 0
        if not mask.any():
            return 1.0

        p = mcts_policy[mask]
        q = nn_prior[mask]

        # Clamp q to avoid log(0)
        q = np.clip(q, 1e-8, 1.0)

        kl = float(np.sum(p * np.log(p / q)))
        return max(0.1, 1.0 + scale * kl)

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
            return new_net, 0.0, {}

        is_transformer = self.config.encoder_type == "transformer"
        total_loss_sum = 0.0
        component_sums: dict[str, float] = {}
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
                for k, v in loss_dict.items():
                    component_sums[k] = component_sums.get(k, 0.0) + v.item()
                num_batches += 1

        n = max(num_batches, 1)
        avg_components = {k: v / n for k, v in component_sums.items()}
        return new_net, total_loss_sum / n, avg_components

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
        """Save a training checkpoint, then prune non-milestone checkpoints.

        Keeps every checkpoint whose iteration number is a multiple of
        ``checkpoint_keep_every`` (default 25).  All others are deleted
        after the new checkpoint is safely written.
        """
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

        keep_every = self.config.checkpoint_keep_every
        if keep_every > 0:
            import glob
            pattern = os.path.join(
                self.config.checkpoint_dir, "hive_gpu_checkpoint_*.pt"
            )
            for ckpt in glob.glob(pattern):
                fname = os.path.basename(ckpt)
                try:
                    ckpt_iter = int(fname.split("_")[-1].split(".")[0])
                except ValueError:
                    continue
                if ckpt_iter % keep_every != 0:
                    os.remove(ckpt)

        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        net_class=None,
        config_overrides: GPUTrainConfig | None = None,
        net_config_overrides: dict | None = None,
    ) -> GPUTrainer:
        """Restore a trainer from a checkpoint.

        Args:
            net_config_overrides: Optional dict of fields to override on the
                saved net_config (e.g. ``{"aux_mobility_enabled": False}``).
                When provided, ``strict=False`` is used for state_dict loading
                so that removed or added heads are silently skipped.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        net_config = checkpoint["net_config"]
        # Apply architecture overrides (e.g. enable/disable auxiliary heads).
        if net_config_overrides:
            for key, val in net_config_overrides.items():
                setattr(net_config, key, val)

        train_config = config_overrides or checkpoint.get(
            "train_config", GPUTrainConfig()
        )
        iteration = checkpoint.get("iteration", 0)

        trainer = cls(
            config=train_config,
            net_config=net_config,
            net_class=net_class,
        )
        # Use strict=False when architecture overrides are present so that
        # extra weights (e.g. removed heads) are silently discarded.
        strict = net_config_overrides is None
        trainer.best_net.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        trainer.best_net = trainer.best_net.to(trainer.device)
        trainer._start_iteration = iteration + 1
        # Treat num_iterations as additional iterations when resuming.
        # e.g. --iterations 30 resumed from checkpoint 28 → runs iters 29-58.
        trainer.config.num_iterations = iteration + train_config.num_iterations

        return trainer
