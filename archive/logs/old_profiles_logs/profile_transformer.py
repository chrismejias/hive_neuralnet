"""
Profile the GPU transformer training pipeline.

Measures time spent in each phase:
  1. Self-play: MCTS loop, NN forward, encoding, action mapping
  2. Training: forward, backward, optimizer step
  3. Per-component breakdown within self-play

Usage:
    cd hive_neuralnet
    .venv/Scripts/python profile_transformer.py
"""

import time
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict

# ── Timer utilities ──────────────────────────────────────────────────

class Timer:
    """Accumulating timer for profiling code sections."""

    def __init__(self):
        self.times: dict[str, list[float]] = defaultdict(list)
        self._stack: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.times[name].append(time.perf_counter() - t0)

    def report(self, title: str = "Profile"):
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

        # Sort by total time descending
        entries = []
        for name, vals in self.times.items():
            total = sum(vals)
            entries.append((name, total, len(vals), total / len(vals)))
        entries.sort(key=lambda x: -x[1])

        total_all = sum(e[1] for e in entries)
        print(f"  {'Section':<40} {'Total':>8} {'Calls':>6} {'Avg':>8} {'%':>6}")
        print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")
        for name, total, count, avg in entries:
            pct = 100 * total / total_all if total_all > 0 else 0
            print(f"  {name:<40} {total:>7.3f}s {count:>6} {avg*1000:>7.2f}ms {pct:>5.1f}%")
        print(f"  {'-'*40} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")
        print(f"  {'TOTAL':<40} {total_all:>7.3f}s")
        print()


# ── 1. Profile self-play ─────────────────────────────────────────────

def profile_self_play():
    """Profile the GPU MCTS self-play pipeline with transformer encoder."""
    from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
    from hive_transformer.transformer_net import HiveTransformer, TransformerConfig

    print("Setting up transformer for self-play profiling...")
    net = HiveTransformer(TransformerConfig.small()).cuda().eval()
    param_count = sum(p.numel() for p in net.parameters())
    print(f"  Model: TransformerConfig.small(), {param_count:,} parameters")

    B = 16  # batch size
    sims = 16  # MCTS simulations (small for profiling)
    max_len = 20  # short games

    config = GPUMCTSConfig(
        num_simulations=sims,
        batch_size=B,
        max_game_length=max_len,
        encoder_type="transformer",
    )

    print(f"  Batch size: {B}, Simulations: {sims}, Max game length: {max_len}")
    print(f"  Running self-play...")

    timer = Timer()
    orch = GPUMCTSOrchestrator(net, config)
    ext = orch.ext

    # ── Instrument the orchestrator ──
    # We'll manually run the self-play loop with timing

    cfg = config

    with timer.section("total_self_play"):
        root_states = ext.create_initial_states(B)
        active = [True] * B
        move_numbers = [0] * B
        histories = [[] for _ in range(B)]

        step = 0
        while any(active) and step < max_len:
            step += 1

            with timer.section("get_turns"):
                current_turns = orch._get_turns(root_states, B)

            with timer.section("mcts_total"):
                policies, graphs, trees = orch._batched_mcts(root_states, B, active, move_numbers)

            with timer.section("generate_legal_mask"):
                all_masks, _ = ext.generate_legal_mask_batch(root_states, B)
                all_masks_np = all_masks.cpu().numpy()

            with timer.section("compute_mobility"):
                mob_tensor, mob_counts = ext.compute_mobility_batch(root_states, B, False)
                mob_np = mob_tensor.cpu().numpy()
                mob_board_counts = mob_counts.cpu().numpy()

            with timer.section("compute_centroids"):
                centroids = ext.compute_centroids_batch(root_states, B).cpu().numpy()

            with timer.section("action_selection"):
                action_move_bytes = np.zeros((B, orch._move_size), dtype=np.uint8)
                for i in range(B):
                    if not active[i]:
                        continue
                    policy = policies[i]
                    turn = current_turns[i]
                    mob_i = mob_np[i, :mob_board_counts[i]].copy()
                    histories[i].append((graphs[i], policy.copy(), turn, mob_i))

                    action = int(np.argmax(policy)) if move_numbers[i] >= cfg.temperature_drop_move else int(np.random.choice(len(policy), p=policy))
                    # Use tree lookup (O(1)) instead of _action_to_gpu_move
                    tree = trees[i]
                    if action in tree.children:
                        action_move_bytes[i] = np.frombuffer(tree.children[action].move_bytes, dtype=np.uint8)
                    else:
                        # Fallback for pass or missing actions
                        cq, cr = int(centroids[i, 0]), int(centroids[i, 1])
                        mask = all_masks_np[i]
                        move_bytes = orch._action_to_gpu_move(root_states, i, action, mask, cq, cr)
                        if move_bytes is not None:
                            action_move_bytes[i] = move_bytes

            with timer.section("apply_moves"):
                moves_t = torch.from_numpy(action_move_bytes).cuda()
                ext.apply_moves_batch(root_states, moves_t, B)

            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            with timer.section("check_results"):
                results = ext.check_results_batch(root_states, B).cpu().numpy()
                for i in range(B):
                    if active[i] and results[i] != 0:
                        active[i] = False

    timer.report(f"Self-Play Profile (B={B}, sims={sims}, steps={step})")

    # Now profile _batched_mcts internals
    print("Profiling MCTS internals...")
    profile_mcts_internals(net, B, sims)


def profile_mcts_internals(net, B, sims):
    """Profile the internals of _batched_mcts."""
    from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig, GPUMCTSNode

    config = GPUMCTSConfig(
        num_simulations=sims,
        batch_size=B,
        max_game_length=50,
        encoder_type="transformer",
    )
    orch = GPUMCTSOrchestrator(net, config)
    ext = orch.ext

    timer = Timer()

    # Set up a position with some pieces
    root_states = ext.create_initial_states(B)
    move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
    # Place queens for all games
    for i in range(B):
        move_bytes[i, 0] = 0  # PLACE
        move_bytes[i, 1] = 1  # Queen
        move_bytes[i, 4] = 144 & 0xFF
        move_bytes[i, 5] = (144 >> 8) & 0xFF
    ext.apply_moves_batch(root_states, torch.from_numpy(move_bytes).cuda(), B)
    # Place ants
    for i in range(B):
        move_bytes[i, 1] = 2  # Ant
        move_bytes[i, 4] = 145 & 0xFF  # cell 145 (east of center)
        move_bytes[i, 5] = (145 >> 8) & 0xFF
    ext.apply_moves_batch(root_states, torch.from_numpy(move_bytes).cuda(), B)

    active = [True] * B

    # Create trees and expand roots
    trees = [GPUMCTSNode() for _ in range(B)]

    with timer.section("expand_roots"):
        root_graphs = orch._expand_roots(root_states, B, trees, active)

    for t in trees:
        if t is not None:
            orch._add_dirichlet_noise(t)

    leaf_states = root_states.clone()

    for sim in range(sims):
        # SELECT
        leaves = [None] * B
        move_paths = [[] for _ in range(B)]
        vl_paths = [[] for _ in range(B)]

        with timer.section("select"):
            for i in range(B):
                if trees[i] is None:
                    continue
                leaf, vl_path, path_moves = orch._select(trees[i])
                leaves[i] = leaf
                vl_paths[i] = vl_path
                move_paths[i] = path_moves

        # PREPARE LEAF STATES
        with timer.section("replay_moves"):
            leaf_states.copy_(root_states)
            max_depth = max((len(p) for p in move_paths), default=0)
            for d in range(max_depth):
                depth_moves = np.zeros((B, orch._move_size), dtype=np.uint8)
                for i in range(B):
                    if d < len(move_paths[i]):
                        depth_moves[i] = move_paths[i][d]
                    else:
                        depth_moves[i][0] = 2  # PASS
                dm_tensor = torch.from_numpy(depth_moves).cuda()
                ext.apply_moves_batch(leaf_states, dm_tensor, B)

        # CHECK TERMINAL
        with timer.section("check_terminal"):
            leaf_results = ext.check_results_batch(leaf_states, B).cpu().numpy()

        needs_eval = []
        for i in range(B):
            if leaves[i] is None:
                continue
            if leaf_results[i] != 0:
                leaves[i].is_terminal = True
                leaves[i].terminal_value = orch._result_to_value(
                    leaf_results[i], orch._get_turn_single(leaf_states, i)
                )
            elif not leaves[i].is_expanded:
                needs_eval.append(i)

        if needs_eval:
            with timer.section("encode_leaves"):
                encoded = orch.encoder.encode_batch(leaf_states, B)

            with timer.section("generate_legal_mask_leaves"):
                masks, _ = ext.generate_legal_mask_batch(leaf_states, B)

            with timer.section("nn_forward"):
                with torch.no_grad():
                    policy_logits, values, *_ = net(encoded)

            with timer.section("postprocess_policy"):
                masks_bool = masks == 0
                policy_logits[masks_bool] = float("-inf")
                action_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()
                masks_np = masks.cpu().numpy()

            with timer.section("generate_legal_moves_leaves"):
                all_legal_moves, all_num_legal = ext.generate_legal_moves_batch(leaf_states, B)
                all_legal_np = all_legal_moves.cpu().numpy()
                all_num_legal_np = all_num_legal.cpu().numpy()

            with timer.section("compute_centroids_leaves"):
                leaf_centroids = ext.compute_centroids_batch(leaf_states, B).cpu().numpy()

            with timer.section("expand_leaves"):
                for i in needs_eval:
                    leaf = leaves[i]
                    if leaf.is_expanded:
                        continue
                    n_legal = all_num_legal_np[i]
                    legal_moves_raw = all_legal_np[i]
                    probs = action_probs[i]
                    cq, cr = int(leaf_centroids[i, 0]), int(leaf_centroids[i, 1])
                    for mi in range(n_legal):
                        move_raw = legal_moves_raw[mi]
                        action_idx = orch._gpu_move_to_action(move_raw, cq, cr)
                        if action_idx is None or action_idx < 0:
                            continue
                        if action_idx not in leaf.children:
                            child = GPUMCTSNode(
                                parent=leaf,
                                parent_action=action_idx,
                                prior=float(probs[action_idx]),
                                move_bytes=move_raw.copy(),
                            )
                            leaf.children[action_idx] = child
                    leaf.is_expanded = True

        # BACKPROPAGATE
        with timer.section("backpropagate"):
            for i in range(B):
                leaf = leaves[i]
                if leaf is None:
                    continue
                if leaf.is_terminal:
                    value = leaf.terminal_value
                elif i in needs_eval:
                    value = float(values_np[i])
                else:
                    value = 0.0
                orch._backpropagate(leaf, value, vl_paths[i])

    timer.report(f"MCTS Internals (B={B}, sims={sims})")


# ── 2. Profile training step ─────────────────────────────────────────

def profile_training():
    """Profile the training phase (forward + backward + optimizer)."""
    from hive_transformer.transformer_net import HiveTransformer, TransformerConfig
    from hive_transformer.transformer_replay_buffer import (
        TransformerTrainingExample, TokenReplayBuffer,
    )
    from hive_transformer.token_encoder import TokenEncoder
    from hive_engine.neural_net import compute_transformer_loss
    from hive_engine.game_state import GameState

    print("\nSetting up training profiling...")
    net = HiveTransformer(TransformerConfig.small()).cuda()
    param_count = sum(p.numel() for p in net.parameters())
    print(f"  Model: {param_count:,} parameters")

    # Generate synthetic training data using CPU encoder
    print("  Generating training data...")
    encoder = TokenEncoder()
    buffer = TokenReplayBuffer(10_000)

    for _ in range(100):
        gs = GameState()
        seq = encoder.encode(gs)
        n = seq.num_board_tokens
        policy = np.zeros(29914, dtype=np.float32)
        policy[0] = 1.0  # dummy
        ex = TransformerTrainingExample(
            sequence=seq,
            policy_target=policy,
            value_target=0.0,
            mobility_target=np.zeros(n, dtype=np.float32),
            queen_surround_target=np.zeros((n, 2), dtype=np.float32),
            queen_surround_mask=np.zeros(2, dtype=np.float32),
            final_mobility_target=np.zeros(n, dtype=np.float32),
            use_for_value=True,
        )
        buffer.add_examples([ex])

    print(f"  Buffer size: {len(buffer)} examples")

    # Profile training loop
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    grad_scaler = torch.amp.GradScaler("cuda")

    timer = Timer()
    batch_size = 32
    num_steps = 20

    print(f"  Profiling {num_steps} training steps with batch_size={batch_size}...")

    for step in range(num_steps):
        with timer.section("sample_batch"):
            batch = buffer.sample_batch(batch_size)

        with timer.section("to_device"):
            batch = batch.to(torch.device("cuda"))

        with timer.section("zero_grad"):
            optimizer.zero_grad()

        with timer.section("forward"):
            with torch.amp.autocast("cuda"):
                policy_logits, value_pred, aux_outputs = net(batch.token_batch)

        with timer.section("compute_loss"):
            with torch.amp.autocast("cuda"):
                total_loss, loss_dict = compute_transformer_loss(
                    policy_logits, value_pred,
                    batch.policy_targets, batch.value_targets,
                    aux_outputs,
                    batch.mobility_targets, batch.queen_surround_targets,
                    batch.queen_surround_mask, batch.final_mobility_targets,
                    batch.value_mask, batch.board_token_batch,
                )

        with timer.section("backward"):
            grad_scaler.scale(total_loss).backward()

        with timer.section("optimizer_step"):
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

    timer.report(f"Training Profile ({num_steps} steps, batch_size={batch_size})")


# ── 3. Profile NN forward pass scaling ───────────────────────────────

def profile_nn_scaling():
    """Profile transformer forward pass at different batch sizes."""
    from hive_transformer.transformer_net import HiveTransformer, TransformerConfig
    from hive_gpu.gpu_encoder import GPUTransformerEncoder
    import hive_gpu

    ext = hive_gpu.load_extension()
    encoder = GPUTransformerEncoder()

    for preset_name, config in [("small", TransformerConfig.small()),
                                 ("large", TransformerConfig.large())]:
        net = HiveTransformer(config).cuda().eval()
        param_count = sum(p.numel() for p in net.parameters())
        print(f"\n  {preset_name} ({param_count:,} params)")
        print(f"  {'Batch':>6} {'Forward':>10} {'Per-item':>10} {'Items/s':>10} {'Encode':>10}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for B in [1, 2, 4, 8, 16, 32, 64]:
            # Create states with some pieces for realistic token counts
            states = ext.create_initial_states(B)
            move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
            for i in range(B):
                move_bytes[i, 0] = 0; move_bytes[i, 1] = 1
                move_bytes[i, 4] = 144 & 0xFF; move_bytes[i, 5] = (144 >> 8) & 0xFF
            ext.apply_moves_batch(states, torch.from_numpy(move_bytes).cuda(), B)
            for i in range(B):
                move_bytes[i, 1] = 2; move_bytes[i, 4] = 145 & 0xFF; move_bytes[i, 5] = (145 >> 8) & 0xFF
            ext.apply_moves_batch(states, torch.from_numpy(move_bytes).cuda(), B)

            # Warmup
            encoded = encoder.encode_batch(states, B)
            with torch.no_grad():
                _ = net(encoded)

            # Measure encode
            N = 10
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                encoded = encoder.encode_batch(states, B)
            torch.cuda.synchronize()
            encode_ms = (time.perf_counter() - t0) / N * 1000

            # Measure forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N):
                with torch.no_grad():
                    _ = net(encoded)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / N * 1000

            per_item = fwd_ms / B
            items_sec = B / (fwd_ms / 1000)
            print(f"  {B:>6} {fwd_ms:>8.2f}ms {per_item:>8.2f}ms {items_sec:>8.0f}/s {encode_ms:>8.2f}ms")

        del net
        torch.cuda.empty_cache()


# ── 4. Profile encode kernel ─────────────────────────────────────────

def profile_encode_kernel():
    """Profile the GPU encode kernel and Python-side assembly."""
    from hive_gpu.gpu_encoder import GPUTransformerEncoder
    import hive_gpu

    ext = hive_gpu.load_extension()
    encoder = GPUTransformerEncoder()

    print("\n  Encode kernel breakdown (B=32):")
    B = 32
    states = ext.create_initial_states(B)
    # Add some pieces
    move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
    for step in range(4):
        for i in range(B):
            move_bytes[i, 0] = 0
            move_bytes[i, 1] = step + 1  # Q, A, G, S
            cell = 144 + step
            move_bytes[i, 4] = cell & 0xFF; move_bytes[i, 5] = (cell >> 8) & 0xFF
        ext.apply_moves_batch(states, torch.from_numpy(move_bytes).cuda(), B)

    timer = Timer()
    N = 20

    for _ in range(N):
        with timer.section("cuda_kernel"):
            raw = ext.encode_states_batch(states, B)
            torch.cuda.synchronize()

        with timer.section("python_assembly"):
            (node_features, node_grid_pos, node_piece_types, global_features,
             num_nodes, num_board_nodes, _ei, _ef, _ne) = raw
            nn_cpu = num_nodes.cpu()
            nb_cpu = num_board_nodes.cpu()
            # ... rest of GPUTransformerEncoder.encode_batch assembly

    timer.report("Encode Kernel Breakdown")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Hive Transformer Pipeline Profiler")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print()

    # 1. NN forward pass scaling
    print("\n--- NN Forward Pass Scaling ---")
    profile_nn_scaling()

    # 2. Encode kernel breakdown
    print("\n--- Encode Kernel Breakdown ---")
    profile_encode_kernel()

    # 3. Self-play profiling
    print("\n--- Self-Play Profiling ---")
    profile_self_play()

    # 4. Training step profiling
    print("\n--- Training Step Profiling ---")
    profile_training()
