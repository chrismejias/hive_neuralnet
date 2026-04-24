"""
Profile 4 model configurations: PRS-v2, MC, FNN-Gumbel, FNN-PUCT.
64 games · 128 simulations · k=16 Gumbel.

Timing strategy: CUDA-synced wall-clock only — no torch.profiler event buffer.
Sub-phases are measured by monkey-patching the CUDA extension proxy and the
NN forward methods on each orchestrator.
"""
from __future__ import annotations

import argparse
import sys, os, gc, time, math
sys.path.insert(0, os.path.dirname(__file__))

import torch

GAMES    = 64
SIMS     = 128
GUMBEL_K = 16
BAR      = "=" * 70


# ─── Timing helpers ───────────────────────────────────────────────────────────

def synced() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def mb_peak() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class Timer:
    """Accumulates CUDA-synced durations per named bucket."""
    def __init__(self):
        self._totals: dict[str, float] = {}
        self._calls:  dict[str, int]   = {}

    def add(self, key: str, dt: float):
        self._totals[key] = self._totals.get(key, 0.0) + dt
        self._calls[key]  = self._calls.get(key, 0) + 1

    def report(self, wall: float) -> str:
        lines = []
        for k, t in sorted(self._totals.items(), key=lambda x: -x[1]):
            pct  = 100.0 * t / wall if wall > 0 else 0.0
            n    = self._calls[k]
            avg  = t / n * 1000 if n else 0.0
            lines.append(f"    {k:<40s}  {t:6.2f}s  {pct:5.1f}%  {n:5d}x  {avg:7.2f}ms/call")
        return "\n".join(lines)


# ─── Timed ext proxy ──────────────────────────────────────────────────────────

class TimedExt:
    """Wraps the CUDA extension so each call is CUDA-synced and timed."""

    _TIMED = {
        "generate_legal_moves_batch",
        "generate_legal_mask_batch",
        "apply_moves_batch",
        "check_results_batch",
        "create_initial_states",
        "extract_fnn_features_batch",
        "prs_v2_classify_batch",
        "legal_moves_to_actions_batch",
        "mcts_select_with_root_mask_batch",
        "mcts_expand_and_backprop_dense_priors_batch",
        "mcts_select_batch",
        "mcts_expand_and_backprop_batch",
        "rebase_tree_batch",
    }

    def __init__(self, real_ext, timer: Timer):
        self._ext   = real_ext
        self._timer = timer

    def __getattr__(self, name: str):
        val = getattr(self._ext, name)
        if name in self._TIMED and callable(val):
            timer = self._timer
            def wrapper(*args, **kwargs):
                t0 = synced()
                r = val(*args, **kwargs)
                timer.add(f"ext.{name}", synced() - t0)
                return r
            return wrapper
        return val


# ─── NN forward wrapper ───────────────────────────────────────────────────────

def wrap_nn_forward(net: torch.nn.Module, timer: Timer, label: str = "nn.forward"):
    """Replace net.forward with a timed version."""
    _orig = net.forward
    def _timed(*args, **kwargs):
        t0 = synced()
        r  = _orig(*args, **kwargs)
        timer.add(label, synced() - t0)
        return r
    net.forward = _timed


def wrap_method(obj, method_name: str, timer: Timer, label: str):
    _orig = getattr(obj, method_name)
    def _timed(*args, **kwargs):
        t0 = synced()
        r  = _orig(*args, **kwargs)
        timer.add(label, synced() - t0)
        return r
    setattr(obj, method_name, _timed)


# ─── Report printer ───────────────────────────────────────────────────────────

def print_report(
    label: str,
    sp_timer: Timer, sp_s: float,
    tr_timer: Timer, tr_s: float,
    peak_mb: float,
):
    print(f"\n{BAR}")
    print(f"  {label}")
    print(BAR)
    print(f"  Self-play : {sp_s:6.2f}s")
    print(f"  Training  : {tr_s:6.2f}s")
    print(f"  Total     : {sp_s + tr_s:6.2f}s")
    print(f"  Peak VRAM : {peak_mb:6.1f} MB")
    print(f"\n  Self-play sub-phases (CUDA-synced):")
    sp_text = sp_timer.report(sp_s)
    if sp_text:
        print(sp_text)
    else:
        print("    (no instrumented sub-phases)")
    print(f"\n  Training sub-phases (CUDA-synced):")
    tr_text = tr_timer.report(tr_s)
    if tr_text:
        print(tr_text)
    else:
        print("    (no instrumented sub-phases)")


# ─── PRS v2 ───────────────────────────────────────────────────────────────────

def run_prs():
    import hive_gpu
    from hive_prs.prs_trainer_v2 import PRSTrainerV2, PRSTrainConfigV2
    from hive_prs.prs_transformer import PRSConfig

    cfg = PRSTrainConfigV2(
        num_iterations=1, games_per_batch=GAMES,
        mcts_simulations=SIMS, max_num_considered=GUMBEL_K,
        num_epochs=3, batch_size=256,
        checkpoint_dir="/tmp/prof_prs", checkpoint_keep_every=0,
    )
    trainer = PRSTrainerV2(cfg, PRSConfig.small())

    sp_timer = Timer()
    tr_timer = Timer()

    # Patch ext inside orchestrator (created fresh each _self_play call, so
    # patch hive_gpu.load_extension to return a timed proxy)
    real_ext = hive_gpu.load_extension()
    timed_ext_sp = TimedExt(real_ext, sp_timer)
    timed_ext_tr = TimedExt(real_ext, tr_timer)

    # ── Self-play ──
    from hive_prs import prs_mcts_orchestrator_v2 as _prs_mod
    _orig_load = hive_gpu.load_extension

    # Inject timed ext into trainer's _forward_train and the orchestrator
    import hive_prs.prs_mcts_orchestrator_v2 as _pv2mod

    _orig_init = _pv2mod.PRSMCTSOrchestratorV2.__init__
    def _patched_init(self_orch, net, config=None):
        _orig_init(self_orch, net, config)
        self_orch.ext = timed_ext_sp
        # Time the NN forward
        wrap_method(self_orch, "_net_forward",           sp_timer, "nn._net_forward (trunk+head)")
        wrap_method(self_orch, "_classify_kernel",       sp_timer, "ext.prs_v2_classify(wrap)")
        wrap_method(self_orch, "_build_legal_priors_v2", sp_timer, "build_legal_priors_v2")
        wrap_method(self_orch, "_run_simulations",       sp_timer, "mcts._run_simulations")
        wrap_method(self_orch, "_expand_root_if_needed", sp_timer, "mcts._expand_root")
        wrap_method(self_orch, "_check_immediate_wins_v2", sp_timer, "check_immediate_wins")

    _pv2mod.PRSMCTSOrchestratorV2.__init__ = _patched_init

    # Also time the PRS encoder
    from hive_prs.prs_encoder import PRSEncoder
    _orig_encode = PRSEncoder.encode_batch
    def _timed_encode(self_enc, *a, **kw):
        t0 = synced()
        r = _orig_encode(self_enc, *a, **kw)
        sp_timer.add("encoder.encode_batch", synced() - t0)
        return r
    PRSEncoder.encode_batch = _timed_encode

    t0 = synced()
    examples, stats = trainer._self_play(1)
    sp_s = synced() - t0

    _pv2mod.PRSMCTSOrchestratorV2.__init__ = _orig_init
    PRSEncoder.encode_batch = _orig_encode
    peak_sp = mb_peak()
    cleanup()

    # ── Training ──
    trainer.buffer.add_examples(examples)
    import hive_prs.prs_trainer_v2 as _pt_mod
    _orig_fwd_train = trainer._forward_train.__func__ if hasattr(trainer._forward_train, '__func__') else None

    # Patch _forward_train sub-phases
    _orig_fwd = type(trainer)._forward_train
    def _timed_fwd(self_t, batch):
        t0 = synced(); states_gpu = torch.from_numpy(batch.state_bytes).cuda()
        tr_timer.add("tr.host_to_gpu", synced() - t0)
        t0 = synced()
        legal_t, nlegal_t = timed_ext_tr._ext.generate_legal_moves_batch(states_gpu, batch.state_bytes.shape[0])
        tr_timer.add("tr.gen_legal_moves", synced() - t0)
        t0 = synced()
        kernel_out = timed_ext_tr._ext.prs_v2_classify_batch(states_gpu, legal_t, nlegal_t, batch.state_bytes.shape[0], legal_t.shape[1])
        tr_timer.add("tr.classify_kernel", synced() - t0)
        from hive_prs.prs_v2_bridge import build_head_inputs_from_kernel
        t0 = synced()
        board_h, cls_h, full_h, value = self_t._train_net.forward_trunk(batch.prs_batch)
        tr_timer.add("tr.forward_trunk", synced() - t0)
        t0 = synced()
        inp, _ = build_head_inputs_from_kernel(board_h, cls_h, full_h, kernel_out)
        logits = self_t._train_net.head(inp)
        tr_timer.add("tr.head_forward", synced() - t0)
        return logits, value
    type(trainer)._forward_train = _timed_fwd

    t0 = synced()
    trainer._train(1)
    tr_s = synced() - t0
    type(trainer)._forward_train = _orig_fwd
    peak_total = mb_peak()

    print_report(
        "PRS v2  (Gumbel MCTS · 813-slot head · ~1.3M params)",
        sp_timer, sp_s, tr_timer, tr_s, max(peak_sp, peak_total),
    )
    cleanup()
    del trainer


# ─── MC ──────────────────────────────────────────────────────────────────────

def run_mc():
    from hive_mc.mc_trainer import MCTrainer, MCTrainConfig
    from hive_mc.mc_transformer import MCTransformerConfig
    import hive_mc.mc_mcts_orchestrator as _mc_mod
    import hive_gpu

    cfg = MCTrainConfig(
        num_iterations=1, games_per_batch=GAMES,
        mcts_simulations=SIMS, max_num_considered=GUMBEL_K,
        num_epochs=3, batch_size=128,
        checkpoint_dir="/tmp/prof_mc", checkpoint_keep_every=0,
    )
    trainer = MCTrainer(cfg, MCTransformerConfig.small())

    sp_timer = Timer()
    tr_timer = Timer()
    real_ext = hive_gpu.load_extension()
    timed_ext_sp = TimedExt(real_ext, sp_timer)

    _orig_init = _mc_mod.MCMCTSOrchestrator.__init__
    def _patched_init(self_orch, net, config=None):
        _orig_init(self_orch, net, config)
        self_orch.ext = timed_ext_sp
        wrap_method(self_orch, "_run_simulations",       sp_timer, "mcts._run_simulations")
        wrap_method(self_orch, "_expand_root_if_needed", sp_timer, "mcts._expand_root")
    _mc_mod.MCMCTSOrchestrator.__init__ = _patched_init

    from hive_gpu.gpu_encoder import GPUTransformerEncoder
    _orig_enc = GPUTransformerEncoder.encode_batch
    def _timed_enc(self_enc, *a, **kw):
        t0 = synced()
        r = _orig_enc(self_enc, *a, **kw)
        sp_timer.add("encoder.encode_batch", synced() - t0)
        return r
    GPUTransformerEncoder.encode_batch = _timed_enc

    # MC self-play calls net.encode / net.screen / net.value_head directly
    _net = trainer.best_net
    _orig_net_encode = _net.encode
    _orig_net_screen = _net.screen
    _orig_vh_fwd     = _net.value_head.forward
    def _timed_encode(*a, **kw):
        t0 = synced(); r = _orig_net_encode(*a, **kw); sp_timer.add("nn.encode", synced() - t0); return r
    def _timed_screen(*a, **kw):
        t0 = synced(); r = _orig_net_screen(*a, **kw); sp_timer.add("nn.screen", synced() - t0); return r
    def _timed_vh_fwd(*a, **kw):
        t0 = synced(); r = _orig_vh_fwd(*a, **kw); sp_timer.add("nn.value_head", synced() - t0); return r
    _net.encode          = _timed_encode
    _net.screen          = _timed_screen
    _net.value_head.forward = _timed_vh_fwd

    t0 = synced()
    examples, stats = trainer._self_play()
    sp_s = synced() - t0

    _mc_mod.MCMCTSOrchestrator.__init__ = _orig_init
    GPUTransformerEncoder.encode_batch = _orig_enc
    _net.encode             = _orig_net_encode
    _net.screen             = _orig_net_screen
    _net.value_head.forward = _orig_vh_fwd
    peak_sp = mb_peak()
    cleanup()

    # ── Training ──
    trainer.buffer.add_examples(examples)
    from hive_mc.mc_utils import build_move_conditioned_batch
    _orig_train = type(trainer)._train

    def _timed_train(self_t, iteration):
        _cfg = self_t.config
        if len(self_t.buffer) < _cfg.batch_size:
            return 0.0, {}
        import torch.optim as optim
        opt = optim.Adam(self_t.best_net.parameters(), lr=self_t._lr(iteration), weight_decay=_cfg.weight_decay)
        total_loss = 0.0; comp_sums: dict = {}; n_batches = 0
        self_t.best_net.train()
        from hive_mc.mc_trainer import compute_mc_loss
        for _epoch in range(_cfg.num_epochs):
            bpe = max(1, len(self_t.buffer) // _cfg.batch_size)
            for _ in range(bpe):
                from hive_mc.mc_replay_buffer import MCTrainingBatch
                batch = self_t.buffer.sample_batch(_cfg.batch_size).to(self_t.device, non_blocking=True)
                opt.zero_grad()
                t0 = synced()
                move_batch = build_move_conditioned_batch(batch.state_bytes, batch.state_bytes.shape[0], encoder=self_t.encoder, ext=None)
                tr_timer.add("tr.build_move_batch", synced() - t0)
                if self_t.use_amp and self_t.scaler:
                    with torch.amp.autocast("cuda"):
                        t0 = synced()
                        sl, al, rv, _av = self_t.best_net(move_batch)
                        tr_timer.add("tr.net_forward", synced() - t0)
                        loss, ld = compute_mc_loss(sl, al, rv, batch.policy_targets, batch.value_targets, batch.num_actions)
                    self_t.scaler.scale(loss).backward()
                    self_t.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self_t.best_net.parameters(), _cfg.max_grad_norm)
                    self_t.scaler.step(opt); self_t.scaler.update()
                else:
                    t0 = synced()
                    sl, al, rv, _av = self_t.best_net(move_batch)
                    tr_timer.add("tr.net_forward", synced() - t0)
                    loss, ld = compute_mc_loss(sl, al, rv, batch.policy_targets, batch.value_targets, batch.num_actions)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(self_t.best_net.parameters(), _cfg.max_grad_norm); opt.step()
                total_loss += float(loss.item()); n_batches += 1
                for k, v in ld.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())
        n = max(n_batches, 1)
        return total_loss / n, {k: v / n for k, v in comp_sums.items()}

    type(trainer)._train = _timed_train
    t0 = synced()
    trainer._train(1)
    tr_s = synced() - t0
    type(trainer)._train = _orig_train
    peak_total = mb_peak()

    print_report(
        "MC  (Move-Conditioned · Gumbel MCTS · ~1.1M params)",
        sp_timer, sp_s, tr_timer, tr_s, max(peak_sp, peak_total),
    )
    cleanup()
    del trainer


# ─── FNN (Gumbel) ────────────────────────────────────────────────────────────

def _run_fnn(use_puct: bool, skip_training: bool = False):
    from hive_fnn.fnn_trainer import FNNTrainer, FNNTrainConfig
    from hive_fnn.fnn_network import FNNConfig
    import hive_fnn.fnn_mcts_orchestrator as _fmcts_mod
    import hive_fnn.fnn_puct_orchestrator as _fpuct_mod
    import hive_gpu

    cfg = FNNTrainConfig(
        num_iterations=1, games_per_batch=GAMES,
        mcts_simulations=SIMS, max_num_considered=GUMBEL_K,
        num_epochs=3, batch_size=128,
        use_puct=use_puct,
        checkpoint_dir="/tmp/prof_fnn", checkpoint_keep_every=0,
    )
    trainer = FNNTrainer(cfg, FNNConfig.small())

    sp_timer = Timer()
    tr_timer = Timer()
    real_ext = hive_gpu.load_extension()
    timed_ext_sp = TimedExt(real_ext, sp_timer)

    _orig_fmcts_init = _fmcts_mod.FNNMCTSOrchestrator.__init__
    def _patched_fmcts_init(self_orch, net, config=None):
        _orig_fmcts_init(self_orch, net, config)
        self_orch.ext = timed_ext_sp
        wrap_method(self_orch, "_eval_states",           sp_timer, "nn._eval_states (FNN encode+score)")
        wrap_method(self_orch, "_find_immediate_wins",   sp_timer, "check_immediate_wins")
        wrap_method(self_orch, "_run_simulations",       sp_timer, "mcts._run_simulations")
        wrap_method(self_orch, "_expand_root_if_needed", sp_timer, "mcts._expand_root")
    _fmcts_mod.FNNMCTSOrchestrator.__init__ = _patched_fmcts_init

    t0 = synced()
    examples, stats = trainer._self_play()
    sp_s = synced() - t0
    variant = "PUCT MCTS" if use_puct else "Gumbel MCTS"
    print(
        f"  FNN {variant} self-play complete: "
        f"{sp_s:.2f}s, {len(examples)} games, "
        f"{sum(len(g) for g in examples)} examples"
    )
    _fmcts_mod.FNNMCTSOrchestrator.__init__ = _orig_fmcts_init
    peak_sp = mb_peak()
    cleanup()

    if skip_training:
        print_report(
            f"FNN  ({variant} · 88-dim features · ~0.6K params)",
            sp_timer, sp_s, Timer(), 0.0, peak_sp,
        )
        cleanup()
        del trainer
        return

    # ── Training ──
    trainer.buffer.add_examples(examples)
    _orig_build = type(trainer)._build_forward_batch
    def _timed_build(self_t, state_bytes, batch_size):
        t0 = synced()
        legal_moves, num_legal = timed_ext_sp._ext.generate_legal_moves_batch(state_bytes, batch_size)
        tr_timer.add("tr.gen_legal_moves(root)", synced() - t0)
        num_actions = num_legal.to(torch.int64)
        device = state_bytes.device
        slot_idx = torch.arange(legal_moves.shape[1], device=device, dtype=torch.int64).unsqueeze(0)
        valid = slot_idx < num_actions.unsqueeze(1)
        action_to_root = torch.arange(batch_size, device=device, dtype=torch.int64).unsqueeze(1).expand_as(valid)[valid]
        move_indices   = slot_idx.expand_as(valid)[valid]
        total_actions  = action_to_root.shape[0]
        t0 = synced()
        root_features = timed_ext_sp._ext.extract_fnn_features_batch(state_bytes, legal_moves, num_legal, batch_size)
        tr_timer.add("tr.extract_fnn_feat(root)", synced() - t0)
        if total_actions == 0:
            return root_features, root_features[:0], action_to_root, num_actions
        child_states = state_bytes[action_to_root].clone()
        moves = legal_moves[action_to_root, move_indices]
        t0 = synced(); timed_ext_sp._ext.apply_moves_batch(child_states, moves, total_actions); tr_timer.add("tr.apply_moves", synced() - t0)
        t0 = synced()
        child_legal, child_nlegal = timed_ext_sp._ext.generate_legal_moves_batch(child_states, total_actions)
        tr_timer.add("tr.gen_legal_moves(child)", synced() - t0)
        t0 = synced()
        succ_features = timed_ext_sp._ext.extract_fnn_features_batch(child_states, child_legal, child_nlegal, total_actions)
        tr_timer.add("tr.extract_fnn_feat(child)", synced() - t0)
        return root_features, succ_features, action_to_root, num_actions
    type(trainer)._build_forward_batch = _timed_build

    wrap_nn_forward(trainer.best_net, tr_timer, "tr.net_forward (encode+score+value)")

    t0 = synced()
    trainer._train(1)
    tr_s = synced() - t0

    type(trainer)._build_forward_batch = _orig_build
    del trainer.best_net.forward
    peak_total = mb_peak()

    print_report(
        f"FNN  ({variant} · 88-dim features · ~0.6K params)",
        sp_timer, sp_s, tr_timer, tr_s, max(peak_sp, peak_total),
    )
    cleanup()
    del trainer


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global GAMES, SIMS, GUMBEL_K

    parser = argparse.ArgumentParser(
        description="CUDA-synced profiler for Hive model/trainer configurations.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["prs", "mc", "fnn-gumbel", "fnn-puct"],
        default=["prs", "mc", "fnn-gumbel", "fnn-puct"],
        help="Which model profiles to run, in order.",
    )
    parser.add_argument("--games", type=int, default=GAMES)
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--gumbel-k", type=int, default=GUMBEL_K)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Profile self-play only and skip the training section.",
    )
    args = parser.parse_args()

    GAMES = args.games
    SIMS = args.sims
    GUMBEL_K = args.gumbel_k

    print(f"Profiling 4 models  |  games={GAMES}  sims={SIMS}  k={GUMBEL_K}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch : {torch.__version__}")

    torch.zeros(1, device="cuda"); torch.cuda.synchronize()  # CUDA warmup

    runners = {
        "prs": run_prs,
        "mc": run_mc,
        "fnn-gumbel": lambda: _run_fnn(use_puct=False, skip_training=args.skip_training),
        "fnn-puct": lambda: _run_fnn(use_puct=True, skip_training=args.skip_training),
    }
    for model_name in args.models:
        cleanup()
        runners[model_name]()

    print(f"\n{BAR}\n  Done.\n{BAR}")


if __name__ == "__main__":
    main()
