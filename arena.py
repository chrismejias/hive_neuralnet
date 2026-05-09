"""General arena script for comparing FNN and PRS checkpoints.

Examples:

  Compare a PRS checkpoint against an FNN checkpoint:
    python3.11 arena.py \
      --white-model prs \
      --black-model fnn \
      --white-checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt \
      --black-checkpoint checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt \
      --white-sims 256 \
      --black-sims 1024

  Compare two FNN checkpoints:
    python3.11 arena.py \
      --white-model fnn \
      --black-model fnn \
      --white-checkpoint checkpoints_fnn_small/hive_fnn_checkpoint_0100.pt \
      --black-checkpoint checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt

  Compare two PRS checkpoints:
    python3.11 arena.py \
      --white-model prs \
      --black-model prs \
      --white-checkpoint checkpoints_prs_v2/prs_v2_iter_0550.pt \
      --black-checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt

Defaults mirror the most common training-time search settings:
  - PRS: 256 sims, k=16, wave parallel on
  - FNN: 1024 sims, k=16, wave parallel on
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Literal

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_puct_orchestrator import FNNPUCTConfig, FNNPUCTOrchestrator
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_transformer_v3 import HivePRSTransformerV3
from hive_prs.slot_map import N_SLOTS

ModelType = Literal["fnn", "prs"]

FNN_DEFAULT_PATTERNS = (
    "checkpoints_fnn_small/hive_fnn_checkpoint_*.pt",
    "checkpoints_fnn_medium/hive_fnn_checkpoint_*.pt",
    "checkpoints_fnn_large/hive_fnn_checkpoint_*.pt",
    "checkpoints_fnn/hive_fnn_checkpoint_*.pt",
)
PRS_DEFAULT_PATTERNS = (
    "checkpoints_prs_v2/prs_v2_iter_*.pt",
)


def _latest_by_mtime(paths: list[str]) -> str | None:
    if not paths:
        return None
    return max(paths, key=lambda p: (os.path.getmtime(p), p))


def latest_checkpoint(model_type: ModelType) -> str | None:
    patterns = FNN_DEFAULT_PATTERNS if model_type == "fnn" else PRS_DEFAULT_PATTERNS
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return _latest_by_mtime(paths)


def default_sims(model_type: ModelType) -> int:
    return 1024 if model_type == "fnn" else 256


def default_k(_: ModelType) -> int:
    return 16


def default_wave_size(model_type: ModelType) -> int:
    return 4 if model_type == "fnn" else 16


def parse_wave_schedule(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return None
    return [max(1, int(p)) for p in parts]


def load_checkpoint(model_type: ModelType, path: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if model_type == "fnn":
        net = HiveFNN(ckpt["net_config"]).cuda().eval()
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        prs_version = ckpt.get("model_version", "v2")
        net_cls = HivePRSTransformerV3 if prs_version == "v3" else HivePRSTransformerV2
        net = net_cls(ckpt["net_config"]).cuda().eval()
        model_state = ckpt["model_state"]
        current = net.state_dict()
        filtered_state = {}
        for key, tensor in model_state.items():
            if key not in current:
                continue
            if current[key].shape != tensor.shape:
                continue
            filtered_state[key] = tensor
        net.load_state_dict(filtered_state, strict=False)
    return net


def build_orchestrator(
    model_type: ModelType,
    net: torch.nn.Module,
    games: int,
    sims: int,
    k: int,
    wave_parallel: bool,
    wave_size: int,
    expansion_mask: int,
    max_game_length: int,
    temperature: float,
    temperature_drop_move: int,
    compile_forward: bool,
    use_puct: bool = False,
    c_puct: float | None = None,
    eval_mode: bool = True,
) -> tuple[object, object]:
    if model_type == "fnn":
        if use_puct:
            cfg = FNNPUCTConfig(
                num_simulations=sims,
                c_puct=c_puct if c_puct is not None else FNNPUCTConfig.c_puct,
                temperature=temperature,
                temperature_drop_move=temperature_drop_move,
                batch_size=games,
                max_game_length=max_game_length,
                expansion_mask=expansion_mask,
                wave_size=wave_size,
            )
            if eval_mode:
                cfg.dirichlet_epsilon = 0.0
                cfg.temperature_drop_move = 0
            orch = FNNPUCTOrchestrator(net, cfg)
        else:
            cfg = FNNMCTSConfig(
                num_simulations=sims,
                max_num_considered_actions=k,
                c_puct=c_puct if c_puct is not None else FNNMCTSConfig.c_puct,
                temperature=temperature,
                temperature_drop_move=temperature_drop_move,
                batch_size=games,
                max_game_length=max_game_length,
                expansion_mask=expansion_mask,
                wave_parallel=wave_parallel,
                wave_size=wave_size,
            )
            if eval_mode:
                cfg.dirichlet_epsilon = 0.0
                cfg.temperature_drop_move = 0
            orch = FNNMCTSOrchestrator(net, cfg)
    else:
        cfg = PRSMCTSConfigV2(
            num_simulations=sims,
            max_num_considered_actions=k,
            c_puct=c_puct if c_puct is not None else PRSMCTSConfigV2.c_puct,
            temperature=temperature,
            temperature_drop_move=temperature_drop_move,
            batch_size=games,
            max_game_length=max_game_length,
            expansion_mask=expansion_mask,
            nn_max_batch=0,
            wave_parallel=wave_parallel,
            compile_forward=compile_forward,
        )
        if eval_mode:
            cfg.dirichlet_epsilon = 0.0
            cfg.temperature_drop_move = 0
        orch = PRSMCTSOrchestratorV2(net, cfg)
    return orch, cfg


def choose_prs_moves(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
    gumbel_noise_scale: float = 0.0,
) -> np.ndarray:
    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    nlegal_np = nlegal_t.cpu().numpy()

    turns_np = orch._get_turns(states, B)
    active = [int(n) > 0 for n in nlegal_np]
    win_overrides = orch._check_immediate_wins_v2(
        states, legal_t, nlegal_np, active, turns_np, B,
    )

    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8]
    policy_logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    policy_logits_813 = policy_logits_813.float()
    priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
        policy_logits_813, slot_of_legal_t, B,
    )

    max_legal = int(nlegal_np.max()) if nlegal_np.size > 0 else 0
    if max_legal <= 0:
        return np.zeros(B, dtype=np.int64)
    max_k = min(cfg.max_num_considered_actions, max_legal)

    if gumbel_noise_scale > 0.0:
        u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        gumbel = -torch.log(-torch.log(u)) * gumbel_noise_scale
    else:
        gumbel = torch.zeros(B, orch._max_legal, dtype=torch.float32, device="cuda")
    score_basis = gumbel + root_logit_per_legal

    _, topk_slots = torch.topk(score_basis, max_k, dim=1)
    active_t = torch.tensor([1 if a else 0 for a in active], dtype=torch.int8, device="cuda")

    orch._expand_root_if_needed(
        tree, states, legal_t, nlegal_t, priors_per_legal, active_t, B,
    )
    if cfg.dirichlet_epsilon > 0:
        orch._apply_root_dirichlet(tree, B, active)

    alive_mask = torch.zeros(B, orch._max_legal, dtype=torch.int8, device="cuda")
    alive_mask.scatter_(1, topk_slots, 1)

    n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
    sims_per_round = max(1, cfg.num_simulations // n_rounds)

    for _ in range(n_rounds):
        orch._run_simulations(tree, states, active_t, alive_mask, B, sims_per_round)
        alive_counts = alive_mask.sum(dim=1)
        cur_alive_max = int(alive_counts.max().item())
        if cur_alive_max <= 1:
            continue
        num_keep = max(1, cur_alive_max // 2)
        slot_visits, slot_q = orch._gather_root_child_stats(tree, B)
        max_n = int(slot_visits.max().item())
        sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
        sigma_score = (score_basis + sigma_norm * slot_q).float()
        sigma_score = torch.where(
            alive_mask.bool(), sigma_score, torch.full_like(sigma_score, -1e30),
        )
        _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
        new_alive = torch.zeros_like(alive_mask)
        new_alive.scatter_(1, keep_slots, 1)
        slot_idx_t = torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
        valid_slot = slot_idx_t < nlegal_t.to(torch.int64).unsqueeze(1)
        alive_mask = new_alive * valid_slot.to(torch.int8)

    leg_visits, _ = orch._gather_root_child_stats(tree, B)
    slot_t = slot_of_legal_t.to(dtype=torch.long)
    valid_legal = slot_t >= 0
    slot_safe = slot_t.clamp(min=0)
    slot_sum = torch.zeros(B, N_SLOTS, dtype=torch.float32, device="cuda")
    slot_sum.scatter_add_(1, slot_safe, leg_visits.float() * valid_legal.float())
    agg_per_legal = slot_sum.gather(1, slot_safe)
    agg_per_legal = torch.where(valid_legal, agg_per_legal, torch.zeros_like(agg_per_legal))
    agg_per_legal = agg_per_legal * alive_mask.float()
    agg_np = agg_per_legal.cpu().numpy()

    chosen = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            chosen[i] = 0
            continue
        if i in win_overrides:
            chosen[i] = int(win_overrides[i])
            continue
        scores = agg_np[i, :n_i].astype(np.float64, copy=True)
        if stochastic and move_numbers is not None and int(move_numbers[i]) < cfg.temperature_drop_move:
            ssum = float(scores.sum())
            if ssum > 0 and np.isfinite(ssum):
                probs = scores / ssum
            else:
                probs = np.full(n_i, 1.0 / n_i, dtype=np.float64)
            chosen[i] = int(np.random.choice(n_i, p=probs))
        else:
            chosen[i] = int(np.argmax(scores))
    return chosen


def choose_prs_policy_moves(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
    gumbel_noise_scale: float = 0.0,
) -> np.ndarray:
    B = int(states.shape[0])
    cfg = orch.config
    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    nlegal_np = nlegal_t.cpu().numpy()

    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8]
    policy_logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    policy_logits_813 = policy_logits_813.float()
    priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
        policy_logits_813, slot_of_legal_t, B,
    )

    nlegal_t_gpu = nlegal_t.to(torch.int64)
    valid_slot = torch.arange(orch._max_legal, device="cuda").unsqueeze(0) < nlegal_t_gpu.unsqueeze(1)
    if gumbel_noise_scale > 0.0:
        u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        gumbel = -torch.log(-torch.log(u)) * gumbel_noise_scale
        scores = torch.where(valid_slot, root_logit_per_legal + gumbel, torch.full_like(root_logit_per_legal, -1e30))
    else:
        scores = torch.where(valid_slot, priors_per_legal, torch.full_like(priors_per_legal, -1.0))

    chosen = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            chosen[i] = 0
            continue
        row_scores = scores[i, :n_i]
        if stochastic and move_numbers is not None and int(move_numbers[i]) < cfg.temperature_drop_move:
            if gumbel_noise_scale > 0.0:
                probs = torch.softmax(row_scores.float(), dim=0)
            else:
                probs = row_scores.float().clamp_min(0)
                probs = probs / probs.sum().clamp_min(1e-8)
            chosen[i] = int(torch.multinomial(probs, 1).item())
        else:
            chosen[i] = int(torch.argmax(row_scores).item())
    return chosen


def choose_fnn_moves(
    orch: FNNMCTSOrchestrator | FNNPUCTOrchestrator,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
    gumbel_noise_scale: float = 0.0,
) -> np.ndarray:
    if isinstance(orch, FNNPUCTOrchestrator):
        return choose_fnn_puct_moves(
            orch,
            states,
            move_numbers=move_numbers,
            stochastic=stochastic,
        )

    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    n_per_game = num_legal.to(torch.int64)
    nlegal_np = num_legal.cpu().numpy()

    has_actions = n_per_game > 0
    immediate_wins = orch._find_immediate_wins(states, legal_moves, num_legal, B)
    has_immediate_win = immediate_wins >= 0

    max_n = int(n_per_game.max().item()) if B > 0 else 0
    if max_n == 0:
        return np.zeros(B, dtype=np.int64)
    max_k = min(cfg.max_num_considered_actions, max_n)

    priors_per_legal, _root_vals, _child_q = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )
    if bool(has_immediate_win.any().item()):
        priors_per_legal.zero_()
        priors_per_legal[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
        _child_q.zero_()

    valid_slot = (
        torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
        < n_per_game.unsqueeze(1)
    )
    safe_prior = priors_per_legal.clamp(min=1e-20)
    legal_logits = torch.where(
        valid_slot, safe_prior.log(),
        torch.full_like(priors_per_legal, -1e30),
    )

    if bool(has_immediate_win.any().item()):
        probs_pad = torch.zeros(B, orch._max_legal, dtype=torch.float32, device="cuda")
        probs_pad[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
    else:
        if gumbel_noise_scale > 0.0:
            u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
            gumbel = -torch.log(-torch.log(u)) * gumbel_noise_scale
        else:
            gumbel = torch.zeros(B, orch._max_legal, dtype=torch.float32, device="cuda")
        gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
        perturbed = gumbel + legal_logits
        _, topk_slots = torch.topk(perturbed, max_k, dim=1)

        active_t = has_actions.to(torch.int8)
        orch._expand_root_if_needed(
            tree, states, legal_moves, num_legal,
            priors_per_legal, _child_q, active_t, B,
        )
        orch._apply_root_dirichlet(tree, B, has_actions)

        alive_mask = torch.zeros(B, orch._max_legal, dtype=torch.int8, device="cuda")
        alive_mask.scatter_(1, topk_slots, 1)

        n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
        sims_per_round = max(1, cfg.num_simulations // n_rounds)

        for _ in range(n_rounds):
            if cfg.wave_parallel:
                custom_schedule = getattr(cfg, "wave_schedule", None)
                if custom_schedule:
                    round_wave_size = custom_schedule[min(_, len(custom_schedule) - 1)]
                else:
                    round_wave_size = orch.config.wave_size
            else:
                round_wave_size = 1
            orch._run_simulations(
                tree, states, active_t, alive_mask, B, sims_per_round,
                wave_size=round_wave_size,
            )
            alive_counts = alive_mask.sum(dim=1)
            cur_alive_max = int(alive_counts.max().item())
            if cur_alive_max <= 1:
                continue
            num_keep = max(1, cur_alive_max // 2)
            slot_visits, slot_q = orch._gather_root_child_stats(tree, B)
            max_n = int(slot_visits.max().item())
            sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
            sigma_score = (gumbel + legal_logits + sigma_norm * slot_q).float()
            sigma_score = torch.where(
                alive_mask.bool(), sigma_score,
                torch.full_like(sigma_score, -1e30),
            )
            _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
            new_alive = torch.zeros_like(alive_mask)
            new_alive.scatter_(1, keep_slots, 1)
            alive_mask = new_alive * valid_slot.to(torch.int8)

        slot_visits, _ = orch._gather_root_child_stats(tree, B)
        visits_f = slot_visits.float()
        vsum = visits_f.sum(dim=1, keepdim=True)
        probs_pad = torch.where(
            vsum > 0,
            visits_f / vsum.clamp(min=1.0),
            torch.zeros_like(visits_f),
        )
        uniform = valid_slot.float()
        ucount = uniform.sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform = uniform / ucount
        probs_pad = torch.where(vsum > 0, probs_pad, uniform)

    chosen = torch.zeros((B,), dtype=torch.int64, device="cuda")
    if bool(has_immediate_win.any().item()):
        chosen[has_immediate_win] = immediate_wins[has_immediate_win]
    else:
        active_idx = torch.nonzero(has_actions, as_tuple=False).squeeze(1)
        if active_idx.numel() > 0:
            if stochastic and move_numbers is not None:
                move_numbers_np = np.asarray(move_numbers)
                active_moves = move_numbers_np[active_idx.cpu().numpy()]
                greedy_mask = active_moves >= cfg.temperature_drop_move
                greedy_rows = active_idx[greedy_mask]
                sample_rows = active_idx[~greedy_mask]
            else:
                greedy_rows = active_idx
                sample_rows = active_idx[:0]

            if greedy_rows.numel() > 0:
                chosen[greedy_rows] = probs_pad[greedy_rows].argmax(dim=1)
            if sample_rows.numel() > 0:
                sample_policy = probs_pad[sample_rows]
                sample_policy = sample_policy / sample_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)
                chosen[sample_rows] = torch.multinomial(sample_policy, 1).squeeze(1)

    chosen_np = chosen.cpu().numpy()
    out = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            out[i] = 0
        else:
            k = int(chosen_np[i])
            out[i] = 0 if k >= n_i else k
    return out


def choose_fnn_policy_moves(
    orch: FNNMCTSOrchestrator | FNNPUCTOrchestrator,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
    gumbel_noise_scale: float = 0.0,
) -> np.ndarray:
    B = int(states.shape[0])
    if isinstance(orch, FNNPUCTOrchestrator):
        cfg = orch.config
        max_legal = orch._max_legal
    else:
        cfg = orch.config
        max_legal = orch._max_legal

    legal_moves, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    priors_per_legal, _root_vals, _child_q = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )
    nlegal_np = num_legal.cpu().numpy()
    nlegal_t = num_legal.to(torch.int64)
    valid_slot = torch.arange(max_legal, device="cuda").unsqueeze(0) < nlegal_t.unsqueeze(1)
    safe_prior = priors_per_legal.clamp(min=1e-20)
    legal_logits = torch.where(valid_slot, safe_prior.log(), torch.full_like(priors_per_legal, -1e30))
    if gumbel_noise_scale > 0.0:
        u = torch.rand(B, max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        scores = legal_logits + (-torch.log(-torch.log(u)) * gumbel_noise_scale)
    else:
        scores = torch.where(valid_slot, priors_per_legal, torch.full_like(priors_per_legal, -1.0))

    chosen = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            chosen[i] = 0
            continue
        row_scores = scores[i, :n_i]
        if stochastic and move_numbers is not None and int(move_numbers[i]) < cfg.temperature_drop_move:
            if gumbel_noise_scale > 0.0:
                probs = torch.softmax(row_scores.float(), dim=0)
            else:
                probs = row_scores.float().clamp_min(0)
                probs = probs / probs.sum().clamp_min(1e-8)
            chosen[i] = int(torch.multinomial(probs, 1).item())
        else:
            chosen[i] = int(torch.argmax(row_scores).item())
    return chosen


def choose_fnn_puct_moves(
    orch: FNNPUCTOrchestrator,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
) -> np.ndarray:
    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    n_per_game = num_legal.to(torch.int64)
    nlegal_np = num_legal.cpu().numpy()
    has_actions = n_per_game > 0

    immediate_wins = orch._find_immediate_wins(states, legal_moves, num_legal, B)
    has_immediate_win = immediate_wins >= 0

    max_n = int(n_per_game.max().item()) if B > 0 else 0
    if max_n == 0:
        return np.zeros(B, dtype=np.int64)

    priors_per_legal, _root_vals, child_q_per_legal = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )
    if bool(has_immediate_win.any().item()):
        priors_per_legal.zero_()
        priors_per_legal[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
        child_q_per_legal.zero_()

    game_active_t = has_actions.to(torch.int8)
    orch._expand_root_if_needed(
        tree, states, legal_moves, num_legal,
        priors_per_legal, child_q_per_legal, game_active_t, B,
    )
    orch._apply_root_dirichlet(tree, B, has_actions)

    if bool(has_immediate_win.any().item()):
        probs_pad = torch.zeros(B, orch._max_legal, dtype=torch.float32, device="cuda")
        probs_pad[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
    else:
        valid_slot = (
            torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
            < n_per_game.unsqueeze(1)
        )
        alive_mask = valid_slot.to(torch.int8)
        orch._run_simulations(tree, states, game_active_t, alive_mask, B, cfg.num_simulations)
        slot_visits, _slot_q = orch._gather_root_child_stats(tree, B)
        visits_f = slot_visits.float()
        vsum = visits_f.sum(dim=1, keepdim=True)
        probs_pad = torch.where(
            vsum > 0,
            visits_f / vsum.clamp(min=1.0),
            torch.zeros_like(visits_f),
        )
        uniform = valid_slot.float()
        ucount = uniform.sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform = uniform / ucount
        probs_pad = torch.where(vsum > 0, probs_pad, uniform)

    chosen = torch.zeros((B,), dtype=torch.int64, device="cuda")
    active_idx = torch.nonzero(has_actions, as_tuple=False).squeeze(1)
    if bool(has_immediate_win.any().item()):
        chosen[has_immediate_win] = immediate_wins[has_immediate_win]
    elif active_idx.numel() > 0:
        if stochastic and move_numbers is not None:
            move_numbers_np = np.asarray(move_numbers)
            active_moves = move_numbers_np[active_idx.cpu().numpy()]
            greedy_mask = active_moves >= cfg.temperature_drop_move
            greedy_rows = active_idx[greedy_mask]
            sample_rows = active_idx[~greedy_mask]
        else:
            greedy_rows = active_idx
            sample_rows = active_idx[:0]
        if greedy_rows.numel() > 0:
            chosen[greedy_rows] = probs_pad[greedy_rows].argmax(dim=1)
        if sample_rows.numel() > 0:
            sample_policy = probs_pad[sample_rows]
            sample_policy = sample_policy / sample_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)
            chosen[sample_rows] = torch.multinomial(sample_policy, 1).squeeze(1)

    chosen_np = chosen.cpu().numpy()
    out = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            out[i] = 0
        else:
            k = int(chosen_np[i])
            out[i] = 0 if k >= n_i else k
    return out


def choose_moves(
    model_type: ModelType,
    orch: object,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
    gumbel_noise_scale: float = 0.0,
    policy_only: bool = False,
) -> np.ndarray:
    if policy_only:
        if model_type == "fnn":
            return choose_fnn_policy_moves(
                orch,
                states,
                move_numbers=move_numbers,
                stochastic=stochastic,
                gumbel_noise_scale=gumbel_noise_scale,
            )
        return choose_prs_policy_moves(
            orch,
            states,
            move_numbers=move_numbers,
            stochastic=stochastic,
            gumbel_noise_scale=gumbel_noise_scale,
        )
    if model_type == "fnn":
        return choose_fnn_moves(
            orch,
            states,
            move_numbers=move_numbers,
            stochastic=stochastic,
            gumbel_noise_scale=gumbel_noise_scale,
        )
    return choose_prs_moves(
        orch,
        states,
        move_numbers=move_numbers,
        stochastic=stochastic,
        gumbel_noise_scale=gumbel_noise_scale,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="General arena: compare FNN and PRS checkpoints in any pairing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Compare PRS vs FNN:\n"
            "    python3.11 arena.py --white-model prs --black-model fnn \\\n"
            "      --white-checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt \\\n"
            "      --black-checkpoint checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt\n\n"
            "  Compare small vs large FNN:\n"
            "    python3.11 arena.py --white-model fnn --black-model fnn \\\n"
            "      --white-checkpoint checkpoints_fnn_small/hive_fnn_checkpoint_0100.pt \\\n"
            "      --black-checkpoint checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt\n\n"
            "  Compare two PRS checkpoints:\n"
            "    python3.11 arena.py --white-model prs --black-model prs \\\n"
            "      --white-checkpoint checkpoints_prs_v2/prs_v2_iter_0550.pt \\\n"
            "      --black-checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt\n"
        ),
    )
    ap.add_argument("--white-model", choices=["fnn", "prs"], default="prs")
    ap.add_argument("--black-model", choices=["fnn", "prs"], default="fnn")
    ap.add_argument("--white-checkpoint", type=str, default=None)
    ap.add_argument("--black-checkpoint", type=str, default=None)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--white-sims", type=int, default=None)
    ap.add_argument("--black-sims", type=int, default=None)
    ap.add_argument("--white-k", type=int, default=None)
    ap.add_argument("--black-k", type=int, default=None)
    ap.add_argument("--white-wave-size", type=int, default=None)
    ap.add_argument("--black-wave-size", type=int, default=None)
    ap.add_argument(
        "--white-fnn-wave-schedule",
        type=str,
        default=None,
        help="Comma-separated per-round FNN wave sizes for white/model A, e.g. 1,2,4,8,16",
    )
    ap.add_argument(
        "--black-fnn-wave-schedule",
        type=str,
        default=None,
        help="Comma-separated per-round FNN wave sizes for black/model B, e.g. 2,4,8,16",
    )
    ap.add_argument("--white-puct", action="store_true", default=False)
    ap.add_argument("--black-puct", action="store_true", default=False)
    ap.add_argument("--white-policy-only", action="store_true", default=False)
    ap.add_argument("--black-policy-only", action="store_true", default=False)
    ap.add_argument("--white-c-puct", type=float, default=None)
    ap.add_argument("--black-c-puct", type=float, default=None)
    ap.add_argument("--wave-parallel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperature-drop-move", type=int, default=20)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--max-game-length", type=int, default=300)
    ap.add_argument(
        "--alternate-colors",
        action="store_true",
        help="Alternate which model is white on a per-game basis.",
    )
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument(
        "--gumbel-noise-scale",
        type=float,
        default=0.0,
        help="Scale for root Gumbel noise in evaluation; 0 disables noise.",
    )
    ap.add_argument("--compile-forward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    eval_mode = not args.stochastic

    white_ckpt_path = args.white_checkpoint or latest_checkpoint(args.white_model)
    black_ckpt_path = args.black_checkpoint or latest_checkpoint(args.black_model)
    if not white_ckpt_path:
        raise SystemExit(f"No checkpoint found for white model type {args.white_model!r}")
    if not black_ckpt_path:
        raise SystemExit(f"No checkpoint found for black model type {args.black_model!r}")

    white_net = load_checkpoint(args.white_model, white_ckpt_path)
    black_net = load_checkpoint(args.black_model, black_ckpt_path)

    white_sims = args.white_sims if args.white_sims is not None else default_sims(args.white_model)
    black_sims = args.black_sims if args.black_sims is not None else default_sims(args.black_model)
    white_k = args.white_k if args.white_k is not None else default_k(args.white_model)
    black_k = args.black_k if args.black_k is not None else default_k(args.black_model)
    white_wave_size = (
        args.white_wave_size if args.white_wave_size is not None else default_wave_size(args.white_model)
    )
    black_wave_size = (
        args.black_wave_size if args.black_wave_size is not None else default_wave_size(args.black_model)
    )
    white_wave_schedule = parse_wave_schedule(args.white_fnn_wave_schedule)
    black_wave_schedule = parse_wave_schedule(args.black_fnn_wave_schedule)

    white_orch, white_cfg = build_orchestrator(
        args.white_model, white_net, args.games, white_sims, white_k, args.wave_parallel,
        white_wave_size, args.expansion_mask, args.max_game_length, args.temperature,
        args.temperature_drop_move, args.compile_forward,
        use_puct=args.white_puct,
        c_puct=args.white_c_puct,
        eval_mode=eval_mode,
    )
    black_orch, black_cfg = build_orchestrator(
        args.black_model, black_net, args.games, black_sims, black_k, args.wave_parallel,
        black_wave_size, args.expansion_mask, args.max_game_length, args.temperature,
        args.temperature_drop_move, args.compile_forward,
        use_puct=args.black_puct,
        c_puct=args.black_c_puct,
        eval_mode=eval_mode,
    )
    if args.white_model == "fnn":
        setattr(white_cfg, "wave_schedule", white_wave_schedule)
    if args.black_model == "fnn":
        setattr(black_cfg, "wave_schedule", black_wave_schedule)
    ext = hive_gpu.load_extension()

    B = args.games
    states = ext.create_initial_states(B, args.expansion_mask)
    active = np.ones(B, dtype=bool)
    move_numbers = np.zeros(B, dtype=np.int32)
    if args.alternate_colors:
        white_is_model_a = np.zeros(B, dtype=bool)
        white_is_model_a[::2] = True
    else:
        white_is_model_a = np.zeros(B, dtype=bool)
        white_is_model_a[: B // 2] = True

    while bool(active.any()):
        if args.white_model == "prs":
            turns = white_orch._get_turns(states, B)
        else:
            turn_off = 3412
            turns = (
                states[:, turn_off].cpu().numpy().astype(np.int32)
                | (states[:, turn_off + 1].cpu().numpy().astype(np.int32) << 8)
            )
        white_to_move = (turns % 2 == 0)
        white_turn = active & (white_to_move == white_is_model_a)
        black_turn = active & ~white_turn

        move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
        for mask, model_type, orch in (
            (white_turn, args.white_model, white_orch),
            (black_turn, args.black_model, black_orch),
        ):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            sub_states = states[idx].clone()
            chosen = choose_moves(
                model_type,
                orch,
                sub_states,
                move_numbers=move_numbers[idx],
                stochastic=args.stochastic,
                gumbel_noise_scale=args.gumbel_noise_scale,
                policy_only=args.white_policy_only if model_type == args.white_model and orch is white_orch else args.black_policy_only,
            )
            legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(sub_states, idx.size)
            legal_np = legal_t.cpu().numpy()
            nlegal_np = nlegal_t.cpu().numpy()
            for j, game_idx in enumerate(idx):
                n_i = int(nlegal_np[j])
                if n_i <= 0:
                    continue
                k = int(chosen[j])
                if k >= n_i:
                    k = 0
                move_bytes[game_idx] = legal_np[j, k]

        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            break
        sub_states = states[active_idx].clone()
        sub_moves = torch.from_numpy(move_bytes[active_idx]).cuda()
        ext.apply_moves_batch(sub_states, sub_moves, int(active_idx.size))
        states[active_idx] = sub_states
        move_numbers[active_idx] += 1

        final_results = ext.check_results_batch(states, B).cpu().numpy()
        active = active & (final_results == 0) & (move_numbers < args.max_game_length)

    white_wins = 0
    black_wins = 0
    draws = 0
    capped = 0
    white_score = 0.0
    model_a_score = 0.0
    model_a_white_games = 0
    model_a_black_games = 0
    model_a_white_score = 0.0
    model_a_black_score = 0.0
    final_results = ext.check_results_batch(states, B).cpu().numpy()
    for i in range(B):
        r = int(final_results[i])
        a_is_white = bool(white_is_model_a[i])
        if r == 0:
            capped += 1
            draws += 1
            white_score += 0.5
            model_a_score += 0.5
            if a_is_white:
                model_a_white_games += 1
                model_a_white_score += 0.5
            else:
                model_a_black_games += 1
                model_a_black_score += 0.5
        elif r == 3:
            draws += 1
            white_score += 0.5
            model_a_score += 0.5
            if a_is_white:
                model_a_white_games += 1
                model_a_white_score += 0.5
            else:
                model_a_black_games += 1
                model_a_black_score += 0.5
        elif (r == 1 and a_is_white) or (r == 2 and not a_is_white):
            white_wins += 1
            white_score += 1.0
            model_a_score += 1.0
            if a_is_white:
                model_a_white_games += 1
                model_a_white_score += 1.0
            else:
                model_a_black_games += 1
                model_a_black_score += 1.0
        else:
            black_wins += 1
            if a_is_white:
                model_a_white_games += 1
            else:
                model_a_black_games += 1

    print(f"White model:      {args.white_model}")
    print(f"White checkpoint: {white_ckpt_path}")
    print(f"Black model:      {args.black_model}")
    print(f"Black checkpoint: {black_ckpt_path}")
    print(
        f"Arena config: games={B}, white_sims={white_sims}, black_sims={black_sims}, "
        f"white_k={white_k}, black_k={black_k}, "
        f"white_root={'puct' if args.white_puct else 'gumbel'}, "
        f"black_root={'puct' if args.black_puct else 'gumbel'}, "
        f"white_policy_only={args.white_policy_only}, "
        f"black_policy_only={args.black_policy_only}, "
        f"white_c_puct={getattr(white_cfg, 'c_puct', 'n/a')}, "
        f"black_c_puct={getattr(black_cfg, 'c_puct', 'n/a')}, "
        f"white_wave_size={white_cfg.wave_size if hasattr(white_cfg, 'wave_size') else 'n/a'}, "
        f"black_wave_size={black_cfg.wave_size if hasattr(black_cfg, 'wave_size') else 'n/a'}, "
        f"expansion_mask={args.expansion_mask}, max_game_length={args.max_game_length}, "
        f"stochastic={args.stochastic}, gumbel_noise_scale={args.gumbel_noise_scale}, "
        f"wave_parallel={args.wave_parallel}, alternate_colors={args.alternate_colors}"
    )
    if args.alternate_colors:
        print(f"Model A score:    {model_a_score:.1f}/{B} = {model_a_score / B:.3f}")
        if model_a_white_games:
            print(
                f"Model A as white: {model_a_white_score:.1f}/{model_a_white_games} = "
                f"{model_a_white_score / model_a_white_games:.3f}"
            )
        if model_a_black_games:
            print(
                f"Model A as black: {model_a_black_score:.1f}/{model_a_black_games} = "
                f"{model_a_black_score / model_a_black_games:.3f}"
            )
    else:
        print(f"White wins:      {white_wins}")
        print(f"Black wins:      {black_wins}")
        print(f"Draws:           {draws}  (capped={capped})")
        print(f"White score:     {white_score:.1f}/{B} = {white_score / B:.3f}")


if __name__ == "__main__":
    main()
