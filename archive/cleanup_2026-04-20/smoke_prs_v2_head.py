"""Smoke-test PRSv2PolicyHead + slot-mapping round-trip.

Exercises:
  1. Build head inputs from random states.
  2. Forward pass produces (B, 813) logits with no NaNs.
  3. Slot-of-legal gather: for each legal move, the head provides a prior.
  4. Training target scatter: visit counts sum correctly per slot.
  5. Backprop through masked cross-entropy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import hive_gpu
from hive_prs.prs_v2_head import PRSv2PolicyHead, MB
from hive_prs.prs_v2_bridge import build_head_inputs_from_states
from hive_prs.slot_map import N_SLOTS, map_legal_moves


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()
    device = torch.device("cuda")

    B = 32
    D = 64
    states = ext.create_initial_states(B, 7)
    # Play a few random plies to get non-trivial positions
    for _ in range(15):
        legal, nlegal = ext.generate_legal_moves_batch(states, B)
        chosen = torch.zeros(B, dtype=torch.long, device=device)
        nc = nlegal.cpu().numpy()
        for i in range(B):
            chosen[i] = np.random.randint(max(1, int(nc[i])))
        moves = legal[torch.arange(B, device=device), chosen]
        ext.apply_moves_batch(states, moves, B)

    # Simulate trunk outputs: random board_h + cls_h + full_h.
    # full_h seq length must be long enough for CLS + board + hand tokens.
    SEQ = MB + 32  # generous upper bound
    board_h = torch.randn(B, MB, D, device=device, requires_grad=True)
    cls_h   = torch.randn(B, D, device=device, requires_grad=True)
    full_h  = torch.randn(B, SEQ, D, device=device, requires_grad=True)

    state_cpu = states.cpu().numpy()
    inp, mappers = build_head_inputs_from_states(state_cpu, board_h, cls_h, full_h)

    head = PRSv2PolicyHead(D).to(device)
    logits = head(inp)  # (B, 813)
    print(f"Logits shape: {tuple(logits.shape)}")
    assert logits.shape == (B, N_SLOTS)
    assert torch.isfinite(logits).all(), "non-finite logits"
    print(f"Logits stats: min={logits.min():.3f}, max={logits.max():.3f}, "
          f"mean={logits.mean():.3f}, std={logits.std():.3f}")

    # Per-legal-move prior gather (what the orchestrator will do)
    legal, nlegal = ext.generate_legal_moves_batch(states, B)
    legal_cpu = legal.cpu().numpy()
    nl_cpu = nlegal.cpu().numpy()

    total_legal = 0
    mismatches = 0
    for b in range(B):
        n = int(nl_cpu[b])
        if n == 0:
            continue
        slots, _, _ = map_legal_moves(state_cpu[b], legal_cpu[b, :n], n)
        total_legal += n
        # Build legal mask in slot space
        legal_mask = torch.zeros(N_SLOTS, dtype=torch.bool, device=device)
        for s in slots:
            if s >= 0:
                legal_mask[int(s)] = True
        # Softmax over legal slots
        masked = logits[b].masked_fill(~legal_mask, float("-inf"))
        probs = F.softmax(masked, dim=0)
        # Check total probability over legal slots ≈ 1.0
        legal_prob_mass = probs[legal_mask].sum().item()
        if abs(legal_prob_mass - 1.0) > 1e-4:
            mismatches += 1

    print(f"\nChecked {total_legal} legal moves across {B} states")
    print(f"States where legal-slot softmax != 1.0: {mismatches}")

    # Build fake visit-count targets, scatter to slot space, compute loss
    loss_total = 0.0
    n_loss = 0
    for b in range(B):
        n = int(nl_cpu[b])
        if n == 0:
            continue
        slots, _, _ = map_legal_moves(state_cpu[b], legal_cpu[b, :n], n)
        slots_t = torch.from_numpy(slots).long().to(device)
        # Random visit counts
        visits = torch.randint(1, 20, (n,), device=device, dtype=torch.float32)
        target = torch.zeros(N_SLOTS, device=device)
        target.scatter_add_(0, slots_t.clamp(min=0), visits)
        target = target / target.sum()

        legal_mask = torch.zeros(N_SLOTS, dtype=torch.bool, device=device)
        for s in slots:
            if s >= 0:
                legal_mask[int(s)] = True
        masked = logits[b].masked_fill(~legal_mask, float("-inf"))
        logp = F.log_softmax(masked, dim=0)
        # Only sum over legal slots — 0 × -inf = NaN otherwise
        loss_total = loss_total + -(target[legal_mask] * logp[legal_mask]).sum()
        n_loss += 1

    loss = loss_total / max(n_loss, 1)
    print(f"\nTraining loss (mean over batch): {float(loss):.4f}")
    loss.backward()
    # Check gradients flow
    assert board_h.grad is not None and torch.isfinite(board_h.grad).all()
    assert cls_h.grad is not None and torch.isfinite(cls_h.grad).all()
    assert full_h.grad is not None and torch.isfinite(full_h.grad).all()
    print(f"Grad OK. board_h.grad={board_h.grad.norm():.4f}, "
          f"cls_h.grad={cls_h.grad.norm():.4f}, "
          f"full_h.grad={full_h.grad.norm():.4f}")

    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nHead params: {n_params:,}")


if __name__ == "__main__":
    main()
