"""
PRS Win-in-one Gumbel analysis.

For positions where the active player has a forced win in one move:
  1. Is the winning PRS action valid (encodable)?
  2. What is its rank by raw policy logit?
  3. What is its rank by Gumbel-perturbed score?
  4. Is it captured in the Gumbel top-K candidates?
  5. Does the halving algorithm select it as the final action?
"""

import math
import sys
import numpy as np
import torch

import hive_gpu
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_transformer import HivePRSTransformer, PRSConfig
from hive_prs.action_space import (
    ACTION_SPACE_SIZE, batch_moves_to_action_indices, decode_action
)
from hive_prs.prs_orchestrator import (
    PRSGumbelOrchestrator, PRSGumbelConfig, _OFF_TURN,
)

# ── Config ─────────────────────────────────────────────────────────────────────

EXPANSION   = 0      # base game
K           = 16     # Gumbel top-k candidates
NUM_SIMS    = 128    # halving simulation budget
C_VISIT     = 50.0
C_SCALE     = 1.0
TEMPERATURE = 1.0
NUM_WANTED  = 40     # win-in-one positions to collect
MAX_STEPS   = 600    # random play steps to search
B_PLAY      = 128    # parallel games used for searching

# ── Setup ──────────────────────────────────────────────────────────────────────

ext     = hive_gpu.load_extension()
encoder = PRSEncoder()
net     = HivePRSTransformer(PRSConfig.small()).cuda().eval()
move_sz = ext.SIZEOF_GPU_MOVE

# ── Step 1: Collect win-in-one positions ───────────────────────────────────────

print(f"Searching for {NUM_WANTED} win-in-one positions "
      f"(B={B_PLAY}, max_steps={MAX_STEPS})...")

states    = ext.create_initial_states(B_PLAY, EXPANSION)
win_list  = []   # list of (state: Tensor(1,S), win_move: ndarray(move_sz,))

for step in range(MAX_STEPS):
    if len(win_list) >= NUM_WANTED:
        break

    legal_t, nlegal_t = ext.generate_legal_moves_batch(states, B_PLAY)
    legal_np  = legal_t.cpu().numpy()    # (B, MAX_L, move_sz)
    nlegal_np = nlegal_t.cpu().numpy()   # (B,)

    rand_np = np.zeros((B_PLAY, move_sz), dtype=np.uint8)

    for i in range(B_PLAY):
        nl = int(nlegal_np[i])
        if nl == 0:
            continue

        # Try all legal moves to detect win-in-one
        trial  = states[i:i+1].expand(nl, -1).clone()   # (nl, S)
        moves_i = torch.from_numpy(legal_np[i, :nl]).cuda()
        ext.apply_moves_batch(trial, moves_i, nl)
        res = ext.check_results_batch(trial, nl)   # (nl,)

        tb       = int(states[i, _OFF_TURN].item())
        is_white = (tb % 2 == 0)
        win_mask = (res == 1) if is_white else (res == 2)

        if win_mask.any():
            j = int(win_mask.nonzero(as_tuple=True)[0][0].item())
            if len(win_list) < NUM_WANTED:
                win_list.append((
                    states[i:i+1].clone(),
                    legal_np[i, j].copy(),
                ))

        # Random move
        j_r = np.random.randint(0, nl)
        rand_np[i] = legal_np[i, j_r]

    ext.apply_moves_batch(states, torch.from_numpy(rand_np).cuda(), B_PLAY)

    # Reset finished games
    done = (ext.check_results_batch(states, B_PLAY) != 0).cpu()
    if done.any():
        fresh = ext.create_initial_states(B_PLAY, EXPANSION)
        for i in range(B_PLAY):
            if done[i]:
                states[i] = fresh[i]

print(f"Collected {len(win_list)} positions.\n")

# ── Step 2: Analyse each position ─────────────────────────────────────────────

rows = []
gumbel_topk_stats = []   # for positions where win move IS in top-K

print(f"{'#':>3}  {'Value':>7}  {'Legal':>5}  {'LogitRk':>8}  "
      f"{'GumRk':>6}  {'InTop16':>7}  {'HalvSel':>7}  {'nPRS':>5}  Action")
print("-" * 80)

for idx, (state1, win_bytes) in enumerate(win_list):
    # ── Encode ──
    prs_batch = encoder.encode_batch(state1, 1)
    occ_cpu   = prs_batch.occupied_cells.cpu().numpy()   # (1, MAX_BOARD)
    nocc_cpu  = prs_batch.num_occupied.cpu().numpy()     # (1,)

    # ── Legal moves + PRS conversion ──
    legal_t, nlegal_t = ext.generate_legal_moves_batch(state1, 1)
    legal_np  = legal_t.cpu().numpy()    # (1, MAX_L, move_sz)
    nlegal_np = nlegal_t.cpu().numpy()   # (1,)
    nl        = int(nlegal_np[0])

    if nl == 0:
        continue

    prs_arr = batch_moves_to_action_indices(
        legal_np, nlegal_np, occ_cpu, nocc_cpu
    )[0]   # (nl,) int32

    # ── Winning move → PRS index ──
    win_np   = win_bytes.reshape(1, 1, move_sz)
    win_nl   = np.array([1], dtype=np.int32)
    win_prs  = int(batch_moves_to_action_indices(
        win_np, win_nl, occ_cpu, nocc_cpu
    )[0][0])

    win_valid = 0 <= win_prs < ACTION_SPACE_SIZE

    # ── NN forward ──
    with torch.no_grad():
        logits, value = net(prs_batch)
    logits = logits[0].float()    # (A,)
    value  = float(value[0, 0])

    # ── Build legal mask ──
    valid_mask = (prs_arr >= 0) & (prs_arr < ACTION_SPACE_SIZE)
    legal_prs  = prs_arr[valid_mask].astype(np.int64)

    prs_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool, device="cuda")
    if len(legal_prs):
        prs_mask[legal_prs] = True

    logits[~prs_mask] = float("-inf")
    n_legal_prs = int(prs_mask.sum().item())

    if n_legal_prs == 0:
        continue

    # ── Logit rank of winning move ──
    if win_valid and prs_mask[win_prs]:
        legal_logits = logits[prs_mask].cpu().numpy()       # (n_legal_prs,)
        win_logit    = float(logits[win_prs].item())
        logit_rank   = int((legal_logits > win_logit).sum()) + 1
    else:
        logit_rank = -1

    # ── Gumbel topk ──
    li  = prs_mask.nonzero(as_tuple=False).squeeze(1)   # (n_legal_prs,)
    legal_log = logits[li].unsqueeze(0)                 # (1, n_legal_prs)
    legal_idx = li.unsqueeze(0)                          # (1, n_legal_prs)

    u       = torch.rand(1, n_legal_prs, device="cuda").clamp(1e-10, 1 - 1e-7)
    gumbel  = -torch.log(-torch.log(u))
    gumbel[legal_log == float("-inf")] = float("-inf")
    perturbed = gumbel + legal_log   # (1, n_legal_prs)

    eff_k  = min(K, n_legal_prs)
    _, topk_local  = torch.topk(perturbed, eff_k, dim=1)
    topk_global    = legal_idx.gather(1, topk_local)    # (1, eff_k)
    perturbed_topk = perturbed.gather(1, topk_local)    # (1, eff_k)
    topk_set       = set(topk_global[0].cpu().numpy().tolist())

    # Gumbel rank (among all legal PRS actions)
    if win_valid and prs_mask[win_prs]:
        gum_vals  = perturbed[0].cpu().numpy()
        win_local = int((li.cpu().numpy() == win_prs).argmax())
        gum_rank  = int((gum_vals > gum_vals[win_local]).sum()) + 1
    else:
        gum_rank = -1

    win_in_topk = win_valid and (win_prs in topk_set)

    # ── Halving selection (full Gumbel AlphaZero round for 1 game) ──
    # Use the orchestrator's _halving_round_batched to get Q-estimates, then
    # apply the temperature-based final selection.
    move_from_prs_local: dict[int, int] = {}
    if len(legal_prs):
        _, first_occ = np.unique(prs_arr[valid_mask], return_index=True)
        orig_j_of_valid = np.where(valid_mask)[0]
        for fi in first_occ:
            p  = int(prs_arr[valid_mask][fi])
            j_ = int(orig_j_of_valid[fi])
            move_from_prs_local[p] = j_

    n_rounds       = max(1, math.ceil(math.log2(eff_k + 1e-9)))
    evals_per_round = max(1, NUM_SIMS // max(n_rounds * eff_k, 1))

    q_sums      = torch.zeros(1, eff_k, device="cuda")
    visit_cnt   = torch.zeros(1, eff_k, dtype=torch.int32, device="cuda")
    cand_mask_h = torch.ones(1, eff_k,  dtype=torch.bool, device="cuda")

    # Build a temporary orchestrator just to call _halving_round_batched
    cfg_tmp = PRSGumbelConfig(
        num_simulations            = NUM_SIMS,
        max_num_considered_actions = K,
        c_visit                    = C_VISIT,
        c_scale                    = C_SCALE,
        temperature                = TEMPERATURE,
        batch_size                 = 1,
        expansion_mask             = EXPANSION,
    )
    orch = PRSGumbelOrchestrator(net, cfg_tmp)

    visits_per_cand = max(1, NUM_SIMS // max(n_rounds * eff_k, 1))
    do_probe = visits_per_cand > 1
    for _rnd in range(n_rounds):
        q_sums, visit_cnt = orch._halving_round_batched(
            state1, topk_global, cand_mask_h,
            q_sums, visit_cnt,
            legal_np, nlegal_np, occ_cpu, nocc_cpu,
            [move_from_prs_local],
            do_reply_probe=do_probe,
        )

        # Halve survivors
        num_cands = int(cand_mask_h.sum().item())
        if num_cands <= 1:
            break
        num_keep = max(1, num_cands // 2)

        max_n  = int(visit_cnt.max().item())
        sigma  = (C_VISIT + max_n) * C_SCALE
        q_mean = q_sums / visit_cnt.clamp(min=1).float()
        scores = perturbed_topk + sigma * q_mean
        scores[~cand_mask_h] = float("-inf")

        _, keep     = torch.topk(scores, num_keep, dim=1)
        new_mask    = torch.zeros_like(cand_mask_h)
        new_mask.scatter_(1, keep, True)
        cand_mask_h = new_mask

    # Final selection (temperature sampling, as per fixed orchestrator)
    max_n  = int(visit_cnt.max().item())
    sigma  = (C_VISIT + max_n) * C_SCALE
    q_mean = q_sums / visit_cnt.clamp(min=1).float()
    final_scores = perturbed_topk + sigma * q_mean
    final_scores[~cand_mask_h] = float("-inf")

    temp_probs  = torch.softmax(final_scores / TEMPERATURE, dim=-1)
    temp_probs  = temp_probs.nan_to_num(nan=1.0 / eff_k)
    sel_local   = int(torch.multinomial(temp_probs, 1).item())
    sel_global  = int(topk_global[0, sel_local].item())

    halv_selected_win = (sel_global == win_prs) and win_valid

    # Q values at final step for winner and win move (if in topk)
    q_at_sel  = float(q_mean[0, sel_local].item()) if visit_cnt[0, sel_local] > 0 else float("nan")
    win_in_topk_local = None
    q_at_win  = float("nan")
    if win_in_topk:
        win_in_topk_local = int((topk_global[0].cpu().numpy() == win_prs).argmax())
        q_at_win = float(q_mean[0, win_in_topk_local].item()) if visit_cnt[0, win_in_topk_local] > 0 else float("nan")

    action_desc = decode_action(win_prs) if win_valid else "INVALID"

    rows.append({
        "idx":              idx,
        "value":            value,
        "win_valid":        win_valid,
        "logit_rank":       logit_rank,
        "gum_rank":         gum_rank,
        "win_in_topk":      win_in_topk,
        "halv_sel_win":     halv_selected_win,
        "sel_global":       sel_global,
        "n_legal_prs":      n_legal_prs,
        "action_desc":      action_desc,
        "q_at_win":         q_at_win,
        "q_at_sel":         q_at_sel,
        "win_prs":          win_prs,
    })

    ok_topk  = "YES" if win_in_topk  else "NO "
    ok_halv  = "WIN" if halv_selected_win else f"#{sel_global}"
    lr_str   = f"{logit_rank}" if logit_rank > 0 else "n/a"
    gr_str   = f"{gum_rank}"  if gum_rank  > 0 else "n/a"

    print(f"{idx:>3}  {value:>+7.3f}  "
          f"{'YES' if win_valid else 'NO ':>5}  "
          f"{lr_str:>8}  {gr_str:>6}  {ok_topk:>7}  {ok_halv:>7}  "
          f"{n_legal_prs:>5}  {action_desc}")

# ── Summary ────────────────────────────────────────────────────────────────────

N = len(rows)
if N == 0:
    print("No data collected.")
    sys.exit(0)

n_valid   = sum(r["win_valid"]    for r in rows)
n_topk    = sum(r["win_in_topk"]  for r in rows)
n_halv    = sum(r["halv_sel_win"] for r in rows)
n_top1l   = sum(r["logit_rank"] == 1 for r in rows if r["logit_rank"] > 0)
n_top3l   = sum(r["logit_rank"] <= 3 for r in rows if r["logit_rank"] > 0)
n_top1g   = sum(r["gum_rank"]   == 1 for r in rows if r["gum_rank"]  > 0)
n_top3g   = sum(r["gum_rank"]   <= 3 for r in rows if r["gum_rank"]  > 0)
mean_val  = np.mean([r["value"] for r in rows])
q_wins    = [r["q_at_win"] for r in rows if not math.isnan(r["q_at_win"])]
q_sels    = [r["q_at_sel"] for r in rows if not math.isnan(r["q_at_sel"])]

print("\n" + "=" * 70)
print(f"=== Summary ({N} positions, K={K}, sims={NUM_SIMS}) ===")
print(f"  Mean value output (untrained, ideal +1) : {mean_val:+.4f}")
print(f"  Win move encodes to valid PRS action     : {n_valid}/{N}")
print(f"  Win move in Gumbel top-{K} candidates   : {n_topk}/{n_valid} (of valid)")
print(f"  Win move selected by halving             : {n_halv}/{n_topk} (of those in top-{K})")
print(f"  Win move has #1 raw logit                : {n_top1l}/{N}")
print(f"  Win move in top-3 raw logit              : {n_top3l}/{N}")
print(f"  Win move has #1 Gumbel score             : {n_top1g}/{N}")
print(f"  Win move in top-3 Gumbel score           : {n_top3g}/{N}")
if q_wins:
    print(f"  Mean Q(win move) when in top-K           : {np.mean(q_wins):+.4f}")
if q_sels:
    print(f"  Mean Q(selected move)                    : {np.mean(q_sels):+.4f}")
print()
print("Interpretation:")
print("  - 'Win in top-16': the key question — does the win move get evaluated?")
print("  - 'Halving selects win': with untrained net, Q≈0 so Gumbel noise decides.")
print("  - For a trained net: Q(win)→+1, Q(others)→0, so halving should reliably pick win.")
