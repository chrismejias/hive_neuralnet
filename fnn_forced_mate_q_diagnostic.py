from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch

import hive_gpu
from hive_engine.game_state import GameResult, GameState, Move, MoveType
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, Piece, PieceType
from hive_fnn.fnn_mcts_orchestrator import (
    FNNMCTSConfig,
    FNNMCTSOrchestrator,
    _GUMBEL_K,
    _GUMBEL_ROUNDS,
    _GUMBEL_WAVE_SCHEDULE,
)
from hive_fnn.fnn_network import HiveFNN
from hive_gpu.endgame_generator import gamestate_to_gpu_bytes, positions_to_tensor


GPU_TO_PT = {
    1: PieceType.QUEEN,
    2: PieceType.ANT,
    3: PieceType.GRASSHOPPER,
    4: PieceType.SPIDER,
    5: PieceType.BEETLE,
    6: PieceType.MOSQUITO,
    7: PieceType.LADYBUG,
    8: PieceType.PILLBUG,
}
PT_TO_GPU = {v: k for k, v in GPU_TO_PT.items()}
BOARD_SIZE = 23
HALF_BOARD = BOARD_SIZE // 2


@dataclass(frozen=True)
class MoveSig:
    move_type: int
    piece_type: int | None
    from_pos: tuple[int, int] | None
    to_pos: tuple[int, int] | None


@dataclass
class ForcedLine:
    black_forced: bool
    line: list[MoveSig]


def piece_label(piece: Piece | None) -> str:
    if piece is None:
        return "PASS"
    return piece.label


def move_sig(move: Move) -> MoveSig:
    if move.move_type == MoveType.PASS:
        return MoveSig(int(MoveType.PASS), None, None, None)
    if move.move_type == MoveType.PLACE:
        return MoveSig(
            int(MoveType.PLACE),
            int(move.piece.piece_type),
            None,
            (move.to.q, move.to.r),
        )
    return MoveSig(
        int(MoveType.MOVE),
        int(move.piece.piece_type),
        (move.from_pos.q, move.from_pos.r),
        (move.to.q, move.to.r),
    )


def format_sig(sig: MoveSig) -> str:
    if sig.move_type == int(MoveType.PASS):
        return "PASS"
    pt = PieceType(sig.piece_type).name if sig.piece_type is not None else "?"
    if sig.move_type == int(MoveType.PLACE):
        return f"PLACE {pt} -> {sig.to_pos}"
    return f"MOVE {pt} {sig.from_pos} -> {sig.to_pos}"


def load_net(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(ckpt["net_config"]).cuda().eval()
    state = ckpt.get("model_state_dict", ckpt.get("model_state"))
    if state is None:
        raise KeyError("checkpoint missing model state")
    net.load_state_dict(state, strict=True)
    return net


def remove_from_hand(gs: GameState, color: Color, pt: PieceType, piece_id: int) -> Piece:
    for piece in list(gs.hand(color)):
        if piece.piece_type == pt and piece.piece_id == piece_id:
            gs._hands[color].remove(piece)
            return piece
    raise KeyError(f"missing piece in hand: {color.name} {pt.name} {piece_id}")


def place(gs: GameState, color: Color, pt: PieceType, piece_id: int, q: int, r: int) -> Piece:
    piece = remove_from_hand(gs, color, pt, piece_id)
    gs.board.place_piece(piece, HexCoord(q, r))
    if pt == PieceType.QUEEN:
        gs._queen_placed[color] = True
    return piece


def build_artificial_base() -> GameState:
    gs = GameState()

    # White queen under a black beetle.
    place(gs, Color.WHITE, PieceType.QUEEN, 0, 0, 0)
    beetle = remove_from_hand(gs, Color.BLACK, PieceType.BEETLE, 0)
    gs.board.place_piece(beetle, HexCoord(0, 0))

    # Four surrounding black pieces around the white queen.
    place(gs, Color.BLACK, PieceType.SPIDER, 0, 1, 0)    # E
    place(gs, Color.BLACK, PieceType.GRASSHOPPER, 0, 1, -1)  # NE
    place(gs, Color.BLACK, PieceType.SPIDER, 1, -1, 0)   # W
    place(gs, Color.BLACK, PieceType.GRASSHOPPER, 1, -1, 1)  # SW

    # Two mobile black ants not already in the four surrounding pieces.
    place(gs, Color.BLACK, PieceType.ANT, 0, 2, 0)
    place(gs, Color.BLACK, PieceType.ANT, 1, -2, 1)

    # Black queen with three white neighbors; white beetles remain off-board.
    place(gs, Color.BLACK, PieceType.QUEEN, 0, 4, 0)
    place(gs, Color.WHITE, PieceType.ANT, 0, 3, 0)
    place(gs, Color.WHITE, PieceType.ANT, 1, 4, -1)
    place(gs, Color.WHITE, PieceType.GRASSHOPPER, 0, 5, -1)

    # A few more white pieces to make the position nontrivial without helping.
    place(gs, Color.WHITE, PieceType.SPIDER, 0, 5, 0)
    place(gs, Color.WHITE, PieceType.GRASSHOPPER, 1, 6, -1)
    place(gs, Color.BLACK, PieceType.ANT, 2, 2, -1)

    gs.turn = 14  # White to move.
    gs.result = GameResult.IN_PROGRESS
    gs._legal_moves_cache = None
    return gs


def state_key(gs: GameState) -> tuple:
    board_key = tuple(
        sorted(
            (
                pos.q,
                pos.r,
                tuple((int(p.piece_type), int(p.color), int(p.piece_id)) for p in stack),
            )
            for pos, stack in gs.board.grid.items()
        )
    )
    hand_key = (
        tuple(sorted((int(p.piece_type), int(p.piece_id)) for p in gs.hand(Color.WHITE))),
        tuple(sorted((int(p.piece_type), int(p.piece_id)) for p in gs.hand(Color.BLACK))),
    )
    return (board_key, hand_key, int(gs.turn), int(gs.result))


def forced_black_win(gs: GameState, depth: int) -> ForcedLine:
    @lru_cache(maxsize=None)
    def solve(key: tuple, to_move: int, rem: int) -> tuple[bool, tuple[MoveSig, ...]]:
        state = key_to_state[key]
        if state.result == GameResult.BLACK_WINS:
            return True, ()
        if state.result == GameResult.WHITE_WINS or state.result == GameResult.DRAW:
            return False, ()
        if rem == 0:
            return False, ()

        moves = state.legal_moves()
        if to_move == int(Color.BLACK):
            for mv in moves:
                nxt = state.copy()
                nxt.apply_move(mv)
                k2 = state_key(nxt)
                key_to_state.setdefault(k2, nxt)
                ok, line = solve(k2, int(nxt.current_player), rem - 1)
                if ok:
                    return True, (move_sig(mv),) + line
            return False, ()

        # White to move: black forced only if every white move still loses.
        best_line: tuple[MoveSig, ...] | None = None
        for mv in moves:
            nxt = state.copy()
            nxt.apply_move(mv)
            k2 = state_key(nxt)
            key_to_state.setdefault(k2, nxt)
            ok, line = solve(k2, int(nxt.current_player), rem - 1)
            if not ok:
                return False, ()
            if best_line is None:
                best_line = (move_sig(mv),) + line
        return True, best_line or ()

    root_key = state_key(gs)
    key_to_state: dict[tuple, GameState] = {root_key: gs.copy()}
    ok, line = solve(root_key, int(gs.current_player), depth)
    return ForcedLine(ok, list(line))


def tensor_from_state(gs: GameState) -> torch.Tensor:
    return positions_to_tensor([gamestate_to_gpu_bytes(gs)], device="cuda")


def decode_gpu_move(raw: np.ndarray) -> MoveSig:
    move_type = int(raw[0])
    piece_type_gpu = int(raw[1]) if move_type != int(MoveType.PASS) else None
    piece_type = int(GPU_TO_PT[piece_type_gpu]) if piece_type_gpu is not None else None
    from_cell = int(raw[2]) | (int(raw[3]) << 8)
    to_cell = int(raw[4]) | (int(raw[5]) << 8)

    def cell_to_qr(cell: int) -> tuple[int, int]:
        row, col = divmod(cell, BOARD_SIZE)
        return (col - HALF_BOARD, row - HALF_BOARD)

    if move_type == int(MoveType.PASS):
        return MoveSig(move_type, None, None, None)
    if move_type == int(MoveType.PLACE):
        return MoveSig(move_type, piece_type, None, cell_to_qr(to_cell))
    return MoveSig(move_type, piece_type, cell_to_qr(from_cell), cell_to_qr(to_cell))


def gather_root_search(
    orch: FNNMCTSOrchestrator,
    gs: GameState,
    sims: int,
    max_considered: int = 16,
    use_gumbel: bool = True,
) -> dict:
    states = tensor_from_state(gs)
    tree = orch._alloc_tree(1)
    orch._reset_tree(tree)

    legal_moves_t, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, 1)
    n = int(num_legal[0].item())
    legal_cpu = gs.legal_moves()
    priors, root_values, child_q_init = orch._eval_states(states, legal_moves_t, num_legal, 1, root_features)

    valid = torch.arange(orch._max_legal, device="cuda").unsqueeze(0) < num_legal.to(torch.int64).unsqueeze(1)
    safe_prior = priors.clamp_min(1e-30)
    legal_logits = torch.where(valid, safe_prior.log(), torch.full_like(priors, -1e30))
    gumbel = torch.zeros_like(legal_logits)
    if use_gumbel:
        perturbed = gumbel + legal_logits
    else:
        perturbed = legal_logits
    k = min(max_considered, n, orch._max_legal)
    _, topk_slots = torch.topk(perturbed, k, dim=1)

    game_active_t = torch.tensor([1], dtype=torch.int8, device="cuda")
    orch._expand_root_if_needed(tree, states, legal_moves_t, num_legal, priors, child_q_init, game_active_t, 1)
    candidate_slots = topk_slots.to(torch.int32)
    candidate_valid = torch.gather(valid, 1, candidate_slots.long())
    candidate_slots = torch.where(candidate_valid, candidate_slots, torch.full_like(candidate_slots, -1))

    rounds = _GUMBEL_ROUNDS if use_gumbel else 1
    sims_per_round = max(1, sims // rounds)
    for round_i in range(rounds):
        num_candidates = int(candidate_slots.shape[1])
        sims_per_candidate = max(1, sims_per_round // max(num_candidates, 1))
        round_wave_size = _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)] if use_gumbel else 1
        orch._run_simulations_for_root_slots(
            tree, states, game_active_t, candidate_slots, 1, sims_per_candidate,
            wave_size=round_wave_size,
        )
        if num_candidates <= 1 or not use_gumbel:
            continue
        candidate_valid = candidate_slots >= 0
        per_game_keep = (candidate_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep = num_candidates // 2
        cand_visits, cand_q = orch._gather_root_candidate_stats(tree, 1, candidate_slots)
        sigma_norm = (orch.config.c_visit + cand_visits.max()) * orch.config.c_scale
        cand_idx = candidate_slots.long().clamp(min=0)
        cand_score = (torch.gather(gumbel + legal_logits, 1, cand_idx) + sigma_norm * cand_q).float()
        cand_score = torch.where(candidate_valid, cand_score, torch.full_like(cand_score, -1e30))
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank = orch._keep_rank(max_keep)
        keep_valid = keep_rank < per_game_keep.unsqueeze(1)
        new_slots = torch.gather(candidate_slots, 1, keep_pos)
        candidate_slots = torch.where(keep_valid, new_slots, torch.full_like(new_slots, -1))

    slot_visits, slot_q = orch._gather_root_child_stats(tree, 1)
    max_n = slot_visits.float().max(dim=1, keepdim=True).values
    sigma_norm = (orch.config.c_visit + max_n) * orch.config.c_scale
    visited = (slot_visits > 0) & valid
    sampled_logits = torch.where(
        visited,
        legal_logits + sigma_norm * slot_q,
        torch.full_like(legal_logits, -1e30),
    )
    search_policy = torch.softmax(sampled_logits, dim=1)
    search_policy = torch.where(valid, search_policy, torch.zeros_like(search_policy))
    expected_score = float((search_policy * slot_q).sum().item())
    expected_winrate = 0.5 * (expected_score + 1.0)
    best_move_q = float(slot_q[0, :n].max().item()) if n > 0 else 0.0
    best_move_winrate = 0.5 * (best_move_q + 1.0)

    visits_np = slot_visits[0, :n].cpu().numpy().astype(int)
    q_np = slot_q[0, :n].cpu().numpy()
    prior_np = priors[0, :n].cpu().numpy()
    init_q_np = child_q_init[0, :n].cpu().numpy()
    search_pi_np = search_policy[0, :n].cpu().numpy()
    legal_gpu_np = legal_moves_t[0, :n].cpu().numpy()

    decoded = [decode_gpu_move(raw) for raw in legal_gpu_np]
    by_sig = {move_sig(mv): i for i, mv in enumerate(legal_cpu)}
    mapped = []
    for i, sig in enumerate(decoded):
        cpu_idx = by_sig.get(sig, None)
        mapped.append(
            {
                "gpu_idx": i,
                "cpu_idx": cpu_idx,
                "sig": sig,
                "prior": float(prior_np[i]),
                "init_q": float(init_q_np[i]),
                "q": float(q_np[i]),
                "search_pi": float(search_pi_np[i]),
                "visits": int(visits_np[i]),
            }
        )
    mapped.sort(key=lambda x: (-x["visits"], -x["q"], -x["prior"]))
    prior_rank = sorted(range(len(mapped)), key=lambda i: -mapped[i]["prior"])
    prior_rank_by_sig = {mapped[i]["sig"]: r + 1 for r, i in enumerate(prior_rank)}
    considered_set = {decoded[i] for i in topk_slots[0, :k].cpu().numpy().tolist()}
    return {
        "root_value": float(root_values[0].item()),
        "search_expected_score": expected_score,
        "search_expected_winrate": expected_winrate,
        "best_move_q": best_move_q,
        "best_move_winrate": best_move_winrate,
        "moves": mapped,
        "considered": considered_set,
        "prior_rank_by_sig": prior_rank_by_sig,
    }


def apply_sig(gs: GameState, sig: MoveSig) -> GameState:
    for mv in gs.legal_moves():
        if move_sig(mv) == sig:
            nxt = gs.copy()
            nxt.apply_move(mv)
            return nxt
    raise KeyError(f"move not legal in state: {sig}")


def summarize_position(
    label: str,
    gs: GameState,
    forced: ForcedLine,
    searches: list[tuple[str, dict]],
) -> None:
    print(f"\n=== {label} ===")
    print(f"turn={gs.turn} to_move={gs.current_player.name} result={gs.result.name}")
    print(f"forced_black_win_within_4={forced.black_forced}")
    if forced.line:
        print("forced_line:")
        for i, sig in enumerate(forced.line, start=1):
            print(f"  {i}. {format_sig(sig)}")

    for variant, search in searches:
        print(f"\n[{variant}] root_raw_value={search['root_value']:.4f}")
        print(
            "search_summary: "
            f"expected_score={search['search_expected_score']:+.4f} "
            f"expected_winrate={search['search_expected_winrate']:.4f} "
            f"best_move_q={search['best_move_q']:+.4f} "
            f"best_move_winrate={search['best_move_winrate']:.4f}"
        )
        top = search["moves"][:8]
        print("top_root_moves:")
        for row in top:
            print(
                f"  visits={row['visits']:5d} q={row['q']:+.4f} init_q={row['init_q']:+.4f} "
                f"pi_search={row['search_pi']:.4f} prior={row['prior']:.4f} "
                f"move={format_sig(row['sig'])}"
            )
        if forced.line:
            first = forced.line[0]
            forced_row = next((row for row in search["moves"] if row["sig"] == first), None)
            if forced_row is not None:
                prior_rank = search["prior_rank_by_sig"].get(first, -1)
                print(
                    "forced_first_move_search_stats: "
                    f"prior_rank={prior_rank} considered={first in search['considered']} "
                    f"visits={forced_row['visits']} q={forced_row['q']:+.4f} "
                    f"init_q={forced_row['init_q']:+.4f} prior={forced_row['prior']:.4f}"
                )
            else:
                print("forced_first_move_search_stats: move not found among legal root moves")


def apply_line(gs: GameState, line: Iterable[MoveSig], plies: int) -> GameState:
    cur = gs.copy()
    for i, sig in enumerate(line):
        if i >= plies:
            break
        cur = apply_sig(cur, sig)
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--simulations", type=int, default=24576)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--c-scale", type=float, default=1.0)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    orch = FNNMCTSOrchestrator(
        net,
        FNNMCTSConfig(
            num_simulations=args.simulations,
            max_num_considered_actions=16,
            c_puct=args.c_puct,
            c_scale=args.c_scale,
            batch_size=1,
            expansion_mask=0,
            wave_parallel=True,
            dirichlet_epsilon=0.0,
        ),
    )

    base = build_artificial_base()
    forced0 = forced_black_win(base, 4)
    variants = [
        ("gumbel16", dict(max_considered=16, use_gumbel=True)),
        ("gumbel64", dict(max_considered=64, use_gumbel=True)),
        ("fullroot_prior", dict(max_considered=orch._max_legal, use_gumbel=False)),
    ]

    searches0 = [(name, gather_root_search(orch, base, args.simulations, **cfg)) for name, cfg in variants]
    summarize_position("base_white_to_move", base, forced0, searches0)

    if forced0.line:
        state1 = apply_sig(base, forced0.line[0])
        forced1 = forced_black_win(state1, 3)
        searches1 = [(name, gather_root_search(orch, state1, args.simulations, **cfg)) for name, cfg in variants]
        summarize_position("after_forced_first_move", state1, forced1, searches1)

    # Additional short-horizon probes derived from the same tactical motif.
    # 1. Black to move immediately in the same shell.
    black_to_move = base.copy()
    black_to_move.turn += 1
    black_to_move._legal_moves_cache = None
    forced2 = forced_black_win(black_to_move, 3)
    searches2 = [(name, gather_root_search(orch, black_to_move, args.simulations, **cfg)) for name, cfg in variants]
    summarize_position("same_shell_black_to_move", black_to_move, forced2, searches2)

    # 2. Later positions derived from the actual forced line, if the shell is really forced.
    if forced0.black_forced and len(forced0.line) >= 3:
        late_white = apply_line(base, forced0.line, len(forced0.line) - 1)
        late_white._legal_moves_cache = None
        forced3 = forced_black_win(late_white, 1)
        searches3 = [(name, gather_root_search(orch, late_white, args.simulations, **cfg)) for name, cfg in variants]
        summarize_position("one_ply_before_terminal_move", late_white, forced3, searches3)

    if forced2.black_forced and len(forced2.line) >= 2:
        mate_in_one = apply_line(black_to_move, forced2.line, len(forced2.line) - 1)
        mate_in_one._legal_moves_cache = None
        forced4 = forced_black_win(mate_in_one, 1)
        searches4 = [(name, gather_root_search(orch, mate_in_one, args.simulations, **cfg)) for name, cfg in variants]
        summarize_position("black_to_move_mate_in_one", mate_in_one, forced4, searches4)


if __name__ == "__main__":
    main()
