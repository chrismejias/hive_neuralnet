/**
 * fnn_features.cuh — CUDA kernel for HiveGo-style FNN feature extraction.
 *
 * Produces a fixed-size feature vector directly from HiveState,
 * bypassing the full encode_states_batch pipeline (no per-node features,
 * no edges, no graph construction).
 *
 * Feature layout (FNN_FEAT_DIM = 140):
 *   [0:16]   count_on_board    — visible top pieces per type(8) × color(2)
 *   [16:32]  count_in_hand     — hand piece counts per type(8) × color(2)
 *   [32:48]  queen_neighbors   — top pieces adjacent to opponent queen, per type(8) × color(2)
 *   [48:64]  avg_dist_to_opp_q — avg hex distance to opponent queen, per type(8) × color(2)
 *   [64:80]  can_move_count    — number of distinct pieces with ≥1 legal MOVE attributable to owner, per type(8) × color(2)
 *   [80:96]  articulation_cnt  — number of ground-level articulation-point top pieces, per type(8) × color(2)
 *   [96:98]  num_single        — board pieces with 0 occupied neighbors, per color(2)
 *   [98:100] queen_covered     — queen not on top, per color(2)
 *   [100:102] num_placement_pos — unique placement destinations from legal moves, per color(2)
 *   [102]    moves_to_draw     — normalized turn count
 *   [103]    move_number       — turn / 100
 *   [104:106] pillbug_capable  — owner has an uncovered pillbug OR ground mosquito
 *                                adjacent to a usable pillbug this turn, per color(2)
 *   [106:108] throwable_own    — own-color pieces adjacent to own pillbug-capable cell, per color(2)
 *   [108:110] throwable_opp    — own-color pieces adjacent to opposing pillbug-capable cell (threatened), per color(2)
 *   [110:116] white_q_surround — one-hot surround count buckets 1..6 for white queen
 *   [116:122] black_q_surround — one-hot surround count buckets 1..6 for black queen
 *   [122:128] sufficient_material — insufficient/exact/surplus free attackers,
 *                                   per color(2) × bucket(3)
 *   [128:134] queen_escape     — no relief/exactly 1/at least 2 fewer occupied
 *                                queen neighbors, per color(2) × bucket(3)
 *   [134:140] queen_ring_access — zero/one/multiple distinct free pieces that
 *                                 can enter an empty opposing queen-ring cell,
 *                                 per color(2) × bucket(3)
 *
 * Must be included from game_logic.cu (needs NEIGHBORS constant memory).
 */

#pragma once

#include "hex_grid.cuh"
#include "hive_state.cuh"
#include "articulation.cuh"
#include "move_gen.cuh"

namespace hive_gpu {

constexpr int FNN_FEAT_DIM = 140;
// Draw is at move 200 in standard Hive
constexpr int DRAW_MOVE_LIMIT = 200;

#ifdef __CUDACC__

__device__ __forceinline__ int hex_distance(int cell_a, int cell_b) {
    // Axial coordinates
    int q_a = cell_a % BOARD_SIZE - HALF_BOARD;
    int r_a = cell_a / BOARD_SIZE - HALF_BOARD;
    int q_b = cell_b % BOARD_SIZE - HALF_BOARD;
    int r_b = cell_b / BOARD_SIZE - HALF_BOARD;
    int dq = q_b - q_a;
    int dr = r_b - r_a;
    // Cube distance: max(|dq|, |dr|, |dq+dr|)
    int ds = -(dq + dr);
    int adq = dq < 0 ? -dq : dq;
    int adr = dr < 0 ? -dr : dr;
    int ads = ds < 0 ? -ds : ds;
    int mx = adq;
    if (adr > mx) mx = adr;
    if (ads > mx) mx = ads;
    return mx;
}

__device__ __forceinline__ bool feature_adjacent_to(
    int cell, uint16_t target
) {
    if (cell < 0 || cell >= NUM_CELLS || target == 0xFFFF) return false;
    for (int d = 0; d < NUM_DIRS; ++d) {
        if (NEIGHBORS[cell][d] == (int16_t)target) return true;
    }
    return false;
}

__device__ __forceinline__ int feature_queen_relief(
    const HiveState& s, Color color, int source, int destination,
    PieceType moving_type, int baseline_surround
) {
    uint16_t queen = s.queen_cell[(int)color];
    if (queen == 0xFFFF || baseline_surround <= 0) return 0;

    if (source == (int)queen && moving_type == PT_QUEEN) {
        int next_surround = 0;
        for (int d = 0; d < NUM_DIRS; ++d) {
            int16_t neighbor = NEIGHBORS[destination][d];
            if (neighbor < 0 || neighbor == source) continue;
            if (s.occupied.get(neighbor)) ++next_surround;
        }
        return max(0, baseline_surround - next_surround);
    }

    int relief = 0;
    if (s.height[source] == 1 && feature_adjacent_to(source, queen)) {
        ++relief;
    }
    if (!s.occupied.get(destination) &&
        feature_adjacent_to(destination, queen)) {
        --relief;
    }
    return max(0, relief);
}

__device__ inline int feature_piece_destinations(
    const HiveState& s, int cell, PieceType type,
    Bitboard& base_perimeter, bool& base_perimeter_ready,
    uint16_t* destinations
) {
    switch (type) {
        case PT_QUEEN:
            return gen_queen_moves(s, cell, destinations);
        case PT_ANT:
            if (!base_perimeter_ready) {
                build_empty_perimeter_mask(s.occupied, base_perimeter);
                base_perimeter_ready = true;
            }
            return gen_ant_moves_with_perimeter(
                s, cell, base_perimeter, destinations, MAX_ANT_DESTS);
        case PT_GRASSHOPPER:
            return gen_grasshopper_moves(s, cell, destinations);
        case PT_SPIDER:
            return gen_spider_moves(s, cell, destinations);
        case PT_BEETLE:
            return gen_beetle_moves(s, cell, destinations);
        case PT_MOSQUITO:
            return gen_mosquito_moves(
                s, cell, destinations, &base_perimeter,
                &base_perimeter_ready);
        case PT_LADYBUG:
            return gen_ladybug_moves(s, cell, destinations);
        case PT_PILLBUG:
            return gen_pillbug_moves(s, cell, destinations);
        default:
            return 0;
    }
}

__device__ inline bool feature_piece_has_move(
    const HiveState& s, int cell, PieceType type,
    Bitboard& base_perimeter, bool& base_perimeter_ready
) {
    switch (type) {
        case PT_QUEEN:
            return has_queen_move(s, cell);
        case PT_ANT:
            if (!base_perimeter_ready) {
                build_empty_perimeter_mask(s.occupied, base_perimeter);
                base_perimeter_ready = true;
            }
            return has_ant_move_with_perimeter(
                s, cell, base_perimeter);
        case PT_GRASSHOPPER:
            return has_grasshopper_move(s, cell);
        case PT_SPIDER:
            return has_spider_move(s, cell);
        case PT_BEETLE:
            return has_beetle_move(s, cell);
        case PT_MOSQUITO:
            return has_mosquito_move(
                s, cell, &base_perimeter, &base_perimeter_ready);
        case PT_LADYBUG:
            return has_ladybug_move(s, cell);
        case PT_PILLBUG:
            return has_pillbug_move(s, cell);
        default:
            return false;
    }
}

__device__ inline void feature_ant_tactical_reach(
    const HiveState& s, int source, const Bitboard& perimeter,
    bool need_ring_access, uint16_t enemy_queen,
    bool need_relief, uint16_t own_queen,
    bool& has_ring_access, int& max_relief
) {
    Bitboard occupied = s.occupied;
    occupied.clr(source);
    Bitboard visited;
    visited.clear();
    visited.set(source);
    uint16_t queue[MAX_ANT_DESTS];
    int read = 0;
    int written = 0;

    for (int d = 0; d < NUM_DIRS && written < MAX_ANT_DESTS; ++d) {
        if (!can_slide_ant_occ(
                occupied, perimeter, source, d)) {
            continue;
        }
        int16_t destination = SLIDE_FLANKS[source][d][0];
        if (!visited.get(destination)) {
            visited.set(destination);
            queue[written++] = (uint16_t)destination;
        }
    }

    while (read < written) {
        int destination = (int)queue[read++];
        if (need_ring_access &&
            feature_adjacent_to(destination, enemy_queen)) {
            has_ring_access = true;
        }
        if (need_relief &&
            !feature_adjacent_to(destination, own_queen)) {
            max_relief = max(max_relief, 1);
        }
        if ((!need_ring_access || has_ring_access) &&
            (!need_relief || max_relief >= 1)) {
            return;
        }

        for (int d = 0; d < NUM_DIRS && written < MAX_ANT_DESTS; ++d) {
            if (!can_slide_ant_occ(
                    occupied, perimeter, destination, d)) {
                continue;
            }
            int16_t next = SLIDE_FLANKS[destination][d][0];
            if (!visited.get(next)) {
                visited.set(next);
                queue[written++] = (uint16_t)next;
            }
        }
    }
}

/**
 * Device function: extract FNN features for a single game state.
 *
 * Can be called from any kernel (selfplay, batch feature extraction, etc.).
 */
__device__ inline void extract_fnn_features_with_ap_device(
    const HiveState& s,
    const GPUMove* my_moves,   // legal moves for this state
    int n_legal,               // number of legal moves
    const Bitboard& ap_mask,
    float* f                   // [FNN_FEAT_DIM] output
) {
    // Zero output
    for (int i = 0; i < FNN_FEAT_DIM; i++) f[i] = 0.0f;

    // ── count_on_board [0:16] + queen_neighbors [32:48] +
    //    avg_dist_to_opp_q [48:64] + articulation_cnt [80:96] +
    //    num_single [96:98] + queen_covered [98:100] ─────────

    // Accumulators for avg distance
    float dist_sum[2][NUM_PIECE_TYPES];  // [color][type]
    int dist_count[2][NUM_PIECE_TYPES];
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            dist_sum[c][t] = 0.0f;
            dist_count[c][t] = 0;
        }
    }

    uint16_t opp_queen[2];  // opponent queen cell for each color
    opp_queen[0] = s.queen_cell[1];  // white's opponent is black's queen
    opp_queen[1] = s.queen_cell[0];  // black's opponent is white's queen
    int queen_surround_counts[2] = {0, 0};
    for (int c = 0; c < 2; ++c) {
        uint16_t queen = s.queen_cell[c];
        if (queen != 0xFFFF) {
            queen_surround_counts[c] =
                num_occupied_neighbors(s, (int)queen);
        }
    }

    constexpr int PB_CELLS_PER_COLOR = 8;   // generous; real games see <= 2
    uint16_t pb_cells[2][PB_CELLS_PER_COLOR];
    int      pb_count[2] = {0, 0};
    Bitboard sufficient_sources[2];
    Bitboard ring_access_sources[2];
    sufficient_sources[0].clear();
    sufficient_sources[1].clear();
    ring_access_sources[0].clear();
    ring_access_sources[1].clear();
    int max_queen_relief[2] = {0, 0};
    Bitboard feature_perimeter;
    bool feature_perimeter_ready = false;

    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t occupied = s.occupied.w[wi];
        while (occupied) {
            int bit = __ffsll(occupied) - 1;
            int cell = wi * 64 + bit;
            occupied &= occupied - 1;
            if (cell >= NUM_CELLS) continue;
            int h = s.height[cell];

            // Top piece
            uint8_t top = s.pieces[h - 1][cell];
            PieceType pt = cell_piece_type(top);
            Color pc = cell_color(top);
            int type_idx = (int)pt - 1;  // 0-indexed
            int tc_idx = type_idx * 2 + (int)pc;  // type×color index

        // count_on_board
        f[tc_idx] += 1.0f;

        // Count occupied neighbors for this cell
        int occ_nb = 0;
        for (int d = 0; d < NUM_DIRS; d++) {
            int16_t nb = NEIGHBORS[cell][d];
            if (nb >= 0 && s.height[nb] > 0) occ_nb++;
        }

        // num_single: top pieces with 0 occupied neighbors
        if (occ_nb == 0) {
            f[96 + (int)pc] += 1.0f;
        }

        // queen_neighbors: is this cell adjacent to opponent's queen?
        uint16_t opp_q = opp_queen[(int)pc];
        if (opp_q != 0xFFFF) {
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[cell][d];
                if (nb >= 0 && (uint16_t)nb == opp_q) {
                    f[32 + tc_idx] += 1.0f;
                    break;
                }
            }
        }

        // avg_dist_to_opp_q: distance to opponent's queen
        if (opp_q != 0xFFFF) {
            int d = hex_distance(cell, (int)opp_q);
            dist_sum[(int)pc][type_idx] += (float)d;
            dist_count[(int)pc][type_idx] += 1;
        }

        // articulation_cnt: only count top pieces on ground-level AP cells.
        // Stacked pieces (e.g. elevated beetles) are intentionally excluded.
        if (h == 1 && ap_mask.get(cell)) {
            f[80 + tc_idx] += 1.0f;
        }

        // queen_covered: queen exists but not on top
        if (pt != PT_QUEEN) {
            // Check all levels below top for a queen at this cell
            for (int lv = 0; lv < h - 1; lv++) {
                uint8_t below = s.pieces[lv][cell];
                if (cell_piece_type(below) == PT_QUEEN) {
                    Color qc = cell_color(below);
                    f[98 + (int)qc] = 1.0f;
                }
            }
        }

        // pillbug_capable [104:106]
        bool is_capable = (s.stunned_cell != (uint16_t)cell) && (pt == PT_PILLBUG);
        if (!is_capable && s.stunned_cell != (uint16_t)cell && pt == PT_MOSQUITO && h == 1) {
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[cell][d];
                if (nb < 0 || s.height[nb] == 0) continue;
                if (s.stunned_cell == (uint16_t)nb) continue;
                if (top_piece_type_at(s, nb) == PT_PILLBUG) {
                    is_capable = true;
                    break;
                }
            }
        }
            if (is_capable) {
                f[104 + (int)pc] = 1.0f;
                if (pb_count[(int)pc] < PB_CELLS_PER_COLOR) {
                    pb_cells[(int)pc][pb_count[(int)pc]++] = (uint16_t)cell;
                }
            }

            // One pass over each visible piece supplies both players' exact
            // normal-move mobility, immediate ring access, and queen relief.
            if (s.result == IN_PROGRESS && is_queen_placed(s, pc) &&
                !is_stunned_cell(s, cell) &&
                !is_pinned(s, ap_mask, cell)) {
                uint16_t enemy_queen = opp_queen[(int)pc];
                bool free_attacker =
                    enemy_queen != 0xFFFF &&
                    !feature_adjacent_to(cell, enemy_queen);
                bool has_normal_move = feature_piece_has_move(
                    s, cell, pt, feature_perimeter,
                    feature_perimeter_ready);
                if (has_normal_move && free_attacker) {
                    sufficient_sources[(int)pc].set(cell);
                }

                uint16_t own_queen = s.queen_cell[(int)pc];
                bool can_change_own_ring =
                    (cell == (int)own_queen && pt == PT_QUEEN) ||
                    (s.height[cell] == 1 &&
                     feature_adjacent_to(cell, own_queen));
                if (has_normal_move &&
                    (free_attacker || can_change_own_ring)) {
                    if (pt == PT_ANT) {
                        bool ant_ring_access = false;
                        feature_ant_tactical_reach(
                            s, cell, feature_perimeter,
                            free_attacker, enemy_queen,
                            can_change_own_ring, own_queen,
                            ant_ring_access,
                            max_queen_relief[(int)pc]);
                        if (ant_ring_access) {
                            ring_access_sources[(int)pc].set(cell);
                        }
                        continue;
                    }
                    uint16_t destinations[MAX_ANT_DESTS];
                    int n_destinations = feature_piece_destinations(
                        s, cell, pt, feature_perimeter,
                        feature_perimeter_ready, destinations);
                    for (int di = 0; di < n_destinations; ++di) {
                        int destination = (int)destinations[di];
                        if (free_attacker &&
                            !s.occupied.get(destination) &&
                            feature_adjacent_to(
                                destination, enemy_queen)) {
                            ring_access_sources[(int)pc].set(cell);
                        }
                        if (can_change_own_ring) {
                            int relief = feature_queen_relief(
                                s, pc, cell, destination, pt,
                                queen_surround_counts[(int)pc]);
                            if (relief > max_queen_relief[(int)pc]) {
                                max_queen_relief[(int)pc] = relief;
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalize avg_dist_to_opp_q [48:64]
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            int tc_idx = t * 2 + c;
            if (dist_count[c][t] > 0) {
                f[48 + tc_idx] = dist_sum[c][t] / (float)dist_count[c][t] / 10.0f;
            }
        }
    }

    // ── count_in_hand [16:32] ───────────────────────────────
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            int tc_idx = t * 2 + c;
            float max_count = (float)pieces_per_type(t);
            f[16 + tc_idx] = (float)s.hands[c][t] / (max_count > 0.0f ? max_count : 1.0f);
        }
    }

    // ── Features from legal moves: can_move_count [64:80], num_placement_pos [100:102] ──

    // One source can have many destinations, but contributes only once to its
    // piece-type/color mobility bucket.
    Bitboard movable_sources;
    movable_sources.clear();
    Bitboard seen_place_dst[2];
    seen_place_dst[0].clear();
    seen_place_dst[1].clear();

    Color cur = current_player(s);

    for (int m = 0; m < n_legal; m++) {
        const GPUMove& mv = my_moves[m];

        if (mv.type == MOVE_MOVE) {
            if (mv.from_cell < NUM_CELLS && s.height[mv.from_cell] > 0) {
                movable_sources.set(mv.from_cell);
            }
        } else if (mv.type == MOVE_PLACE) {
            if (mv.to_cell < NUM_CELLS) {
                seen_place_dst[(int)cur].set(mv.to_cell);
            }
        }
    }

    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t sources = movable_sources.w[wi];
        while (sources) {
            int bit = __ffsll(sources) - 1;
            int cell = wi * 64 + bit;
            sources &= sources - 1;
            if (cell >= NUM_CELLS || s.height[cell] == 0) continue;
            uint8_t packed = s.pieces[s.height[cell] - 1][cell];
            int bucket = ((int)cell_piece_type(packed) - 1) * 2 +
                (int)cell_color(packed);
            if (bucket >= 0 && bucket < 16) f[64 + bucket] += 1.0f;
        }
    }

    // Write num_placement_pos (normalized by ~10 typical positions)
    for (int c = 0; c < 2; c++) {
        f[100 + c] = (float)seen_place_dst[c].popcount() / 10.0f;
    }

    // ── moves_to_draw [102] ───────────────────────────────────
    int moves_left = DRAW_MOVE_LIMIT - (int)s.turn;
    if (moves_left < 0) moves_left = 0;
    f[102] = (float)moves_left / (float)DRAW_MOVE_LIMIT;

    // ── move_number [103] ────────────────────────────────────
    float t = (float)s.turn / 100.0f;
    f[103] = t < 1.0f ? t : 1.0f;

    // ── pillbug_capable [104:106], throwable_own [106:108], throwable_opp [108:110] ──
    //
    // A cell is "pillbug-capable" for color c if:
    //   (a) c owns a pillbug on top of its stack, OR
    //   (b) c owns a ground-level mosquito adjacent to ANY pillbug (friendly
    //       or enemy) — per Hive rules, mosquitoes copy any adjacent piece's
    //       ability, so such a mosquito inherits the pillbug's throw ability
    //       for its owner's turn.
    //
    // Gate/pin legality is intentionally NOT enforced here — this is a
    // structural board feature. The `can_move` block already captures the
    // turn-specific legality for the current player. Stunned pillbug cells
    // are excluded because they are not usable this turn.
    //
    // throwable_own[c] = count of c-color top pieces adjacent to c's own
    //                    pillbug-capable cells (repositioning material).
    // throwable_opp[c] = count of c-color top pieces adjacent to (1−c)'s
    //                    pillbug-capable cells (threatened pieces).
    //
    // Pieces adjacent to both sides' pillbugs get counted in both buckets,
    // which is the intended signal (double pressure).
    for (int c = 0; c < 2; c++) {
        for (int pi = 0; pi < pb_count[c]; pi++) {
            int pbc = (int)pb_cells[c][pi];
            uint16_t enemy_queen = opp_queen[c];
            int actor_distance =
                enemy_queen == 0xFFFF ? 99 :
                hex_distance(pbc, (int)enemy_queen);
            bool actor_near_enemy_queen =
                actor_distance >= 1 && actor_distance <= 2;
            int pillbug_height = s.height[pbc];
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[pbc][d];
                if (nb < 0 || s.height[nb] == 0) continue;
                if (s.stunned_cell == (uint16_t)nb) continue;
                uint8_t ntop = s.pieces[s.height[nb] - 1][nb];
                Color nc = cell_color(ntop);
                if ((int)nc == c) {
                    // c's own piece adjacent to c's pillbug — c could reposition it
                    f[106 + c] += 1.0f;
                } else {
                    // Piece owned by nc adjacent to c's pillbug — nc's piece is
                    // under threat from c's pillbug.
                    f[108 + (int)nc] += 1.0f;
                }

                // Merge exact throw-derived features into this existing
                // pillbug-neighbor pass. Throws are legal only after the
                // actor's queen has been placed.
                if (s.result != IN_PROGRESS ||
                    !is_queen_placed(s, (Color)c) ||
                    s.height[nb] != 1 ||
                    is_pinned(s, ap_mask, nb)) {
                    continue;
                }
                int lift_height = max(s.height[nb] - 1, pillbug_height);
                int lift_direction = find_direction(nb, pbc);
                if (lift_direction < 0 ||
                    elevated_gate_blocked(
                        s, nb, lift_direction, lift_height)) {
                    continue;
                }

                PieceType target_type = cell_piece_type(ntop);
                bool free_owned_target =
                    (int)nc == c && enemy_queen != 0xFFFF &&
                    !feature_adjacent_to(nb, enemy_queen);
                bool has_legal_throw = false;
                for (int dd = 0; dd < NUM_DIRS; ++dd) {
                    int16_t destination = NEIGHBORS[pbc][dd];
                    if (destination < 0 || destination == nb ||
                        s.occupied.get(destination)) {
                        continue;
                    }
                    int drop_height = max(pillbug_height, 0);
                    if (elevated_gate_blocked(
                            s, pbc, dd, drop_height)) {
                        continue;
                    }
                    has_legal_throw = true;
                    if (free_owned_target &&
                        feature_adjacent_to(destination, enemy_queen)) {
                        ring_access_sources[c].set(nb);
                    }
                    int relief = feature_queen_relief(
                        s, (Color)c, nb, destination, target_type,
                        queen_surround_counts[c]);
                    if (relief > max_queen_relief[c]) {
                        max_queen_relief[c] = relief;
                    }
                }
                if (has_legal_throw && free_owned_target &&
                    actor_near_enemy_queen) {
                    sufficient_sources[c].set(nb);
                }
            }
        }
    }

    // ── queen surround one-hot buckets [110:122] ────────────────
    for (int qc = 0; qc < 2; qc++) {
        int surround = queen_surround_counts[qc];
        if (surround >= 1 && surround <= 6) {
            f[110 + qc * 6 + (surround - 1)] = 1.0f;
        }
    }

    // ── sufficient material [122:128] ───────────────────────────
    // A player's free material excludes pieces already on the opposing
    // queen's ring. Normal movers and nearby legal pillbug throws share one
    // source bitboard, so a piece can never be counted twice.
    for (int c = 0; c < 2; ++c) {
        Color target = (Color)(1 - c);
        if (!is_queen_placed(s, target)) continue;
        int vacancies = 6 - queen_surround_counts[(int)target];
        int material = sufficient_sources[c].popcount();
        int bucket = material < vacancies ? 0 :
            (material == vacancies ? 1 : 2);
        f[122 + c * 3 + bucket] = 1.0f;
    }

    // ── available queen escape [128:134] ────────────────────────
    for (int c = 0; c < 2; ++c) {
        if (!is_queen_placed(s, (Color)c)) continue;
        int bucket = max_queen_relief[c] <= 0 ? 0 :
            (max_queen_relief[c] == 1 ? 1 : 2);
        f[128 + c * 3 + bucket] = 1.0f;
    }

    // ── immediate queen-ring access [134:140] ───────────────────
    for (int c = 0; c < 2; ++c) {
        Color target = (Color)(1 - c);
        if (!is_queen_placed(s, target)) continue;
        int access = ring_access_sources[c].popcount();
        int bucket = access == 0 ? 0 : (access == 1 ? 1 : 2);
        f[134 + c * 3 + bucket] = 1.0f;
    }
}

__device__ inline void extract_fnn_features_device(
    const HiveState& s,
    const GPUMove* my_moves,
    int n_legal,
    float* f
) {
    Bitboard ap_mask = find_articulation_points(s);
    extract_fnn_features_with_ap_device(s, my_moves, n_legal, ap_mask, f);
}

#ifndef HIVE_CPU_NATIVE
__global__ void extract_fnn_features_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,   // [B, MAX_LEGAL_MOVES]
    const int* num_legal,         // [B]
    float* features_out,          // [B, FNN_FEAT_DIM]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    extract_fnn_features_device(
        states[idx],
        legal_moves + idx * MAX_LEGAL_MOVES,
        num_legal[idx],
        features_out + idx * FNN_FEAT_DIM
    );
}
#endif  // HIVE_CPU_NATIVE

#endif  // __CUDACC__

}  // namespace hive_gpu
