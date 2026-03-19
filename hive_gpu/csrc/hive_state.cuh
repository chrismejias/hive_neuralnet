/**
 * hive_state.cuh — Compact Hive game state for GPU.
 *
 * Designed for efficient GPU storage and manipulation:
 * - Fixed 17×17 grid (289 cells)
 * - Full stack tracking: up to MAX_STACK=5 pieces per cell (for beetle play)
 * - Bitboard occupancy for fast neighbor/placement queries
 * - ~1700 bytes total per state
 *
 * Stack tracking is critical for Hive strategy:
 * - Beetles stack on top of pieces, covering them
 * - Middle pieces (covered but covering others) affect connectivity
 * - Queen win detection works regardless of beetles on top
 * - When a beetle moves off, the piece underneath is revealed
 */

#pragma once

#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include "hex_grid.cuh"

namespace hive_gpu {

// ── Piece types and colors ─────────────────────────────────────────

enum PieceType : uint8_t {
    PT_EMPTY  = 0,
    PT_QUEEN  = 1,
    PT_ANT    = 2,
    PT_GRASSHOPPER = 3,
    PT_SPIDER = 4,
    PT_BEETLE = 5,
    PT_MOSQUITO  = 6,
    PT_LADYBUG   = 7,
    PT_PILLBUG   = 8,
};

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
};

constexpr int NUM_PIECE_TYPES = 8;  // Q, A, G, S, B, M, L, P
constexpr int NUM_BASE_PIECE_TYPES = 5;  // Q, A, G, S, B

// Pieces per player per type (matches hive_engine/pieces.py)
// Note: __device__ constexpr arrays aren't supported in older CUDA, so we use
// a constexpr function instead for device-side access.
constexpr uint8_t PIECES_PER_TYPE[NUM_PIECE_TYPES] = {
    1,  // Queen
    3,  // Ant
    3,  // Grasshopper
    2,  // Spider
    2,  // Beetle
    1,  // Mosquito
    1,  // Ladybug
    1,  // Pillbug
};

#ifdef __CUDACC__
__device__ __forceinline__ uint8_t pieces_per_type(int idx) {
    constexpr uint8_t table[NUM_PIECE_TYPES] = {1, 3, 3, 2, 2, 1, 1, 1};
    return table[idx];
}
#endif
constexpr int PIECES_PER_PLAYER_BASE = 11;
constexpr int PIECES_PER_PLAYER = 14;  // max with all expansions
constexpr int MAX_PIECES = 28;

// Maximum stack height. Theoretical max = ground piece + 4 beetles (all 4 beetles
// in the game stacked on one cell). In practice stacks > 3 are extremely rare.
constexpr int MAX_STACK = 5;

// ── Cell encoding ──────────────────────────────────────────────────

/**
 * Each cell at each level is packed as a single byte:
 *   bits 0-3: PieceType (0=empty, 1-8=Q/A/G/S/B/M/L/P)
 *   bit  4:   Color (0=WHITE, 1=BLACK)
 *   bits 5-7: reserved (0)
 *
 * Empty cells have value 0x00.
 */
constexpr uint8_t CELL_EMPTY = 0;

#ifdef __CUDACC__
__device__ __host__
#endif
inline uint8_t make_cell(PieceType pt, Color c) {
    return (uint8_t)pt | ((uint8_t)c << 4);
}

#ifdef __CUDACC__
__device__ __host__
#endif
inline PieceType cell_piece_type(uint8_t cell_val) {
    return (PieceType)(cell_val & 0x0F);
}

#ifdef __CUDACC__
__device__ __host__
#endif
inline Color cell_color(uint8_t cell_val) {
    return (Color)((cell_val >> 4) & 0x01);
}

// ── Bitboard helpers (289 bits in BB_WORDS × uint64) ─────────────────

struct Bitboard {
    uint64_t w[BB_WORDS];  // w[0]=bits 0-63, w[1]=64-127, ..., w[4]=256-288

#ifdef __CUDACC__
    __device__ __host__
#endif
    void clear() {
        for (int i = 0; i < BB_WORDS; i++) w[i] = 0;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    bool get(int bit) const {
        return (w[bit >> 6] >> (bit & 63)) & 1;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    void set(int bit) {
        w[bit >> 6] |= (1ULL << (bit & 63));
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    void clr(int bit) {
        w[bit >> 6] &= ~(1ULL << (bit & 63));
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    int popcount() const {
        int total = 0;
#ifdef __CUDA_ARCH__
        for (int i = 0; i < BB_WORDS; i++) total += __popcll(w[i]);
#elif defined(_MSC_VER)
        for (int i = 0; i < BB_WORDS; i++) total += (int)__popcnt64(w[i]);
#else
        for (int i = 0; i < BB_WORDS; i++) total += __builtin_popcountll(w[i]);
#endif
        return total;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    bool is_zero() const {
        uint64_t all = 0;
        for (int i = 0; i < BB_WORDS; i++) all |= w[i];
        return all == 0;
    }

    // Bitwise operations
#ifdef __CUDACC__
    __device__ __host__
#endif
    Bitboard operator&(const Bitboard& o) const {
        Bitboard r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = w[i] & o.w[i];
        return r;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    Bitboard operator|(const Bitboard& o) const {
        Bitboard r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = w[i] | o.w[i];
        return r;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    Bitboard operator~() const {
        Bitboard r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = ~w[i];
        // Mask off unused bits in the last word
        constexpr int tail_bits = NUM_CELLS & 63;  // 289 % 64 = 33
        if (tail_bits > 0) {
            r.w[BB_WORDS - 1] &= (1ULL << tail_bits) - 1;
        }
        return r;
    }

#ifdef __CUDACC__
    __device__ __host__
#endif
    Bitboard andnot(const Bitboard& o) const {
        // this & ~o
        Bitboard r;
        for (int i = 0; i < BB_WORDS; i++) r.w[i] = w[i] & ~o.w[i];
        return r;
    }
};

// ── Game result ────────────────────────────────────────────────────

enum GameResult : uint8_t {
    IN_PROGRESS = 0,
    WHITE_WINS  = 1,
    BLACK_WINS  = 2,
    DRAW        = 3,
};

// ── Move representation ────────────────────────────────────────────

enum MoveType : uint8_t {
    MOVE_PLACE = 0,
    MOVE_MOVE  = 1,
    MOVE_PASS  = 2,
};

struct GPUMove {
    MoveType  type;
    PieceType piece_type;
    uint16_t  from_cell;   // source cell (for MOVE_MOVE), unused for PLACE
    uint16_t  to_cell;     // destination cell
};

constexpr int MAX_LEGAL_MOVES = 256;

// ── HiveState ──────────────────────────────────────────────────────

struct HiveState {
    // ── Full stack per cell ──────────────────────────────────────
    // pieces[level][cell] = packed piece byte (type|color), 0 = empty
    // Level 0 = ground, level 1 = first beetle on top, etc.
    // This preserves ALL pieces including those covered by beetles.
    uint8_t pieces[MAX_STACK][NUM_CELLS];  // 5 × 289 = 1445 bytes

    // height[cell] = number of pieces stacked at this cell (0 = empty)
    uint8_t height[NUM_CELLS];             // 289 bytes

    // ── Bitboards ───────────────────────────────────────────────
    Bitboard occupied;    // cells with at least one piece
    Bitboard white_top;   // cells where top piece is white
    Bitboard black_top;   // cells where top piece is black

    // ── Queen tracking ──────────────────────────────────────────
    uint16_t queen_cell[2];   // queen_cell[color] = cell index (0xFFFF = not placed)

    // ── Hands ───────────────────────────────────────────────────
    // hands[color][piece_type_idx] where piece_type_idx = PieceType - 1
    // (0=Q, 1=A, 2=G, 3=S, 4=B, 5=M, 6=L, 7=P)
    uint8_t hands[2][NUM_PIECE_TYPES];

    // ── Game metadata ───────────────────────────────────────────
    uint16_t turn;
    uint8_t queen_placed;    // bit 0 = white queen placed, bit 1 = black queen placed
    GameResult result;
    int8_t center_q;
    int8_t center_r;

    // Padding to align to 8 bytes (optional, for perf)
    uint8_t _pad[2];
};
// Total: ~1445 + 289 + 3*40 + 14 + 10 + 8 ≈ ~1886 bytes

// ── Device functions for HiveState manipulation ────────────────────

#ifdef __CUDACC__

__device__ __forceinline__ Color current_player(const HiveState& s) {
    return (Color)(s.turn & 1);
}

__device__ __forceinline__ int player_turn_number(const HiveState& s) {
    return s.turn / 2;
}

__device__ __forceinline__ bool is_queen_placed(const HiveState& s, Color c) {
    return (s.queen_placed >> (int)c) & 1;
}

__device__ __forceinline__ bool is_occupied(const HiveState& s, int cell) {
    return s.occupied.get(cell);
}

__device__ __forceinline__ PieceType top_piece_type_at(const HiveState& s, int cell) {
    int h = s.height[cell];
    if (h == 0) return PT_EMPTY;
    return cell_piece_type(s.pieces[h - 1][cell]);
}

__device__ __forceinline__ Color top_piece_color_at(const HiveState& s, int cell) {
    int h = s.height[cell];
    if (h == 0) return WHITE;  // shouldn't be called on empty cells
    return cell_color(s.pieces[h - 1][cell]);
}

__device__ __forceinline__ uint8_t top_piece_packed(const HiveState& s, int cell) {
    int h = s.height[cell];
    if (h == 0) return CELL_EMPTY;
    return s.pieces[h - 1][cell];
}

__device__ __forceinline__ int num_occupied_neighbors(const HiveState& s, int cell) {
    int count = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        int16_t nb = NEIGHBORS[cell][d];
        if (nb >= 0 && s.occupied.get(nb)) count++;
    }
    return count;
}

/**
 * Place a piece from hand onto the board at cell (on top of any existing stack).
 */
__device__ inline void place_piece(HiveState& s, int cell, PieceType pt, Color c) {
    uint8_t packed = make_cell(pt, c);
    int h = s.height[cell];

    // Push onto stack
    s.pieces[h][cell] = packed;
    s.height[cell] = h + 1;

    // Update occupancy
    if (h == 0) {
        s.occupied.set(cell);
    }

    // Update color bitboards based on new top piece
    if (c == WHITE) {
        s.white_top.set(cell);
        s.black_top.clr(cell);
    } else {
        s.black_top.set(cell);
        s.white_top.clr(cell);
    }

    // Track queen position
    if (pt == PT_QUEEN) {
        s.queen_cell[c] = (uint16_t)cell;
        s.queen_placed |= (1 << (int)c);
    }

    // Decrement hand count
    s.hands[c][pt - 1]--;
}

/**
 * Remove the top piece from cell. Returns the packed piece value.
 * Reveals the piece underneath (if any).
 */
__device__ inline uint8_t remove_top(HiveState& s, int cell) {
    int h = s.height[cell];
    uint8_t removed = s.pieces[h - 1][cell];
    s.pieces[h - 1][cell] = CELL_EMPTY;
    s.height[cell] = h - 1;

    if (h - 1 == 0) {
        // Cell is now empty
        s.occupied.clr(cell);
        s.white_top.clr(cell);
        s.black_top.clr(cell);
    } else {
        // Update color bitboard to reflect newly revealed top piece
        uint8_t new_top = s.pieces[h - 2][cell];
        Color new_top_c = cell_color(new_top);
        if (new_top_c == WHITE) {
            s.white_top.set(cell);
            s.black_top.clr(cell);
        } else {
            s.black_top.set(cell);
            s.white_top.clr(cell);
        }
    }

    return removed;
}

/**
 * Move the top piece from `from_cell` to `to_cell`.
 * Pops from source stack, pushes onto destination stack.
 */
__device__ inline void move_piece(HiveState& s, int from_cell, int to_cell) {
    uint8_t packed = remove_top(s, from_cell);
    PieceType pt = cell_piece_type(packed);
    Color c = cell_color(packed);

    // Push at destination
    int h = s.height[to_cell];
    s.pieces[h][to_cell] = packed;
    s.height[to_cell] = h + 1;

    if (h == 0) {
        s.occupied.set(to_cell);
    }

    // Update color bitboards
    if (c == WHITE) {
        s.white_top.set(to_cell);
        s.black_top.clr(to_cell);
    } else {
        s.black_top.set(to_cell);
        s.white_top.clr(to_cell);
    }

    // Update queen position tracking
    if (pt == PT_QUEEN) {
        s.queen_cell[c] = (uint16_t)to_cell;
    }
}

/**
 * Check if a queen is surrounded (all 6 neighbors occupied) → game over check.
 */
__device__ inline bool is_queen_surrounded(const HiveState& s, Color c) {
    if (!is_queen_placed(s, c)) return false;
    int qcell = s.queen_cell[c];
    return num_occupied_neighbors(s, qcell) == 6;
}

/**
 * Check and update game result after a move.
 */
__device__ inline void check_game_over(HiveState& s) {
    bool w_surr = is_queen_surrounded(s, WHITE);
    bool b_surr = is_queen_surrounded(s, BLACK);
    if (w_surr && b_surr)      s.result = DRAW;
    else if (w_surr)           s.result = BLACK_WINS;
    else if (b_surr)           s.result = WHITE_WINS;
}

/**
 * Initialize a fresh game state.
 * expansion_mask: 3-bit mask (bit 0=Mosquito, 1=Ladybug, 2=Pillbug)
 */
__device__ inline void init_state(HiveState& s, uint8_t expansion_mask = 0) {
    // Zero all piece data
    for (int level = 0; level < MAX_STACK; level++) {
        for (int i = 0; i < NUM_CELLS; i++) {
            s.pieces[level][i] = CELL_EMPTY;
        }
    }
    for (int i = 0; i < NUM_CELLS; i++) {
        s.height[i] = 0;
    }

    s.occupied.clear();
    s.white_top.clear();
    s.black_top.clear();

    s.queen_cell[0] = 0xFFFF;
    s.queen_cell[1] = 0xFFFF;

    // Initialize hands: base types always get full counts,
    // expansion types get count 1 only if corresponding mask bit is set
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < NUM_BASE_PIECE_TYPES; p++) {
            s.hands[c][p] = pieces_per_type(p);
        }
        // Mosquito (index 5): enabled if bit 0 set
        s.hands[c][5] = (expansion_mask & 1) ? 1 : 0;
        // Ladybug (index 6): enabled if bit 1 set
        s.hands[c][6] = (expansion_mask & 2) ? 1 : 0;
        // Pillbug (index 7): enabled if bit 2 set
        s.hands[c][7] = (expansion_mask & 4) ? 1 : 0;
    }

    s.turn = 0;
    s.queen_placed = 0;
    s.result = IN_PROGRESS;
    s.center_q = 0;
    s.center_r = 0;
    s._pad[0] = s._pad[1] = 0;
}

/**
 * Apply a move to the state and advance the turn.
 */
__device__ inline void apply_move(HiveState& s, const GPUMove& m) {
    if (m.type == MOVE_PLACE) {
        place_piece(s, m.to_cell, m.piece_type, current_player(s));
    } else if (m.type == MOVE_MOVE) {
        move_piece(s, m.from_cell, m.to_cell);
    }
    // MOVE_PASS: do nothing to the board

    s.turn++;
    check_game_over(s);
}

#endif  // __CUDACC__

}  // namespace hive_gpu
