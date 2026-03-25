"""
NNUE-style feature extraction for Hive.

Inspired by Stockfish NNUE's HalfKP feature set, this encoder creates
a flat feature vector that captures piece-queen spatial relationships.
The key insight: a beetle adjacent to the opponent's queen is far more
valuable than one on the periphery. By encoding distance-to-queen as
one-hot buckets, the MLP can learn these positional values automatically.

Feature vector layout (428 features total):
    [0, 209):   Current player's 11 pieces (19 features each)
    [209, 418): Opponent's 11 pieces (19 features each)
    [418, 428): Global game state features (10 features)

Per-piece features (19 each):
    [0, 4):     Status: is_on_board, is_in_hand, is_mobile, is_on_top
    [4, 11):    Distance to own queen (one-hot buckets 0-6+)
    [11, 18):   Distance to opponent queen (one-hot buckets 0-6+)
    [18]:       Neighbor density (occupied neighbors / 6)
"""

from __future__ import annotations

import numpy as np

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, PieceType, Piece, create_player_pieces


# ── Constants ─────────────────────────────────────────────────────

FEATURES_PER_PIECE: int = 19
NUM_PIECES_PER_PLAYER: int = 11
NUM_DISTANCE_BUCKETS: int = 7       # distances 0, 1, 2, 3, 4, 5, 6+
MAX_DISTANCE_BUCKET: int = 6
NUM_GLOBAL_FEATURES: int = 10

FEATURE_DIM: int = (
    FEATURES_PER_PIECE * NUM_PIECES_PER_PLAYER * 2  # both players
    + NUM_GLOBAL_FEATURES
)  # 19 * 11 * 2 + 10 = 428


# ── Canonical piece ordering ──────────────────────────────────────

# Pieces are always presented in this fixed order per player:
# Q, A1, A2, A3, G1, G2, G3, S1, S2, B1, B2
# This matches create_player_pieces() output order.

_WHITE_PIECES: list[Piece] = create_player_pieces(Color.WHITE)
_BLACK_PIECES: list[Piece] = create_player_pieces(Color.BLACK)


# ── Feature Encoder ───────────────────────────────────────────────


class NNUEFeatureEncoder:
    """
    Extract NNUE-style features from a Hive game state.

    Features are presented from the current player's perspective:
    the first 220 features describe the current player's pieces,
    the next 220 describe the opponent's pieces. This symmetry means
    the network learns a single evaluation function that works for
    both colors.

    The queen-relative distance features are the key innovation:
    they allow the MLP to learn positional piece values (e.g., a
    beetle near the opponent's queen is worth more than one far away)
    without any explicit spatial encoding like convolutions or graphs.
    """

    def encode(self, game_state: GameState) -> np.ndarray:
        """
        Encode a game state as a flat NNUE feature vector.

        Args:
            game_state: The current game state.

        Returns:
            Float32 array of shape (FEATURE_DIM,) = (428,).
        """
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        board = game_state.board
        current = game_state.current_player
        opponent = current.other()

        # ── Find queen positions ──────────────────────────────
        own_queen_pos: HexCoord | None = None
        opp_queen_pos: HexCoord | None = None

        if game_state._queen_placed[current]:
            for piece, pos in board.piece_positions.items():
                if (
                    piece.piece_type == PieceType.QUEEN
                    and piece.color == current
                ):
                    own_queen_pos = pos
                    break

        if game_state._queen_placed[opponent]:
            for piece, pos in board.piece_positions.items():
                if (
                    piece.piece_type == PieceType.QUEEN
                    and piece.color == opponent
                ):
                    opp_queen_pos = pos
                    break

        # ── Compute articulation points for pinning ───────────
        # A piece is pinned if removing it disconnects the hive.
        # Pieces on top of a stack (height > 1) are never pinned
        # since removing them doesn't empty the position.
        articulation_points: set[HexCoord] = set()
        if board.grid:
            articulation_points = board.find_articulation_points()

        # ── Get canonical piece lists ─────────────────────────
        if current == Color.WHITE:
            own_pieces = _WHITE_PIECES
            opp_pieces = _BLACK_PIECES
        else:
            own_pieces = _BLACK_PIECES
            opp_pieces = _WHITE_PIECES

        # ── Encode own pieces (slots 0..10) ───────────────────
        offset = 0
        for piece in own_pieces:
            self._encode_piece(
                features, offset, piece, board,
                own_queen_pos, opp_queen_pos,
                articulation_points,
                game_state._queen_placed[current],
            )
            offset += FEATURES_PER_PIECE

        # ── Encode opponent pieces (slots 11..21) ─────────────
        for piece in opp_pieces:
            self._encode_piece(
                features, offset, piece, board,
                opp_queen_pos, own_queen_pos,
                articulation_points,
                game_state._queen_placed[opponent],
            )
            offset += FEATURES_PER_PIECE

        # ── Global features ───────────────────────────────────
        g = offset  # should be 440

        # Current player identity (for potential asymmetry)
        features[g + 0] = 1.0 if current == Color.WHITE else 0.0

        # Turn progress
        features[g + 1] = min(game_state.turn / 100.0, 1.0)

        # Queen placement
        features[g + 2] = 1.0 if game_state._queen_placed[current] else 0.0
        features[g + 3] = 1.0 if game_state._queen_placed[opponent] else 0.0

        # Queen surround counts
        features[g + 4] = game_state.queen_surrounded_count(current) / 6.0
        features[g + 5] = game_state.queen_surrounded_count(opponent) / 6.0

        # Hand counts
        features[g + 6] = len(game_state.hand(current)) / 11.0
        features[g + 7] = len(game_state.hand(opponent)) / 11.0

        # Board piece counts
        own_on_board = len(board.pieces_of_color(current))
        opp_on_board = len(board.pieces_of_color(opponent))
        features[g + 8] = own_on_board / 11.0
        features[g + 9] = opp_on_board / 11.0

        return features

    @staticmethod
    def _encode_piece(
        features: np.ndarray,
        offset: int,
        piece: Piece,
        board,
        own_queen_pos: HexCoord | None,
        opp_queen_pos: HexCoord | None,
        articulation_points: set[HexCoord],
        queen_placed: bool,
    ) -> None:
        """
        Encode a single piece's features into the feature vector.

        Args:
            features: The output feature array to write into.
            offset: Starting index in the feature array for this piece.
            piece: The piece to encode.
            board: The game board.
            own_queen_pos: Position of this piece's player's queen.
            opp_queen_pos: Position of the opposing queen.
            articulation_points: Set of pinned positions.
            queen_placed: Whether this piece's player has placed their queen.
        """
        pos = board.position_of(piece)

        if pos is None:
            # Piece is in hand
            features[offset + 1] = 1.0   # is_in_hand
            return

        # ── Status features ───────────────────────────────────
        features[offset + 0] = 1.0   # is_on_board

        is_on_top = board.is_on_top(piece)
        features[offset + 3] = 1.0 if is_on_top else 0.0

        # Mobility: on top + not pinned + queen placed
        if is_on_top and queen_placed:
            stack_h = board.stack_height(pos)
            is_pinned = (stack_h == 1) and (pos in articulation_points)
            if not is_pinned:
                features[offset + 2] = 1.0   # is_mobile

        # ── Distance to own queen (one-hot buckets) ───────────
        if own_queen_pos is not None:
            dist = pos.distance(own_queen_pos)
            bucket = min(dist, MAX_DISTANCE_BUCKET)
            features[offset + 4 + bucket] = 1.0

        # ── Distance to opponent queen (one-hot buckets) ──────
        if opp_queen_pos is not None:
            dist = pos.distance(opp_queen_pos)
            bucket = min(dist, MAX_DISTANCE_BUCKET)
            features[offset + 11 + bucket] = 1.0

        # ── Neighbor density ──────────────────────────────────
        features[offset + 18] = board.num_occupied_neighbors(pos) / 6.0


# ── Encoder Wrapper (MCTS-compatible interface) ───────────────────


class NNUEEncoder:
    """
    Encoder wrapper for the NNUE pipeline.

    Provides the same interface as HiveEncoder / GNNEncoder but returns
    flat NNUE feature vectors for state encoding. Action encoding/decoding
    and legal mask generation are delegated to HiveEncoder.
    """

    # Re-export constants for compatibility
    ACTION_SPACE_SIZE: int = HiveEncoder.ACTION_SPACE_SIZE
    BOARD_SIZE: int = HiveEncoder.BOARD_SIZE
    NUM_CHANNELS: int = HiveEncoder.NUM_CHANNELS
    PASS_ACTION_INDEX: int = HiveEncoder.PASS_ACTION_INDEX

    def __init__(self) -> None:
        self._feature_encoder = NNUEFeatureEncoder()
        self._hive_encoder = HiveEncoder()

    def encode_state(self, game_state: GameState) -> np.ndarray:
        """
        Encode a game state as an NNUE feature vector.

        Args:
            game_state: The current game state.

        Returns:
            Float32 array of shape (FEATURE_DIM,) = (428,).
        """
        return self._feature_encoder.encode(game_state)

    def encode_action(self, move: Move, game_state: GameState) -> int:
        """Encode a move as a flat action index (delegated to HiveEncoder)."""
        return self._hive_encoder.encode_action(move, game_state)

    def decode_action(
        self, action_index: int, game_state: GameState
    ) -> Move:
        """Decode a flat action index to a Move (delegated to HiveEncoder)."""
        return self._hive_encoder.decode_action(action_index, game_state)

    def get_legal_action_mask(
        self,
        game_state: GameState,
        legal_moves: list[Move] | None = None,
    ) -> np.ndarray:
        """Get binary mask over action space for legal moves."""
        return self._hive_encoder.get_legal_action_mask(
            game_state, legal_moves
        )
