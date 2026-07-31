"""Small Universal Hive Protocol client and move translator."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import Direction, HexCoord
from hive_engine.pieces import Piece


_RELATIVE = {
    Direction.NW: ("\\", ""),
    Direction.W: ("-", ""),
    Direction.SW: ("/", ""),
    Direction.NE: ("", "/"),
    Direction.E: ("", "-"),
    Direction.SE: ("", "\\"),
}


def uhp_piece_name(piece: Piece) -> str:
    """Return the standard UHP name for a local piece."""
    color = "w" if piece.color.value == 0 else "b"
    suffix = str(piece.piece_id + 1) if piece.piece_type.count_per_player > 1 else ""
    return f"{color}{piece.piece_type.short}{suffix}"


def move_to_uhp(state: GameState, move: Move) -> str:
    """Serialize a legal local move using an occupied tile as its anchor."""
    if move.move_type == MoveType.PASS:
        return "pass"
    assert move.piece is not None and move.to is not None
    piece_name = uhp_piece_name(move.piece)
    if state.turn == 0:
        return piece_name

    stack = state.board.stack_at(move.to)
    if stack:
        return f"{piece_name} {uhp_piece_name(stack[-1])}"

    for direction in Direction:
        anchor_pos = move.to.neighbor(direction.opposite())
        anchor = state.board.top_piece_at(anchor_pos)
        if anchor is not None:
            prefix, suffix = _RELATIVE[direction]
            return f"{piece_name} {prefix}{uhp_piece_name(anchor)}{suffix}"
    raise ValueError(f"move destination has no UHP anchor: {move!r}")


def move_from_uhp(state: GameState, text: str) -> Move:
    """Map a UHP move onto the canonical object in ``state.legal_moves()``."""
    text = text.strip()
    if text == "pass":
        candidates = [m for m in state.legal_moves() if m.move_type == MoveType.PASS]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("UHP engine passed when pass is not the sole legal move")

    fields = text.split()
    if len(fields) not in (1, 2):
        raise ValueError(f"invalid UHP move: {text!r}")
    piece_name = fields[0]
    destination: HexCoord | None = None
    if len(fields) == 2:
        relative = fields[1]
        direction = None
        if relative[0] in "\\-/":
            direction = {"\\": Direction.NW, "-": Direction.W, "/": Direction.SW}[relative[0]]
            anchor_name = relative[1:]
        elif relative[-1] in "\\-/":
            direction = {"/": Direction.NE, "-": Direction.E, "\\": Direction.SE}[relative[-1]]
            anchor_name = relative[:-1]
        else:
            anchor_name = relative
        anchors = [
            (piece, pos) for piece, pos in state.board.piece_positions.items()
            if uhp_piece_name(piece) == anchor_name
        ]
        if len(anchors) != 1:
            raise ValueError(f"unknown or ambiguous UHP anchor: {anchor_name!r}")
        destination = anchors[0][1]
        if direction is not None:
            destination = destination.neighbor(direction)

    candidates = [
        move for move in state.legal_moves()
        if move.piece is not None
        and uhp_piece_name(move.piece) == piece_name
        and (destination is None or move.to == destination)
    ]
    if len(candidates) != 1:
        raise ValueError(f"UHP move {text!r} maps to {len(candidates)} local legal moves")
    return candidates[0]


class UHPClient:
    """Synchronous subprocess client for a line-oriented UHP engine."""

    def __init__(self, executable: str | Path) -> None:
        self.process = subprocess.Popen(
            [str(executable), "uhp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.info = self._read_response()

    def _read_response(self) -> list[str]:
        assert self.process.stdout is not None
        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"UHP engine exited unexpectedly: {stderr.strip()}")
            line = line.rstrip("\r\n")
            if line == "ok":
                return lines
            lines.append(line)

    def command(self, command: str) -> list[str]:
        if self.process.poll() is not None:
            raise RuntimeError(f"UHP engine has exited with code {self.process.returncode}")
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        errors = [line for line in response if line.startswith(("err ", "invalidmove "))]
        if errors:
            raise RuntimeError(f"UHP command {command!r} failed: {'; '.join(errors)}")
        return response

    def new_game(self, *, threads: int = 1, table_mb: int = 64) -> None:
        self.command(f"options set NumThreads {threads}")
        self.command(f"options set TableSizeMiB {table_mb}")
        self.command("options set RandomOpening False")
        self.command("newgame Base")

    def play(self, state: GameState, move: Move) -> None:
        self.command("play " + move_to_uhp(state, move))

    def valid_moves(self) -> list[str]:
        """Return the engine's current legal moves as UHP strings."""
        response = self.command("validmoves")
        if len(response) != 1:
            raise RuntimeError(f"unexpected validmoves response: {response!r}")
        return [] if response[0] == "pass" else response[0].split(";")

    def score(self) -> int:
        """Return Nokamute.s backed-up score from its last completed search."""
        response = self.command("score")
        if len(response) != 1:
            raise RuntimeError(f"unexpected score response: {response!r}")
        return int(response[0])

    def best_move(self, state: GameState, search: str) -> Move:
        response = self.command("bestmove " + search)
        if len(response) != 1:
            raise RuntimeError(f"unexpected bestmove response: {response!r}")
        return move_from_uhp(state, response[0])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("exit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=2)

    def __enter__(self) -> "UHPClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
