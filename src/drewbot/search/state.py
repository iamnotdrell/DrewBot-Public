"""Reversible board state, deterministic Zobrist hashing and exact cache witnesses."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import chess

_RANDOM = random.Random(0xD2E5B07)
_PIECES = tuple(tuple(_RANDOM.getrandbits(64) for _ in range(64)) for _ in range(12))
_TURN = _RANDOM.getrandbits(64)
_CASTLING = tuple(_RANDOM.getrandbits(64) for _ in range(64))
_EP = tuple(_RANDOM.getrandbits(64) for _ in range(64))


def _state_hash(board: chess.Board) -> int:
    """WHAT: Hash nonpieces. HOW: XOR turn/rights/raw EP. WHY: Preserve legal distinctions."""
    value = _TURN if board.turn else 0
    for square in chess.scan_forward(board.castling_rights):
        value ^= _CASTLING[square]
    if board.ep_square is not None:
        value ^= _EP[board.ep_square]
    return value


def _bitboards(board: chess.Board) -> tuple[int, ...]:
    """WHAT: Index twelve piece sets. HOW: Color then type. WHY: Compare XOR deltas."""
    return tuple(board.pieces_mask(piece, color) for color in chess.COLORS for piece in range(1, 7))


def zobrist(board: chess.Board) -> int:
    """WHAT: Recompute a 64-bit position hash. HOW: XOR occupied piece/state keys.

    WHY: Independent reference for incremental make/unmake. Clocks and repetition
    live in the exact cache witness, not this O(pieces) hash.
    """
    value = _state_hash(board)
    for index, occupied in enumerate(_bitboards(board)):
        for square in chess.scan_forward(occupied):
            value ^= _PIECES[index][square]
    return value


def repetition_identity(board: chess.Board) -> str:
    """WHAT: Exact rule identity. HOW: Legal-EP FEN first four fields. WHY: Ignore phantom EP."""
    return " ".join(board.fen().split()[:4])


def _frame_identity(board: chess.Board) -> tuple[str, bool]:
    return board.board_fen(), board.turn


@dataclass(frozen=True)
class PositionInput:
    """WHAT: Inference snapshot. HOW: Copy current board/history. WHY: Safe sibling batching.

    Frames are chronological, each with its own mover, at most H<=8. Repetitions
    refers to the current rule identity; no result/target enters this structure.
    """

    board: chess.Board
    frames: tuple[chess.Board, ...]
    repetitions: int
    key: tuple


class SearchState:
    """WHAT: Own and restore a search position with exact known history.

    HOW: Replay the caller's stack once, keep H frames and repetition counts since
    irreversible moves, update a Zobrist hash by piece/state XOR deltas, undo in
    finally blocks. WHY: Neural history and draw rights invalidate piece-only TTs.
    Push costs O(12+changed pieces), plus rule/frame identity generation. Cache
    witnesses cost O(H+reversible history); this correctness-first overhead is measured.
    """

    def __init__(self, board: chess.Board, history_frames: int = 8):
        if (
            type(board) is not chess.Board
            or board.chess960
            or not board.is_valid()
            or type(history_frames) is not int
            or not 1 <= history_frames <= 8
        ):
            raise ValueError("Search requires valid standard chess and 1..8 history frames")
        self.history_frames = history_frames
        self.board = board.root()
        if not self.board.is_valid():
            raise ValueError("Invalid replay root")
        self.frames = [self.board.copy(stack=False)]
        self.repetitions = Counter({repetition_identity(self.board): 1})
        self.hash = zobrist(self.board)
        self._undo = []
        for move in board.move_stack:
            if self.board.is_game_over() or move not in self.board.legal_moves:
                raise ValueError("Illegal move or move after automatic terminal in replay")
            self.push(move)
        if self.board.fen(en_passant="fen") != board.fen(en_passant="fen"):
            raise ValueError("Current board disagrees with its replay history")
        self._undo.clear()

    def key(self) -> tuple:
        """WHAT: Exact cache identity. HOW: Hash plus context witness. WHY: No draw/history
        alias.
        """
        return (
            self.hash,
            self.board.fen(en_passant="fen"),
            tuple(_frame_identity(frame) for frame in self.frames),
            tuple(sorted(self.repetitions.items())),
        )

    def snapshot(self) -> PositionInput:
        """WHAT: Retain inference inputs. HOW: Stackless copies. WHY: Subsequent pushes are safe."""
        return PositionInput(
            self.board.copy(stack=False),
            tuple(self.frames),
            self.repetitions[repetition_identity(self.board)],
            self.key(),
        )

    def push(self, move: chess.Move) -> None:
        """WHAT: Make a legal move. HOW: XOR changed squares and save undo. WHY: Exact restoration.

        Internal callers generate legal moves; replay validation checks external moves.
        python-chess supplies castling/EP/promotion state changes without reimplementing rules.
        """
        before = _bitboards(self.board)
        old_hash = self.hash
        self.hash ^= _state_hash(self.board)
        old_repetitions = self.repetitions
        irreversible = self.board.is_irreversible(move)
        self.board.push(move)
        for index, (old, new) in enumerate(zip(before, _bitboards(self.board), strict=True)):
            for square in chess.scan_forward(old ^ new):
                self.hash ^= _PIECES[index][square]
        self.hash ^= _state_hash(self.board)
        if irreversible:
            self.repetitions = Counter()
        identity = repetition_identity(self.board)
        self.repetitions[identity] += 1
        dropped = self.frames.pop(0) if len(self.frames) == self.history_frames else None
        self.frames.append(self.board.copy(stack=False))
        self._undo.append((old_hash, old_repetitions, irreversible, identity, dropped))

    def pop(self) -> None:
        """WHAT: Unmake. HOW: Restore hash/counters/frames then board. WHY: Interruptions are
        safe.
        """
        old_hash, repetitions, irreversible, identity, dropped = self._undo.pop()
        if irreversible:
            self.repetitions = repetitions
        else:
            self.repetitions[identity] -= 1
            if not self.repetitions[identity]:
                del self.repetitions[identity]
        self.frames.pop()
        if dropped is not None:
            self.frames.insert(0, dropped)
        self.board.pop()
        self.hash = old_hash
