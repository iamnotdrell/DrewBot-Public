"""Sequential, deterministic sample generation using the educational encoding contract."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from collections.abc import Iterator
from time import perf_counter

import chess
import numpy as np

from drewbot.config import DataConfig
from drewbot.data.ingest import GameRecord
from drewbot.data.walkthrough import (
    HISTORY,
    encode_history,
    legal_mask,
    orient_square,
    policy_target,
    value_target,
)


def selected_plies(record: GameRecord, config: DataConfig) -> list[int]:
    """WHAT: Select pre-move plies. HOW: Per-ID sampling. WHY: Order-independent reproducibility.

    Candidates use range(min_ply, move_count, stride); chosen plies are sorted for O(plies) replay.
    O(max_game_plies) worst-case output, bounded by the validated per-game limit.
    """
    candidates = range(config.min_ply, len(record.moves), config.sample_stride)
    budget = config.max_positions_per_game
    if budget is None or len(candidates) <= budget:
        return list(candidates)
    seed = hashlib.sha256(f"sample:{config.seed}:{record.game_id}".encode()).digest()
    return sorted(random.Random(seed).sample(candidates, budget))


def auxiliary(board: chess.Board, repetitions: int) -> np.ndarray:
    """WHAT: Float32 (72,) legal-state companion features.

    HOW: Encode absolute turn/rights, canonical raw EP, scaled clocks/repetition count.
    WHY: Piece channels alone alias positions with different legal moves/draw state.
    This is a documented DrewBot extension; no future model is implemented here.
    """
    features = np.zeros(72, dtype=np.float32)
    features[0] = not board.turn
    features[1:5] = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    if board.ep_square is not None:
        features[5 + orient_square(board.ep_square, board.turn)] = 1
    features[69:] = [
        board.halfmove_clock / 100,
        board.fullmove_number / 100,
        min(repetitions, 5) / 5,
    ]
    if not np.isfinite(features).all():
        raise ValueError("Auxiliary state features must be finite")
    return features


def iter_samples(record: GameRecord, plies: list[int], split: str) -> Iterator[dict]:
    """WHAT: Emit selected input/target pairs in a single legal replay.

    HOW: Maintain eight boards plus a repetition counter; encode before pushing each move.
    WHY: Avoid O(plies²) prefix replays and target leakage. Memory is bounded by one
    game (replay/counters) and H=8 history frames; output arrays are (64,96), (72,),
    scalar int64 labels and (544,) packed legal bits. Bit order is explicitly little.
    """
    board = chess.Board(record.initial_fen)
    history = deque([board.copy(stack=False)], maxlen=HISTORY)
    selected = set(plies)
    repetitions: Counter[str] = Counter()
    for ply, uci in enumerate(record.moves):
        # FEN's default legal-EP mode correctly ignores non-capturable EP fields in
        # repetition identity, while exact raw EP remains in metadata and auxiliary.
        key = " ".join(board.fen().split()[:4])
        repetitions[key] += 1
        if ply in selected:
            move = chess.Move.from_uci(uci)
            index, _ = policy_target(board, move)
            started = perf_counter()
            tokens = encode_history(list(history))
            encoding_seconds = perf_counter() - started
            value = value_target(record.result, board.turn)
            yield {
                "tokens": tokens,
                "auxiliary": auxiliary(board, repetitions[key]),
                "policy": index,
                "value": value["value_class"],
                "legal": np.packbits(legal_mask(board), bitorder="little"),
                "metadata": {
                    "game_id": record.game_id,
                    "ply": ply,
                    "fen": board.fen(en_passant="fen"),
                    "played_uci": uci,
                    "split": split,
                },
                "encoding_seconds": encoding_seconds,
            }
        board.push_uci(uci)
        history.append(board.copy(stack=False))
