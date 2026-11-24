"""PGN transformations and the version 1 board and action encodings.

WHAT: Build one auditable supervised sample from a small standard-chess PGN.
HOW: Replay legal moves, tokenize historical boards, and label the next move/outcome.
WHY: Make the data contract understandable before implementing a production pipeline.
"""

from __future__ import annotations

import io
from collections import deque
from typing import Any

import chess
import chess.pgn
import numpy as np

HISTORY = 8
ACTION_COUNT = 64 * 64 + 8 * 8 * 4
PROMOTIONS = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
VALUE_IGNORE = -100


def read_game(text: str) -> chess.pgn.Game:
    """WHAT: Parse exactly one small PGN, retaining headers and rejecting parser errors.

    HOW: Inspect python-chess's error list and starting-board validity; reject variants.
    WHY: A forgiving parser can otherwise yield a plausible but truncated game.
    This bounded teaching helper accepts text; it is not a streaming archive reader.
    """
    stream = io.StringIO(text)
    game = chess.pgn.read_game(stream)
    if game is None:
        raise ValueError("PGN contains no game")
    if game.errors:
        raise ValueError(f"Malformed PGN: {game.errors}")
    if game.headers.get("Variant", "Standard").lower() not in {"standard", "chess"}:
        raise ValueError("Only standard chess is supported")
    board = game.board()
    if board.chess960 or type(board) is not chess.Board or not board.is_valid():
        raise ValueError("Unsupported variant or invalid starting position")
    if game.headers.get("Result", "*") not in {"1-0", "0-1", "1/2-1/2", "*"}:
        raise ValueError("Unsupported result label")
    if chess.pgn.read_game(stream) is not None:
        raise ValueError("Supply exactly one game for this walkthrough")

    # push() itself does not validate legality. Validate before every push, and
    # reject moves after automatic game termination, even if geometrically legal.
    for move in game.mainline_moves():
        if board.is_game_over() or move not in board.legal_moves:
            raise ValueError(f"Illegal move or move after game end: {move.uci()}")
        board.push(move)
    outcome = board.outcome()
    if outcome is not None and game.headers["Result"] not in {"*", outcome.result()}:
        raise ValueError("PGN result contradicts the terminal board")
    return game


def orient_square(square: int, turn: chess.Color) -> int:
    """WHAT: Map a square to/from mover coordinates. HOW: Reflect ranks for Black.

    WHY: This involution preserves files and matches the inspected reference source.
    """
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_frame(board: chess.Board) -> np.ndarray:
    """WHAT: Return one board as uint8 (64, 12) square tokens.

    HOW: Mirror Black's board (ranks and colors), then one-hot P,N,B,R,Q,K per color.
    WHY: Empty squares remain zero and the mover always occupies channels 0..5.
    O(64*12) allocation; no FEN state or move stack is encoded in these channels.
    """
    canonical = board if board.turn == chess.WHITE else board.mirror()
    tokens = np.zeros((64, 12), dtype=np.uint8)
    for square, piece in canonical.piece_map().items():
        channel = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tokens[square, channel] = 1
    return tokens


def legal_state(board: chess.Board) -> dict[str, Any]:
    """WHAT: Preserve absolute legal state outside the piece tensor.

    HOW: Store turn, castling bitboard, raw en passant square, and FEN counters.
    WHY: Identical piece placements can have different legal moves and draw status.
    Repetition additionally requires the replay prefix, not just these fields.
    """
    return {
        "turn": board.turn,
        "castling_rights": int(board.castling_rights),
        "ep_square": board.ep_square,
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
    }


def decode_frame(tokens: np.ndarray, state: dict[str, Any]) -> chess.Board:
    """WHAT: Recover a standard board's pieces and FEN state from (64,12) tokens.

    HOW: Validate one-hot/zero vectors and invert the recorded frame's orientation.
    WHY: Round trips expose coordinate mistakes; full repetition needs replay instead.
    Time/space are O(64*12); the returned board has no historical move stack.
    """
    if tokens.shape != (64, 12) or tokens.dtype != np.uint8:
        raise ValueError("Expected uint8 tokens with shape (64, 12)")
    if not np.isin(tokens, [0, 1]).all() or np.any(tokens.sum(axis=1) > 1):
        raise ValueError("Each square must be one-hot or zero")
    board = chess.Board.empty()
    turn = bool(state["turn"])
    for square, channel in zip(*np.nonzero(tokens), strict=True):
        color = bool(channel < 6)
        if turn == chess.BLACK:
            color = not color
        board.set_piece_at(
            orient_square(int(square), turn), chess.Piece(int(channel % 6) + 1, color)
        )
    board.turn = turn
    board.castling_rights = state["castling_rights"]
    board.ep_square = state["ep_square"]
    board.halfmove_clock = state["halfmove_clock"]
    board.fullmove_number = state["fullmove_number"]
    return board


def encode_history(boards: list[chess.Board]) -> np.ndarray:
    """WHAT: Return uint8 (64,96) from up to eight chronological boards.

    HOW: Keep the last eight, left-pad the earliest, concatenate each frame's channels.
    WHY: Match the reference history ordering without future-position leakage.
    Each frame uses its OWN turn, not the current board's turn. O(H*64*12).
    """
    if not boards:
        raise ValueError("History must contain the current position")
    frames = boards[-HISTORY:]
    frames = [frames[0]] * (HISTORY - len(frames)) + frames
    return np.concatenate([encode_frame(board) for board in frames], axis=1)


def policy_target(board: chess.Board, move: chess.Move) -> tuple[np.int64, np.ndarray]:
    """WHAT: Encode a legal observed move as class index and uint8 (from,to,promotion).

    HOW: Mirror coordinates for Black; reserve 4096 base and 256 promotion classes.
    WHY: Source/destination alone cannot distinguish four promotion choices.
    Tuple codes: none=0, Q=1, R=2, B=3, N=4. Vocabulary matches reference utils.py.
    """
    if move not in board.legal_moves:
        raise ValueError(f"Training move is illegal: {move.uci()}")
    source = orient_square(move.from_square, board.turn)
    destination = orient_square(move.to_square, board.turn)
    promotion = PROMOTIONS.index(move.promotion) + 1 if move.promotion else 0
    if promotion:
        index = 4096 + 4 * (8 * chess.square_file(source) + chess.square_file(destination))
        index += promotion - 1
    else:
        index = 64 * source + destination
    return np.int64(index), np.array([source, destination, promotion], dtype=np.uint8)


def decode_action(index: int, turn: chess.Color) -> chess.Move:
    """WHAT: Recover an absolute UCI move. HOW: Invert vocabulary and rank reflection.

    WHY: Check a policy label against its original PGN move; legality is board-dependent.
    Reserved vocabulary entries may decode to illegal moves and must remain masked.
    """
    if not 0 <= index < ACTION_COUNT:
        raise ValueError("Action index is outside the vocabulary")
    promotion = None
    if index < 4096:
        source, destination = divmod(int(index), 64)
    else:
        pair, piece = divmod(int(index) - 4096, 4)
        source_file, destination_file = divmod(pair, 8)
        source = chess.square(source_file, 6)
        destination = chess.square(destination_file, 7)
        promotion = PROMOTIONS[piece]
    return chess.Move(
        orient_square(source, turn), orient_square(destination, turn), promotion=promotion
    )


def legal_mask(board: chess.Board) -> np.ndarray:
    """WHAT: Return bool (4352,) legal actions. HOW: Encode python-chess legal moves.

    WHY: Training/inference must exclude impossible actions, including false promotions.
    Automatic terminal boards have no actions; claimable draws may still have moves.
    """
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    if not board.is_game_over():
        for move in board.legal_moves:
            index, _ = policy_target(board, move)
            mask[index] = True
    return mask


def value_target(result: str, turn: chess.Color) -> dict[str, Any]:
    """WHAT: Generate mover-relative loss/draw/win targets, including missing labels.

    HOW: Negate White's signed outcome for Black, then class = signed + 1.
    WHY: The same final result is a win for one input player and a loss for the other.
    Unknown '*' uses ignore_index=-100 and a false mask, never a fabricated draw.
    """
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        raise ValueError(f"Unsupported outcome: {result}")
    one_hot = np.zeros(3, dtype=np.uint8)
    if result == "*":
        return {
            "value_class": np.int64(VALUE_IGNORE),
            "value_valid": np.bool_(False),
            "value_one_hot": one_hot,
            "value_signed": None,
        }
    white_outcome = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]
    signed = white_outcome if turn == chess.WHITE else -white_outcome
    one_hot[signed + 1] = 1
    return {
        "value_class": np.int64(signed + 1),
        "value_valid": np.bool_(True),
        "value_one_hot": one_hot,
        "value_signed": signed,
    }


def make_sample(game: chess.pgn.Game, ply: int) -> dict[str, Any]:
    """WHAT: Build a sample immediately BEFORE zero-based ply's observed move.

    HOW: Replay the prefix, keep eight boards, encode input and separate policy/value.
    WHY: A pre-move input prevents target leakage; replay metadata retains draw history.
    This single-game example keeps O(H) boards plus O(ply) replay metadata and the PGN
    tree. It is not an archive-scale dataset interface. See read_game for validation.
    """
    if ply < 0:
        raise ValueError("Ply must be nonnegative")
    board = game.board()
    initial_fen = board.fen(en_passant="fen")
    history = deque([(0, board.copy(stack=False))], maxlen=HISTORY)
    prefix: list[str] = []
    target = None
    for index, move in enumerate(game.mainline_moves()):
        if board.is_game_over() or move not in board.legal_moves:
            raise ValueError(f"Invalid mainline move: {move.uci()}")
        if index == ply:
            target = move
            break
        board.push(move)
        prefix.append(move.uci())
        history.append((index + 1, board.copy(stack=False)))
    if target is None:
        raise ValueError("Ply has no observed next move (terminal boards have no policy label)")
    index, coordinates = policy_target(board, target)
    frames = list(history)
    padded = [frames[0]] * (HISTORY - len(frames)) + frames
    mask = legal_mask(board)
    if not mask[index]:
        raise ValueError("Observed action is absent from the legal mask")
    return {
        "schema_version": "drewbot-example-v1",
        "tokens": encode_history([frame for _, frame in frames]),
        "policy_index": index,
        "policy_from_to_promotion": coordinates,
        "legal_mask": mask,
        **value_target(game.headers.get("Result", "*"), board.turn),
        "state": legal_state(board),
        "history_plies": [number for number, _ in padded],
        "history_turns": [frame.turn for _, frame in padded],
        "initial_fen": initial_fen,
        "prefix_uci": prefix,
        "headers": dict(game.headers),
        "fen": board.fen(en_passant="fen"),
        "played_uci": target.uci(),
        "ply": ply,
    }
