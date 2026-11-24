"""Bounded streaming PGN ingestion, metadata preservation and conservative validation."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import lzma
import re
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import zstandard

from drewbot.config import DataConfig


def check_zstd_frames(raw) -> None:
    """WHAT: Reject incomplete .zst frames. HOW: Walk bounded frame/block headers without decoding.

    WHY: zstandard.stream_reader can return EOF for truncated input without raising.
    This preflight uses seek over payloads, checks frame lengths and caps window size;
    decompression still verifies encoded blocks/checksums while streaming the PGN.
    Memory O(1), time O(compressed blocks); no archive-sized allocation.
    """
    size = raw.seek(0, 2)
    raw.seek(0)

    def exact(count):
        data = raw.read(count)
        if len(data) != count:
            raise ValueError("Truncated Zstandard frame")
        return data

    def skip(count):
        if raw.tell() + count > size:
            raise ValueError("Truncated Zstandard block")
        raw.seek(count, 1)

    while raw.tell() < size:
        magic = struct.unpack("<I", exact(4))[0]
        if 0x184D2A50 <= magic <= 0x184D2A5F:
            skip(struct.unpack("<I", exact(4))[0])
            continue
        if magic != 0xFD2FB528:
            raise ValueError("Invalid Zstandard magic or trailing data")
        header = struct.pack("<I", magic) + exact(1)
        header += exact(zstandard.frame_header_size(header) - len(header))
        parameters = zstandard.get_frame_parameters(header)
        if parameters.window_size > 128 * 1024 * 1024:
            raise ValueError("Zstandard window exceeds 128 MiB limit")
        while True:
            block = int.from_bytes(exact(3), "little")
            kind, count = (block >> 1) & 3, block >> 3
            if kind == 3 or count > 128 * 1024:
                raise ValueError("Invalid Zstandard block header")
            skip(1 if kind == 1 else count)
            if block & 1:
                break
        if parameters.has_checksum:
            exact(4)
    raw.seek(0)


@dataclass
class GameRecord:
    """WHAT: Mainline and metadata. HOW: Store once/game. WHY: Avoid repeated replay text."""

    headers: dict[str, str]
    initial_fen: str
    moves: list[str]
    result: str
    game_id: str
    player_color: str | None


def identity(initial_fen: str, moves: list[str]) -> str:
    """WHAT: Trajectory ID. HOW: JSON SHA256. WHY: Header-independent duplicate isolation."""
    return hashlib.sha256(
        json.dumps([initial_fen, moves], separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def open_pgn(path: Path) -> Iterator[TextIO]:
    """WHAT: Stream PGN text. HOW: Incremental UTF-8 decoders. WHY: Archive-sized IO.

    Zstd reads concatenated frames with a bounded 128 MiB decompression window.
    Invalid UTF-8 or damaged compressed streams fail visibly, never decode with replacement.
    """
    with path.open("rb") as raw:
        suffix = path.suffix.lower()
        if suffix == ".zst":
            check_zstd_frames(raw)
            binary = zstandard.ZstdDecompressor(max_window_size=131072).stream_reader(
                raw, read_across_frames=True
            )
        elif suffix == ".gz":
            binary = gzip.GzipFile(fileobj=raw)
        elif suffix in {".bz2", ".xz"}:
            binary = {".bz2": bz2.BZ2File, ".xz": lzma.LZMAFile}[suffix](raw)
        else:
            binary = raw
        with io.TextIOWrapper(binary, encoding="utf-8-sig", errors="strict") as text:
            yield text


class BoundedReader:
    """WHAT: Bound one parser call. HOW: Cap lines and total characters. WHY: Bound comments/trees.

    Exceeding limits aborts this source because finding a trustworthy new game boundary
    inside malformed comments is unsafe. Callers log the unprocessed remainder explicitly.
    """

    def __init__(self, stream: TextIO, config: DataConfig):
        self.stream, self.config = stream, config
        self.lines: list[str] = []
        self.count = 0

    def readline(self) -> str:
        line = self.stream.readline(self.config.max_line_chars + 1)
        self.count += len(line)
        if len(line) > self.config.max_line_chars or self.count > self.config.max_game_chars:
            raise ValueError(
                "PGN line/game character limit exceeded; source remainder not processed"
            )
        self.lines.append(line)
        return line


class MainlineBuilder(chess.pgn.GameBuilder):
    """WHAT: Ignore side variations. HOW: Return parser SKIP. WHY: Train on actual played moves."""

    def begin_variation(self):
        return chess.pgn.SKIP

    def end_variation(self):
        # All side variations are skipped, so GameBuilder's variation stack was
        # never extended. Its default end_variation would pop the mainline node.
        pass

    def __init__(self, max_plies=4096):
        super().__init__()
        self.max_plies = max_plies
        self.parsed_plies = 0

    def begin_parse_san(self, board, san):
        if getattr(self.game, "has_movetext_result", False):
            self.game.errors.append(ValueError("Move after movetext termination marker"))
            return chess.pgn.SKIP
        if self.parsed_plies >= self.max_plies:
            if self.parsed_plies == self.max_plies:
                self.game.errors.append(ValueError("Game exceeds max_game_plies"))
                self.parsed_plies += 1
            return chess.pgn.SKIP

    def visit_move(self, board, move):
        self.parsed_plies += 1
        super().visit_move(board, move)

    def handle_error(self, error):
        self.game.errors.append(error)

    def visit_header(self, tagname, tagvalue):
        if tagname == "Result":
            self.explicit_result = tagvalue
        super().visit_header(tagname, tagvalue)

    def visit_result(self, result):
        if getattr(self.game, "has_movetext_result", False):
            self.game.errors.append(ValueError("Multiple movetext termination markers"))
        self.game.has_movetext_result = True
        declared = getattr(self, "explicit_result", None)
        if declared is not None and declared != result:
            self.game.errors.append(ValueError("Header and movetext outcome disagree"))
        super().visit_result(result)


def check_lexical_text(text: str) -> None:
    """WHAT: Reject silently ignored garbage. HOW: Scan PGN token gaps. WHY: Parser is forgiving.

    Comments, percent/semicolon lines and move-number punctuation are allowed. This
    complements chess legality checking; it is not a full PGN conformance implementation.
    Work and temporary text are bounded by max_game_chars.
    """
    in_comment = False
    variation_depth = 0
    for line in text.splitlines():
        stripped = line.lstrip("\ufeff")
        if not in_comment and stripped.startswith("%"):
            continue
        if not in_comment and stripped.startswith("["):
            if not chess.pgn.TAG_REGEX.fullmatch(stripped.strip()):
                raise ValueError("Malformed PGN header")
            continue
        offset = 0
        while offset < len(line):
            if in_comment:
                end = line.find("}", offset)
                if end == -1:
                    break
                in_comment, offset = False, end + 1
            elif line[offset].isspace():
                offset += 1
            elif line[offset] == ";":
                break
            elif line[offset] == "{":
                in_comment, offset = True, offset + 1
            else:
                number = re.match(r"(?:\d+\.(?:\.\.)?|\.\.\.)", line[offset:])
                token = chess.pgn.MOVETEXT_REGEX.match(line, offset) if number is None else None
                if number:
                    offset += number.end()
                elif token:
                    if token.group(0) == "(":
                        variation_depth += 1
                    elif token.group(0) == ")":
                        variation_depth -= 1
                        if variation_depth < 0:
                            raise ValueError("Unbalanced PGN variations")
                    offset = token.end()
                    if token.group(0) not in {"1-0", "0-1", "1/2-1/2", "*"}:
                        while offset < len(line) and line[offset] in "+#":
                            offset += 1
                else:
                    raise ValueError(f"Unrecognized PGN text near {line[offset : offset + 30]!r}")
    if in_comment:
        raise ValueError("Unterminated PGN comment")
    if variation_depth:
        raise ValueError("Unbalanced PGN variations")


def record_game(game: chess.pgn.Game, config: DataConfig) -> GameRecord:
    """WHAT: Validate a full game before yielding any positions.

    HOW: Reconstruct all legal mainline positions and check result/variant/limits.
    WHY: A later illegal move must not leave earlier positions from a corrupt game.
    Time O(plies*rule-check cost), bounded by max_game_plies and max_game_chars.
    """
    if game.errors:
        raise ValueError(f"PGN parser errors: {game.errors}")
    if not getattr(game, "has_movetext_result", False):
        raise ValueError("Missing movetext termination marker; possible truncated game")
    if game.headers.get("Variant", "Standard").casefold() not in {"standard", "chess"}:
        raise ValueError("Only standard chess is supported")
    board = game.board()
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError("Invalid or unsupported initial board")
    maximum_scaled_clock = int(np.finfo(np.float32).max) * 100
    if (
        max(board.halfmove_clock, board.fullmove_number) + config.max_game_plies
        > maximum_scaled_clock
    ):
        raise ValueError("FEN clocks cannot be represented as finite auxiliary features")
    initial_fen = board.fen(en_passant="fen")
    result = game.headers.get("Result", "*")
    if result not in {"1-0", "0-1", "1/2-1/2", "*"}:
        raise ValueError("Invalid outcome label")
    moves = []
    for move in game.mainline_moves():
        if len(moves) >= config.max_game_plies:
            raise ValueError("Game exceeds max_game_plies")
        if board.is_game_over() or move not in board.legal_moves:
            raise ValueError(f"Illegal move or move after automatic termination: {move}")
        moves.append(move.uci())
        board.push(move)
    if board.outcome() is not None and result not in {"*", board.result()}:
        raise ValueError("Outcome contradicts terminal position")
    color = None
    if config.player:
        matches = [
            name
            for name in ("White", "Black")
            if game.headers.get(name, "").casefold() == config.player.casefold()
        ]
        color = matches[0].lower() if len(matches) == 1 else "ambiguous" if matches else "absent"
    return GameRecord(
        dict(game.headers), initial_fen, moves, result, identity(initial_fen, moves), color
    )


def stream_games(
    path: Path, config: DataConfig
) -> Iterator[tuple[int, GameRecord | None, str | None]]:
    """WHAT: Yield valid/invalid games one at a time with ordinal and explicit reason.

    HOW: Parse from a bounded streaming reader, validate lexemes and replay before yield.
    WHY: Malformed games become auditable rejected occurrences, never partial training data.
    Fatal stream/limit errors propagate so the caller records the unread source remainder.
    """
    with open_pgn(path) as stream:
        ordinal = 0
        while True:
            reader = BoundedReader(stream, config)
            game = chess.pgn.read_game(
                reader, Visitor=partial(MainlineBuilder, max_plies=config.max_game_plies)
            )
            if game is None:
                break
            ordinal += 1
            try:
                check_lexical_text("".join(reader.lines))
                # PGN movetext and header results must agree, including '*' headers.
                # GameBuilder may fill absent results; capture explicit declarations.
                declared = re.findall(r'^\[Result\s+"([^"\n]+)"\]', "".join(reader.lines), re.M)
                if declared and declared[0] != game.headers.get("Result", "*"):
                    raise ValueError("Header and movetext outcome disagree")
                yield ordinal, record_game(game, config), None
            except ValueError as error:
                yield ordinal, None, str(error)


def safe_games(path: Path, config: DataConfig):
    """WHAT: Mark source failures. HOW: Catch parser/IO iteration. WHY: Never hide writer errors."""
    try:
        yield from stream_games(path, config)
    except (ValueError, OSError, EOFError, UnicodeError, zstandard.ZstdError) as error:
        yield -1, None, str(error)


def filter_reason(record: GameRecord, config: DataConfig) -> str | None:
    """WHAT: Filter whole games. HOW: Outcome/length/player rules. WHY: Reproducible selection.

    Filtering is separate from malformed-game rejection. Defaults preserve every valid
    game; filtered occurrences retain headers/provenance without emitting positions.
    """
    if len(record.moves) < config.min_game_plies:
        return "Game is shorter than min_game_plies"
    if record.result not in config.allowed_results:
        return "Game outcome excluded by allowed_results"
    if config.require_player and record.player_color not in {"white", "black"}:
        return "Configured player absent or ambiguous"
    return None
