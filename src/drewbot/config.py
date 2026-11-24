"""Central, validated configuration and repository-relative output paths."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")


def project_path(relative: str, root: Path = ROOT) -> Path:
    """WHAT: Contain outputs in the project. HOW: Resolve symlinks/parents. WHY: Avoid escapes."""
    path = Path(relative)
    resolved = (root / path).resolve()
    if (
        path.is_absolute()
        or not resolved.is_relative_to(root.resolve())
        or resolved == root.resolve()
    ):
        raise ValueError("Output must be a non-root, relative path contained in the project")
    return resolved


@dataclass(frozen=True)
class DataConfig:
    """WHAT: Reproducible pipeline settings. HOW: Reject invalid types/ranges at construction.

    WHY: Immutable processing settings prevent incremental schema/sampling drift.
    Game limits and progress cadence may change between invocations, not dataset semantics.
    """

    output: str = "data/processed/dataset"
    seed: int = 42
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    shard_positions: int = 2048
    max_games: int | None = None
    min_ply: int = 0
    sample_stride: int = 1
    max_positions_per_game: int | None = 32
    max_game_plies: int = 4096
    max_game_chars: int = 2_097_152
    max_line_chars: int = 262_144
    progress_every: int = 100
    player: str | None = None
    source: str = "user-supplied"
    min_game_plies: int = 0
    allowed_results: tuple[str, ...] = ("1-0", "0-1", "1/2-1/2", "*")
    require_player: bool = False

    def __post_init__(self) -> None:
        for key in (
            "seed",
            "min_ply",
            "shard_positions",
            "sample_stride",
            "max_game_plies",
            "max_game_chars",
            "max_line_chars",
            "progress_every",
            "min_game_plies",
        ):
            value = getattr(self, key)
            minimum = 0 if key in {"seed", "min_ply", "min_game_plies"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("max_games", "max_positions_per_game"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be null or a positive integer")
        if (
            not isinstance(self.split, (list, tuple))
            or len(self.split) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in self.split)
            or not math.isclose(sum(self.split), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("split must contain three nonnegative finite weights summing to 1")
        object.__setattr__(self, "split", tuple(self.split))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty label")
        if self.player is not None and (
            not isinstance(self.player, str) or not self.player.strip()
        ):
            raise ValueError("player must be null or a nonempty name")
        if (
            not isinstance(self.allowed_results, (list, tuple))
            or not self.allowed_results
            or any(
                not isinstance(result, str) or result not in {"1-0", "0-1", "1/2-1/2", "*"}
                for result in self.allowed_results
            )
        ):
            raise ValueError("allowed_results must contain known PGN outcomes")
        object.__setattr__(self, "allowed_results", tuple(self.allowed_results))
        if type(self.require_player) is not bool or self.require_player and self.player is None:
            raise ValueError("require_player must be boolean and needs a configured player name")
        if self.min_game_plies > self.max_game_plies:
            raise ValueError("min_game_plies cannot exceed max_game_plies")
        if not isinstance(self.output, str):
            raise ValueError("output must be a relative string path")
        project_path(self.output)

    @classmethod
    def load(cls, path: Path) -> DataConfig:
        """WHAT: Read JSON settings. HOW: Validate fields. WHY: No hidden defaults."""
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Data configuration must be a JSON object")
        try:
            return cls(**raw)
        except TypeError as error:
            raise ValueError(f"Invalid data configuration fields: {error}") from error

    def semantic(self) -> dict:
        """WHAT: Dataset identity settings. HOW: Remove run-local knobs. WHY: Safe append checks."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"output", "max_games", "progress_every", "source"}
        }


def split_for(game_id: str, config: DataConfig) -> str:
    """WHAT: Stable game-level split. HOW: Hash into cumulative weights. WHY: No position leakage.

    Assignment uses U in [0,1); split sizes converge statistically, not exact small counts.
    O(1) RAM/time per fixed-size identity, independent of corpus size and input order.
    """
    value = int.from_bytes(hashlib.sha256(f"{config.seed}:{game_id}".encode()).digest()[:8], "big")
    value /= 2**64
    if value < config.split[0]:
        return "train"
    return "validation" if value < sum(config.split[:2]) else "test"
