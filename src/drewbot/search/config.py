"""Executable resource profiles for the authorized Phase 5 engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from drewbot.validation import finite, positive


@dataclass(frozen=True)
class SearchLimits:
    """WHAT: Bound a decision. HOW: Validate every capacity. WHY: Predictable local work.

    Time is cooperative, nodes count visited alpha-beta nodes, and cache capacities
    are entry counts. Zero cache entries disables reuse.
    """

    depth: int = 2
    time_budget_seconds: float = 1.0
    node_budget: int = 500
    inference_batch_size: int = 1
    cache_entries: int = 2048
    transposition_entries: int = 2048

    def __post_init__(self):
        for name in ("depth", "node_budget", "inference_batch_size"):
            positive(getattr(self, name), name)
        finite(self.time_budget_seconds, "time_budget_seconds", minimum=1e-9)
        if self.depth > 64 or self.inference_batch_size > 256:
            raise ValueError("depth must be <=64 and inference_batch_size <=256")
        for name in ("cache_entries", "transposition_entries"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 1_000_000:
                raise ValueError(f"{name} must be an integer in [0,1000000]")


@dataclass(frozen=True)
class SearchConfig:
    """WHAT: Central search configuration. HOW: Freeze three profiles. WHY: No inactive bypass."""

    implemented: bool = True
    phase: int = 5
    algorithm: str = "iterative_deepening_alpha_beta"
    modes: tuple[str, ...] = ("fast", "balanced", "deep")
    fast: SearchLimits = field(default_factory=SearchLimits)
    balanced: SearchLimits = field(
        default_factory=lambda: SearchLimits(3, 3.0, 3000, 4, 8192, 8192)
    )
    deep: SearchLimits = field(
        default_factory=lambda: SearchLimits(4, 10.0, 10000, 8, 32768, 32768)
    )

    def __post_init__(self):
        if (
            self.implemented is not True
            or type(self.phase) is not int
            or self.phase != 5
            or self.algorithm != "iterative_deepening_alpha_beta"
            or not isinstance(self.modes, (tuple, list))
            or tuple(self.modes) != ("fast", "balanced", "deep")
        ):
            raise ValueError("Invalid implemented Phase 5 search configuration")
        object.__setattr__(self, "modes", tuple(self.modes))
        for mode in self.modes:
            profile = getattr(self, mode)
            if isinstance(profile, dict):
                try:
                    profile = SearchLimits(**profile)
                except TypeError as error:
                    raise ValueError(f"Invalid {mode} profile: {error}") from error
            if not isinstance(profile, SearchLimits):
                raise ValueError(f"{mode} must be a SearchLimits profile")
            object.__setattr__(self, mode, profile)

    def limits(self, mode: str) -> SearchLimits:
        """WHAT: Select a profile. HOW: Exact names only. WHY: No silent mode fallback."""
        if mode not in self.modes:
            raise ValueError("mode must be fast, balanced or deep")
        return getattr(self, mode)
