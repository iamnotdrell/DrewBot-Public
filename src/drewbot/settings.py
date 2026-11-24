"""Validated settings for the model, search, and local benchmarks."""

from dataclasses import dataclass, field

from drewbot.search.config import SearchConfig
from drewbot.validation import finite, positive


@dataclass(frozen=True)
class HardwareSettings:
    cpu_threads: int = 1
    allow_cpu_fallback: bool = True
    warmup: int = 2
    iterations: int = 3


@dataclass(frozen=True)
class ModelSettings:
    """WHAT: Executable Phase 3 architecture contract. HOW: Validate dimensions and modes.

    WHY: Keep experiment shapes explicit without reinterpreting persisted v1 tensors.
    Defaults are DrewBot engineering choices.
    """

    implemented: bool = True
    phase: int = 3
    layers: int = 6
    board_tokens: int = 64
    history_frames: int = 8
    auxiliary_features: int = 72
    embedding_dim: int = 128
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.0
    gab_enabled: bool = True
    gab_d1: int = 8
    gab_d2: int = 32
    gab_templates: int = 32
    auxiliary_mode: str = "broadcast"
    value_hidden_dim: int = 128
    benchmark_batch_sizes: tuple[int, ...] = (1, 8)

    def __post_init__(self):
        if self.implemented is not True or type(self.phase) is not int or self.phase != 3:
            raise ValueError("Model settings describe implemented Phase 3 only")
        if (
            type(self.board_tokens) is not int
            or self.board_tokens != 64
            or type(self.auxiliary_features) is not int
            or self.auxiliary_features != 72
        ):
            raise ValueError("Model requires 64 board tokens and 72 auxiliary features")
        for name in (
            "layers",
            "history_frames",
            "embedding_dim",
            "attention_heads",
            "feedforward_dim",
            "gab_d1",
            "gab_d2",
            "gab_templates",
            "value_hidden_dim",
        ):
            positive(getattr(self, name), name)
        finite(self.dropout, "dropout")
        if self.dropout >= 1 or type(self.gab_enabled) is not bool:
            raise ValueError("dropout must be < 1 and gab_enabled must be boolean")
        if self.embedding_dim % self.attention_heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        if not isinstance(self.auxiliary_mode, str) or self.auxiliary_mode not in {
            "broadcast",
            "none",
        }:
            raise ValueError("auxiliary_mode must be broadcast or none (explicit ablation)")
        if (
            not isinstance(self.benchmark_batch_sizes, (list, tuple))
            or not self.benchmark_batch_sizes
        ):
            raise ValueError("benchmark_batch_sizes must be a nonempty list of positive integers")
        for size in self.benchmark_batch_sizes:
            positive(size, "benchmark_batch_sizes")
        object.__setattr__(self, "benchmark_batch_sizes", tuple(self.benchmark_batch_sizes))


SearchPlan = SearchConfig


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    hardware: HardwareSettings = field(default_factory=HardwareSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    search: SearchConfig = field(default_factory=SearchConfig)

    @classmethod
    def load(cls):
        """Return the public defaults."""
        return cls()
