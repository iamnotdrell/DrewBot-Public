"""Validated, serializable optimization settings for bounded local experiments."""

from dataclasses import dataclass

from drewbot.config import project_path
from drewbot.validation import finite, positive


@dataclass(frozen=True)
class TrainingConfig:
    """WHAT: Define an experiment. HOW: Validate immutable fields. WHY: Reject ambiguous runs.

    Defaults are a short pilot, not tuned research hyperparameters. Epoch/step limits
    count optimizer updates; validation is deterministic and never samples the test split.
    """

    dataset: str = "data/processed/dataset"
    objective: str = "joint"
    seed: int = 42
    device: str = "auto"
    allow_cpu_fallback: bool = True
    deterministic: bool = True
    cpu_threads: int = 1
    batch_size: int = 16
    workers: int = 0
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    policy_weight: float = 1.0
    value_weight: float = 1.0
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    epochs: int = 2
    max_steps: int | None = 8
    max_validation_batches: int | None = None
    warmup_steps: int = 2
    schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    patience: int = 5
    validation_every: int = 1
    checkpoint_every: int = 1
    tensorboard: bool = True
    implemented: bool = True
    phase: int = 4

    def __post_init__(self):
        if self.objective not in {"policy", "value", "joint"}:
            raise ValueError("objective must be policy, value or joint")
        if self.device not in {"auto", "cpu", "mps"} or self.schedule not in {"cosine", "constant"}:
            raise ValueError("Unknown device or schedule")
        for key in ("allow_cpu_fallback", "deterministic", "tensorboard", "implemented"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if not self.implemented or type(self.phase) is not int or self.phase != 4:
            raise ValueError("TrainingConfig implements Phase 4")
        for key in ("seed", "workers", "warmup_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.seed >= 2**63:
            raise ValueError("seed out of range")
        for key in (
            "cpu_threads",
            "batch_size",
            "accumulation_steps",
            "epochs",
            "patience",
            "validation_every",
            "checkpoint_every",
        ):
            positive(getattr(self, key), key)
        for key in ("max_steps", "max_validation_batches"):
            positive(getattr(self, key), key, optional=True)
        for key in ("learning_rate", "gradient_clip"):
            finite(getattr(self, key), key, minimum=1e-12)
        for key in ("weight_decay", "policy_weight", "value_weight", "min_lr_ratio"):
            finite(getattr(self, key), key)
        if self.min_lr_ratio > 1:
            raise ValueError("min_lr_ratio must be <=1")
        if (self.objective in {"policy", "joint"} and self.policy_weight == 0) or (
            self.objective in {"value", "joint"} and self.value_weight == 0
        ):
            raise ValueError("Active objectives require positive weights")
        if not isinstance(self.dataset, str):
            raise ValueError("dataset must be a relative path")
        project_path(self.dataset)
