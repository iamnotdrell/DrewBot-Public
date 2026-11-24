"""Local experiment records and TensorBoard preparation without a training loop."""

from __future__ import annotations

import json
import math
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from drewbot.config import ROOT, project_path
from drewbot.observability import event, write_json

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


def seed_everything(seed: int) -> None:
    """WHAT: Repeat local RNG sequences. HOW: Seed Python/NumPy/torch. WHY: Reproducible
    experiments.

    This seeds generators, not a guarantee of deterministic future MPS kernels. A
    model milestone must test its own operations and document deterministic support.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class ExperimentLogger:
    """WHAT: Identified JSONL/console runs and optional TensorBoard scalar events.

    HOW: Create an exclusive run directory, snapshot configuration, filter verbosity,
    and write finite metric records with step/time/run ID. WHY: Prepare future
    experiments with real auditable files, without implementing optimization.
    """

    def __init__(
        self,
        directory: str = "experiments",
        experiment_id: str | None = None,
        level: str = "INFO",
        tensorboard: bool = False,
        configuration: dict | None = None,
        root: Path = ROOT,
    ):
        self.experiment_id = experiment_id or uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.experiment_id):
            raise ValueError("experiment_id must be a short filename-safe identifier")
        if level not in LEVELS or type(tensorboard) is not bool:
            raise ValueError("Invalid log verbosity or TensorBoard toggle")
        self.level = LEVELS[level]
        self.directory = project_path(directory, root) / self.experiment_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.writer = None
        self.stream = (self.directory / "events.jsonl").open("w")
        try:
            write_json(self.directory / "configuration.json", configuration or {})
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.directory / "tensorboard"))
        except BaseException:
            self.stream.close()
            raise

    def log(self, event_name: str, *, level: str = "INFO", **fields) -> None:
        """WHAT: Log an event. HOW: Filter severity and serialize. WHY: One JSON record per
        event."""
        if level not in LEVELS:
            raise ValueError("Unknown log level")
        record = {
            **fields,
            "event": event_name,
            "experiment_id": self.experiment_id,
            "level": level,
            "time": datetime.now(UTC).isoformat(),
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        self.stream.write(encoded + "\n")
        self.stream.flush()
        if LEVELS[level] >= self.level:
            event(event_name, experiment_id=self.experiment_id, level=level, **fields)

    def metric(self, name: str, value: float, step: int = 0) -> None:
        """WHAT: Record a finite scalar. HOW: JSONL plus TensorBoard. WHY: Comparable local
        evidence."""
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or type(step) is not int
            or step < 0
        ):
            raise ValueError("Metrics require a finite scalar and nonnegative integer step")
        self.log("metric", name=name, value=value, step=step)
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        """WHAT: Flush and release run files. HOW: Close writers. WHY: Visible complete
        artifacts."""
        if self.writer is not None:
            self.writer.close()
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
