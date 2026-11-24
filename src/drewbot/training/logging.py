"""Durable run metadata, JSONL, tidy CSV and TensorBoard exports."""

import csv
import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from uuid import uuid4

from drewbot.config import ROOT, project_path
from drewbot.experiments import ExperimentLogger
from drewbot.observability import write_json


def software_provenance(root=ROOT):
    """WHAT: Record software/source identity. HOW: Versions and Git. WHY: Reproduce dirty trees.

    A commit alone does not identify uncommitted work: record dirty status explicitly.
    Source file hashes are additionally saved by Trainer, including untracked Python.
    """

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "python-chess", "tensorboard", "psutil")
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
    }


class RunLogger:
    """WHAT: Persist progress during training. HOW: Reuse foundation logger plus CSV/JSON.

    WHY: Metrics survive interrupted runs. Resume creates a new run linked to its
    parent; existing logs remain immutable and steps retain checkpoint numbering.
    Scalar CSV is tidy: step, phase, metric, value. Arrays stay in structured JSONL.
    """

    def __init__(self, metadata, tensorboard=True, root=ROOT):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.id = f"phase4-{stamp}-{uuid4().hex[:8]}"
        self.logger = ExperimentLogger(
            experiment_id=self.id,
            tensorboard=tensorboard,
            configuration=metadata,
            level="WARNING",
            root=root,
        )
        self.directory = self.logger.directory
        self.csv_stream = (self.directory / "metrics.csv").open("w", newline="")
        self.csv = csv.writer(self.csv_stream)
        self.csv.writerow(["step", "phase", "metric", "value"])
        self.csv_stream.flush()
        self.checkpoint_directory = project_path("checkpoints", root) / self.id

    def record(self, phase, metrics, step):
        """WHAT: Save each progress record. HOW: Flush JSON/CSV/TB. WHY: Live local monitoring."""
        self.logger.log("progress", phase=phase, step=step, metrics=metrics)
        for name, value in metrics.items():
            if type(value) in (float, int):
                self.csv.writerow([step, phase, name, value])
                self.logger.metric(f"{phase}/{name}", value, step)
        self.csv_stream.flush()
        if self.logger.writer:
            self.logger.writer.flush()

    def finish(self, summary):
        """WHAT: Publish final summary. HOW: Atomic JSON. WHY: Easy comparison tooling."""
        write_json(self.directory / "summary.json", summary)

    def close(self):
        """WHAT: Release writers. HOW: Close every stream. WHY: Flush final events."""
        self.csv_stream.close()
        self.logger.close()
