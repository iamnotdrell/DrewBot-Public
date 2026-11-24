"""Atomic full-state checkpoints and weights-only inference loading."""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from drewbot.models import DrewBot
from drewbot.settings import ModelSettings

FORMAT = "drewbot-training-v1"


def rng_state(device: str) -> dict:
    """WHAT: Snapshot RNGs. HOW: Save Python/NumPy/torch/MPS. WHY: Continue dropout streams.

    NumPy state is converted to safe builtins/tensors so weights_only=True suffices;
    checkpoint loading never enables arbitrary pickle execution.
    """
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], torch.tensor(numpy[1].astype(np.int64)), numpy[2], numpy[3], numpy[4]],
        "torch": torch.get_rng_state(),
        "mps": torch.mps.get_rng_state().cpu() if device == "mps" else None,
    }


def restore_rng(state: dict, device: str):
    """WHAT: Restore RNG sequences. HOW: Reverse safe snapshot. WHY: Matched continuation."""
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if device == "mps" and state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])


def atomic_save(path: Path, payload: dict) -> int:
    """WHAT: Publish a complete checkpoint. HOW: Unique sibling, fsync, atomic replace.

    WHY: A killed writer leaves the previous checkpoint intact. Atomicity assumes a
    local filesystem. Parent fsync persists rename where supported; no remote guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def read_checkpoint(path: Path) -> dict:
    """WHAT: Read constrained training state. HOW: CPU weights-only deserializer. WHY: Safe API."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def load_model(path: Path, device: str = "cpu"):
    """WHAT: Independent inference model. HOW: Strict state load. WHY: No optimizer required."""
    state = read_checkpoint(path)
    model = DrewBot(ModelSettings(**state["model_config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state


def architecture_id(model) -> dict:
    """WHAT: Identify architecture. HOW: Versioned type/config. WHY: Reject accidental redesign."""
    return {"name": "drewbot-phase3-v1", "config": asdict(model.config)}
