"""Audited Phase 2 data, deterministic worker seeds and lazy epoch permutations."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from drewbot.config import ROOT, project_path
from drewbot.data.dataset import ChessDataset
from drewbot.data.storage import connect, read_manifest
from drewbot.data.validate import validate_dataset
from drewbot.observability import digest_file
from drewbot.training.config import TrainingConfig


def audit_dataset(directory) -> dict:
    """WHAT: Establish provenance/isolation. HOW: Full Phase 2 replay plus registry audit.

    WHY: Validate historical orientation, labels, legal masks, dtypes and duplicate
    trajectories before optimization. RAM is bounded by Phase 2's game/shard inventory.
    Dataset identity hashes the manifest (including shard hashes), not just schema v1.
    All accepted trajectories have a single primary key and a deterministic game split.
    Shared opening positions may occur in distinct games; this is documented, not removed.
    """
    validation = validate_dataset(directory)
    manifest = read_manifest(directory)
    connection = connect(directory)
    try:
        occurrences = dict(
            connection.execute("SELECT status,COUNT(*) FROM occurrences GROUP BY status")
        )
        splits = {}
        classes = {}
        for split in ("train", "validation", "test"):
            digest = hashlib.sha256()
            for row in connection.execute(
                "SELECT id FROM games WHERE split=? ORDER BY id", (split,)
            ):
                digest.update(row[0].encode())
            splits[split] = digest.hexdigest()
            counts = [0, 0, 0]
            dataset = ChessDataset(directory, split)
            try:
                for i in range(len(dataset)):
                    value = int(dataset.raw(i)["value"])
                    if value != -100:
                        counts[value] += 1
            finally:
                dataset.close()
            classes[split] = counts
        return {
            "validation": validation,
            "dataset_version": manifest["dataset_version"],
            "encoding_version": manifest["encoding_version"],
            "fingerprint": digest_file(directory / "manifest.json"),
            "split_identifiers": splits,
            "class_counts_loss_draw_win": classes,
            "occurrences": occurrences,
            "preprocessing": manifest["configuration"],
        }
    finally:
        connection.close()


def seed_worker(worker_id: int) -> None:
    """WHAT: Seed worker libraries. HOW: Derive from torch initial_seed. WHY: Spawn parity."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class EpochSampler(Sampler):
    """WHAT: Permute positions without an O(N) index list. HOW: Keyed Feistel permutation.

    WHY: Streaming-compatible mmap access and exact restart from batch offsets. Six
    balanced Feistel rounds permute a power-of-four domain; cycle walking restricts
    it to [0,N). This is deterministic sampling, not a cryptographic primitive.
    Unlike a coprime stride, it avoids fixed adjacent-index correlations. O(1) RAM.
    """

    def __init__(self, size: int, seed: int, epoch: int, offset: int = 0):
        self.size, self.seed, self.epoch, self.offset = size, seed, epoch, offset
        self.bits = max(1, ((max(1, size) - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.bits) - 1
        self.keys = [
            int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{r}".encode()).digest()[:8], "little")
            for r in range(6)
        ]

    def permutation(self, index: int) -> int:
        """WHAT: Map one index bijectively. HOW: Feistel + cycle walk. WHY: Lazy shuffle."""
        while True:
            left, right = index >> self.bits, index & self.mask
            for key in self.keys:
                mixed = ((right ^ key) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                mixed ^= mixed >> 29
                left, right = right, (left ^ mixed) & self.mask
            index = (left << self.bits) | right
            if index < self.size:
                return index

    def __iter__(self):
        return (self.permutation(i) for i in range(self.offset, self.size))

    def __len__(self):
        return max(0, self.size - self.offset)


def make_loader(config: TrainingConfig, split: str, *, epoch=0, offset=0, root=ROOT):
    """WHAT: Load existing shards lazily. HOW: Worker-local mmap, lazy sampler, fixed seeds.

    WHY: Do not materialize a corpus or alter the schema. An epoch-specific generator
    isolates worker seeding from model dropout RNG, including during resumed epochs.
    No random transforms are applied; preprocessing has no fitted statistics.
    """
    dataset = ChessDataset(project_path(config.dataset, root), split)
    sampler = EpochSampler(len(dataset), config.seed, epoch, offset) if split == "train" else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed + epoch),
        multiprocessing_context="spawn" if config.workers else None,
    )


def to_device(batch: dict, device: str) -> dict:
    """WHAT: Place tensors together. HOW: Preserve metadata on CPU. WHY: No label inputs."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
