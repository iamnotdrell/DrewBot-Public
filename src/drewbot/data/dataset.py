"""Storage-independent tensor contract over bounded, worker-local shard access."""

from __future__ import annotations

import bisect
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from drewbot.data.storage import contained, read_manifest


class ShardReader:
    """WHAT: Access one mmap-backed shard. HOW: NPY mappings and indexed JSONL. WHY: Bounded reads.

    Metadata offsets are uint64 (N+1,). Files are trusted only after dataset validation;
    np.load forbids pickle. Individual reads copy arrays before exposing writable tensors.
    """

    def __init__(self, path: Path):
        self.arrays = {
            name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("tokens", "auxiliary", "policy", "value", "legal", "offsets")
        }
        self.metadata = (path / "samples.jsonl").open("rb")

    def row(self, index: int) -> dict:
        """WHAT: Fetch arrays and metadata. HOW: Seek byte offsets. WHY: No JSONL full scan."""
        start, end = map(int, self.arrays["offsets"][index : index + 2])
        self.metadata.seek(start)
        return {
            **{
                name: value[index].copy()
                for name, value in self.arrays.items()
                if name != "offsets"
            },
            "metadata": json.loads(self.metadata.read(end - start)),
        }

    def close(self) -> None:
        """WHAT: Release descriptors. HOW: Close mmap/file handles. WHY: Bounded worker cache."""
        self.metadata.close()
        for array in self.arrays.values():
            array._mmap.close()


class ChessDataset(Dataset):
    """WHAT: Map-style PyTorch dataset for chess samples with identical fields.

    HOW: Binary-search a manifest's shard lengths and use an LRU of worker-local readers.
    WHY: Random access and DataLoader multiprocessing without loading corpus-sized arrays.
    Returns float32 tokens (64,96), auxiliary (72,), int64 class scalars, bool legal
    mask (4352,) and value_valid, plus collatable JSON metadata. O(log(shards)) lookup.
    """

    def __init__(self, directory: Path, split: str | None = None, cache_shards: int = 2):
        if cache_shards < 1:
            raise ValueError("cache_shards must be positive")
        self.directory = directory.resolve()
        manifest = read_manifest(self.directory)
        if split not in {None, "train", "validation", "test"}:
            raise ValueError("Unknown split")
        self.shards = [
            shard for shard in manifest["shards"] if split is None or shard["split"] == split
        ]
        self.ends = np.cumsum(
            [shard["positions"] for shard in self.shards], dtype=np.int64
        ).tolist()
        self.cache_shards = cache_shards
        self._readers: OrderedDict[int, ShardReader] = OrderedDict()
        self._pid = os.getpid()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def raw(self, index: int) -> dict:
        """WHAT: Read a row. HOW: Per-process LRU. WHY: Separate storage and tensor layers."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard_index = bisect.bisect_right(self.ends, index)
        if shard_index not in self._readers:
            self._readers[shard_index] = ShardReader(
                contained(self.directory, self.shards[shard_index]["path"])
            )
            if len(self._readers) > self.cache_shards:
                _, oldest = self._readers.popitem(last=False)
                oldest.close()
        self._readers.move_to_end(shard_index)
        start = self.ends[shard_index - 1] if shard_index else 0
        return self._readers[shard_index].row(index - start)

    def __getitem__(self, index: int) -> dict:
        row = self.raw(index)
        return {
            "tokens": torch.from_numpy(row["tokens"]).to(torch.float32),
            "auxiliary": torch.from_numpy(row["auxiliary"]),
            "policy": torch.tensor(row["policy"], dtype=torch.int64),
            "value": torch.tensor(row["value"], dtype=torch.int64),
            "value_valid": torch.tensor(row["value"] != -100, dtype=torch.bool),
            "legal_mask": torch.from_numpy(
                np.unpackbits(row["legal"], bitorder="little").astype(bool)
            ),
            "metadata": json.dumps(row["metadata"], sort_keys=True),
        }

    def close(self) -> None:
        """WHAT: Release caches. HOW: Close every reader. WHY: Predictable resources."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self):
        return {**self.__dict__, "_readers": OrderedDict()}
