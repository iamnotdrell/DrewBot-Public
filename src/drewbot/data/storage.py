"""Disk-backed registry and atomically published, versioned NumPy shards."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from drewbot.config import SPLITS, DataConfig
from drewbot.observability import digest_file, write_json

DATASET_VERSION = "drewbot-shards-v1"
ENCODING_VERSION = "chessformer-board-history-v1+aux72"
ARRAY_SCHEMA = {
    "tokens": (np.dtype("uint8"), (64, 96)),
    "auxiliary": (np.dtype("float32"), (72,)),
    "policy": (np.dtype("int64"), ()),
    "value": (np.dtype("int64"), ()),
    "legal": (np.dtype("uint8"), (544,)),
}


def check_versions(metadata: dict) -> None:
    """WHAT: Enforce stored versions. HOW: Compare both identifiers. WHY: Prevent mixed
    encodings."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset_version") != DATASET_VERSION
        or metadata.get("encoding_version") != ENCODING_VERSION
    ):
        raise ValueError("Unsupported dataset or encoding version; use a separate output")


def read_manifest(directory: Path) -> dict:
    """WHAT: Read a supported inventory. HOW: Validate split/count/version fields. WHY: Fail
    early."""
    manifest = json.loads((directory / "manifest.json").read_text())
    check_versions(manifest)
    if (
        not isinstance(manifest.get("shards"), list)
        or not isinstance(manifest.get("counts"), dict)
        or set(manifest["counts"]) != set(SPLITS)
    ):
        raise ValueError("Invalid manifest inventory or split counts")
    for shard in manifest["shards"]:
        check_versions(shard)
        if (
            shard.get("split") not in SPLITS
            or type(shard.get("positions")) is not int
            or shard["positions"] < 1
        ):
            raise ValueError("Invalid shard split or position count")
    for count in manifest["counts"].values():
        if (
            not isinstance(count, dict)
            or set(count) != {"games", "positions"}
            or any(type(v) is not int or v < 0 for v in count.values())
        ):
            raise ValueError("Invalid manifest game/position totals")
    return manifest


@contextmanager
def writer_lock(directory: Path):
    """WHAT: Exclude concurrent writers. HOW: OS advisory lock. WHY: Crash-safe lock release."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Dataset is already being written") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def connect(directory: Path, config: DataConfig | None = None) -> sqlite3.Connection:
    """WHAT: Open a bounded-cache registry. HOW: Indexed SQLite. WHY: Dedup scales on disk.

    Read-only callers pass no config. Mutable dataset settings are checked before any append.
    Schema creation and initial settings commit precede a caller-managed run transaction.
    """
    path = directory / "registry.sqlite3"
    if config is not None and (directory / "manifest.json").exists():
        read_manifest(directory)
    connection = sqlite3.connect(path if config else f"{path.as_uri()}?mode=ro", uri=config is None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA cache_size=-2048")
    if config is not None:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, path TEXT, bytes INTEGER);
            CREATE TABLE IF NOT EXISTS source_aliases (
                id TEXT NOT NULL, path TEXT NOT NULL, label TEXT NOT NULL,
                PRIMARY KEY(id, path, label)
            );
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY, split TEXT NOT NULL, result TEXT NOT NULL,
                record TEXT NOT NULL, plies TEXT NOT NULL, positions INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS occurrences (
                source_id TEXT NOT NULL, ordinal INTEGER NOT NULL, game_id TEXT,
                status TEXT NOT NULL, detail TEXT NOT NULL,
                PRIMARY KEY(source_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        """)
        expected = json.dumps(config.semantic(), sort_keys=True)
        stored = connection.execute("SELECT value FROM settings WHERE key='config'").fetchone()
        if (
            stored
            and json.dumps(DataConfig(**json.loads(stored[0])).semantic(), sort_keys=True)
            != expected
        ):
            connection.close()
            raise ValueError("Processing settings differ from existing dataset; use a new output")
        versions = {"dataset_version": DATASET_VERSION, "encoding_version": ENCODING_VERSION}
        try:
            for key, expected_version in versions.items():
                stored_version = connection.execute(
                    "SELECT value FROM settings WHERE key=?", (key,)
                ).fetchone()
                if stored_version and stored_version[0] != expected_version:
                    raise ValueError("Unsupported registry dataset or encoding version")
            # Legacy v1 registries predate explicit version keys. Accept only when
            # every already committed shard declares the current supported versions.
            for row in connection.execute("SELECT metadata FROM shards"):
                check_versions(json.loads(row[0]))
            for key, expected_version in versions.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings VALUES (?,?)", (key, expected_version)
                )
        except BaseException:
            connection.rollback()
            connection.close()
            raise
        # Original v1 configs predate additive no-op filtering defaults. Canonicalize
        # those defaults only after semantic equality was established above.
        connection.execute("INSERT OR REPLACE INTO settings VALUES ('config', ?)", (expected,))
        connection.commit()
    else:
        for key, expected_version in {
            "dataset_version": DATASET_VERSION,
            "encoding_version": ENCODING_VERSION,
        }.items():
            stored = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            if stored and stored[0] != expected_version:
                connection.close()
                raise ValueError("Unsupported registry dataset or encoding version")
    return connection


def publish_manifest(directory: Path, connection: sqlite3.Connection) -> dict:
    """WHAT: Publish inventory. HOW: Registry query and atomic JSON. WHY: Recoverable snapshots.

    Manifest RAM grows with shard count, not with games/positions. It is small at million-game
    scale with normal shard sizes. Uncommitted/orphan shard directories are never discovered.
    """
    shards = [
        dict(path=row[0], **json.loads(row[1]))
        for row in connection.execute("SELECT path,metadata FROM shards ORDER BY rowid")
    ]
    counts = {name: {"games": 0, "positions": 0} for name in SPLITS}
    for row in connection.execute(
        "SELECT split,COUNT(*),COALESCE(SUM(positions),0) FROM games GROUP BY split"
    ):
        counts[row[0]] = {"games": row[1], "positions": row[2]}
    manifest = {
        "dataset_version": DATASET_VERSION,
        "encoding_version": ENCODING_VERSION,
        "configuration": json.loads(
            connection.execute("SELECT value FROM settings WHERE key='config'").fetchone()[0]
        ),
        "counts": counts,
        "shards": shards,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def contained(directory: Path, relative: str) -> Path:
    """WHAT: Validate a stored path. HOW: Resolve and contain. WHY: Never load outside a dataset."""
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError("Shard path escapes dataset directory")
    return path


class ShardWriter:
    """WHAT: Buffer bounded samples and stage shards. HOW: One buffer per split, NPY + JSONL.

    WHY: Memory maps support later random access without decompression of entire shards.
    Buffer RAM is O(3*shard_positions*sample_bytes). Commit/publish belongs to the caller;
    all files have content checksums and are loaded with allow_pickle=False.
    """

    def __init__(self, directory: Path, config: DataConfig, connection: sqlite3.Connection):
        self.directory, self.config, self.connection = directory, config, connection
        self.run = uuid4().hex
        self.buffers = {split: [] for split in SPLITS}
        self.pending: list[tuple[Path, str, dict]] = []
        self.write_seconds = 0.0

    def add(self, sample: dict) -> None:
        """WHAT: Stage a sample. HOW: Flush at capacity. WHY: Fixed buffer memory."""
        split = sample["metadata"]["split"]
        self.buffers[split].append(sample)
        if len(self.buffers[split]) >= self.config.shard_positions:
            self.flush(split)

    def flush(self, split: str) -> None:
        """WHAT: Write a shard. HOW: Stack arrays and index JSONL. WHY: Fast loading."""
        samples = self.buffers[split]
        if not samples:
            return
        started = perf_counter()
        name = f"{self.run}-{len(self.pending):06d}"
        path = self.directory / "_staging" / name
        path.mkdir(parents=True)
        for key, (dtype, _) in ARRAY_SCHEMA.items():
            np.save(
                path / f"{key}.npy",
                np.asarray([sample[key] for sample in samples], dtype=dtype),
                allow_pickle=False,
            )
        offsets = [0]
        with (path / "samples.jsonl").open("wb") as stream:
            for sample in samples:
                raw = (json.dumps(sample["metadata"], sort_keys=True) + "\n").encode()
                stream.write(raw)
                offsets.append(offsets[-1] + len(raw))
        np.save(path / "offsets.npy", np.array(offsets, dtype=np.uint64), allow_pickle=False)
        files = {
            file.name: {"bytes": file.stat().st_size, "sha256": digest_file(file)}
            for file in sorted(path.iterdir())
        }
        metadata = {
            "dataset_version": DATASET_VERSION,
            "encoding_version": ENCODING_VERSION,
            "split": split,
            "positions": len(samples),
            "games": len({sample["metadata"]["game_id"] for sample in samples}),
            "sources": [self.config.source],
            "configuration": asdict(self.config),
            "files": files,
        }
        write_json(path / "metadata.json", metadata)
        self.pending.append((path, f"shards/{name}", metadata))
        self.buffers[split] = []
        self.write_seconds += perf_counter() - started

    def finish(self) -> None:
        """WHAT: Prepare files for commit. HOW: Unique renames. WHY: Publish complete shards."""
        for split in SPLITS:
            self.flush(split)
        (self.directory / "shards").mkdir(exist_ok=True)
        for staged, relative, metadata in self.pending:
            os.rename(staged, self.directory / relative)
            self.connection.execute(
                "INSERT INTO shards VALUES (?,?)", (relative, json.dumps(metadata, sort_keys=True))
            )
