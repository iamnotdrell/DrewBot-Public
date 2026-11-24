"""Small structured logging and file-integrity utilities shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def event(event_name: str, **fields) -> None:
    """WHAT: JSON stderr record. HOW: Timestamp fields. WHY: Auditable failures."""
    print(
        json.dumps(
            {"time": datetime.now(UTC).isoformat(), "event": event_name, **fields},
            sort_keys=True,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def digest_file(path: Path) -> str:
    """WHAT: Stream SHA256. HOW: Read 1 MiB blocks. WHY: Bounded integrity checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """WHAT: Publish JSON atomically. HOW: Sibling temporary + replace. WHY: No partial manifest."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
