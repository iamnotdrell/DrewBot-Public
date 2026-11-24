"""Command-line processing, inspection and full dataset validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from drewbot.config import DataConfig
from drewbot.data.dataset import ChessDataset
from drewbot.data.pipeline import process
from drewbot.data.storage import connect
from drewbot.data.validate import validate_dataset
from drewbot.hardware import inspect_hardware
from drewbot.observability import event


def main(argv=None) -> int:
    """WHAT: Data workflows. HOW: Subcommands and JSON output. WHY: Reproducible operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("process")
    ingest.add_argument("inputs", nargs="+", type=Path)
    ingest.add_argument("--config", type=Path)
    ingest.add_argument("--output")
    ingest.add_argument("--source")
    ingest.add_argument("--player")
    ingest.add_argument("--max-games", type=int)
    ingest.add_argument("--shard-positions", type=int)
    for name in ("validate", "inspect"):
        subparser = commands.add_parser(name)
        subparser.add_argument("directory", type=Path)
        if name == "inspect":
            subparser.add_argument("--index", type=int, default=0)
    commands.add_parser("hardware")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", default="experiments/phase2-benchmark")
    benchmark.add_argument("--games", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.command == "process":
            raw = asdict(DataConfig.load(args.config) if args.config else DataConfig())
            for key in ("output", "source", "player", "max_games", "shard_positions"):
                if getattr(args, key) is not None:
                    raw[key] = getattr(args, key)
            result = process(args.inputs, DataConfig(**raw))
        elif args.command == "validate":
            result = validate_dataset(args.directory)
        elif args.command == "inspect":
            dataset = ChessDataset(args.directory)
            try:
                if not len(dataset):
                    result = {"positions": 0}
                else:
                    sample = dataset[args.index]
                    meta = json.loads(sample["metadata"])
                    connection = connect(args.directory.resolve())
                    try:
                        game = connection.execute(
                            "SELECT record FROM games WHERE id=?", (meta["game_id"],)
                        ).fetchone()
                    finally:
                        connection.close()
                    result = {
                        "positions": len(dataset),
                        "sample": meta,
                        "game": json.loads(game[0]),
                        "tensors": {
                            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for key, value in sample.items()
                            if key != "metadata"
                        },
                    }
            finally:
                dataset.close()
        elif args.command == "hardware":
            result = inspect_hardware()
        else:
            from drewbot.data.benchmark import benchmark

            result = benchmark(args.output, args.games)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 2 if result.get("source_errors") else 0
    except (ValueError, OSError, KeyError) as error:
        event("command_failed", reason=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
