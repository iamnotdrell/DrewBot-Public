"""Short, isolated synthetic preprocessing benchmarks with JSON/CSV export."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import resource
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, process_time

import chess
import chess.pgn
import torch
from torch.utils.data import DataLoader

from drewbot.config import ROOT, DataConfig, project_path
from drewbot.data.dataset import ChessDataset
from drewbot.data.pipeline import process
from drewbot.data.validate import validate_dataset
from drewbot.data.walkthrough import encode_history
from drewbot.hardware import inspect_hardware
from drewbot.observability import write_json


def synthetic_pgn(path: Path, games: int, plies: int = 12) -> None:
    """WHAT: Small synthetic fixtures. HOW: Seeded legal moves. WHY: Run without a game archive.

    Recorded nonterminal results here are synthetic labels, not predicted evaluations.
    Each game uses its own seed and is explicitly named synthetic in PGN metadata.
    """
    if games < 1 or plies < 1:
        raise ValueError("Synthetic game/ply counts must be positive")
    with path.open("w") as stream:
        for index in range(games):
            rng = random.Random(index)
            board = chess.Board()
            for _ in range(plies):
                if board.is_game_over():
                    break
                board.push(rng.choice(sorted(board.legal_moves, key=lambda move: move.uci())))
            game = chess.pgn.Game.from_board(board)
            game.headers.update(
                Event="Synthetic benchmark",
                White=f"Synthetic W {index}",
                Black=f"Synthetic B {index}",
            )
            game.headers["Result"] = (
                board.result()
                if board.is_game_over()
                else ("1-0", "0-1", "1/2-1/2", "*")[index % 4]
            )
            print(game, file=stream, end="\n\n")


def tensor_baseline(
    hardware: dict, seed: int = 42, size: int = 128, warmup: int = 2, iterations: int = 3
) -> list[dict]:
    """WHAT: Foundation tensor smoke benchmark. HOW: Warm up, synchronize, time forward/backward.

    WHY: Verify actual device execution without any chess model or training loop.
    Shape (128,128), float32, seed 42, two warmups and three timed repetitions.
    MPS has no supported peak-allocated API here; do not mislabel current usage as peak.
    """
    if any(type(value) is not int or value < 1 for value in (size, warmup, iterations)):
        raise ValueError("Tensor benchmark size/warmup/iterations must be positive integers")
    results = []
    for device in ("cpu", "mps"):
        if not hardware[device]["operational"]:
            results.append({"device": device, "skipped": hardware[device]["error"]})
            continue
        torch.manual_seed(seed)
        x = torch.randn(size, size, device=device, requires_grad=True)
        weight = torch.randn(size, size, device=device)

        def synchronize(device=device):
            if device == "mps":
                torch.mps.synchronize()

        for operation in ("matmul", "forward", "backward"):

            def execute(x=x, weight=weight, operation=operation):
                x.grad = None
                output = x @ weight
                if operation != "matmul":
                    output = torch.nn.functional.gelu(output).square().mean()
                if operation == "backward":
                    output.backward()

            for _ in range(warmup):
                execute()
            synchronize()
            wall, cpu = perf_counter(), process_time()
            for _ in range(iterations):
                execute()
            synchronize()
            elapsed, cpu_elapsed = perf_counter() - wall, process_time() - cpu
            results.append(
                {
                    "device": device,
                    "operation": operation,
                    "latency_seconds": elapsed / iterations,
                    "cpu_seconds": cpu_elapsed / iterations,
                    "matmul_flops_per_second": 2 * size**3 * iterations / elapsed
                    if operation == "matmul"
                    else None,
                    "peak_allocated_mps_bytes": None,
                }
            )
    return results


def worker(source: Path, output: str, shard_positions: int) -> dict:
    """WHAT: Measure one fresh process. HOW: Time real work. WHY: Comparable RSS peaks."""
    torch.set_num_threads(1)
    hardware = inspect_hardware()
    baseline = tensor_baseline(hardware)
    config = DataConfig(
        output=output,
        shard_positions=shard_positions,
        max_positions_per_game=None,
        source="synthetic-benchmark",
        progress_every=1000,
    )
    for _ in range(8):
        encode_history([chess.Board()])
    stats = process([source], config)
    directory = project_path(output)
    validation_started = perf_counter()
    validation = validate_dataset(directory)
    validation_seconds = perf_counter() - validation_started
    dataset = ChessDataset(directory)
    try:
        # Warm up conversion/loading, then measure the entire small dataset once.
        if len(dataset):
            dataset[0]
        load_started = perf_counter()
        loaded = sum(batch["tokens"].shape[0] for batch in DataLoader(dataset, batch_size=64))
        loading_seconds = perf_counter() - load_started
    finally:
        dataset.close()
    size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    positions = stats["positions"]
    return {
        "synthetic": True,
        "shard_positions": shard_positions,
        "games": stats["games"],
        "positions": positions,
        "processing_seconds": stats["seconds"],
        "games_per_second": stats["games"] / stats["seconds"],
        "positions_per_second": positions / stats["seconds"],
        "encoding_latency_us": 1e6 * stats["encoding_seconds"] / positions,
        "peak_process_rss_bytes": int(peak_rss),
        "processed_bytes": size,
        "bytes_per_position": size / positions,
        "shard_write_seconds": stats["shard_write_seconds"],
        "shard_writing_positions_per_second": positions / stats["shard_write_seconds"],
        "loading_seconds": loading_seconds,
        "loading_positions_per_second": loaded / loading_seconds,
        "validation_seconds": validation_seconds,
        "validated": validation["valid"],
        "configuration": asdict(config),
        "hardware": hardware,
        "tensor_baseline": baseline,
    }


def benchmark(output: str, games: int = 64) -> dict:
    """WHAT: Compare shard capacities. HOW: Isolated workers, JSON/CSV. WHY: Reproducible evidence.

    Synthetic-only bounded run; refuse to overwrite prior benchmark artifacts. Reported
    throughput includes parsing/validation/dedup/encoding/writing/publication but excludes
    the separate semantic validator. Loading includes tensor conversion and collation.
    """
    directory = project_path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "synthetic.pgn"
    synthetic_pgn(source, games)
    results = []
    for capacity in (32, 128):
        dataset_output = str((directory / f"shards-{capacity}").relative_to(ROOT))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "drewbot.data.benchmark",
                "--worker",
                str(source),
                "--output",
                dataset_output,
                "--shard-positions",
                str(capacity),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(completed.stdout))
    report = {
        "benchmark_version": 1,
        "synthetic": True,
        "games_requested": games,
        "configurations": results,
    }
    write_json(directory / "metrics.json", report)
    with (directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in result.items()
                }
            )
    return {
        "json": str(directory / "metrics.json"),
        "csv": str(directory / "metrics.csv"),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-positions", type=int, required=True)
    arguments = parser.parse_args()
    print(json.dumps(worker(arguments.worker, arguments.output, arguments.shard_positions)))
