"""Short paired search measurements, separate from model quality and chess strength."""

import csv
from dataclasses import asdict
from statistics import mean

import chess

from drewbot.search.config import SearchLimits
from drewbot.search.engine import SearchEngine
from drewbot.search.state import SearchState

POSITIONS = {
    "opening": chess.STARTING_FEN,
    "tactic": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
    "endgame": "8/8/7k/8/8/8/P6P/4K3 w - - 0 1",
}


def benchmark_search(evaluator, config, repeats=3) -> dict:
    """WHAT: Compare bounded batching/cache costs on one fixed evaluator/device.

    HOW: Warm two model calls, then repeat each position with batch1/cache-off,
    batch1/cache-on and batch4/cache-on. Each decision resets caches. Record all
    raw results, limits and paired comparable-depth summaries. WHY: Reduced neural
    calls alone is not a latency benefit; speculative leaves may cost more than saved
    dispatch. No model loading, training or playing-strength test is timed.
    """
    warm = SearchState(chess.Board(), evaluator.history_frames).snapshot()
    for _ in range(2):
        evaluator.evaluate([warm])
    samples = []
    for case, batch, cache in (("uncached", 1, 0), ("cached", 1, 2048), ("batched", 4, 2048)):
        limits = SearchLimits(2, 2.0, 256, batch, cache, cache)
        for name, fen in POSITIONS.items():
            for repeat in range(repeats):
                result = SearchEngine(evaluator).search(chess.Board(fen), limits)
                samples.append(
                    {
                        "case": case,
                        "position": name,
                        "repeat": repeat,
                        "limits": asdict(limits),
                        **result.to_dict(),
                    }
                )
    profiles = []
    # A small legal endgame exercises each actual configured profile and its budgets.
    for mode in config.modes:
        limit = config.limits(mode)
        result = SearchEngine(evaluator).search(chess.Board(POSITIONS["endgame"]), limit)
        profiles.append({"mode": mode, "limits": asdict(limit), **result.to_dict()})
    summaries = []
    for name in POSITIONS:
        for case in ("uncached", "cached", "batched"):
            selected = [s for s in samples if s["position"] == name and s["case"] == case]
            summaries.append(
                {
                    "position": name,
                    "case": case,
                    **{
                        key: mean(s["metrics"][key] for s in selected)
                        for key in (
                            "nodes",
                            "nodes_per_second",
                            "completed_depth",
                            "cutoff_rate",
                            "inference_calls",
                            "evaluated_positions",
                            "inference_seconds",
                            "evaluation_latency_ms",
                            "total_seconds",
                        )
                    },
                    "scores": [s["score"] for s in selected],
                    "stop_reasons": [s["stop_reason"] for s in selected],
                }
            )
    return {
        "methodology": "Fixed six-layer evaluator; two inference warmups; three repeats; "
        "fresh caches per decision; synchronized device; decision includes replay/encoding/rules. "
        "Speculative inference positions differ from visited nodes. Scores are uncalibrated.",
        "positions": POSITIONS,
        "samples": samples,
        "summaries": summaries,
        "profiles": profiles,
    }


def write_benchmark_csv(path, samples):
    """WHAT: Export raw matching rows. HOW: Flatten metrics. WHY: Independent artifact readback."""
    rows = [
        {
            "case": s["case"],
            "position": s["position"],
            "repeat": s["repeat"],
            "move": s["move"],
            "score": s["score"],
            "stop_reason": s["stop_reason"],
            **s["metrics"],
        }
        for s in samples
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
