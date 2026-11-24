"""Local Phase 5 search, short benchmarks and validation-only calibration diagnostics."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import chess
import torch

from drewbot.config import ROOT, project_path
from drewbot.hardware import inspect_hardware
from drewbot.observability import write_json
from drewbot.search.benchmark import benchmark_search, write_benchmark_csv
from drewbot.search.calibration import diagnose_validation
from drewbot.search.engine import SearchEngine
from drewbot.search.evaluator import Evaluation, NeuralEvaluator
from drewbot.search.tactical import TacticalSearch
from drewbot.settings import Settings
from drewbot.training.checkpoints import load_model


class UniformPolicy:
    """Give the handcrafted tactical search equal root priors."""

    history_frames = 1

    def evaluate(self, positions):
        evaluations = []
        for position in positions:
            moves = list(position.board.legal_moves)
            policy = dict.fromkeys(moves, 1 / len(moves)) if moves else {}
            evaluations.append(Evaluation(0.0, policy))
        return evaluations


def main(argv=None):
    """WHAT: Run search tools. HOW: Validated central settings. WHY: Reproducible use."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "benchmark", "calibrate"):
        command = sub.add_parser(name)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--checkpoint", type=Path)
        if name != "calibrate":
            source.add_argument(
                "--random-model",
                action="store_true",
                help="Explicit untrained mechanics demonstration",
            )
        command.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
        command.add_argument("--output", required=name != "search")
        if name == "search":
            source.add_argument(
                "--baseline", action="store_true",
                help="Use tactical search with handcrafted evaluation and a uniform root policy",
            )
            command.add_argument(
                "--tactical", action="store_true", help="Use opt-in tactical search"
            )
            command.add_argument(
                "--tactical-prior-cp",
                type=int,
                default=35,
                help="Maximum root-policy penalty for tactical search",
            )
            command.add_argument("--fen", default=chess.STARTING_FEN)
            command.add_argument("--moves", nargs="*", default=[])
            command.add_argument("--mode", choices=("fast", "balanced", "deep"), default="fast")
            for flag, kind in (
                ("depth", int),
                ("time-budget-seconds", float),
                ("node-budget", int),
                ("inference-batch-size", int),
                ("cache-entries", int),
                ("transposition-entries", int),
            ):
                command.add_argument("--" + flag, type=kind)
    args = parser.parse_args(argv)
    try:
        settings = Settings.load()
        destination = project_path(args.output) if args.output else None
        if destination is not None and destination.exists():
            raise ValueError("Output directory already exists; choose a fresh path")
        torch.set_num_threads(settings.hardware.cpu_threads)
        torch.manual_seed(settings.seed)
        hardware = inspect_hardware(args.device, settings.hardware.allow_cpu_fallback)
        device = hardware["selected_device"]
        baseline = getattr(args, "baseline", False)
        if baseline:
            network, state = None, None
            evaluator = UniformPolicy()
            identity = {"evaluation": "handcrafted tactical baseline", "root_policy": "uniform"}
        elif args.checkpoint:
            network, state = load_model(args.checkpoint, device)
            identity = {
                "checkpoint": str(args.checkpoint),
                "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            }
        else:
            from drewbot.models import DrewBot

            network, state = DrewBot(settings.model).to(device).eval(), None
            identity = {
                "checkpoint": None,
                "initialization": "explicit random mechanics demo",
                "seed": settings.seed,
            }
        if network is not None:
            evaluator = NeuralEvaluator(network)
        source_hashes = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src").rglob("*.py"))
        }
        common = {
            "model": identity,
            "model_config": asdict(network.config) if network is not None else None,
            "hardware": hardware,
            "source_sha256": source_hashes,
            "search_config": asdict(settings.search),
        }
        if args.command == "search":
            limits = settings.search.limits(args.mode)
            overrides = {
                key: getattr(args, key) for key in asdict(limits) if getattr(args, key) is not None
            }
            limits = replace(limits, **overrides)
            board = chess.Board(args.fen)
            for uci in args.moves:
                if board.is_game_over():
                    raise ValueError("Move supplied after automatic terminal")
                board.push_uci(uci)
            use_tactical = baseline or args.tactical
            prior_cp = 0 if baseline else args.tactical_prior_cp
            engine = (
                TacticalSearch(evaluator, prior_cp=prior_cp)
                if use_tactical
                else SearchEngine(evaluator)
            )
            result = {
                **common,
                "mode": args.mode,
                **(
                    {"selector": "tactical", "tactical_prior_cp": prior_cp}
                    if use_tactical
                    else {}
                ),
                "limits": asdict(limits),
                "initial_fen": args.fen,
                "moves": args.moves,
                **engine.search(board, limits).to_dict(),
            }
        elif args.command == "benchmark":
            result = {**common, **benchmark_search(evaluator, settings.search)}
        else:
            result = {**common, **diagnose_validation(network, state, device)}
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
            write_json(destination / f"{args.command}.json", result)
            if args.command == "benchmark":
                write_benchmark_csv(destination / "benchmark.csv", result["samples"])
        print(json.dumps(result, indent=2, allow_nan=False))
        return result
    except (ValueError, OSError) as error:
        parser.error(str(error))


def console_main():
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(console_main())
