"""Local train/evaluate/resume/ablate/benchmark/compare/plot workflows."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from drewbot.config import ROOT, project_path
from drewbot.hardware import inspect_hardware
from drewbot.observability import write_json
from drewbot.settings import ModelSettings
from drewbot.training.config import TrainingConfig


def main(argv=None):
    """WHAT: Execute Phase 4 workflows. HOW: Strict JSON settings. WHY: Reproducible commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--objective", choices=("policy", "value", "joint"))
    train.add_argument("--device", choices=("cpu", "mps", "auto"))
    train.add_argument("--resume", type=Path)
    ablate = sub.add_parser("ablate")
    ablate.add_argument("--output", required=True)
    ablate.add_argument("--seeds", nargs="+", type=int)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    evaluation.add_argument("--inspection-positions", type=int, default=5)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--output", required=True)
    for name in ("compare", "plot"):
        p = sub.add_parser(name)
        p.add_argument("runs", type=Path, nargs="+")
        p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig()
        model = ModelSettings(**json.loads(args.model_config.read_text())) if args.model_config else ModelSettings()
        if args.command == "train":
            from drewbot.training.trainer import Trainer

            if args.objective:
                config = replace(config, objective=args.objective)
            if args.device:
                config = replace(config, device=args.device)
            result = Trainer(config, model, resume=args.resume).fit()
        elif args.command == "ablate":
            from drewbot.evaluation.reports import compare_runs
            from drewbot.training.trainer import Trainer

            if model.layers != 6:
                raise ValueError("Phase 4 attention ablation requires six layers")
            runs = []
            for seed in args.seeds or [config.seed]:
                for gab in (False, True):
                    result = Trainer(
                        replace(config, seed=seed), replace(model, gab_enabled=gab)
                    ).fit()
                    runs.append(ROOT / "experiments" / result["run_id"])
            result = compare_runs(runs, project_path(args.output))
        elif args.command == "evaluate":
            from drewbot.evaluation.evaluator import evaluate, inspect_predictions
            from drewbot.training.checkpoints import load_model
            from drewbot.training.data import audit_dataset, make_loader

            device = inspect_hardware(args.device)["selected_device"]
            network, state = load_model(args.checkpoint, device)
            config = TrainingConfig(**state["training_config"])
            import torch

            torch.set_num_threads(config.cpu_threads)
            audit = audit_dataset(project_path(config.dataset))
            if audit["fingerprint"] != state["dataset_fingerprint"]:
                raise ValueError("Evaluation dataset differs from checkpoint provenance")
            loader = make_loader(config, args.split)
            try:
                if not len(loader.dataset):
                    raise ValueError("Requested evaluation split is empty")
                result = {
                    "split": args.split,
                    "checkpoint": str(args.checkpoint),
                    "dataset_fingerprint": audit["fingerprint"],
                    "device": device,
                    "metrics": evaluate(network, loader, config, device),
                    "inspection": inspect_predictions(
                        network, next(iter(loader)), device, args.inspection_positions
                    ),
                }
            finally:
                loader.dataset.close()
            destination = project_path(args.output)
            destination.mkdir(parents=True, exist_ok=False)
            write_json(destination / "evaluation.json", result)
        elif args.command == "benchmark":
            from drewbot.training.benchmark import benchmark

            result = benchmark(config, model, args.output)
        else:
            from drewbot.evaluation.reports import compare_runs, export_curves

            call = compare_runs if args.command == "compare" else export_curves
            result = call(args.runs, project_path(args.output))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, FloatingPointError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
