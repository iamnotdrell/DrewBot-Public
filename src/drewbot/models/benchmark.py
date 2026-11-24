"""Short matched-shape Phase 3 measurements, never a training/optimizer workflow."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, process_time

import psutil
import torch

from drewbot.config import ROOT, project_path
from drewbot.experiments import seed_everything
from drewbot.hardware import inspect_hardware
from drewbot.observability import write_json
from drewbot.settings import ModelSettings, Settings, positive

from .model import DrewBot


def synchronize(device: str) -> None:
    """WHAT: Wait for completed work. HOW: MPS synchronization. WHY: Honest wall timing."""
    if device == "mps":
        torch.mps.synchronize()


def memory_metrics(device: str) -> dict:
    """WHAT: Report supported memory observations. HOW: OS RSS and public MPS APIs.

    WHY: Current allocation, driver allocation and process lifetime peak are different.
    PyTorch has no supported peak-allocated MPS counter here: keep it null.
    """
    result = {
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "mps_current_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
        "peak_allocated_mps_bytes": None,
    }
    if device == "mps":
        for field, api in (
            ("mps_current_allocated_bytes", "current_allocated_memory"),
            ("mps_driver_allocated_bytes", "driver_allocated_memory"),
        ):
            try:
                result[field] = int(getattr(torch.mps, api)())
            except (AttributeError, RuntimeError, NotImplementedError):
                pass
    return result


def measure(call, module, device: str, phase: str, warmup: int, iterations: int) -> dict:
    """WHAT: Isolate forward or backward latency. HOW: Warm, synchronize, repeat.

    WHY: Backward setup runs outside the timer so the two measurements are explicit.
    Forward uses eval/no_grad; backward uses eval/autograd on a synthetic squared-logit
    scalar, zeroing gradients before each graph. There is no optimizer or parameter step.
    Each call includes Python dispatch; complete-model forward includes input validation.
    """
    wall_samples, cpu_samples = [], []
    for index in range(warmup + iterations):
        module.zero_grad(set_to_none=True)
        if phase == "backward":
            outputs = call()
            loss = sum(value.square().mean() for value in outputs)
            synchronize(device)
            cpu_started, wall_started = process_time(), perf_counter()
            loss.backward()
        else:
            synchronize(device)
            cpu_started, wall_started = process_time(), perf_counter()
            with torch.no_grad():
                outputs = call()
        synchronize(device)
        wall, cpu = perf_counter() - wall_started, process_time() - cpu_started
        if not all(torch.isfinite(value).all().item() for value in outputs):
            raise RuntimeError("Nonfinite benchmark output")
        if phase == "backward" and any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
            for parameter in module.parameters()
        ):
            raise RuntimeError("Missing/nonfinite benchmark gradient")
        if index >= warmup:
            wall_samples.append(wall)
            cpu_samples.append(cpu)
        del outputs
        if phase == "backward":
            del loss
    return {
        "latency_seconds": sum(wall_samples) / iterations,
        "cpu_seconds": sum(cpu_samples) / iterations,
        "wall_samples_seconds": wall_samples,
        **memory_metrics(device),
    }


def run_benchmarks(
    output: str,
    config: ModelSettings,
    *,
    seed: int = 42,
    warmup: int = 2,
    iterations: int = 3,
    cpu_threads: int = 1,
    devices: tuple[str, ...] = ("cpu", "mps"),
    root: Path = ROOT,
) -> dict:
    """WHAT: Export attention/full-model GAB versus standard measurements.

    HOW: Match shapes, seeds, inputs, batch sizes and timing budget; record exact config,
    distinct parameter counts, hardware/software and memory scope in JSON/CSV.
    WHY: Establish local execution/overhead without accuracy or playing-strength claims.
    Outputs refuse overwrite. MPS absence/unsupported operations are recorded, with CPU
    evidence retained; unsupported requested device execution is never called a success.
    """
    for name, value in (
        ("warmup", warmup),
        ("iterations", iterations),
        ("cpu_threads", cpu_threads),
    ):
        positive(value, name)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if not devices or any(device not in {"cpu", "mps"} for device in devices):
        raise ValueError("devices must contain cpu and/or mps")
    destination = project_path(output, root)
    destination.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    try:
        hardware = inspect_hardware()
        records, device_status = [], []
        for device in devices:
            if not hardware[device]["operational"]:
                device_status.append(
                    {"device": device, "status": "unavailable", "reason": hardware[device]["error"]}
                )
                continue
            try:
                for batch in config.benchmark_batch_sizes:
                    for gab in (False, True):
                        seed_everything(seed)
                        settings = replace(config, gab_enabled=gab)
                        model = DrewBot(settings).to(device).eval()
                        # A separate attention-only wrapper registers the one template
                        # so its parameter/gradient checks count exactly what it uses.
                        attention = model.layers[0].attention
                        wrapper = torch.nn.Module()
                        wrapper.add_module("attention", attention)
                        wrapper.register_parameter("templates", model.templates)
                        generator = torch.Generator().manual_seed(seed + 1)
                        tokens = torch.randn(
                            batch, 64, 12 * config.history_frames, generator=generator
                        ).to(device)
                        auxiliary = torch.randn(batch, 72, generator=generator).to(device)
                        hidden = torch.randn(
                            batch, 64, config.embedding_dim, generator=generator
                        ).to(device)

                        def attention_call(
                            attention=attention, hidden=hidden, templates=model.templates
                        ):
                            return (attention(hidden, templates)[0],)

                        def model_call(model=model, tokens=tokens, auxiliary=auxiliary):
                            result = model(tokens, auxiliary)
                            return result.policy_logits, result.value_logits

                        for component, module, call in (
                            ("attention", wrapper, attention_call),
                            ("model", model, model_call),
                        ):
                            for phase in ("forward", "backward"):
                                measurement = measure(
                                    call, module, device, phase, warmup, iterations
                                )
                                records.append(
                                    {
                                        "device": device,
                                        "variant": "gab" if gab else "standard",
                                        "component": component,
                                        "phase": phase,
                                        "batch_size": batch,
                                        "parameter_count": sum(
                                            p.numel() for p in module.parameters()
                                        ),
                                        "model_parameter_count": sum(
                                            p.numel() for p in model.parameters()
                                        ),
                                        "positions_per_second": batch
                                        / measurement["latency_seconds"],
                                        "seed": seed,
                                        "warmup": warmup,
                                        "iterations": iterations,
                                        "cpu_threads": cpu_threads,
                                        "configuration": asdict(settings),
                                        **measurement,
                                    }
                                )
                        del attention_call, model_call, wrapper, attention, model
                device_status.append({"device": device, "status": "executed", "reason": None})
            except (RuntimeError, NotImplementedError) as error:
                if device == "cpu":
                    raise
                device_status.append({"device": device, "status": "failed", "reason": str(error)})
        result = {
            "hardware": hardware,
            "device_status": device_status,
            "records": records,
            "measurement_scope": {
                "forward": "eval/no_grad; complete model includes finite input validation",
                "backward": "eval/autograd backward only; graph setup outside timer; no optimizer",
                "memory": "Process RSS; resident MPS current/driver; MPS peak unsupported",
                "inputs": "seeded synthetic float32; 64 tokens; not a chess-strength evaluation",
            },
        }
        write_json(destination / "metrics.json", result)
        fields = list(records[0]) + ["hardware"] if records else ["hardware"]
        with (destination / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {
                        **row,
                        "configuration": json.dumps(row["configuration"], sort_keys=True),
                        "wall_samples_seconds": json.dumps(row["wall_samples_seconds"]),
                        "hardware": json.dumps(hardware, sort_keys=True),
                    }
                )
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None) -> int:
    """WHAT: Run Phase 3 microbenchmarks. HOW: Central settings and short CLI overrides.

    WHY: Reproduce timing evidence without importing or starting a trainer.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--devices", choices=("cpu", "mps"), nargs="+", default=["cpu", "mps"])
    args = parser.parse_args(argv)
    settings = Settings.load()
    config = settings.model
    if args.batches:
        config = replace(config, benchmark_batch_sizes=tuple(args.batches))
    result = run_benchmarks(
        args.output,
        config,
        seed=settings.seed,
        warmup=settings.hardware.warmup,
        iterations=settings.hardware.iterations,
        cpu_threads=settings.hardware.cpu_threads,
        devices=tuple(args.devices),
    )
    print(
        json.dumps(
            {"device_status": result["device_status"], "measurements": len(result["records"])},
            indent=2,
        )
    )
    return int(any(row["status"] == "failed" for row in result["device_status"]))


if __name__ == "__main__":
    raise SystemExit(main())
