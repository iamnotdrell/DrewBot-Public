"""Synchronized actual-update and inference microbenchmarks over processed data."""

from dataclasses import asdict, replace
from time import perf_counter

import torch

from drewbot.config import ROOT, project_path
from drewbot.experiments import seed_everything
from drewbot.hardware import inspect_hardware
from drewbot.models import DrewBot
from drewbot.models.benchmark import memory_metrics, synchronize
from drewbot.observability import write_json
from drewbot.training.data import make_loader, to_device
from drewbot.training.losses import compute_losses
from drewbot.training.optim import checked_gradient_norm, make_optimizer


def benchmark(config, model_config, output, *, root=ROOT, warmup=2, iterations=3):
    """WHAT: Investigate device/batch/worker/accumulation cost. HOW: Matched short sweeps.

    WHY: Measure actual effects before recommending an optimization. All variants use
    the same real training partition. CPU threads are fixed, float32 only; MPS timers
    synchronize at both boundaries. Warmup updates are discarded from timings. The
    sweep changes one workload factor around B=16/workers=0/accumulation=1, and includes
    B=1. RSS high water is process-lifetime, not isolated model memory; utilization
    percent is process CPU-time/wall-time (100% = one core). GPU utilization unavailable.
    Inference uses model.eval/no_grad; batch and single-position latency are separate.
    """
    if warmup < 1 or iterations < 1:
        raise ValueError("Use positive warmup and measurement iterations")
    destination = project_path(output, root)
    destination.mkdir(parents=True, exist_ok=False)
    hardware = inspect_hardware()
    previous = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    records = []
    try:
        for device in ("cpu", "mps"):
            if not hardware[device]["operational"]:
                continue
            for gab in (False, True):
                for batch_size, workers, accumulation in (
                    (1, 0, 1),
                    (16, 0, 1),
                    (16, 1, 1),
                    (16, 0, 2),
                ):
                    c = replace(
                        config,
                        batch_size=batch_size,
                        workers=workers,
                        accumulation_steps=accumulation,
                    )
                    seed_everything(c.seed)
                    mc = replace(model_config, gab_enabled=gab)
                    model = DrewBot(mc).to(device)
                    optimizer = make_optimizer(model, c)
                    loader = make_loader(c, "train", root=root)
                    iterator = iter(loader)
                    samples, data_samples = [], []
                    cpu_started = __import__("time").process_time()
                    wall_started = perf_counter()
                    try:
                        for index in range(warmup + iterations):
                            synchronize(device)
                            started = perf_counter()
                            batches = []
                            loading_started = perf_counter()
                            for _ in range(accumulation):
                                batch = next(iterator, None)
                                if batch is None:
                                    iterator = iter(loader)
                                    batch = next(iterator)
                                batches.append(to_device(batch, device))
                            data_seconds = perf_counter() - loading_started
                            pc = sum(int((b["policy"] != -100).sum()) for b in batches)
                            vc = sum(int(b["value_valid"].sum()) for b in batches)
                            if c.objective == "value":
                                pc = 0
                            if c.objective == "policy":
                                vc = 0
                            optimizer.zero_grad(set_to_none=True)
                            model.train()
                            for batch in batches:
                                loss = compute_losses(
                                    model.forward_batch(batch), batch, c
                                ).weighted(c, pc, vc)
                                if loss is not None:
                                    loss.backward()
                            checked_gradient_norm(model, c.gradient_clip)
                            optimizer.step()
                            synchronize(device)
                            elapsed = perf_counter() - started
                            if index >= warmup:
                                samples.append(
                                    {
                                        "seconds": elapsed,
                                        "positions": sum(len(b["policy"]) for b in batches),
                                    }
                                )
                                data_samples.append(data_seconds)
                        cpu_percent = (
                            100
                            * (__import__("time").process_time() - cpu_started)
                            / (perf_counter() - wall_started)
                        )
                        model.eval()
                        inference = {}
                        with torch.no_grad():
                            for label, size in (("single", 1), ("batch", len(batch["policy"]))):
                                inputs = {
                                    k: v[:size] if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                                timings = []
                                for index in range(warmup + iterations):
                                    synchronize(device)
                                    started = perf_counter()
                                    model.forward_batch(inputs)
                                    synchronize(device)
                                    if index >= warmup:
                                        timings.append(perf_counter() - started)
                                inference[label + "_inference_seconds"] = sum(timings) / iterations
                                inference[label + "_inference_positions_per_second"] = (
                                    size * iterations / sum(timings)
                                )
                        records.append(
                            {
                                "device": device,
                                "gab": gab,
                                "batch_size": batch_size,
                                "workers": workers,
                                "accumulation": accumulation,
                                "parameter_count": sum(p.numel() for p in model.parameters()),
                                "step_seconds": sum(s["seconds"] for s in samples) / iterations,
                                "positions_per_second": sum(s["positions"] for s in samples)
                                / sum(s["seconds"] for s in samples),
                                "data_loading_seconds": sum(data_samples) / iterations,
                                "process_cpu_percent": cpu_percent,
                                "gpu_utilization_percent": None,
                                "samples": samples,
                                **inference,
                                **memory_metrics(device),
                            }
                        )
                    finally:
                        del iterator
                        loader.dataset.close()
                    del optimizer, model, batch, batches
        result = {
            "hardware": hardware,
            "warmup": warmup,
            "iterations": iterations,
            "training": asdict(config),
            "model": asdict(model_config),
            "records": records,
            "methodology": benchmark.__doc__,
        }
        write_json(destination / "benchmark.json", result)
        import csv

        with (destination / "benchmark.csv").open("w", newline="") as stream:
            fields = [k for k in records[0] if k != "samples"] if records else []
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in records)
        return result
    finally:
        torch.set_num_threads(previous)
