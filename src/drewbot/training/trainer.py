"""Small explicit training engine with restartable optimizer-update boundaries."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from drewbot.config import ROOT, project_path
from drewbot.evaluation.evaluator import evaluate
from drewbot.experiments import seed_everything
from drewbot.hardware import inspect_hardware
from drewbot.models import DrewBot
from drewbot.models.benchmark import memory_metrics, synchronize
from drewbot.observability import digest_file, write_json
from drewbot.training.checkpoints import (
    FORMAT,
    architecture_id,
    atomic_save,
    read_checkpoint,
    restore_rng,
    rng_state,
)
from drewbot.training.data import audit_dataset, make_loader, to_device
from drewbot.training.logging import RunLogger, software_provenance
from drewbot.training.losses import compute_losses, target_masks
from drewbot.training.metrics import Metrics
from drewbot.training.optim import WarmupCosine, checked_gradient_norm, make_optimizer


class Trainer:
    """WHAT: Optimize policy/value/joint models. HOW: Small composable PyTorch components.

    WHY: Make every update, validation and checkpoint inspectable. Accumulation holds
    only a configured window of CPU batches, computes label denominators first, then
    backpropagates microbatches separately: sum_i CE_i / observed_window_labels.
    This matches a larger labelled batch even with uneven/missing labels (dropout may
    differ). Checkpoints occur only after complete windows, so no partial gradients
    need serialization. Mid-window failures replay from the prior committed boundary.
    """

    def __init__(self, config, model_config, *, resume: Path | None = None, root=ROOT):
        self.config, self.model_config, self.root = config, model_config, root
        if model_config.history_frames > 8:
            raise ValueError("v1 training datasets support history_frames<=8")
        self.previous_threads = torch.get_num_threads()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(config.deterministic)
        seed_everything(config.seed)
        try:
            self.hardware = inspect_hardware(config.device, config.allow_cpu_fallback)
            self.device = self.hardware["selected_device"]
            self.dataset_info = audit_dataset(project_path(config.dataset, root))
            counts = self.dataset_info["validation"]["counts"]
            if not counts["train"]["positions"] or not counts["validation"]["positions"]:
                raise ValueError("Training requires nonempty isolated train and validation splits")
            if config.objective == "value" and not sum(
                self.dataset_info["class_counts_loss_draw_win"]["train"]
            ):
                raise ValueError("No observed value targets in training partition")
            self.model = DrewBot(model_config).to(self.device)
            self.optimizer = make_optimizer(self.model, config)
            windows = math.ceil(
                math.ceil(counts["train"]["positions"] / config.batch_size)
                / config.accumulation_steps
            )
            budget = min(config.epochs * windows, config.max_steps or config.epochs * windows)
            self.scheduler = WarmupCosine(self.optimizer, config, budget)
            self.epoch = self.offset = self.global_step = self.bad_validations = 0
            self.best_loss = math.inf
            self.best_metrics = None
            self.epoch_metrics = Metrics()
            self.epoch_seconds = 0.0
            self.last_validation = None
            self.last_training = None
            self.skipped_windows = 0
            self.efficiency = {
                name: 0.0
                for name in (
                    "positions",
                    "steps",
                    "step_seconds",
                    "forward_seconds",
                    "backward_seconds",
                    "optimizer_seconds",
                    "data_loading_seconds",
                )
            }
            self.total_training_seconds = 0.0
            self.early_stopped = False
            parent = self.restore(resume) if resume else None
            metadata = {
                "training": asdict(config),
                "model": asdict(model_config),
                "architecture": architecture_id(self.model),
                "dataset": self.dataset_info,
                "hardware": self.hardware,
                "software": software_provenance(root),
                "initialization": "PyTorch defaults; GAB Xavier-normal; seed before model",
                "resume_from": str(resume) if resume else None,
                "parent_run": parent,
                "source_sha256": {
                    str(p.relative_to(root)): digest_file(p)
                    for p in sorted((root / "src").rglob("*.py"))
                },
            }
            self.log = RunLogger(metadata, config.tensorboard, root)
        except BaseException:
            self.restore_runtime()
            raise

    def restore_runtime(self):
        """WHAT: Undo process-wide settings. HOW: Restore original flags. WHY: Library hygiene."""
        torch.set_num_threads(self.previous_threads)
        torch.use_deterministic_algorithms(
            self.previous_deterministic, warn_only=self.previous_warn_only
        )

    def restore(self, path):
        """WHAT: Resume a committed boundary. HOW: Strict config/data/device checks and RNG.

        WHY: Changing budget, objective, corpus or architecture invalidates an exact
        continuation. Cross-device continuation is intentionally refused; inference
        loading may choose either device. New experiments should start explicitly.
        """
        state = read_checkpoint(path)
        if (
            state["training_config"] != asdict(self.config)
            or state["architecture"] != architecture_id(self.model)
            or state["dataset_fingerprint"] != self.dataset_info["fingerprint"]
            or state["device"] != self.device
        ):
            raise ValueError("Resume configuration, architecture, dataset or device differs")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        for key, value in state["progress"].items():
            setattr(self, key, value)
        self.epoch_metrics.load_state_dict(state["epoch_metrics"])
        restore_rng(state["rng"], self.device)
        return state["run_id"]

    def checkpoint(self, name):
        """WHAT: Save complete training state. HOW: CPU-safe RNG + atomic persistence.

        WHY: Keep final/best/periodic states independent of Git. Models are serialized
        on their current device; constrained loading maps everything to CPU first.
        """
        progress = {
            key: getattr(self, key)
            for key in (
                "epoch",
                "offset",
                "global_step",
                "bad_validations",
                "best_loss",
                "best_metrics",
                "epoch_seconds",
                "last_validation",
                "last_training",
                "skipped_windows",
                "efficiency",
                "total_training_seconds",
                "early_stopped",
            )
        }
        payload = {
            "format": FORMAT,
            "run_id": self.log.id,
            "architecture": architecture_id(self.model),
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "progress": progress,
            "epoch_metrics": self.epoch_metrics.state_dict(),
            "rng": rng_state(self.device),
            "dataset_version": self.dataset_info["dataset_version"],
            "dataset_fingerprint": self.dataset_info["fingerprint"],
            "device": self.device,
        }
        destination = self.log.checkpoint_directory / f"{name}.pt"
        size = atomic_save(destination, payload)
        self.log.record("checkpoint", {"bytes": size}, self.global_step)
        return destination

    def validate(self):
        """WHAT: Monitor held-out games. HOW: Shared validation evaluator.

        WHY: Select checkpoints without tuning against the test partition.
        """
        loader = make_loader(self.config, "validation", root=self.root)
        try:
            result = evaluate(
                self.model, loader, self.config, self.device, self.config.max_validation_batches
            )
        finally:
            loader.dataset.close()
        if result["loss"] is None:
            raise ValueError("Validation has no labels for the selected objective")
        self.last_validation = result
        self.log.record("validation", result, self.global_step)
        if result["loss"] < self.best_loss:
            self.best_loss, self.best_metrics, self.bad_validations = result["loss"], result, 0
            self.checkpoint("best")
        else:
            self.bad_validations += 1
        self.early_stopped = self.bad_validations >= self.config.patience

    def train_window(self, batches):
        """WHAT: Apply one weighted update. HOW: Exact window denominators, finite guard, clipping.

        WHY: Missing values cannot shrink policy gradients or produce decay-only value
        updates. An all-unlabelled active window is skipped without changing the
        optimizer or scheduler budget. Timings synchronize MPS and include validation guards.
        """
        config = self.config
        counts = [target_masks(batch) for batch in batches]
        pc = sum(int(pm.sum()) for pm, _ in counts) if config.objective != "value" else 0
        vc = sum(int(vm.sum()) for _, vm in counts) if config.objective != "policy" else 0
        if not pc and not vc:
            self.optimizer.zero_grad(set_to_none=True)
            positions = sum(len(batch["policy"]) for batch in batches)
            self.offset += positions
            self.skipped_windows += 1
            return {"skipped": True, "positions": positions}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        forward = backward = 0.0
        window_metrics = Metrics()
        for batch in batches:
            batch = to_device(batch, self.device)
            synchronize(self.device)
            started = perf_counter()
            output = self.model.forward_batch(batch)
            losses = compute_losses(output, batch, config)
            synchronize(self.device)
            forward += perf_counter() - started
            total = losses.weighted(config, pc, vc)
            if total is not None:
                started = perf_counter()
                total.backward()
                synchronize(self.device)
                backward += perf_counter() - started
            window_metrics.update(output, batch, losses)
            self.epoch_metrics.update(output, batch, losses)
            self.offset += len(batch["policy"])
        started = perf_counter()
        norm = checked_gradient_norm(self.model, config.gradient_clip)
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        synchronize(self.device)
        optimizer_seconds = perf_counter() - started
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError(
                "Non-finite parameters after update; restore previous checkpoint"
            )
        self.scheduler.step()
        self.global_step += 1
        return {
            **window_metrics.result(config),
            "gradient_norm": norm,
            "learning_rate": learning_rate,
            "forward_seconds": forward,
            "backward_seconds": backward,
            "optimizer_seconds": optimizer_seconds,
        }

    @torch.no_grad()
    def profile_inference(self):
        """WHAT: Measure trained single/batch inference. HOW: Warmup and synchronize.

        WHY: Compare both architectures on the same validation batch, after optimization.
        Includes forward validation guards; excludes DataLoader and legal normalization.
        Three repeats are noisy pilot observations, not a capacity/strength result.
        """
        loader = make_loader(self.config, "validation", root=self.root)
        previous = self.model.training
        self.model.eval()
        try:
            batch = to_device(next(iter(loader)), self.device)
            result = {}
            for label, size in (("single", 1), ("batch", len(batch["policy"]))):
                inputs = {
                    k: v[:size] if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                samples = []
                for index in range(5):
                    synchronize(self.device)
                    started = perf_counter()
                    self.model.forward_batch(inputs)
                    synchronize(self.device)
                    if index >= 2:
                        samples.append(perf_counter() - started)
                result[label] = {
                    "batch_size": size,
                    "warmup": 2,
                    "iterations": 3,
                    "samples_seconds": samples,
                    "seconds": sum(samples) / 3,
                    "positions_per_second": size * 3 / sum(samples),
                }
            return result
        finally:
            loader.dataset.close()
            self.model.train(previous)

    def fit(self, *, stop_after_updates=None) -> dict:
        """WHAT: Run bounded epochs. HOW: Windows, validation, early stopping and durable logs.

        WHY: A stop_after_updates boundary simulates interruption without changing the
        configured schedule/budget. Arbitrary process death recovers from the most
        recent periodic checkpoint. This API supports recovery tests and short demos.
        """
        session_start = perf_counter()
        interrupted = False
        try:
            while (
                self.epoch < self.config.epochs
                and self.global_step < self.scheduler.total_steps
                and not self.early_stopped
            ):
                loader = make_loader(
                    self.config, "train", epoch=self.epoch, offset=self.offset, root=self.root
                )
                iterator = iter(loader)
                exhausted = False
                try:
                    while self.global_step < self.scheduler.total_steps:
                        started = perf_counter()
                        batches = []
                        for _ in range(self.config.accumulation_steps):
                            batch = next(iterator, None)
                            if batch is None:
                                exhausted = True
                                break
                            batches.append(batch)
                        loading = perf_counter() - started
                        if not batches:
                            break
                        metrics = self.train_window(batches)
                        if metrics.get("skipped"):
                            self.log.record("train_skipped", metrics, self.global_step)
                            continue
                        duration = perf_counter() - started
                        self.epoch_seconds += duration
                        self.total_training_seconds += duration
                        metrics.update(
                            {
                                "data_loading_seconds": loading,
                                "step_seconds": duration,
                                "steps_per_second": 1 / duration,
                                "positions_per_second": metrics["positions"] / duration,
                                **memory_metrics(self.device),
                            }
                        )
                        for name in self.efficiency:
                            if name == "steps":
                                self.efficiency[name] += 1
                            else:
                                self.efficiency[name] += metrics[name]
                        self.log.record("train_step", metrics, self.global_step)
                        if self.global_step % self.config.checkpoint_every == 0:
                            self.checkpoint(f"step-{self.global_step:08d}")
                        if stop_after_updates and self.global_step >= stop_after_updates:
                            interrupted = True
                            break
                finally:
                    del iterator
                    loader.dataset.close()
                if interrupted:
                    break
                epoch_complete = (
                    exhausted
                    or self.offset
                    == self.dataset_info["validation"]["counts"]["train"]["positions"]
                )
                result = {
                    **self.epoch_metrics.result(self.config),
                    "epoch_index": self.epoch,
                    "epoch_complete": epoch_complete,
                    "duration_seconds": self.epoch_seconds,
                }
                self.last_training = result
                self.log.record("train_epoch", result, self.global_step)
                if epoch_complete:
                    self.epoch += 1
                    self.offset = 0
                    self.epoch_metrics = Metrics()
                    self.epoch_seconds = 0.0
                if (
                    self.epoch % self.config.validation_every == 0
                    or self.global_step == self.scheduler.total_steps
                    or self.epoch == self.config.epochs
                ):
                    self.validate()
                self.checkpoint("latest")
            final = self.checkpoint("final")
            summary = {
                "run_id": self.log.id,
                "interrupted": interrupted,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "early_stopped": self.early_stopped,
                "skipped_windows": self.skipped_windows,
                "device": self.device,
                "model": asdict(self.model_config),
                "training": asdict(self.config),
                "dataset_fingerprint": self.dataset_info["fingerprint"],
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "validation": self.last_validation,
                "training_metrics": self.last_training,
                "efficiency": {
                    **self.efficiency,
                    "positions_per_second": (
                        self.efficiency["positions"] / self.efficiency["step_seconds"]
                        if self.efficiency["step_seconds"]
                        else None
                    ),
                },
                "inference": self.profile_inference() if not interrupted else None,
                "best_validation": self.best_metrics,
                "training_update_seconds": self.total_training_seconds,
                "session_seconds": perf_counter() - session_start,
                "checkpoint": str(final),
                "checkpoint_bytes": final.stat().st_size,
                **memory_metrics(self.device),
            }
            self.log.finish(summary)
            return summary
        except BaseException as error:
            write_json(
                self.log.directory / "failure.json",
                {
                    "error": str(error),
                    "step": self.global_step,
                    "recovery": "Resume last complete periodic checkpoint; failed window unsaved",
                },
            )
            raise
        finally:
            self.log.close()
            self.restore_runtime()
