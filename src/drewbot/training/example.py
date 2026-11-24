"""Executable WHAT/HOW/WHY walkthrough using an explicitly selected processed corpus."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from drewbot.config import ROOT
from drewbot.evaluation.evaluator import inspect_predictions
from drewbot.observability import write_json
from drewbot.settings import ModelSettings
from drewbot.training.checkpoints import load_model
from drewbot.training.config import TrainingConfig
from drewbot.training.data import make_loader, to_device
from drewbot.training.losses import compute_losses
from drewbot.training.optim import checked_gradient_norm
from drewbot.training.trainer import Trainer


def demonstrate(config, model_config, *, root=ROOT):
    """WHAT: Teach one actual update and reload. HOW: Expose each operation. WHY: Defend the math.

    Input data is required, audited Phase 2 data; there is no synthetic fallback. The
    provenance snapshot identifies preprocessing/source. Demonstrations use an isolated
    experiment and checkpoint directory and never consume the held-out test split.
    """
    trainer = Trainer(replace(config, objective="joint", max_steps=1), model_config, root=root)
    trace = []

    def explain(what, how, why, **evidence):
        trace.append({"WHAT": what, "HOW": how, "WHY": why, **evidence})

    loader = make_loader(config, "train", root=root)
    try:
        batch = to_device(next(iter(loader)), trainer.device)
        explain(
            "Load a training batch",
            "Lazy mmap shards and seeded game-isolated partition",
            "Use measured data rather than fabricated evaluations",
            positions=len(batch["policy"]),
        )
        explain(
            "Inspect input tensors",
            "Read shapes/dtypes before forward",
            "Catch schema/device mismatches",
            shapes={k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
        )
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        output = trainer.model.forward_batch(batch)
        explain(
            "Forward pass",
            "Six transformer blocks then two heads",
            "Share representations without feeding labels",
            policy_shape=list(output.policy_logits.shape),
            value_shape=list(output.value_logits.shape),
        )
        inspection = inspect_predictions(trainer.model, batch, trainer.device)
        explain(
            "Inspect policy predictions",
            "Mask illegal logits then normalize and decode UCI",
            "Keep underpromotions distinct",
            examples=inspection,
        )
        explain(
            "Inspect value predictions",
            "Softmax of loss/draw/win logits",
            "Forecast from the mover's perspective; not centipawns",
            example=inspection[0]["value_probabilities_loss_draw_win"],
        )
        losses = compute_losses(output, batch, trainer.config)
        explain(
            "Compute individual losses",
            "Categorical CE over observed targets",
            "Missing outcomes do not invent labels",
            policy_sum=float(losses.policy_sum.detach()),
            value_sum=float(losses.value_sum.detach()) if losses.value_sum is not None else None,
            policy_count=losses.policy_count,
            value_count=losses.value_count,
        )
        total = losses.weighted(trainer.config)
        explain(
            "Combine weighted losses",
            "lambda_policy * mean_policy + lambda_value * mean_value",
            "Control the two gradient contributions",
            total_loss=float(total.detach()),
            policy_weight=config.policy_weight,
            value_weight=config.value_weight,
        )
        total.backward()
        explain(
            "Backpropagate",
            "Autograd applies the chain rule through heads and encoder",
            "Obtain parameter derivatives without changing weights",
        )
        norm = checked_gradient_norm(trainer.model, config.gradient_clip)
        explain(
            "Inspect and clip gradients",
            "Reject nonfinite derivatives; clip global L2",
            "Stabilize the update and expose its scale",
            pre_clip_norm=norm,
        )
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.global_step = 1
        trainer.offset = len(batch["policy"])
        explain(
            "Take an optimizer step",
            "AdamW applies moment estimates and decoupled decay",
            "Reduce observed training loss over repeated updates",
        )
        trainer.epoch_metrics.update(output, batch, losses)
        trainer.log.record(
            "demonstration",
            {**trainer.epoch_metrics.result(trainer.config), "gradient_norm": norm},
            1,
        )
        explain(
            "Record training metrics",
            "Flush JSONL, CSV and TensorBoard events",
            "Retain inspectable evidence during execution",
        )
        checkpoint = trainer.checkpoint("demonstration")
        restored, _ = load_model(checkpoint, trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            before = trainer.model.forward_batch(batch)
            after = restored.forward_batch(batch)
        torch.testing.assert_close(before.policy_logits, after.policy_logits, rtol=0, atol=0)
        torch.testing.assert_close(before.value_logits, after.value_logits, rtol=0, atol=0)
        explain(
            "Save and restore checkpoint",
            "Atomic full-state save, strict inference reload",
            "Recover weights and preserve the experiment",
            checkpoint=str(checkpoint),
            exact_output_match=True,
        )
        result = {"run_id": trainer.log.id, "dataset": trainer.dataset_info, "trace": trace}
        write_json(trainer.log.directory / "demonstration.json", result)
        return result
    finally:
        loader.dataset.close()
        trainer.log.close()
        trainer.restore_runtime()


def main(argv=None):
    """WHAT: Run the annotated example. HOW: Read existing configs. WHY: Verified invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(argv)
    result = demonstrate(
        TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig(),
        ModelSettings(**json.loads(args.model_config.read_text())) if args.model_config else ModelSettings(),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
