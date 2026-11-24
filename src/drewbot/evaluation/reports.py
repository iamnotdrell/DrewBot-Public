"""Comparable run reports and static training-curve exports."""

import csv
import json
from pathlib import Path

from drewbot.observability import write_json


def compare_runs(directories, output: Path):
    """WHAT: Compare actual experiments. HOW: Validate matched designs and export differences.

    WHY: Same interface does not imply same parameter count or statistically conclusive
    findings. Refuse differences in data, budget, objectives and optimizer settings;
    allow seeds to differ for repeated trials. No test metrics are used for selection.
    """
    runs = [json.loads((d / "summary.json").read_text()) for d in directories]
    if not runs or any(r["interrupted"] or r["validation"] is None for r in runs):
        raise ValueError("Comparison requires completed, evaluated experiments")

    def signature(run):
        training = {k: v for k, v in run["training"].items() if k != "seed"}
        model = {k: v for k, v in run["model"].items() if k != "gab_enabled"}
        return training, model, run["dataset_fingerprint"], run["device"], run["global_step"]

    if any(signature(r) != signature(runs[0]) for r in runs[1:]):
        raise ValueError("Unmatched dataset, objective, hardware, architecture or training budget")
    rows = [
        {
            "run_id": r["run_id"],
            "gab": r["model"]["gab_enabled"],
            "seed": r["training"]["seed"],
            "parameters": r["parameter_count"],
            "validation_loss": r["validation"]["loss"],
            "policy_top1": r["validation"]["policy_top1"],
            "policy_top5": r["validation"]["policy_top5"],
            "value_accuracy": r["validation"]["value_accuracy"],
            "training_loss": r["training_metrics"]["loss"],
            "training_positions_per_second": r["efficiency"]["positions_per_second"],
            "single_inference_seconds": r["inference"]["single"]["seconds"],
            "batch_inference_positions_per_second": r["inference"]["batch"]["positions_per_second"],
            "session_seconds": r["session_seconds"],
            "training_update_seconds": r["training_update_seconds"],
            "process_lifetime_peak_rss_bytes": r["process_lifetime_peak_rss_bytes"],
        }
        for r in runs
    ]
    baseline = rows[0]
    for row in rows:
        row["validation_loss_delta_from_first"] = (
            row["validation_loss"] - baseline["validation_loss"]
        )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "runs": rows,
        "configurations": runs,
        "limitations": "Short pilot; unequal parameter counts; single-seed results are "
        "not conclusive; shared opening positions can recur across distinct games; "
        "imitation accuracy is not playing strength. Inference timings are in the "
        "separate matched benchmark report. Peak RSS is process-lifetime.",
    }
    write_json(output / "comparison.json", result)
    fields = list(rows[0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 4 pilot comparison",
        "",
        result["limitations"],
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines += ["| " + " | ".join(str(r[k]) for k in fields) + " |" for r in rows]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def export_curves(directories, output: Path):
    """WHAT: Export portable curves. HOW: Plot tidy CSV with matplotlib. WHY: Share local evidence.

    Matplotlib is an optional visualization dependency. TensorBoard remains the live
    interactive monitor; these files can later feed a research dashboard. Validation
    and training losses have different sampling scope and are labelled separately.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    panels = [
        ("loss", "Loss"),
        ("policy_top1", "Policy top-1"),
        ("value_accuracy", "Value accuracy"),
        ("learning_rate", "Learning rate"),
        ("gradient_norm", "Gradient L2 norm"),
        ("positions_per_second", "Positions / second"),
        ("process_rss_bytes", "Process RSS bytes"),
        ("mps_current_allocated_bytes", "Current MPS bytes"),
    ]
    figure, axes = plt.subplots(4, 2, figsize=(14, 16))
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        label = f"{'GAB' if summary['model']['gab_enabled'] else 'standard'} / {summary['run_id']}"
        with (directory / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for ax, (metric, title) in zip(axes.flat, panels, strict=True):
            for phase in ("train_step", "validation"):
                points = [
                    (int(r["step"]), float(r["value"]))
                    for r in rows
                    if r["metric"] == metric and r["phase"] == phase
                ]
                if points:
                    ax.plot(*zip(*points, strict=True), marker=".", label=f"{label} {phase}")
            ax.set(title=title, xlabel="Optimizer update")
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    figure.suptitle("DrewBot bounded pilot — validation games held out; no strength claim")
    figure.tight_layout()
    figure.savefig(output / "curves.png", dpi=140)
    figure.savefig(output / "curves.svg")
    plt.close(figure)
    return {"png": str(output / "curves.png"), "svg": str(output / "curves.svg")}
