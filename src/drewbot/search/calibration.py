"""Validation-only diagnostics for uncalibrated mover-relative outcome forecasts."""

import torch
from torch.utils.data import DataLoader

from drewbot.config import project_path
from drewbot.data.dataset import ChessDataset
from drewbot.search.evaluator import outcome_scalar
from drewbot.training.data import audit_dataset


def calibration_metrics(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> dict:
    """WHAT: Measure categorical forecast reliability without fitting a calibration map.

    HOW: For p (N,3), y (N,), NLL=-mean(log p_y); multiclass Brier is
    mean(sum_c (p_c-1[y=c])²). Confidence ECE=sum_b n_b/N*|accuracy_b-confidence_b|.
    Bins partition [0,1] uniformly; right endpoint 1 belongs to the last bin.
    WHY: Accuracy alone hides confidence errors. ECE is bin/sample dependent and
    positions within a game are correlated; no competitive-use guarantee follows.
    O(N*3) time/space; bounded pilot caller accumulates validation predictions only.
    """
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 3
        or len(probabilities) < 1
        or targets.shape != (len(probabilities),)
        or targets.dtype != torch.long
        or (targets < 0).any()
        or (targets > 2).any()
        or type(bins) is not int
        or not 1 <= bins <= 100
    ):
        raise ValueError("Expected nonempty (N,3) probabilities, (N,) int64 labels and 1..100 bins")
    outcome_scalar(probabilities)
    p = probabilities.detach().cpu().double()
    y = targets.detach().cpu()
    confidence, predicted = p.max(-1)
    correct = predicted.eq(y)
    one_hot = torch.nn.functional.one_hot(y, 3)
    indices = (confidence * bins).long().clamp(max=bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        accuracy = float(correct[selected].double().mean()) if count else None
        mean_confidence = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    chosen = p[torch.arange(len(y)), y]
    # Preserve strict JSON even for a confidently impossible observed class.
    nll = float(-chosen.log().mean()) if (chosen > 0).all() else None
    return {
        "positions": len(y),
        "class_support_loss_draw_win": torch.bincount(y, minlength=3).tolist(),
        "predicted_support_loss_draw_win": torch.bincount(predicted, minlength=3).tolist(),
        "accuracy": float(correct.double().mean()),
        "nll": nll,
        "zero_target_probability_count": int((chosen == 0).sum()),
        "multiclass_brier": float(((p - one_hot) ** 2).sum(-1).mean()),
        "confidence_ece": ece,
        "bins": rows,
        "signed_outcome_mse": float(((p[:, 2] - p[:, 0]) - (y - 1)).square().mean()),
    }


@torch.inference_mode()
def diagnose_validation(model, state: dict, device: str, batch_size: int = 16) -> dict:
    """WHAT: Audit the existing validation pilot. HOW: Frozen model, full known labels.

    WHY: Investigate calibration without another held-out selection or retraining.
    Dataset fingerprint must match checkpoint provenance. This explicitly bounded
    diagnostic refuses >4096 validation positions; streaming larger studies is deferred.
    """
    directory = project_path(state["training_config"]["dataset"])
    audit = audit_dataset(directory)
    if audit["fingerprint"] != state["dataset_fingerprint"]:
        raise ValueError("Calibration dataset differs from checkpoint provenance")
    dataset = ChessDataset(directory, "validation")
    training_dataset = ChessDataset(directory, "train")
    previous = model.training
    try:
        if not 0 < len(dataset) <= 4096:
            raise ValueError("Calibration pilot requires 1..4096 validation positions")
        counts = torch.zeros(3, dtype=torch.long)
        for i in range(len(training_dataset)):
            label = int(training_dataset.raw(i)["value"])
            if label >= 0:
                counts[label] += 1
        model.eval()
        probabilities, labels = [], []
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            output = model.forward_batch(
                {key: batch[key].to(device) for key in ("tokens", "auxiliary")}
            )
            observed = batch["value_valid"]
            probabilities.append(output.value_logits.softmax(-1).cpu()[observed])
            labels.append(batch["value"][observed])
        p, y = torch.cat(probabilities), torch.cat(labels)
        metrics = calibration_metrics(p, y)
        majority = int(counts.argmax()) if int(counts.sum()) else None
        return {
            "split": "validation",
            "dataset_fingerprint": audit["fingerprint"],
            "training_support_loss_draw_win": counts.tolist(),
            "training_majority_class": majority,
            "majority_validation_accuracy": float((y == majority).float().mean())
            if majority is not None
            else None,
            "metrics": metrics,
            "interpretation": "No calibration fitted. Two validation games and absent draws "
            "cannot establish positional utility or competitive strength.",
        }
    finally:
        model.train(previous)
        dataset.close()
        training_dataset.close()
