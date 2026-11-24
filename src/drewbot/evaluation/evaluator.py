"""Reusable validation/test evaluation and legal per-position prediction inspection."""

import json
from itertools import islice
from time import perf_counter

import chess
import torch

from drewbot.data.walkthrough import decode_action
from drewbot.models.benchmark import synchronize
from drewbot.models.inference import legal_policy
from drewbot.training.data import to_device
from drewbot.training.losses import compute_losses
from drewbot.training.metrics import Metrics


@torch.no_grad()
def evaluate(model, loader, config, device="cpu", max_batches=None) -> dict:
    """WHAT: Measure fixed-model forecasts. HOW: Eval/no_grad and streaming sufficient stats.

    WHY: Shared methodology for validation and one-off held-out tests. Does not update
    parameters, preprocessing or RNG sampling. The caller selects the split; Trainer
    only ever requests validation. Limits are explicit in exported reports.
    """
    previous = model.training
    model.eval()
    metrics = Metrics()
    synchronize(device)
    started = perf_counter()
    try:
        for batch in islice(loader, max_batches):
            batch = to_device(batch, device)
            output = model.forward_batch(batch)
            losses = compute_losses(output, batch, config)
            metrics.update(output, batch, losses)
        synchronize(device)
        duration = perf_counter() - started
        return {
            **metrics.result(config),
            "duration_seconds": duration,
            "positions_per_second": metrics.positions / duration,
            "max_batches": max_batches,
        }
    finally:
        model.train(previous)


@torch.no_grad()
def inspect_predictions(model, batch, device="cpu", limit=5) -> list[dict]:
    """WHAT: Inspect legal moves/confidence. HOW: Decode mover coordinates with stored FEN.

    WHY: Human-readable checks preserve promotion identities and Black orientation.
    Metadata is used only to explain outputs, never as model input. Bare FEN cannot
    reconstruct repetition; use the persisted legal mask rather than recomputing it.
    """
    previous = model.training
    model.eval()
    try:
        batch = to_device(batch, device)
        output = model.forward_batch(batch)
        probabilities, _ = legal_policy(output.policy_logits, batch["legal_mask"])
        rows = []
        for i in range(min(limit, len(batch["policy"]))):
            metadata = json.loads(batch["metadata"][i])
            board = chess.Board(metadata["fen"])
            count = min(5, int(batch["legal_mask"][i].sum()))
            values, indices = probabilities[i].topk(count)
            rows.append(
                {
                    "position": metadata,
                    "policy_target": int(batch["policy"][i]),
                    "candidates": [
                        {
                            "uci": decode_action(int(index), board.turn).uci(),
                            "probability": float(prob),
                        }
                        for index, prob in zip(indices, values, strict=True)
                    ],
                    "value_probabilities_loss_draw_win": output.value_logits[i]
                    .softmax(-1)
                    .cpu()
                    .tolist(),
                    "value_target": int(batch["value"][i]),
                }
            )
        return rows
    finally:
        model.train(previous)
