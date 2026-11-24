"""Existing model-to-search adapter; no new learned architecture or calibrated score."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np
import torch

from drewbot.data.samples import auxiliary
from drewbot.data.walkthrough import encode_history, legal_mask, policy_target
from drewbot.models.inference import legal_policy
from drewbot.search.state import PositionInput


@dataclass(frozen=True)
class Evaluation:
    """WHAT: Mover utility and legal ordering. HOW: Scalar and move probabilities. WHY: Small
    API.
    """

    value: float
    policy: dict[chess.Move, float]


class Evaluator(Protocol):
    """WHAT: Injectable deterministic inference. HOW: Batched snapshots. WHY: Independent tests."""

    history_frames: int

    def evaluate(self, positions: list[PositionInput]) -> list[Evaluation]: ...


def outcome_scalar(probabilities: torch.Tensor) -> torch.Tensor:
    """WHAT: Map (...,3) loss/draw/win to (...,) mover utility in [-1,1].

    HOW: E[reward]=[-1,0,+1] dot p=p_win-p_loss; validate finite probabilities.
    WHY: Negamax needs a signed scalar; this assumes no centipawn or calibration map.
    O(number of distributions). Perspective inversion negates the resulting scalar.
    """
    if (
        probabilities.ndim < 1
        or probabilities.shape[-1] != 3
        or not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not torch.allclose(
            probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-6, rtol=0
        )
    ):
        raise ValueError("Expected finite loss/draw/win probabilities summing to one")
    return probabilities[..., 2] - probabilities[..., 0]


class NeuralEvaluator:
    """WHAT: Adapt unchanged DrewBot heads to search.

    HOW: Encode v1 history (B,64,96) and state (B,72), select H through forward_batch,
    legal-softmax (B,4352), outcome-softmax (B,3), copy results to CPU. WHY: Reuse
    audited contracts without labels or redesign. Calls use eval/inference_mode and
    restore every prior module training flag. Already-evaluating modules avoid recursive
    mode writes; modules are still inspected each call so mixed-mode callers work.
    CPU copies complete asynchronous MPS work before
    return; synchronize() also makes benchmark timing explicit.
    """

    def __init__(self, model):
        self.model = model
        self.history_frames = model.config.history_frames
        if not 1 <= self.history_frames <= 8:
            raise ValueError("Search v1 adapter requires history_frames in [1,8]")
        self.device = next(model.parameters()).device

    def synchronize(self) -> None:
        """WHAT: Finish device work. HOW: MPS barrier. WHY: Honest inference timers."""
        if self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def _evaluation_mode(self):
        """WHAT: Temporarily disable training without redundant recursive mode writes.

        HOW: Inspect all modules, change only true flags, restore those exact flags
        in finally, including mixed-mode trees and failed inference. WHY: Profiling
        found two full recursive train() traversals per leaf even for eval models.
        DrewBot uses standard nn.Module training flags (no custom train hooks).
        O(modules) inspection and O(training modules) writes; no persistent cache.
        """
        training_modules = [module for module in self.model.modules() if module.training]
        for module in training_modules:
            module.training = False
        try:
            yield
        finally:
            for module in training_modules:
                module.training = True

    @torch.inference_mode()
    def evaluate(self, positions: list[PositionInput]) -> list[Evaluation]:
        """WHAT: Infer a bounded batch. HOW: Encode/forward/legal normalize. WHY: Share dispatch."""
        if not positions:
            return []
        tokens = np.stack([encode_history(list(p.frames)) for p in positions])
        state = np.stack([auxiliary(p.board, p.repetitions) for p in positions])
        masks = np.stack([legal_mask(p.board) for p in positions])
        with self._evaluation_mode():
            output = self.model.forward_batch(
                {
                    "tokens": torch.as_tensor(tokens, dtype=torch.float32, device=self.device),
                    "auxiliary": torch.as_tensor(state, device=self.device),
                }
            )
            policy, _ = legal_policy(
                output.policy_logits, torch.as_tensor(masks, device=self.device)
            )
            values = outcome_scalar(output.value_logits.softmax(-1)).cpu().tolist()
            probabilities = policy.cpu().numpy()
        return [
            Evaluation(
                float(values[i]),
                {
                    move: float(probabilities[i, policy_target(p.board, move)[0]])
                    for move in p.board.legal_moves
                },
            )
            for i, p in enumerate(positions)
        ]
