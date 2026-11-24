"""Legality-aware mechanics, separate from raw training-facing head outputs."""

from __future__ import annotations

import chess
import torch

from drewbot.data.walkthrough import ACTION_COUNT, decode_action, legal_mask


def legal_policy(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """WHAT: Return legal probabilities (B,4352) and best indices (B,), -1 if no action.

    HOW: Mask illegal logits before stable softmax. Replace all-false rows with zeros
    BEFORE softmax, then set their probabilities to zero. Illegal mass is exactly zero.
    WHY: Checkmate/stalemate must not yield NaNs or a fabricated playable move.
    Finite float32 logits, boolean same-device masks and positive B are required.
    Cost O(B*4352). This is inference normalization, not a training loss.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] < 1
        or logits.shape[1] != ACTION_COUNT
        or logits.dtype != torch.float32
    ):
        raise ValueError("policy logits must be float32 (B,4352), B>0")
    if (
        not isinstance(mask, torch.Tensor)
        or mask.shape != logits.shape
        or mask.dtype != torch.bool
        or mask.device != logits.device
    ):
        raise ValueError("legal mask must be bool, matching policy shape and device")
    if not torch.isfinite(logits).all():
        raise ValueError("policy logits must be finite")
    playable = mask.any(dim=1)
    masked = logits.masked_fill(~mask, -torch.inf)
    masked = torch.where(playable[:, None], masked, torch.zeros_like(masked))
    probabilities = masked.softmax(dim=-1).masked_fill(~mask, 0.0)
    indices = probabilities.argmax(dim=-1).masked_fill(~playable, -1)
    return probabilities, indices


@torch.no_grad()
def choose_move(logits: torch.Tensor, board: chess.Board) -> chess.Move | None:
    """WHAT: Decode the best legal action or None from a single board's (4352,) logits.

    HOW: Use existing terminal-aware legal mask and rank-reflection inverse for Black.
    WHY: Promotion identity, castling and EP must obey the same vocabulary as labels.
    Random model outputs demonstrate mechanics only; this does not implement search.
    """
    if not isinstance(logits, torch.Tensor) or logits.shape != (ACTION_COUNT,):
        raise ValueError("choose_move requires one (4352,) policy vector")
    mask = torch.as_tensor(legal_mask(board), device=logits.device).unsqueeze(0)
    _, indices = legal_policy(logits.unsqueeze(0), mask)
    index = int(indices.item())
    return None if index == -1 else decode_action(index, board.turn)
