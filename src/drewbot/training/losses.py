"""Legality-aware categorical objectives with explicit missing-target masks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from drewbot.models.model import ModelOutput
from drewbot.training.config import TrainingConfig


@dataclass
class Losses:
    """WHAT: Independent loss sums/counts. HOW: Named tensors. WHY: Correct window means."""

    policy_sum: torch.Tensor | None
    value_sum: torch.Tensor | None
    policy_count: int
    value_count: int

    def weighted(self, config: TrainingConfig, policy_count=None, value_count=None):
        """WHAT: Weighted means. HOW: Divide by observed labels. WHY: No fabricated gradient.

        Window counts allow exact gradient accumulation even when microbatches have
        unequal sizes or missing values. An entirely unlabelled window returns None.
        """
        total = None
        for loss, count, weight in (
            (
                self.policy_sum,
                self.policy_count if policy_count is None else policy_count,
                config.policy_weight,
            ),
            (
                self.value_sum,
                self.value_count if value_count is None else value_count,
                config.value_weight,
            ),
        ):
            if loss is not None and count:
                term = weight * loss / count
                total = term if total is None else total + term
        return total


def target_masks(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """WHAT: Validate targets. HOW: Check shapes/types/ranges/legality. WHY: Fail before CE.

    Policy -100 is explicitly missing; v1 provides every policy target. Value -100
    must match value_valid=False. Classes 0/1/2 mean mover-relative loss/draw/win.
    Terminal rows are allowed only when policy is missing. No false terminal softmax.
    """
    policy, value, mask, valid = (
        batch[k] for k in ("policy", "value", "legal_mask", "value_valid")
    )
    if (
        policy.ndim != 1
        or value.shape != policy.shape
        or valid.shape != value.shape
        or mask.shape != (len(policy), 4352)
        or policy.dtype != torch.int64
        or value.dtype != torch.int64
        or valid.dtype != torch.bool
        or mask.dtype != torch.bool
        or len({t.device for t in (policy, value, mask, valid)}) != 1
    ):
        raise ValueError("Invalid target shape/dtype/device")
    pm, vm = policy != -100, value != -100
    if not torch.equal(vm, valid):
        raise ValueError("value_valid disagrees with missing value sentinel")
    if (
        (policy[pm] < 0).any()
        or (policy[pm] >= 4352).any()
        or (value[vm] < 0).any()
        or (value[vm] >= 3).any()
    ):
        raise ValueError("Target class out of range")
    if pm.any() and not mask[pm].gather(1, policy[pm, None]).all():
        raise ValueError("Policy target is illegal")
    return pm, vm


def compute_losses(output: ModelOutput, batch: dict, config: TrainingConfig) -> Losses:
    """WHAT: Compute CE sums for active observed labels. HOW: Legal masking before logsumexp.

    Inputs logits (B,4352)/(B,3), labels (B,), legal bool (B,4352). CE =
    logsumexp(z_legal)-z_target, minimized through heads and shared encoder.
    torch CE uses stable log-softmax. Illegal logits receive exactly zero gradient.
    Missing heads return None and get no gradient/AdamW decay. Cost O(B*4352).
    Alternatives: soft engine targets or regression require data we do not have.
    """
    pm, vm = target_masks(batch)
    p, v = output.policy_logits, output.value_logits
    if (
        p.shape != batch["legal_mask"].shape
        or v.shape != (len(pm), 3)
        or p.dtype != torch.float32
        or v.dtype != torch.float32
        or p.device != batch["policy"].device
        or v.device != p.device
    ):
        raise ValueError("Invalid logits shape/dtype/device")
    if not torch.isfinite(p).all() or not torch.isfinite(v).all():
        raise FloatingPointError("Non-finite logits; update refused")
    policy_sum = value_sum = None
    pc = int(pm.sum()) if config.objective in {"policy", "joint"} else 0
    vc = int(vm.sum()) if config.objective in {"value", "joint"} else 0
    if pc:
        masked = p[pm].masked_fill(~batch["legal_mask"][pm], -torch.inf)
        policy_sum = F.cross_entropy(masked, batch["policy"][pm], reduction="sum")
    if vc:
        value_sum = F.cross_entropy(v[vm], batch["value"][vm], reduction="sum")
    for loss in (policy_sum, value_sum):
        if loss is not None and not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; update refused")
    return Losses(policy_sum, value_sum, pc, vc)
