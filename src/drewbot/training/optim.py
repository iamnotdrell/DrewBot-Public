"""AdamW, explicit warmup/cosine updates and checked gradients."""

import math

import torch


def make_optimizer(model, config):
    """WHAT: Initialize AdamW. HOW: Decoupled decay and configured LR. WHY: Simple baseline.

    Uniform decay is explicit (including norms/biases); parameter-group exemptions
    are an experiment alternative. Parameters with grad=None are not updated/decayed.
    """
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


class WarmupCosine:
    """WHAT: Set LR for each optimizer update. HOW: Linear warmup then cosine floor.

    WHY: Smooth transformer startup without complex adaptive schedules. Index 0 is
    the first update; warmup=2 uses 0.5,1.0 base LR. State includes the original
    budget, so interrupted runs must keep epochs/max_steps unchanged.
    """

    def __init__(self, optimizer, config, total_steps):
        self.optimizer, self.config = optimizer, config
        self.total_steps = total_steps
        self.completed = 0
        self.apply()

    def ratio(self, index):
        """WHAT: Compute multiplier. HOW: Warmup and bounded cosine. WHY: Predictable endpoints."""
        warmup = min(self.config.warmup_steps, self.total_steps)
        if index < warmup:
            return (index + 1) / warmup
        if self.config.schedule == "constant":
            return 1.0
        progress = min(1.0, (index - warmup) / max(1, self.total_steps - warmup - 1))
        return (
            self.config.min_lr_ratio
            + (1 - self.config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )

    def apply(self):
        """WHAT: Prepare next update LR. HOW: Set groups. WHY: No scheduler off-by-one."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * self.ratio(self.completed)

    def step(self):
        """WHAT: Advance after update. HOW: Increment then apply. WHY: Correct resume."""
        self.completed += 1
        self.apply()

    def state_dict(self):
        """WHAT: Persist schedule. HOW: Save update/budget. WHY: Resume identical trajectory."""
        return {"completed": self.completed, "total_steps": self.total_steps}

    def load_state_dict(self, state):
        """WHAT: Restore schedule. HOW: Reject changed budget. WHY: Comparable continuation."""
        if state["total_steps"] != self.total_steps:
            raise ValueError("Resume changes the scheduler training budget")
        self.completed = state["completed"]
        self.apply()


def checked_gradient_norm(model, maximum):
    """WHAT: Reject nonfinite gradients then clip. HOW: Torch finite checks and global L2.

    WHY: Fail before any optimizer mutation. Returns the pre-clip norm for diagnosis.
    Checking each gradient also catches inactive-path anomalies; no silent NaN skipping.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Missing or non-finite gradients; update refused")
    norm = torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)
    return float(norm)
