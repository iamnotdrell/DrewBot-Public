"""Streaming sufficient statistics for policy imitation and outcome classification."""

import torch

from drewbot.models.inference import legal_policy
from drewbot.training.losses import target_masks


class Metrics:
    """WHAT: Aggregate without retaining predictions. HOW: Sums/counts and 3x3 confusion.

    WHY: Batch means bias small final batches and missing-label metrics. Policy top-k
    is measured within the legal action set; raw argmax legality is a separate diagnostic.
    Value precision/recall use zero for undefined classes and expose support counts.
    Accuracy measures imitation/outcome forecasting, not playing strength or centipawns.
    """

    def __init__(self):
        self.positions = self.pc = self.vc = self.p_loss_n = self.v_loss_n = 0
        self.p_loss = self.v_loss = self.top1 = self.top5 = 0.0
        self.p_conf = self.p_entropy = self.raw_legal = self.masked_legal = self.v_conf = 0.0
        self.confusion = torch.zeros(3, 3, dtype=torch.int64)

    @torch.no_grad()
    def update(self, output, batch, losses):
        """WHAT: Add one batch. HOW: Detached stable probabilities. WHY: Bounded memory."""
        pm, vm = target_masks(batch)
        self.positions += len(pm)
        if losses.policy_sum is not None:
            self.p_loss += float(losses.policy_sum.detach())
            self.p_loss_n += losses.policy_count
        if losses.value_sum is not None:
            self.v_loss += float(losses.value_sum.detach())
            self.v_loss_n += losses.value_count
        if pm.any():
            logits, mask, target = (
                output.policy_logits[pm],
                batch["legal_mask"][pm],
                batch["policy"][pm],
            )
            probs, best = legal_policy(logits, mask)
            top = logits.masked_fill(~mask, -torch.inf).topk(5, dim=1).indices
            self.pc += len(target)
            self.top1 += int((best == target).sum())
            self.top5 += int((top == target[:, None]).any(1).sum())
            self.p_conf += float(probs.max(1).values.sum())
            self.p_entropy += float(-(probs * probs.masked_fill(probs == 0, 1.0).log()).sum())
            self.raw_legal += int(mask.gather(1, logits.argmax(1)[:, None]).sum())
            self.masked_legal += int(mask.gather(1, best[:, None]).sum())
        if vm.any():
            probs = output.value_logits[vm].softmax(-1)
            target = batch["value"][vm]
            pred = probs.argmax(1)
            self.vc += len(target)
            self.v_conf += float(probs.max(1).values.sum())
            indices = (target * 3 + pred).cpu()
            self.confusion += torch.bincount(indices, minlength=9).reshape(3, 3)

    def result(self, config) -> dict:
        """WHAT: Export comparable metrics. HOW: Normalize observed counts. WHY: Honest nulls."""
        p = self.p_loss / self.p_loss_n if self.p_loss_n else None
        v = self.v_loss / self.v_loss_n if self.v_loss_n else None
        diagonal = self.confusion.diag()
        support = self.confusion.sum(1)
        precision = diagonal / self.confusion.sum(0).clamp_min(1)
        recall = diagonal / support.clamp_min(1)
        terms = [
            weight * loss
            for weight, loss in ((config.policy_weight, p), (config.value_weight, v))
            if loss is not None
        ]
        return {
            "positions": self.positions,
            "policy_labels": self.pc,
            "value_labels": self.vc,
            "loss": sum(terms) if terms else None,
            "policy_loss": p,
            "value_loss": v,
            "policy_top1": self.top1 / self.pc if self.pc else None,
            "policy_top5": self.top5 / self.pc if self.pc else None,
            "legal_move_accuracy": self.masked_legal / self.pc if self.pc else None,
            "raw_argmax_legal_fraction": self.raw_legal / self.pc if self.pc else None,
            "policy_confidence": self.p_conf / self.pc if self.pc else None,
            "policy_entropy_nats": self.p_entropy / self.pc if self.pc else None,
            "value_accuracy": int(diagonal.sum()) / self.vc if self.vc else None,
            "value_confidence": self.v_conf / self.vc if self.vc else None,
            "value_confusion_true_rows": self.confusion.tolist(),
            "value_support": support.tolist(),
            "value_precision": precision.tolist(),
            "value_recall": recall.tolist(),
        }

    def state_dict(self):
        """WHAT: Serialize running stats. HOW: Copy fields. WHY: Mid-epoch resume metrics."""
        return dict(self.__dict__)

    def load_state_dict(self, state):
        """WHAT: Restore sufficient stats. HOW: Assign snapshot. WHY: Continuous epochs."""
        self.__dict__.update(state)
