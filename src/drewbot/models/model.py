"""DrewBot Phase 3 model and the explicit v1 Dataset history boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from drewbot.models.attention import AttentionInspection, EncoderBlock
from drewbot.models.heads import PolicyHead, ValueHead
from drewbot.settings import ModelSettings


def latest_v1_history(tokens: torch.Tensor, frames: int) -> torch.Tensor:
    """WHAT: Adapt v1 (B,64,96) to (B,64,12H) without modifying storage.

    HOW: Select the latest contiguous H frames (oldest first, current last).
    WHY: An explicit adapter avoids silently reinterpreting eight-frame shards.
    H>8 requires newly encoded input through direct forward. This returns a view.
    """
    if type(frames) is not int or not 1 <= frames <= 8:
        raise ValueError("v1 shards support only 1..8 latest history frames")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[1:] != (64, 96):
        raise ValueError("v1 history adapter requires (B,64,96) tokens")
    return tokens[:, :, -12 * frames :]


@dataclass(frozen=True)
class ModelOutput:
    """WHAT: Stable training-independent result. HOW: Named raw logits and optional maps.

    WHY: Separate (B,4352) policy and (B,3) loss/draw/win logits permit future
    independent or joint objectives; diagnostics are absent by default.
    """

    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    inspections: tuple[AttentionInspection, ...] | None = None


class InputProjection(nn.Module):
    """WHAT: Embed piece history and explicit state into 64 tokens of depth D.

    HOW: Concatenate (B,72) state broadcast over (B,64,12H), then one affine
    projection. Explicit 'none' mode omits state for ablation. No labels are used.
    WHY: Same pieces can have different castling/EP rights. This DrewBot extension
    preserves the square-token count without claiming Maia rating compatibility.
    Cost O(B*64*(12H+72)*D), with expanded state a view until concatenation.
    """

    def __init__(self, config: ModelSettings):
        super().__init__()
        self.auxiliary_mode = config.auxiliary_mode
        width = 12 * config.history_frames + (72 if self.auxiliary_mode == "broadcast" else 0)
        self.projection = nn.Linear(width, config.embedding_dim)

    def forward(self, tokens: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """WHAT: Condition square representations. HOW: Broadcast state. WHY: Retain legal state."""
        if self.auxiliary_mode == "broadcast":
            tokens = torch.cat((tokens, auxiliary[:, None, :].expand(-1, 64, -1)), dim=-1)
        return self.projection(tokens)


class DrewBot(nn.Module):
    """WHAT: Configurable custom encoder, source/destination policy and outcome heads.

    HOW: Validate float32 inputs, project state, apply post-norm blocks and final LN,
    then independent heads. GAB layers all receive the SAME model-owned template
    parameter. Standard mode instead adds a learned absolute square embedding.
    WHY: Keep faithful dynamic GAB mechanics separate from documented DrewBot
    adaptations and from any future training or search workflow. See MODEL_SPEC.md
    for every shape, formula, research reference and complexity term.
    """

    def __init__(self, config: ModelSettings | None = None):
        super().__init__()
        self.config = config if config is not None else ModelSettings()
        config = self.config
        self.input_projection = InputProjection(config)
        if config.gab_enabled:
            self.templates = nn.Parameter(torch.empty(4096, config.gab_templates))
            nn.init.xavier_normal_(self.templates)
            self.register_parameter("square_embedding", None)
        else:
            self.register_parameter("templates", None)
            self.square_embedding = nn.Parameter(torch.empty(1, 64, config.embedding_dim))
            nn.init.normal_(self.square_embedding, std=0.02)
        self.layers = nn.ModuleList(EncoderBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.policy_head = PolicyHead(config.embedding_dim)
        self.value_head = ValueHead(config.embedding_dim, config.value_hidden_dim)

    def _validate(self, tokens: torch.Tensor, auxiliary: torch.Tensor) -> None:
        """WHAT: Enforce inputs. HOW: Check shape/type/device/finiteness. WHY: Clear failures."""
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] < 1
            or tokens.shape[1:] != (64, 12 * self.config.history_frames)
        ):
            raise ValueError("tokens must have shape (B,64,12*history_frames), B>0")
        if not isinstance(auxiliary, torch.Tensor) or auxiliary.shape != (tokens.shape[0], 72):
            raise ValueError("auxiliary must have shape (B,72)")
        parameter = self.input_projection.projection.weight
        if tokens.dtype != torch.float32 or auxiliary.dtype != torch.float32:
            raise ValueError("tokens and auxiliary must be float32")
        if parameter.dtype != torch.float32:
            raise ValueError("Phase 3 model parameters must remain float32")
        if tokens.device != parameter.device or auxiliary.device != parameter.device:
            raise ValueError("tokens, auxiliary and model must be on the same device")
        if not torch.isfinite(tokens).all() or not torch.isfinite(auxiliary).all():
            raise ValueError("tokens and auxiliary must be finite")

    def forward(
        self, tokens: torch.Tensor, auxiliary: torch.Tensor, *, inspect: bool = False
    ) -> ModelOutput:
        """WHAT: Return raw logits. HOW: Encode then branch. WHY: Keep labels outside forward."""
        self._validate(tokens, auxiliary)
        if type(inspect) is not bool:
            raise ValueError("inspect must be boolean")
        x = self.input_projection(tokens, auxiliary)
        if self.square_embedding is not None:
            x = x + self.square_embedding
        inspections = []
        for layer in self.layers:
            x, details = layer(x, self.templates, inspect=inspect)
            if details is not None:
                inspections.append(details)
        x = self.final_norm(x)
        return ModelOutput(
            self.policy_head(x), self.value_head(x), tuple(inspections) if inspect else None
        )

    def forward_batch(self, batch: Mapping, *, inspect: bool = False) -> ModelOutput:
        """WHAT: Consume v1 inputs. HOW: Select latest H explicitly. WHY: Prevent label leakage.

        Caller moves tokens/auxiliary to the model's device; labels, legal masks and
        metadata are intentionally ignored, including value=-100/value_valid=False.
        """
        tokens = latest_v1_history(batch["tokens"], self.config.history_frames)
        return self(tokens, batch["auxiliary"], inspect=inspect)
