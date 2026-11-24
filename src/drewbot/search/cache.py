"""Bounded LRU reuse and explicit alpha-beta bound semantics."""

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

import chess

MATE = 10000.0
MATE_THRESHOLD = MATE - 128
T = TypeVar("T")


class LRU(Generic[T]):
    """WHAT: Bound cache entries. HOW: Evict least recently used. WHY: Local RAM limits.

    Keys include exact context witnesses (SEARCH_SPEC.md). Zero disables storage.
    Entry limits bound retained entries, not exact Python byte usage.
    """

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        self.capacity = capacity
        self.entries: OrderedDict[tuple, T] = OrderedDict()

    def get(self, key: tuple) -> T | None:
        """WHAT: Reuse an entry. HOW: Touch LRU order. WHY: Keep active subtrees."""
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key: tuple, value: T) -> None:
        """WHAT: Store within capacity. HOW: Replace/evict. WHY: Never grow unbounded."""
        if not self.capacity:
            return
        self.entries[key] = value
        self.entries.move_to_end(key)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class Bound(StrEnum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True)
class TTEntry:
    """WHAT: Store a search fact. HOW: Depth/bound/score/move. WHY: Bounds are not exact values."""

    depth: int
    score: float
    bound: Bound
    move: chess.Move | None
    claim_draw: bool = False


def store_score(score: float, ply: int) -> float:
    """WHAT: Normalize mate distance. HOW: Remove root ply. WHY: Re-root TT entries safely."""
    return (
        score + ply
        if score >= MATE_THRESHOLD
        else score - ply
        if score <= -MATE_THRESHOLD
        else score
    )


def load_score(score: float, ply: int) -> float:
    """WHAT: Restore root mate distance. HOW: Invert normalization. WHY: Prefer shorter wins."""
    return (
        score - ply
        if score >= MATE_THRESHOLD
        else score + ply
        if score <= -MATE_THRESHOLD
        else score
    )
