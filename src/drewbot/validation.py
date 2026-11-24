"""Shared strict numeric setting validators; no training/model import cycle."""

import math


def positive(value, name: str, optional: bool = False) -> None:
    """WHAT: Validate integer capacities. HOW: Reject booleans and nonpositive values. WHY:
    Clear errors."""
    if optional and value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or null" if optional else ""))


def finite(value, name: str, minimum: float = 0, optional: bool = False) -> None:
    """WHAT: Validate rates. HOW: Check types, finiteness, lower bound. WHY: No NaN
    configuration."""
    if optional and value is None:
        return
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
