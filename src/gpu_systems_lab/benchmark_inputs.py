"""Shared validation helpers for benchmark command-line inputs."""

from __future__ import annotations

import argparse
from collections.abc import Collection


def positive_integer(value: str) -> int:
    """Parse one positive integer for ``argparse``."""

    message = "expected a positive integer"
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(message) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(message)
    return parsed


def positive_integer_list(value: str) -> list[int]:
    """Parse a comma-separated, non-empty unique list of positive integers."""

    message = "expected a comma-separated list of positive integers"
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(message) from error
    if not parsed or any(item <= 0 for item in parsed) or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(message)
    return parsed


def positive_float(value: str) -> float:
    """Parse one finite positive floating-point value for ``argparse``."""

    message = "expected a finite positive number"
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(message) from error
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError(message)
    return parsed


def element_count(message_bytes: int, element_size: int) -> int:
    """Return the exact element count represented by a byte-sized message."""

    if message_bytes <= 0 or element_size <= 0 or message_bytes % element_size != 0:
        raise ValueError("message bytes must be a positive multiple of the dtype size")
    return message_bytes // element_size


def choice_list(value: str, *, allowed: Collection[str], label: str) -> list[str]:
    """Parse a unique comma-separated list constrained to known values."""

    parsed = [item.strip() for item in value.split(",")]
    if not parsed or any(not item for item in parsed):
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of {label}")
    unknown = [item for item in parsed if item not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported {label.rstrip('s')}: {unknown[0]}")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"duplicate {label.rstrip('s')} are not allowed")
    return parsed
