"""Shared validation helpers for benchmark command-line inputs."""

from __future__ import annotations

import argparse


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
    """Parse a comma-separated, non-empty list of positive integers."""

    message = "expected a comma-separated list of positive integers"
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(message) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(message)
    return parsed


def element_count(message_bytes: int, element_size: int) -> int:
    """Return the exact element count represented by a byte-sized message."""

    if message_bytes <= 0 or element_size <= 0 or message_bytes % element_size != 0:
        raise ValueError("message bytes must be a positive multiple of the dtype size")
    return message_bytes // element_size
