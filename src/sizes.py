from __future__ import annotations

import argparse
import re

SIZE_PATTERN = r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*"
SIZE_UNITS = ("B", "KB", "MB", "GB", "TB")
SIZE_FACTORS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_size(value: str) -> int:
    match = re.fullmatch(SIZE_PATTERN, value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    if unit not in SIZE_FACTORS:
        raise argparse.ArgumentTypeError(f"Invalid size unit: {unit}")
    size = int(number * SIZE_FACTORS[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError("Size must be greater than zero")
    return size


def human_size(size: int) -> str:
    value = float(size)
    for unit in SIZE_UNITS:
        if value < 1000 or unit == SIZE_UNITS[-1]:
            return f"{value:.2f}{unit}" if unit != "B" else f"{size}B"
        value /= 1000
    return f"{size}B"
