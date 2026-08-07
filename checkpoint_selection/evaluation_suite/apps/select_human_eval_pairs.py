#!/usr/bin/env python3
"""Organized entry point forwarding to legacy selector script."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    legacy = repo_root / "human evaluation" / "select_human_eval_pairs.py"
    if not legacy.is_file():
        raise FileNotFoundError(f"Legacy selector not found: {legacy}")
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__":
    main()

