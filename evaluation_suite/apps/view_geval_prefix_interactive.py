#!/usr/bin/env python3
"""Organized entry point forwarding to legacy interactive viewer script."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    legacy = repo_root / "Other" / "view_geval_prefix_interactive.py"
    if not legacy.is_file():
        raise FileNotFoundError(f"Legacy interactive script not found: {legacy}")
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__":
    main()

