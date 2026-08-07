"""G-Eval prompt text for human annotators (instruction rubric only)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from human_annotation.config import DATA_ROOT

DEFAULT_PROMPTS_DIR = DATA_ROOT / "prompts" / "geval"

# Everything from the document/summary placeholders through output format is excluded.
_DOCUMENT_SECTION_MARKERS = (
    "\nDocument:\n",
    "\nSource Text:\n",
)


def instruction_only_geval_prompt(full_template: str) -> str:
    """Return rubric/instructions without document slots or JSON output spec."""
    for marker in _DOCUMENT_SECTION_MARKERS:
        if marker in full_template:
            return full_template.split(marker, 1)[0].rstrip()
    if "{{DOCUMENT}}" in full_template:
        head = full_template.split("{{DOCUMENT}}", 1)[0].rstrip()
        # Drop a trailing "Document:" or "Source Text:" label if present.
        for label in ("Document:", "Source Text:"):
            if head.endswith(label):
                return head[: -len(label)].rstrip()
        return head
    return full_template.rstrip()


@lru_cache(maxsize=None)
def load_annotation_prompt(
    dimension: str,
    prompts_dir: str | None = None,
) -> str:
    """Load the human-facing criterion prompt for one dimension."""
    base = Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
    full = (base / f"{dimension}.txt").read_text(encoding="utf-8")
    return instruction_only_geval_prompt(full)
