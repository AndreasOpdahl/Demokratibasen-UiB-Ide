"""
Local LLM G-Eval judge: fill a prompts/geval template and POST to a local chat API.

Typical use:

    from geval_local_judge import local_llm_geval_judge, make_model_judge

    text = local_llm_geval_judge(
        document,
        summary_a,
        summary_b,
        judge="faithfulness",
        model="google/gemma-3-4b",
    )

    judge_fn = make_model_judge("google/gemma-3-4b", "faithfulness")
    text = judge_fn(document, summary_a, summary_b)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import requests

JudgeName = Literal["faithfulness", "correctness", "completeness"]

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_PROMPTS_DIR = _REPO_ROOT / "Data" / "prompts" / "geval"
_DEFAULT_CHAT_URL = "http://localhost:1234/api/v1/chat"

_JUDGE_FILES: Dict[str, str] = {
    "faithfulness": "faithfulness.txt",
    "correctness": "correctness.txt",
    "completeness": "completeness.txt",
}


def fill_geval_prompt(
    template: str,
    document: str,
    summary_a: str,
    summary_b: str,
) -> str:
    """Substitute placeholders in a G-Eval template string.

    Input: template text, document body, summary A/B. Output: single user message for the LLM.
    """
    return (
        template.replace("{{DOCUMENT}}", document)
        .replace("{{SUMMARY_A}}", summary_a)
        .replace("{{SUMMARY_B}}", summary_b)
    )


def load_geval_template(
    judge: Union[JudgeName, str, Path],
    *,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Load a ``.txt`` prompt from ``prompts_dir`` (default ``Data/prompts/geval``).

    Input: dimension key (e.g. ``faithfulness``), filename, or absolute path; optional base dir.
    Output: template file contents as a string.
    """
    base = prompts_dir if prompts_dir is not None else _DEFAULT_PROMPTS_DIR
    if isinstance(judge, Path):
        path = judge
    else:
        key = str(judge)
        if key in _JUDGE_FILES:
            path = base / _JUDGE_FILES[key]
        else:
            p = Path(key)
            path = p if p.is_absolute() else base / p
    return path.read_text(encoding="utf-8")


def local_llm_chat(
    user_input: str,
    *,
    model: str,
    system_prompt: str = "",
    base_url: str = _DEFAULT_CHAT_URL,
    timeout_s: float = 300.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Send a chat completion request to a LM Studio–style local server.

    Input: user message, model id, system prompt, base URL, timeout, optional session.
    Output: parsed JSON response body (dict).
    """
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "input": user_input,
    }
    req = session or requests
    r = req.post(
        base_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def extract_assistant_text(data: Dict[str, Any]) -> str:
    """Pull assistant text from API JSON (``output`` list entries with ``type == 'message'``).

    Input: parsed chat response dict. Output: concatenated assistant string (may be empty).
    """
    parts: list[str] = []
    for block in data.get("output") or []:
        if isinstance(block, dict) and block.get("type") == "message":
            parts.append(str(block.get("content", "")))
    return "\n".join(parts).strip()


def _sanitize_identifier(model: str) -> str:
    """Normalize a model id into a lowercase alphanumeric+underscore fragment.

    Input: model string. Output: safe suffix for generated function names.
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", model)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "model"


def local_llm_geval_judge(
    document: str,
    summary_a: str,
    summary_b: str,
    *,
    judge: Union[JudgeName, str, Path] = "faithfulness",
    model: str,
    base_url: str = _DEFAULT_CHAT_URL,
    prompts_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    timeout_s: float = 300.0,
    session: Optional[requests.Session] = None,
) -> str:
    """Run one pairwise G-Eval: load template for ``judge`` dimension, fill, call local LLM.

    Input: document + two summaries; ``judge`` selects prompt file; ``model`` is the API model id.
    Output: assistant reply text (expected to contain A, B, or Tie).
    """
    template = load_geval_template(judge, prompts_dir=prompts_dir)
    user_message = fill_geval_prompt(template, document, summary_a, summary_b)
    sys_msg = system_prompt or (
        "You are an evaluator. Follow the user instructions exactly. "
        "If asked for one word (A, B, or Tie), reply with only that word."
    )
    data = local_llm_chat(
        user_message,
        model=model,
        system_prompt=sys_msg,
        base_url=base_url,
        timeout_s=timeout_s,
        session=session,
    )
    return extract_assistant_text(data)


def make_model_judge(
    model: str,
    judge: JudgeName,
    *,
    base_url: str = _DEFAULT_CHAT_URL,
    prompts_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    timeout_s: float = 300.0,
    session: Optional[requests.Session] = None,
) -> Callable[[str, str, str], str]:
    """Return a callable that runs :func:`local_llm_geval_judge` with fixed model + dimension.

    Input: model id, judge dimension, optional API options. Output: ``(doc, sum_a, sum_b) -> str``.
    """
    safe = _sanitize_identifier(model)

    def fn(document: str, summary_a: str, summary_b: str) -> str:
        """Bound G-Eval call for one (model, judge) pair."""
        return local_llm_geval_judge(
            document,
            summary_a,
            summary_b,
            judge=judge,
            model=model,
            base_url=base_url,
            prompts_dir=prompts_dir,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            session=session,
        )

    fn.__name__ = f"{safe}_{judge}_judge"
    fn.__qualname__ = fn.__name__
    return fn


__all__ = [
    "fill_geval_prompt",
    "load_geval_template",
    "local_llm_chat",
    "extract_assistant_text",
    "local_llm_geval_judge",
    "make_model_judge",
]
