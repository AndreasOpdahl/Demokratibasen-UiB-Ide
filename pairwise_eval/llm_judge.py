"""Pairwise row evaluator: local LM Studio chat and/or OpenAI Chat Completions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    """Prepend repo root to ``sys.path`` so ``import geval_local_judge`` works from the package."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


def parse_geval_ab_tie(raw: str, *, left_model: str, right_model: str) -> Dict[str, Any]:
    """Parse G-Eval reply text into pipeline fields (first A/B/Tie token wins).

    Input: raw LLM string; ``left_model`` / ``right_model`` ids for the pair. Output: dict with
    ``choice_side``, ``chosen``, ``rationale``.
    """
    text = (raw or "").strip()
    upper = text.upper()
    tokens = re.findall(r"\b(A|B|TIE)\b", upper)
    if not tokens:
        return {
            "choice_side": "tie",
            "chosen": pd.NA,
            "rationale": f"[unparseable] {text}",
        }
    w = tokens[0]
    if w == "TIE":
        return {"choice_side": "tie", "chosen": pd.NA, "rationale": text}
    if w == "A":
        return {"choice_side": "left", "chosen": left_model, "rationale": text}
    return {"choice_side": "right", "chosen": right_model, "rationale": text}


def _openai_assistant_text(message: Mapping[str, Any]) -> str:
    """Normalize Chat Completions ``message.content`` (string, None, or list of parts) to plain text."""
    raw = message.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(raw).strip()


def _openai_chat_completion(
    *,
    user_content: str,
    system_content: str,
    model: str,
    api_key: str,
    url: str,
    timeout_s: float,
) -> str:
    """POST OpenAI Chat Completions (``/v1/chat/completions``); return assistant message text."""
    key = api_key.strip()
    if not key:
        raise RuntimeError("OpenAI API key is empty after strip()")

    r = requests.post(
        url.strip(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        },
        timeout=timeout_s,
    )

    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError(f"OpenAI response is not JSON (HTTP {r.status_code}): {r.text[:400]}") from e

    if not r.ok:
        err = data.get("error") if isinstance(data, dict) else None
        detail = ""
        if isinstance(err, dict) and err.get("message"):
            detail = str(err["message"])
        elif isinstance(err, str):
            detail = err
        else:
            detail = r.text[:500] if r.text else r.reason
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {detail}")

    choices = data.get("choices")
    if not choices:
        raise RuntimeError("OpenAI response has no choices[]")
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError("OpenAI response choice missing message object")
    return _openai_assistant_text(msg)


def openai_llm_geval_judge(
    document: str,
    summary_a: str,
    summary_b: str,
    *,
    judge: str,
    model: str,
    api_key: str,
    completions_url: str,
    prompts_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    timeout_s: float = 300.0,
) -> str:
    """Same templates as local G-Eval; send via OpenAI Chat Completions."""
    _ensure_repo_on_path()
    from geval_local_judge import fill_geval_prompt, load_geval_template

    template = load_geval_template(judge, prompts_dir=prompts_dir)
    user_message = fill_geval_prompt(template, document, summary_a, summary_b)
    sys_msg = system_prompt or (
        "You are an evaluator. Follow the user instructions exactly. "
        "If asked for one word (A, B, or Tie), reply with only that word."
    )
    return _openai_chat_completion(
        user_content=user_message,
        system_content=sys_msg,
        model=model,
        api_key=api_key,
        url=completions_url,
        timeout_s=timeout_s,
    )


def make_local_llm_evaluate_fn(
    *,
    base_url: Optional[str] = None,
    timeout_s: Optional[float] = None,
    prompts_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    openai_api_key: Optional[str] = None,
    openai_completions_url: Optional[str] = None,
    openai_judges: Optional[frozenset[str]] = None,
):
    """Factory: return an ``evaluate_pair`` for local and/or OpenAI judges.

    Judges listed in ``openai_judges`` (default: :data:`pairwise_eval.config.OPENAI_JUDGE_IDS`)
    use OpenAI Chat Completions and require ``openai_api_key``. Other judges use the local
    LM Studio–style endpoint (``LOCAL_LLM_CHAT_URL``).

    Checkpoints are one JSONL per (``judge_id``, dimension), so adding an OpenAI judge does not
    overwrite existing local-judge checkpoint files.
    """
    from pairwise_eval.config import (
        LOCAL_LLM_CHAT_URL,
        LOCAL_LLM_TIMEOUT_S,
        OPENAI_API_KEY as _cfg_openai_key,
        OPENAI_CHAT_COMPLETIONS_URL,
        OPENAI_JUDGE_IDS,
    )

    url = base_url or LOCAL_LLM_CHAT_URL
    t_out = LOCAL_LLM_TIMEOUT_S if timeout_s is None else timeout_s
    oai_url = openai_completions_url or OPENAI_CHAT_COMPLETIONS_URL
    oai_j = openai_judges if openai_judges is not None else OPENAI_JUDGE_IDS
    oai_key = openai_api_key if openai_api_key is not None else _cfg_openai_key

    def evaluate_pair(
        row: Mapping[str, Any],
        dimension: str,
        judge_id: str,
        rng: np.random.Generator,
    ) -> Dict[str, object]:
        """One LLM comparison: fill prompt for ``dimension``, POST, parse A/B/Tie."""
        del rng  # LLM judge is deterministic given server state
        _ensure_repo_on_path()

        if verbose:
            backend = "openai" if judge_id in oai_j else "local"
            print(
                f"[LLM G-Eval] {backend} | {judge_id} | {dimension} | {row['doc_id']}",
                flush=True,
            )

        try:
            if judge_id in oai_j:
                if not oai_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is not set (or pass openai_api_key=...) for OpenAI judges"
                    )
                assistant = openai_llm_geval_judge(
                    str(row["source_text"]),
                    str(row["sumleft"]),
                    str(row["sumright"]),
                    judge=dimension,
                    model=judge_id,
                    api_key=oai_key,
                    completions_url=oai_url,
                    prompts_dir=prompts_dir,
                    system_prompt=system_prompt,
                    timeout_s=t_out,
                )
            else:
                from geval_local_judge import local_llm_geval_judge

                assistant = local_llm_geval_judge(
                    str(row["source_text"]),
                    str(row["sumleft"]),
                    str(row["sumright"]),
                    judge=dimension,
                    model=judge_id,
                    base_url=url,
                    prompts_dir=prompts_dir,
                    system_prompt=system_prompt,
                    timeout_s=t_out,
                )
        except Exception as e:
            return {
                "choice_side": "tie",
                "chosen": pd.NA,
                "rationale": f"[api_error] {e!s}",
            }
        return parse_geval_ab_tie(
            assistant, left_model=str(row["left"]), right_model=str(row["right"])
        )

    return evaluate_pair
