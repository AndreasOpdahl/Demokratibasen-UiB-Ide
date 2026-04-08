"""Pairwise row evaluator: local LM Studio chat and/or OpenAI Chat Completions."""

from __future__ import annotations

import json
import re
import sys
import time
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


def _casefold_keys(obj: Any) -> dict[str, Any] | None:
    """If ``obj`` is a dict, return a copy with lowercase string keys (first wins on clash)."""
    if not isinstance(obj, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = str(k).lower()
        if key not in out:
            out[key] = v
    return out


def _try_parse_json_object_with_decision(text: str) -> tuple[dict[str, Any], str] | None:
    """Parse assistant text as JSON object; return (normalized_key_dict, outer_text) or None."""
    outer = text.strip()
    if not outer:
        return None
    s = outer
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()

    def _load() -> Any:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            start = s.find("{")
            end = s.rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(s[start : end + 1])

    try:
        obj = _load()
    except json.JSONDecodeError:
        return None
    norm = _casefold_keys(obj)
    if norm is None or "decision" not in norm:
        return None
    return norm, outer


def _normalize_ab_tie_token(decision: str) -> str | None:
    """Map a decision string to ``A``, ``B``, or ``TIE``; return None if not recognized."""
    u = decision.strip().upper()
    if u in ("A", "B", "TIE"):
        return u
    tokens = re.findall(r"\b(A|B|TIE)\b", u)
    return tokens[0] if tokens else None


def parse_geval_ab_tie(raw: str, *, left_model: str, right_model: str) -> Dict[str, Any]:
    """Parse G-Eval reply: JSON ``{ "decision", "rationale" }`` if present, else first A/B/Tie token.

    Input: raw LLM string; ``left_model`` / ``right_model`` ids for the pair. Output: dict with
    ``choice_side``, ``chosen``, ``rationale``.
    """
    text = (raw or "").strip()
    parsed = _try_parse_json_object_with_decision(text)
    if parsed is not None:
        obj, outer = parsed
        dec_raw = obj.get("decision")
        rat = obj.get("rationale")
        rationale_str = (
            str(rat).strip()
            if rat is not None
            else outer
        )
        if dec_raw is None:
            return {
                "choice_side": "tie",
                "chosen": pd.NA,
                "rationale": f"[unparseable] {outer}",
            }
        w = _normalize_ab_tie_token(str(dec_raw))
        if not w:
            return {
                "choice_side": "tie",
                "chosen": pd.NA,
                "rationale": f"[unparseable decision] {dec_raw!r} | {rationale_str}",
            }
        if w == "TIE":
            return {"choice_side": "tie", "chosen": pd.NA, "rationale": rationale_str}
        if w == "A":
            return {"choice_side": "left", "chosen": left_model, "rationale": rationale_str}
        return {"choice_side": "right", "chosen": right_model, "rationale": rationale_str}

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


def _gemini_response_text(data: Mapping[str, Any]) -> str:
    """Extract plain text from a Gemini ``generateContent`` JSON body."""
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        return ""
    first = cands[0] if isinstance(cands[0], dict) else {}
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(str(p["text"]))
    return "\n".join(texts).strip()


def _gemini_generate_content(
    *,
    user_content: str,
    system_content: str,
    model: str,
    api_key: str,
    api_base: str,
    timeout_s: float,
) -> str:
    """POST Gemini ``generateContent``; return model text."""
    key = api_key.strip()
    if not key:
        raise RuntimeError("Google API key is empty after strip()")

    base = api_base.strip().rstrip("/")
    url = f"{base}/models/{model}:generateContent"
    r = requests.post(
        url,
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system_content}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {"temperature": 0},
        },
        timeout=timeout_s,
    )

    try:
        data = r.json()
    except ValueError as e:
        raise RuntimeError(
            f"Gemini response is not JSON (HTTP {r.status_code}): {r.text[:400]}"
        ) from e

    if not r.ok:
        err = data.get("error") if isinstance(data, dict) else None
        detail = ""
        if isinstance(err, dict) and err.get("message"):
            detail = str(err["message"])
        elif isinstance(err, str):
            detail = err
        else:
            detail = r.text[:500] if r.text else r.reason
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {detail}")

    text = _gemini_response_text(data)
    if not text:
        raise RuntimeError(f"Gemini response has no text (HTTP {r.status_code})")
    return text


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
        "When the prompt asks for JSON, reply with only a valid JSON object "
        'with \"decision\" (A, B, or Tie) and \"rationale\" (string). '
        "Otherwise, if asked for one word (A, B, or Tie), reply with only that word."
    )
    return _openai_chat_completion(
        user_content=user_message,
        system_content=sys_msg,
        model=model,
        api_key=api_key,
        url=completions_url,
        timeout_s=timeout_s,
    )


def gemini_llm_geval_judge(
    document: str,
    summary_a: str,
    summary_b: str,
    *,
    judge: str,
    model: str,
    api_key: str,
    api_base: str,
    prompts_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    timeout_s: float = 300.0,
) -> str:
    """Same templates as local G-Eval; send via Gemini ``generateContent`` (``model`` = API id, no ``google/``)."""
    _ensure_repo_on_path()
    from geval_local_judge import fill_geval_prompt, load_geval_template

    template = load_geval_template(judge, prompts_dir=prompts_dir)
    user_message = fill_geval_prompt(template, document, summary_a, summary_b)
    sys_msg = system_prompt or (
        "You are an evaluator. Follow the user instructions exactly. "
        "When the prompt asks for JSON, reply with only a valid JSON object "
        'with \"decision\" (A, B, or Tie) and \"rationale\" (string). '
        "Otherwise, if asked for one word (A, B, or Tie), reply with only that word."
    )
    return _gemini_generate_content(
        user_content=user_message,
        system_content=sys_msg,
        model=model,
        api_key=api_key,
        api_base=api_base,
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
    google_api_key: Optional[str] = None,
    gemini_api_base: Optional[str] = None,
    gemini_judges: Optional[frozenset[str]] = None,
    gemini_max_requests_per_minute: Optional[float] = None,
):
    """Factory: return an ``evaluate_pair`` for local, OpenAI, and/or Gemini judges.

    Judges listed in ``openai_judges`` (default: :data:`pairwise_eval.config.OPENAI_JUDGE_IDS`)
    use OpenAI Chat Completions and require ``openai_api_key``. Judges in ``gemini_judges``
    (default: :data:`pairwise_eval.config.GEMINI_JUDGE_IDS`) use Gemini ``generateContent`` and
    require ``google_api_key`` (or env ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``). Consecutive Gemini
    calls are spaced using :data:`pairwise_eval.config.GEMINI_MAX_REQUESTS_PER_MINUTE` (or
    ``gemini_max_requests_per_minute=`` here) to stay under per-minute quotas. All other judges
    use the local LM Studio–style endpoint (``LOCAL_LLM_CHAT_URL``).

    Checkpoints are one JSONL per (``judge_id``, dimension), so adding cloud judges does not
    overwrite existing local-judge checkpoint files.
    """
    from pairwise_eval.config import (
        GEMINI_API_BASE,
        GEMINI_JUDGE_IDS,
        GEMINI_JUDGE_TO_API_MODEL,
        GEMINI_MAX_REQUESTS_PER_MINUTE as _cfg_gemini_rpm,
        GOOGLE_API_KEY as _cfg_google_key,
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
    gem_j = gemini_judges if gemini_judges is not None else GEMINI_JUDGE_IDS
    g_key = google_api_key if google_api_key is not None else _cfg_google_key
    gem_base = gemini_api_base or GEMINI_API_BASE
    rpm = (
        _cfg_gemini_rpm
        if gemini_max_requests_per_minute is None
        else float(gemini_max_requests_per_minute)
    )
    gem_min_interval_s = (60.0 / rpm) if rpm > 0 else 0.0
    _last_gemini_monotonic: list[float] = [0.0]

    def _throttle_gemini() -> None:
        """Sleep if needed so consecutive Gemini calls are at least ``gem_min_interval_s`` apart."""
        if gem_min_interval_s <= 0.0:
            return
        now = time.monotonic()
        elapsed = now - _last_gemini_monotonic[0]
        wait = gem_min_interval_s - elapsed
        if wait > 0.0:
            time.sleep(wait)

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
            if judge_id in oai_j:
                backend = "openai"
            elif judge_id in gem_j:
                backend = "gemini"
            else:
                backend = "local"
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
            elif judge_id in gem_j:
                if not g_key:
                    raise RuntimeError(
                        "GOOGLE_API_KEY or GEMINI_API_KEY is not set (or pass google_api_key=...) "
                        "for Gemini judges"
                    )
                if "/" not in judge_id:
                    raise RuntimeError(f"Gemini judge id must be google/<model-id>, got {judge_id!r}")
                _, suffix = judge_id.split("/", 1)
                gem_model = GEMINI_JUDGE_TO_API_MODEL.get(judge_id, suffix)
                _throttle_gemini()
                _last_gemini_monotonic[0] = time.monotonic()
                assistant = gemini_llm_geval_judge(
                    str(row["source_text"]),
                    str(row["sumleft"]),
                    str(row["sumright"]),
                    judge=dimension,
                    model=gem_model,
                    api_key=g_key,
                    api_base=gem_base,
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
