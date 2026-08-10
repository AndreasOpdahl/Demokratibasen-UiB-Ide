"""Shared generation helpers for evaluation and hygiene regeneration."""

import re
from typing import Any, Dict

import torch


def sync_model_tokenizer_special_tokens(model: torch.nn.Module, tokenizer) -> None:
    """Align pad/eos on model config + generation_config with the tokenizer."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None and eos_id is not None:
        pad_id = eos_id
    candidates = [model]
    try:
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()
            if base_model is not None:
                candidates.append(base_model)
    except Exception:
        pass
    inner = getattr(model, "base_model", None)
    if inner is not None:
        candidates.append(inner)
        nested = getattr(inner, "model", None)
        if nested is not None:
            candidates.append(nested)
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        cfg = getattr(candidate, "config", None)
        if cfg is not None:
            if pad_id is not None:
                cfg.pad_token_id = pad_id
            if eos_id is not None:
                cfg.eos_token_id = eos_id
        gen_cfg = getattr(candidate, "generation_config", None)
        if gen_cfg is not None:
            if pad_id is not None:
                gen_cfg.pad_token_id = pad_id
            if eos_id is not None:
                gen_cfg.eos_token_id = eos_id


def clean_generated_summary_text(text: str) -> str:
    """Canonical cleaning for decoded payload summaries."""
    if '[/SAK]' in text:
        text = text.split('[/SAK]')[0].strip()
    text = re.sub(r'(\n\s*)#{3,}\s*$', '', text).strip()
    text = text.replace('[/INST]', '').replace('[INST]', '')
    text = text.replace('</s>', '').replace('<s>', '')
    text = text.replace('<|begin_of_text|>', '')
    text = text.replace('<|end_of_text|>', '')
    text = text.replace('<|eot_id|>', '')
    text = text.replace('<|start_header_id|>', '')
    text = text.replace('<|end_header_id|>', '')
    text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', text)
    text = text.replace('<|im_start|>', '').replace('<|im_end|>', '')
    while text.startswith('<unk>'):
        text = text[5:].lstrip()
    text = re.sub(r'^assistant\s*\n*\s*', '', text, flags=re.IGNORECASE)
    if text.lstrip().startswith('Oppsummering:'):
        text = text.lstrip()[len('Oppsummering:'):].lstrip()
    for _ in range(6):
        if not text.startswith('###'):
            break
        text = text[3:].lstrip()
    text = text.replace('\\', '')
    text = re.sub(r'\s*#+\s*$', '', text)
    text = ' '.join(text.split())
    return text.strip()


def _minimal_generated_summary_clean(text: str) -> str:
    """Less aggressive fallback if canonical cleaning removes everything."""
    fallback = text.replace('[/INST]', '').replace('[INST]', '')
    fallback = fallback.replace('</s>', '').replace('<s>', '')
    fallback = fallback.replace('<|begin_of_text|>', '')
    fallback = fallback.replace('<|end_of_text|>', '')
    fallback = fallback.replace('<|eot_id|>', '')
    fallback = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', fallback)
    fallback = fallback.replace('<|im_start|>', '').replace('<|im_end|>', '')
    fallback = fallback.replace('\\', '')
    return ' '.join(fallback.split()).strip()


def truncate_repeated_paragraphs(text: str, min_words: int = 12) -> str:
    """Truncate at first repeated paragraph/chunk."""
    if not text or len(text.split()) < min_words * 2:
        return text
    words = text.split()
    for i in range(0, len(words) - min_words):
        chunk = ' '.join(words[i:i + min_words])
        rest = ' '.join(words[i + min_words:])
        if len(rest) < len(chunk) * 0.5:
            continue
        if chunk in rest:
            return ' '.join(words[:i + min_words]).strip()
    return text


def fix_mid_sentence_start(text: str) -> str:
    """Try to recover a complete sentence when decoded text starts mid-sentence."""
    if not text or len(text.strip()) < 10:
        return text
    first_char = text.strip()[0] if text.strip() else ''
    if first_char in [',', '.', ' ', '\n'] or (first_char and first_char.islower()):
        match = re.search(r'[.!?]\s+[A-ZÆØÅ]', text)
        if match:
            return text[match.end() - 1:].strip()
        capital_match = re.search(r'[A-ZÆØÅ]', text)
        if capital_match:
            return text[capital_match.start():].strip()
    return text


def postprocess_generated_summary_text(text: str) -> str:
    """Canonical full post-processing for one decoded generation."""
    cleaned = clean_generated_summary_text(text)
    if not cleaned and text and text.strip():
        cleaned = _minimal_generated_summary_clean(text)
    cleaned = truncate_repeated_paragraphs(cleaned)
    return fix_mid_sentence_start(cleaned)


def extract_generated_continuations(generated_ids: torch.Tensor, input_width: int) -> torch.Tensor:
    """Extract generated continuation tokens from decoder-only generate() output."""
    if generated_ids.shape[1] <= input_width:
        return generated_ids[:, 0:0]
    return generated_ids[:, input_width:]


def make_inputs_refs_preds_record(
    *,
    input_text: Any,
    reference: Any,
    prediction: Any,
    prompt: Any = "",
) -> Dict[str, str]:
    """Build the canonical inputs-refs-preds JSONL record."""
    return {
        "input_text": input_text if isinstance(input_text, str) else ("" if input_text is None else str(input_text)),
        "prompt": prompt if isinstance(prompt, str) else ("" if prompt is None else str(prompt)),
        "reference": reference if isinstance(reference, str) else ("" if reference is None else str(reference)),
        "prediction": prediction if isinstance(prediction, str) else ("" if prediction is None else str(prediction)),
    }
