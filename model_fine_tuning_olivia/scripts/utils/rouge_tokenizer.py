"""
Norwegian-aware ROUGE tokenizer.

The default rouge_score tokenizer strips all non-ASCII characters (regex [^a-z0-9]+),
which destroys Norwegian words containing æ, ø, å (e.g. "økonomi" → "konomi",
"på" → "p"). This module provides a Unicode-aware tokenizer with optional
lemmatization for fairer ROUGE evaluation of Norwegian text.

Usage with HuggingFace evaluate:
    from utils.rouge_tokenizer import norwegian_tokenize
    rouge = evaluate.load("rouge")
    rouge.compute(predictions=preds, references=refs,
                  use_stemmer=False, tokenizer=norwegian_tokenize)

The tokenizer callable signature matches what evaluate's ROUGE metric expects:
    f(text: str) -> list[str]
"""

import re
import warnings
from typing import List, Callable, Optional

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# ---------------------------------------------------------------------------
# Lemmatizer / stemmer backends (resolved lazily on first call)
# ---------------------------------------------------------------------------
_backend: Optional[Callable[[str], str]] = None
_backend_resolved = False
_backend_name: Optional[str] = None


def _resolve_backend() -> None:
    """Try simplemma (true lemmatisation), fall back to NLTK Norwegian Snowball stemmer."""
    global _backend, _backend_resolved, _backend_name

    # 1. simplemma – lightweight, dictionary-based Norwegian lemmatiser
    try:
        import simplemma

        def _lemmatize(token: str) -> str:
            return simplemma.lemmatize(token, lang="nb")

        # Smoke-test
        assert _lemmatize("regjeringen") == "regjering"
        _backend = _lemmatize
        _backend_name = "simplemma"
        _backend_resolved = True
        return
    except Exception:
        pass

    # 2. NLTK Norwegian Snowball stemmer (already a rouge_score dependency)
    try:
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer("norwegian")

        def _stem(token: str) -> str:
            return stemmer.stem(token) if len(token) > 3 else token

        _backend = _stem
        _backend_name = "nltk-snowball-norwegian"
        _backend_resolved = True
        return
    except Exception:
        pass

    # 3. No morphological normalisation available
    _backend = None
    _backend_name = None
    _backend_resolved = True
    warnings.warn(
        "Neither simplemma nor NLTK SnowballStemmer('norwegian') available. "
        "ROUGE will use Unicode-aware tokenisation without lemmatisation.",
        stacklevel=2,
    )


def _get_backend():
    global _backend_resolved
    if not _backend_resolved:
        _resolve_backend()
    return _backend


def get_backend_name() -> Optional[str]:
    """Return the name of the active morphological backend, or None."""
    _get_backend()
    return _backend_name


# ---------------------------------------------------------------------------
# Tokenizer function (callable expected by evaluate's rouge.compute())
# ---------------------------------------------------------------------------

def norwegian_tokenize(text: str) -> List[str]:
    """Tokenize Norwegian text for ROUGE scoring.

    1. Lowercase
    2. Extract Unicode word tokens (preserves æ, ø, å and other letters)
    3. Lemmatize / stem each token (if a backend is available)
    """
    tokens = _WORD_RE.findall(text.lower())
    backend = _get_backend()
    if backend is not None:
        tokens = [backend(t) for t in tokens]
    return tokens
