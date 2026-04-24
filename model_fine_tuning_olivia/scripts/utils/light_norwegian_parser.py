"""Lightweight Norwegian sentence-quality estimator.

Uses spaCy (nb_core_news_md) for:
  - sentence segmentation
  - POS / dependency-based completeness heuristics
  - vocabulary lookup (words with vectors = known Norwegian words)

Uses the shared _HYGIENE_TOKEN_RE tokenizer from utils.metrics for
word/non-word accounting (consistent with other hygiene metrics).

Install:
    pip install spacy
    python -m spacy download nb_core_news_md

For broader word coverage, supply an external lexicon file
(e.g. Norsk Ordbank from Språkrådet) via load_lexicon().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Any

import spacy

from .metrics import _HYGIENE_TOKEN_RE

_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_HAS_LETTER_RE = re.compile(r"[A-Za-zÆØÅæøå]")
_ONLY_LETTERS_SEPARATORS_RE = re.compile(
    r"^[A-Za-zÆØÅæøå]+(?:[-/][A-Za-zÆØÅæøå]+)*$"
)

# Norwegian abbreviations whose periods should be treated as neutral.
_ABBREV_RE = re.compile(
    r"\b(?:"
    r"bl\.a\.|f\.eks\.|jf\.|pbl\.|mfl\.|osv\.|dvs\.|nr\.|gbnr\.|"
    r"ca\.|etc\.|evt\.|kl\.|mn\.|kap\.|hhv\.|jnr\.|mrd\.|"
    r"sst\.|ibid\.|resp\.|tlf\.|vedk\.|inkl\.|ekskl\.|"
    r"[a-zæøå]\.(?:[a-zæøå]\.)+|"  # multi-letter: a.s., o.l.
    r"[A-ZÆØÅ]\.(?:[A-ZÆØÅ]\.)*"   # initials: A.S., N.N.
    r")",
    re.IGNORECASE,
)

_BRACKET_PAIRS = [("(", ")"), ("«", "»"), ("[", "]"), ("{", "}")]
_INTERNAL_PUNCT = {",", ";", ":"}


@dataclass
class SentenceStats:
    text: str
    is_complete: bool
    score: int
    known_words: int
    unknown_words: int
    known_ratio: float


@dataclass
class TextStats:
    sentences_total: int
    complete_sentences: int
    incomplete_sentences: int
    complete_ratio: float
    total_word_candidates: int
    known_words: int
    unknown_words: int
    known_word_ratio: float
    sentence_details: List[SentenceStats]


def load_lexicon(path: str) -> Set[str]:
    """Load a newline-delimited word list (lowercased)."""
    lex: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                lex.add(word)
    return lex


class NorwegianLightParser:
    """Lightweight Norwegian sentence-quality estimator.

    Lexicon strategy (checked in order):
      1. External lexicon set (if supplied via *lexicon* or load_lexicon())
      2. spaCy's word vectors — a word that has a vector in nb_core_news_md
         (~20 k entries) is treated as a known Norwegian word.
    """

    def __init__(
        self,
        lexicon: Set[str] | None = None,
        model: str = "nb_core_news_md",
    ) -> None:
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"spaCy model '{model}' not found — downloading …")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
        self._external_lexicon: Set[str] = lexicon or set()

    # ------------------------------------------------------------------
    # Tokenisation (shared regex from utils.metrics)
    # ------------------------------------------------------------------

    @staticmethod
    def regex_tokenize(text: str) -> List[str]:
        return _HYGIENE_TOKEN_RE.findall(text)

    @staticmethod
    def normalize_word(token_text: str) -> str:
        return token_text.lower().replace("\u2019", "'").strip()

    @staticmethod
    def is_candidate_word(tok: str) -> bool:
        """Strict wordhood filter on top of the shared tokeniser.

        Accepts alphabetic tokens and internal hyphen/slash compounds
        (e.g. "e-post", "og/eller").  Rejects numbers, punctuation,
        underscores, and mixed alphanumeric junk.
        """
        if not tok:
            return False
        if _NUMERIC_RE.fullmatch(tok):
            return False
        if "_" in tok:
            return False
        if not _HAS_LETTER_RE.search(tok):
            return False
        if not _ONLY_LETTERS_SEPARATORS_RE.fullmatch(tok):
            return False
        return True

    def is_known_word(self, tok: str) -> bool:
        """Check whether *tok* is probably a real Norwegian word.

        Uses the external lexicon first, then falls back to spaCy's
        word-vector vocabulary.  Hyphen/slash compounds pass when every
        part is individually known.
        """
        if not self.is_candidate_word(tok):
            return False

        norm = self.normalize_word(tok)

        if norm in self._external_lexicon:
            return True

        if self.nlp.vocab[norm].has_vector:
            return True

        if "-" in norm or "/" in norm:
            parts = [p for p in re.split(r"[-/]", norm) if p]
            if parts and all(
                p in self._external_lexicon or self.nlp.vocab[p].has_vector
                for p in parts
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Word-level stats
    # ------------------------------------------------------------------

    def word_stats(self, text: str) -> tuple[int, int]:
        """Return (known_words, unknown_words) for *text*."""
        known = unknown = 0
        for tok in self.regex_tokenize(text):
            if not self.is_candidate_word(tok):
                continue
            if self.is_known_word(tok):
                known += 1
            else:
                unknown += 1
        return known, unknown

    # ------------------------------------------------------------------
    # Sentence-level completeness heuristic (spaCy features only)
    # ------------------------------------------------------------------

    def _sentence_score(self, sent) -> int:
        score = 0
        alpha_tokens = [t for t in sent if t.is_alpha]
        content_tokens = [t for t in alpha_tokens if not t.is_stop]

        finite_like = [
            t for t in sent
            if t.pos_ in {"VERB", "AUX"}
            and ("VerbForm=Fin" in t.morph or t.tag_.startswith("V"))
        ]
        if finite_like:
            score += 3
        else:
            score -= 3

        has_subject_dep = any(
            t.dep_ in {"nsubj", "csubj", "expl"} for t in sent
        )
        has_subject_candidate = any(
            t.pos_ in {"NOUN", "PROPN", "PRON"} for t in sent
        )
        if has_subject_dep:
            score += 2
        elif has_subject_candidate:
            score += 1
        else:
            score -= 1

        roots = [t for t in sent if t.dep_ == "ROOT"]
        if roots and roots[0].pos_ in {"VERB", "AUX", "ADJ", "NOUN"}:
            score += 1

        if len(alpha_tokens) >= 3:
            score += 1
        else:
            score -= 2

        stripped = sent.text.strip()
        if stripped.endswith((".", "!", "?")):
            score += 1
        else:
            score -= 1

        known, unknown = self.word_stats(sent.text)
        total_candidates = known + unknown
        if total_candidates:
            unknown_ratio = unknown / total_candidates
            if unknown_ratio > 0.5:
                score -= 3
            elif unknown_ratio > 0.25:
                score -= 1
            else:
                score += 1

        lower = stripped.lower()
        if (
            lower.startswith(("og ", "men ", "eller ", "fordi ", "som ", "at "))
            and not finite_like
        ):
            score -= 2

        if len(content_tokens) == 0:
            score -= 2

        return score

    def _classify_sentence(self, sent) -> SentenceStats:
        score = self._sentence_score(sent)
        known, unknown = self.word_stats(sent.text)
        total = known + unknown
        return SentenceStats(
            text=sent.text,
            is_complete=(score >= 2),
            score=score,
            known_words=known,
            unknown_words=unknown,
            known_ratio=(known / total) if total else 0.0,
        )

    # ------------------------------------------------------------------
    # Punctuation quality score
    # ------------------------------------------------------------------

    def _punctuation_score(self, text: str, doc) -> float:
        """Length-normalised punctuation quality. 0.0 = perfect, negative = worse.

        Penalties:
          - mismatched brackets / quotes
          - run-on sentences (15+ alpha tokens with no internal ,;:)

        Abbreviation periods (bl.a., jf., etc.) are stripped before analysis
        so they count neither positively nor negatively.
        """
        text_len = len(text)
        if text_len == 0:
            return 0.0

        # Strip abbreviation periods so they don't pollute bracket/quote counts.
        clean = _ABBREV_RE.sub(lambda m: m.group().replace(".", ""), text)

        score = 0.0

        # Mismatched brackets
        for open_ch, close_ch in _BRACKET_PAIRS:
            diff = abs(clean.count(open_ch) - clean.count(close_ch))
            if diff:
                score -= diff * 10.0 / text_len

        # Mismatched double quotes (expect even count)
        if clean.count('"') % 2 != 0:
            score -= 10.0 / text_len

        # Run-on sentences: long sentence without internal punctuation
        for sent in doc.sents:
            alpha_count = sum(1 for t in sent if t.is_alpha)
            if alpha_count < 15:
                continue
            has_internal = any(t.text in _INTERNAL_PUNCT for t in sent)
            if not has_internal:
                score -= (alpha_count - 14) * 2.0 / text_len

        return round(score, 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        include_sentence_details: bool = False,
    ) -> Dict[str, Any]:
        """Analyse *text* and return a JSON-serialisable dict.

        Keys:
            sentences_total, complete_sentences, incomplete_sentences,
            complete_ratio, total_word_candidates, known_words,
            unknown_words, known_word_ratio.

        If *include_sentence_details* is True, a ``sentence_details``
        list is included (one dict per sentence with text, score, etc.).
        """
        doc = self.nlp(text)
        details = [self._classify_sentence(sent) for sent in doc.sents]

        complete = sum(1 for s in details if s.is_complete)
        known, unknown = self.word_stats(text)
        total_candidates = known + unknown

        result: Dict[str, Any] = {
            "sentences_total": len(details),
            "complete_sentences": complete,
            "incomplete_sentences": len(details) - complete,
            "complete_ratio": (complete / len(details)) if details else 0.0,
            "starts_with_complete_sent": details[0].is_complete if details else False,
            "ends_with_complete_sent": details[-1].is_complete if details else False,
            "punctuation_score": self._punctuation_score(text, doc),
            "total_word_candidates": total_candidates,
            "known_words": known,
            "unknown_words": unknown,
            "known_word_ratio": (
                (known / total_candidates) if total_candidates else 0.0
            ),
        }
        if include_sentence_details:
            result["sentence_details"] = [asdict(s) for s in details]
        return result
