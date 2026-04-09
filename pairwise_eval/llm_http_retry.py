"""Shared HTTP retry backoff for cloud LLM APIs (429 rate limit, 503 overload).

Use :class:`LLMHttpRetryPolicy` to compute per-attempt delays (honors ``Retry-After`` when
present) and :func:`request_with_retry` to wrap ``requests``-style POSTs.
"""

from __future__ import annotations

import email.utils
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, TypeVar

TResp = TypeVar("TResp")


def parse_retry_after_header(value: Optional[str]) -> Optional[float]:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date). Returns seconds to wait, or ``None``."""
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if v.isdigit():
        return float(v)
    try:
        dt = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    wait = (dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, wait)


@dataclass(frozen=True)
class LLMHttpRetrySettings:
    """Numeric knobs for backoff (tune per provider via :meth:`LLMHttpRetryPolicy.for_provider`)."""

    max_retries: int = 5
    """How many times to retry after a retryable response (at most ``max_retries + 1`` HTTP calls)."""

    base_delay_s: float = 2.0
    max_delay_s: float = 120.0
    jitter_ratio: float = 0.2
    """Uniform jitter in ``[0, jitter_ratio * exponential_component]``."""

    retry_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 503})
    )
    status_multiplier: dict[int, float] = field(
        default_factory=lambda: {429: 1.0, 503: 1.25}
    )
    """Applied to the exponential component (503 often benefits from slightly longer spacing)."""


def _settings_for_provider_id(provider_id: str) -> LLMHttpRetrySettings:
    """Defaults for ``gemini``, ``openai``, ``anthropic``, ``mistral``, or anything else."""
    pid = provider_id.strip().lower()
    if pid in ("google", "gemini"):
        return LLMHttpRetrySettings(
            max_retries=6,
            base_delay_s=2.0,
            max_delay_s=120.0,
            jitter_ratio=0.25,
        )
    if pid == "openai":
        return LLMHttpRetrySettings(
            max_retries=5,
            base_delay_s=1.0,
            max_delay_s=90.0,
            jitter_ratio=0.2,
        )
    if pid in ("anthropic", "claude"):
        return LLMHttpRetrySettings(
            max_retries=5,
            base_delay_s=1.5,
            max_delay_s=120.0,
            jitter_ratio=0.2,
        )
    if pid == "mistral":
        return LLMHttpRetrySettings(
            max_retries=5,
            base_delay_s=1.5,
            max_delay_s=90.0,
            jitter_ratio=0.2,
        )
    return LLMHttpRetrySettings()


class LLMHttpRetryPolicy:
    """Compute wait time before the next HTTP retry; provider-aware defaults."""

    def __init__(
        self,
        *,
        provider: str,
        model: Optional[str] = None,
        settings: Optional[LLMHttpRetrySettings] = None,
    ) -> None:
        pid = provider.strip().lower()
        self.provider = pid
        self.model = model
        self.settings = settings if settings is not None else _settings_for_provider_id(pid)

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        model: Optional[str] = None,
        overrides: Optional[LLMHttpRetrySettings] = None,
    ) -> LLMHttpRetryPolicy:
        """Build a policy with defaults for ``gemini``, ``openai``, ``anthropic``, or ``generic``."""
        pid = provider.strip().lower()
        base = overrides if overrides is not None else _settings_for_provider_id(pid)
        return cls(provider=pid, model=model, settings=base)

    def should_retry(self, status_code: int, *, failure_count: int) -> bool:
        """``failure_count`` = how many retryable responses have already occurred (0 on first 429/503)."""
        s = self.settings
        return status_code in s.retry_status_codes and failure_count < s.max_retries

    def wait_before_retry_s(
        self,
        *,
        failure_count: int,
        status_code: int,
        retry_after_header: Optional[str],
    ) -> float:
        """Seconds to sleep before the next attempt (call only if :meth:`should_retry` is true)."""
        s = self.settings
        parsed = parse_retry_after_header(retry_after_header)
        if parsed is not None:
            # Respect the server; still cap so a bad header cannot stall for hours.
            return max(0.5, min(parsed, s.max_delay_s))

        mult = s.status_multiplier.get(status_code, 1.0)
        exp = s.base_delay_s * (2**failure_count) * mult
        exp = min(exp, s.max_delay_s)
        jitter = random.uniform(0.0, exp * s.jitter_ratio) if s.jitter_ratio > 0 else 0.0
        return max(0.5, exp + jitter)

    def wait_before_retry(
        self,
        *,
        failure_count: int,
        status_code: int,
        retry_after_header: Optional[str],
    ) -> None:
        time.sleep(
            self.wait_before_retry_s(
                failure_count=failure_count,
                status_code=status_code,
                retry_after_header=retry_after_header,
            )
        )


def request_with_retry(
    send: Callable[[], TResp],
    *,
    policy: LLMHttpRetryPolicy,
    get_status: Callable[[TResp], int],
    get_retry_after: Callable[[TResp], Optional[str]],
) -> TResp:
    """Call ``send()`` until the response is not a configured retry code or retries are exhausted.

    ``get_status`` / ``get_retry_after`` adapt the response type (e.g. ``requests.Response``).
    """
    failure_count = 0
    while True:
        r = send()
        code = int(get_status(r))
        if policy.should_retry(code, failure_count=failure_count):
            policy.wait_before_retry(
                failure_count=failure_count,
                status_code=code,
                retry_after_header=get_retry_after(r),
            )
            failure_count += 1
            continue
        return r
