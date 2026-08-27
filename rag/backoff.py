"""Retry and rate-limiting for the Gemini API.

Two separate concerns that are easy to conflate:

  with_retry   -- reactive. Something failed; wait and try again. Only for
                  transient errors, so a real bug fails fast instead of
                  taking five doublings to surface.
  RateLimiter  -- proactive. Don't send the request that would breach the
                  quota in the first place. Much better than discovering the
                  limit by being rejected, because the server counts rejected
                  requests against you too.
"""

import re
import time
from collections import deque

RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE",
             "DEADLINE", "INTERNAL")

# Gemini returns e.g. "retryDelay': '31s'" -- obey it rather than guessing.
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")

# A *daily* quota cannot be waited out inside a process. The server still says
# "please retry in 38s", which is misleading -- retrying just burns minutes and
# then fails anyway. Detect it and surface it immediately.
_DAILY = re.compile(r"PerDay|per day", re.I)


class DailyQuotaExceeded(RuntimeError):
    pass


def with_retry(fn, max_attempts: int = 6, base_delay: float = 2.0,
               max_delay: float = 65.0):
    delay = base_delay
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg and _DAILY.search(msg):
                raise DailyQuotaExceeded(
                    "Daily free-tier quota exhausted for this model. This is "
                    "a per-DAY cap, not per-minute, so waiting will not help "
                    "today.\nEach model has its own separate daily allowance, "
                    "so the quickest fix is to switch:\n"
                    "  RAG_EMBED_MODEL=gemini-embedding-2   (embedding)\n"
                    "  RAG_GEN_MODEL=gemini-3.6-flash       (generation)\n"
                    "Set it in .env, or enable billing to remove the cap.\n"
                    + msg[:300]) from e
            if attempt == max_attempts - 1 or not any(s in msg
                                                      for s in RETRYABLE):
                raise
            m = _RETRY_DELAY.search(msg)
            wait = min(float(m.group(1)) + 1 if m else delay, max_delay)
            print(f"    retry {attempt + 1}/{max_attempts - 1} in "
                  f"{wait:.0f}s ({msg[:70]})")
            time.sleep(wait)
            delay = min(delay * 2, max_delay)


class RateLimiter:
    """Sliding-window limiter over *items* consumed per period.

    The free-tier embedding quota counts individual texts, not HTTP calls, so
    a batch of 32 spends 32 units. Call `reserve(n)` before each request.
    """

    def __init__(self, limit: int, period: float = 60.0):
        self.limit, self.period = limit, period
        self.events: deque[tuple[float, int]] = deque()

    def reserve(self, n: int) -> None:
        while True:
            now = time.monotonic()
            while self.events and now - self.events[0][0] >= self.period:
                self.events.popleft()
            used = sum(c for _, c in self.events)
            if used + n <= self.limit or not self.events:
                self.events.append((now, n))
                return
            sleep_for = self.period - (now - self.events[0][0]) + 0.25
            print(f"    rate limit: {used}/{self.limit} used, "
                  f"pausing {sleep_for:.0f}s")
            time.sleep(sleep_for)
