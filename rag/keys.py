"""Round-robin across multiple Gemini API keys when one hits its daily cap.

The free tier's generation quota is as low as ~20 requests/day, and that
quota is scoped to *(key, model)* -- one key burns out fast under any real
use. If more than one key is configured (several of your own free-tier
projects, or several teammates'), rotating to the next one when the current
one hits `DailyQuotaExceeded` lets the pipeline keep working today instead
of stopping until the quota resets. With exactly one key configured, this
degrades to exactly today's behaviour: no key to rotate to, so the original
error still surfaces, unchanged.

What this deliberately does NOT do: split one logical request across keys,
pool quota, or share state between keys in any way. Each key is independent
and rate-limited independently (see backoff.py's RateLimiter, which is
per-process and per-model, not per-key -- see the note on that below).
"""
from google import genai

from .backoff import DailyQuotaExceeded
from .config import API_KEYS


class KeyRotator:
    """Tracks, per model, which configured keys are known exhausted today.

    Exhaustion is remembered only for the life of this process -- there is no
    file on disk recording it. That is deliberate: the daily reset time is
    not something this project should guess at, so a fresh run always tries
    every key again from the start rather than trusting a stale on-disk
    "exhausted until" timestamp that might be wrong by hours.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise SystemExit(
                "No Gemini API key configured. Copy .env.example to .env "
                "and set GEMINI_API_KEY (or GEMINI_API_KEYS for several, "
                "comma- or newline-separated) -- "
                "https://aistudio.google.com/apikey")
        self.keys = keys
        self._clients: dict[str, genai.Client] = {}
        self._exhausted: dict[str, set[str]] = {}   # model -> exhausted keys
        self._cursor = 0

    def _client_for(self, key: str) -> genai.Client:
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def current(self, model: str) -> genai.Client:
        """The client to use right now for `model`.

        Skips any key already marked exhausted for this specific model --
        a key exhausted for generation may still have embedding quota left,
        since the two are entirely separate allowances.
        """
        exhausted = self._exhausted.get(model, set())
        for _ in range(len(self.keys)):
            key = self.keys[self._cursor % len(self.keys)]
            if key not in exhausted:
                return self._client_for(key)
            self._cursor += 1
        raise DailyQuotaExceeded(
            f"All {len(self.keys)} configured key(s) have hit their daily "
            f"quota for {model}. Add another key to GEMINI_API_KEYS, wait "
            f"for the quota to reset, or enable billing.")

    def mark_exhausted(self, model: str) -> bool:
        """Record the CURRENT key as exhausted for `model` and advance past
        it. Returns True if a not-yet-exhausted key remains to try."""
        key = self.keys[self._cursor % len(self.keys)]
        used = self._exhausted.setdefault(model, set())
        used.add(key)
        self._cursor += 1
        return len(used) < len(self.keys)

    def status(self) -> str:
        if len(self.keys) == 1:
            return "1 key configured (no rotation possible with only one)"
        lines = [f"{len(self.keys)} keys configured"]
        for model, used in self._exhausted.items():
            lines.append(f"  {model}: {len(used)}/{len(self.keys)} "
                         f"exhausted this run")
        return "\n".join(lines)


_rotator: KeyRotator | None = None


def rotator() -> KeyRotator:
    global _rotator
    if _rotator is None:
        _rotator = KeyRotator(API_KEYS)
    return _rotator


def call_with_rotation(fn, model: str):
    """Call fn(). On DailyQuotaExceeded, rotate to the next configured key
    for `model` and call fn() again -- fn() must look up its client via
    rotator().current(model) itself, so it picks up the new key on retry.

    Only DailyQuotaExceeded triggers rotation. Every other error (a real
    bug, a malformed request, a transient 5xx already retried by
    with_retry) propagates immediately -- rotating keys would not fix any
    of those, and silently retrying a broken request N times across N keys
    would only hide the real error behind N times the noise.
    """
    r = rotator()
    for _ in range(len(r.keys)):
        try:
            return fn()
        except DailyQuotaExceeded:
            if not r.mark_exhausted(model):
                raise
            print(f"    key exhausted for {model}; rotating to the next "
                 f"configured key")
    raise DailyQuotaExceeded(  # unreachable in practice; satisfies the type
        f"All configured keys exhausted for {model}.")
