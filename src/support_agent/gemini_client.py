"""LLM client with a Gemini primary and a Groq fallback.

One place that talks to a model, so the rest of the code is provider-agnostic. Features:
- Two providers: Google **Gemini** (`google-genai`) and **Groq** (OpenAI-compatible REST,
  stdlib only). A client can be pinned to one provider, or run Gemini-first-Groq-fallback
  so a Gemini quota wall degrades gracefully instead of failing the request.
- Per-provider, thread-safe rate throttle to stay under free-tier per-minute limits.
- Exponential-backoff retries; on a quota 429 it waits out the window.
- Content-addressed disk cache (keyed by provider+model+prompt) so re-runs are instant,
  reproducible, and quota-free.
- Lenient JSON parsing that tolerates code fences / surrounding prose.

The class keeps the name `GeminiClient` for backward-compat; `provider="groq"` pins it to
Groq, and `fallback=...` attaches a secondary client to try when the primary fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from .config import SETTINGS

# Per-provider throttles. Free tiers cap requests-per-minute; we pace to stay just under
# so we never burst into 429s and thrash in backoff. Gemini free is tighter than Groq.
_INTERVALS = {
    "gemini": float(os.getenv("GEMINI_MIN_INTERVAL_S", "4.2")),  # ~14/min
    "groq": float(os.getenv("GROQ_MIN_INTERVAL_S", "2.1")),      # ~28/min
}
_locks: dict[str, threading.Lock] = {"gemini": threading.Lock(), "groq": threading.Lock()}
_last: dict[str, float] = {"gemini": 0.0, "groq": 0.0}


def _throttle(provider: str):
    lock = _locks.setdefault(provider, threading.Lock())
    with lock:
        gap = _INTERVALS.get(provider, 2.0) - (time.time() - _last.get(provider, 0.0))
        if gap > 0:
            time.sleep(gap)
        _last[provider] = time.time()


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq sits behind Cloudflare, which blocks the default urllib UA (error 1010).
_UA = "Mozilla/5.0 (compatible; support-agent/1.0)"


class GeminiClient:
    def __init__(
        self,
        cache: bool = True,
        model: str | None = None,
        provider: str = "gemini",
        fallback: "GeminiClient | None" = None,
    ):
        cfg = SETTINGS.gemini
        self._cfg = cfg
        self._provider = provider
        self._fallback = fallback
        self._use_cache = cache
        self._cache_dir = SETTINGS.cache_dir / "gemini"
        if cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Keys are validated lazily at call time, not here — so results can be reproduced
        # from the committed cache (GEMINI_CACHE_ONLY=1) with no keys configured at all.
        if provider == "gemini":
            self._client = None
            self._model = model or cfg.model
        elif provider == "groq":
            self._groq_key = os.getenv("GROQ_API_KEY", "")
            self._model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        else:
            raise ValueError(f"unknown provider {provider}")

    # ---- caching ----
    def _cache_key(self, prompt: str, json_mode: bool, temperature: float) -> Path:
        h = hashlib.sha256(
            f"{self._provider}|{self._model}|{temperature}|{json_mode}|{prompt}".encode()
        ).hexdigest()
        return self._cache_dir / f"{h}.json"

    # ---- public API ----
    def generate(self, prompt: str, *, json_mode: bool = False,
                 temperature: float | None = None, system: str | None = None) -> str:
        temperature = self._cfg.temperature if temperature is None else temperature
        full = (system + "\n\n" + prompt) if system else prompt

        if self._use_cache:
            key = self._cache_key(full, json_mode, temperature)
            if key.exists():
                return json.loads(key.read_text(encoding="utf-8"))["text"]
            if os.getenv("GEMINI_CACHE_ONLY") == "1":
                raise RuntimeError("cache-miss in GEMINI_CACHE_ONLY mode")

        try:
            # If a fallback exists, don't waste long quota-backoff on the primary — fail
            # over fast so a Gemini quota wall degrades to Groq in ~1 call, not ~2 minutes.
            text = self._call(prompt, system, json_mode, temperature,
                              fast_fail=self._fallback is not None)
        except Exception:  # noqa: BLE001
            if self._fallback is None:
                raise
            text = self._fallback.generate(
                prompt, json_mode=json_mode, temperature=temperature, system=system
            )

        if self._use_cache:
            key.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text

    def generate_json(self, prompt: str, **kw) -> Any:
        return _loads_lenient(self.generate(prompt, json_mode=True, **kw))

    # ---- providers ----
    def _call(self, prompt: str, system: str | None, json_mode: bool, temperature: float,
              fast_fail: bool = False) -> str:
        last_err: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                _throttle(self._provider)
                if self._provider == "gemini":
                    text = self._call_gemini(prompt, system, json_mode, temperature)
                else:
                    text = self._call_groq(prompt, system, json_mode, temperature)
                if not text.strip():
                    raise RuntimeError("empty response")
                return text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                s = str(e)
                is_quota = ("429" in s or "RESOURCE_EXHAUSTED" in s
                            or "quota" in s.lower() or "rate" in s.lower())
                if is_quota and fast_fail:
                    break  # let the caller fail over to the fallback provider immediately
                time.sleep((20 + random.uniform(0, 8)) if is_quota
                           else min(2 ** attempt, 10) + random.uniform(0, 1))
        raise RuntimeError(f"{self._provider} call failed after retries: {last_err}")

    def _call_gemini(self, prompt, system, json_mode, temperature) -> str:
        if self._client is None:
            if not self._cfg.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env.")
            from google import genai

            self._client = genai.Client(api_key=self._cfg.api_key)
        from google.genai import types

        full = (system + "\n\n" + prompt) if system else prompt
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self._client.models.generate_content(model=self._model, contents=full, config=cfg)
        return resp.text or ""

    def _call_groq(self, prompt, system, json_mode, temperature) -> str:
        if not self._groq_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self._model, "messages": messages, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._groq_key}",
                "Content-Type": "application/json",
                "User-Agent": _UA,
            },
        )
        with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"] or ""


def make_client(cache: bool = True) -> GeminiClient:
    """Factory for the live agent/app: Gemini primary, Groq fallback when configured."""
    fallback = None
    if os.getenv("GROQ_API_KEY"):
        try:
            fallback = GeminiClient(cache=cache, provider="groq")
        except Exception:
            fallback = None
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        return GeminiClient(cache=cache, provider="groq")
    return GeminiClient(cache=cache, provider="gemini", fallback=fallback)


def _loads_lenient(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise
