"""Multi-provider LLM client with disk cache and cost cap.

Currently routes to Gemini by default; OpenAI / Anthropic supported but disabled
(no quota on the available keys). Cached responses are provider-tagged.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

# Patch broken macOS DNS resolver (fallback to `host` command)
try:
    from . import dns_patch
    dns_patch.install()
except Exception:
    pass

# Load .env if present (so OPENAI_API_KEY / GOOGLE_API_KEY are picked up automatically)
_ROOT = Path(__file__).resolve().parents[1]
_envf = _ROOT / ".env"
if _envf.exists():
    for line in _envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

CACHE_DIR = _ROOT / "cache" / "llm"
COST_LOG = _ROOT / "cache" / "llm_costs.jsonl"
COST_CAP_USD = 20.0

# Per-1M-token prices (USD)
PRICES = {
    "gpt-4o-mini":      {"in": 0.15, "out": 0.60},
    "gemini-2.5-flash": {"in": 0.075, "out": 0.30},
    "gemini-2.5-pro":   {"in": 1.25, "out": 5.00},
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
}


def _hash_key(model: str, prompt: str, system: str | None = None) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x1f")
    if system:
        h.update(system.encode())
    h.update(b"\x1f")
    h.update(prompt.encode())
    return h.hexdigest()


def _read_cache(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _write_cache(key: str, payload: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))


def _total_cost() -> float:
    if not COST_LOG.exists():
        return 0.0
    total = 0.0
    for line in COST_LOG.read_text().splitlines():
        try:
            total += json.loads(line)["cost_usd"]
        except Exception:
            pass
    return total


def _log_cost(model: str, in_tokens: int, out_tokens: int):
    price = PRICES.get(model, {"in": 1.0, "out": 5.0})
    cost = (in_tokens / 1e6) * price["in"] + (out_tokens / 1e6) * price["out"]
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "model": model, "in": in_tokens, "out": out_tokens, "cost_usd": cost}) + "\n")
    return cost


def _gemini_call(model: str, prompt: str, system: Optional[str] = None, max_tokens: int = 200) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    # Disable Gemini-2.5 "thinking" tokens — they eat the output budget.
    thinking = types.ThinkingConfig(thinking_budget=0) if "2.5" in model else None
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.0,
        system_instruction=system if system else None,
        thinking_config=thinking,
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    text = resp.text or ""
    usage = resp.usage_metadata
    in_tokens = getattr(usage, "prompt_token_count", 0) or 0
    out_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return {"text": text, "in": in_tokens, "out": out_tokens}


def _openai_call(model: str, prompt: str, system: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.0) -> dict:
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    text = resp.choices[0].message.content
    usage = resp.usage
    return {"text": text, "in": usage.prompt_tokens, "out": usage.completion_tokens}


def call_llm(model: str, prompt: str, system: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.0) -> dict:
    """Unified entry. Dispatches to Gemini / OpenAI based on model name."""
    key = _hash_key(model, prompt, system)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    if _total_cost() >= COST_CAP_USD:
        raise RuntimeError(f"LLM cost cap of ${COST_CAP_USD} reached")

    # Retry on transient failures (503, 429, network) with exponential backoff
    last_err = None
    for attempt in range(5):
        try:
            if model.startswith("gemini"):
                raw = _gemini_call(model, prompt, system, max_tokens)
            elif model.startswith("gpt"):
                raw = _openai_call(model, prompt, system, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown model: {model}")
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            retry_markers = (
                "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate limit", "timeout",
                "nodename", "servname",            # DNS failure (Errno 8)
                "Errno 8", "Errno 60", "Errno 54", # network errors
                "Connection reset", "Connection aborted", "Temporary failure",
                "ConnectionError", "TimeoutError",
            )
            if any(s in msg for s in retry_markers):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise
    else:
        raise last_err

    cost = _log_cost(model, raw["in"], raw["out"])
    payload = {"text": raw["text"], "in": raw["in"], "out": raw["out"], "cost": cost, "model": model}
    _write_cache(key, payload)
    return payload


# Backwards-compatible alias used by older scripts
def call_openai(model: str, prompt: str, system: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.0) -> dict:
    return call_llm(model, prompt, system, max_tokens, temperature)


def remaining_budget_usd() -> float:
    return COST_CAP_USD - _total_cost()
