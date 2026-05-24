"""H&M deliberation prompt + LLM call.

Two prompt variants (audit-required ablation):
  - F-base: includes the H&M per-bucket 30-day repeat-purchase base-rate table.
  - F-nobase: identical otherwise; the table is redacted.

This ablation isolates the cognition pipeline's contribution from the
test-set marginal leakage that would otherwise contaminate the headline.
"""

from __future__ import annotations
import json
import re
from ..llm_client import call_llm

# H&M per-activity-bucket 30-day repeat-purchase rates from Phase 1 EDA.
# These come from the TEST split — including them in the prompt is the
# specific leak the F-base/F-nobase ablation is designed to isolate.
HM_BASE_RATES = {
    "1": 0.027,
    "2-5": 0.049,
    "6-20": 0.123,
    "21-100": 0.326,
    "101+": 0.598,
}

SYSTEM_PROMPT_BASE = (
    "You are a behavioural-prediction model embodying a single H&M customer. "
    "Given their purchase history and pre-computed cognitive signals (attended features, "
    "retrieved memories, gut friction estimate), you reason in first-person about how "
    "likely you are to make any new H&M purchase in the next 30 days, then output JSON.\n\n"
    "Required output keys:\n"
    "  reasoning            : 1-2 sentences, first-person, referencing concrete trace facts.\n"
    "  verbatim_reaction    : ONE first-person sentence describing what you would do this month.\n"
    "  key_objection        : ONE phrase naming the strongest reason you might NOT buy.\n"
    "  baseline_30d_buy_likelihood : 0-100 percent. Your unconditional intent.\n"
    "  stimulus_30d_buy_likelihood : 0-100 percent. With normal H&M browsing/marketing context.\n"
    "  friction_score       : 0-100. Higher = more friction against purchasing.\n"
    "  confidence           : 0-100. How sure you are about the above.\n\n"
    "Calibrate to your trace. Heavy buyers (101+ prior tx) should land near 60-80%. "
    "Single-purchase or lapsed customers should land below 10%. Mid-cohorts in between. "
    "Output strict JSON only — no markdown, no preamble, no code fences."
)


def _format_base_rate_table() -> str:
    rows = "\n".join(f"  - prior-tx-count {k}: {v*100:.1f}%" for k, v in HM_BASE_RATES.items())
    return (
        "Empirical base rates from H&M test data (use as calibration anchor):\n" + rows
    )


def _trace_block(trace: dict, attention: dict, memories: list[dict], affect: dict) -> str:
    ps = trace["purchase_stats"]
    ps_lines = [
        f"  - total_orders: {ps['total_orders']}",
        f"  - recency_days: {ps['recency_days']}",
        f"  - tenure_days: {ps['tenure_days']}",
        f"  - aov: {ps['aov']:.4f}",
        f"  - distinct_articles: {ps['distinct_articles']}",
        f"  - channel2_share: {ps['channel2_share']:.2f}",
    ]
    sums = trace["product_summary"]
    pr_lines = [
        f"  - top_section: {sums.get('top_section')}",
        f"  - top_garment_group: {sums.get('top_garment_group')}",
        f"  - top_color: {sums.get('top_color')}",
        f"  - top_product_type: {sums.get('top_product_type')}",
        f"  - sections_seen: {sums['sections_seen']}",
    ]

    identity = trace["identity"]
    timeline = trace["timeline"]
    recent_lines = []
    for r in trace["recent_purchases"][:8]:
        recent_lines.append(
            f"    - {r['days_ago']}d ago: {r['prod_name']} ({r['product_type']}, {r['color']}, {r['section']}, {r['channel']})"
        )

    attn_str = f"primary: {attention['primary_focus']}, secondary: {attention['secondary_focus']}"

    mem_lines = []
    for m in memories:
        mem_lines.append(f"    - [{m['type']} rel={m['relevance']}] {m['summary']}")

    parts = [
        f"IDENTITY: age={identity.get('age')}",
        "PURCHASE STATS:\n" + "\n".join(ps_lines),
        "PRODUCT SUMMARY:\n" + "\n".join(pr_lines),
        f"TIMELINE: first_tx={timeline['first_tx_date']}, last_tx={timeline['last_tx_date']}, "
        f"avg_inter_purchase_days={timeline['avg_inter_purchase_days']}",
        f"PERSONALITY: {trace['personality']}",
        f"DERIVED FLAGS: {trace['derived_flags']}",
        "ATTENDED FEATURES (deterministic salience ranking):\n  " + attn_str,
        "RETRIEVED MEMORIES (deterministic, recency + pattern-weighted, top-5):\n" + "\n".join(mem_lines),
        f"GUT REACTION (deterministic affect pre-compute): {affect['gut_reaction']} "
        f"(blended friction = {affect['blended_friction']})",
        "RECENT PURCHASES (last 8):\n" + "\n".join(recent_lines),
    ]
    return "\n\n".join(parts)


def build_prompt(trace: dict, attention: dict, memories: list[dict], affect: dict,
                 include_base_rate: bool) -> str:
    body = _trace_block(trace, attention, memories, affect)
    pieces = [body]
    if include_base_rate:
        pieces.append(_format_base_rate_table())
    pieces.append(
        "Question: in the next 30 days starting from your last_tx_date+1, will you make any new H&M purchase?\n"
        "Reason briefly first-person, then output JSON with the required keys."
    )
    return "\n\n".join(pieces)


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _brace_balanced_extract(s: str) -> str | None:
    """Scan `s` and return the first top-level brace-balanced object substring, or None.
    AUDIT FIX (Agent C APPLY NOW): replaces a `\\{[^{}]*\\}` regex that failed on nested JSON."""
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return s[start:i + 1]
    return None


def parse_response(text: str) -> dict:
    """Parse the LLM's JSON response. Returns dict with the required fields and
    `stated_intent_prob` ∈ [0, 1] (canonical extraction).
    AUDIT FIX (Agent C): replace broken regex with brace-balanced scan; add `parse_ok`
    flag so downstream analyses can filter silent fallbacks."""
    s = _strip_code_fence(text)
    parsed: dict = {}
    parse_ok = False
    try:
        parsed = json.loads(s)
        parse_ok = True
    except Exception:
        block = _brace_balanced_extract(s)
        if block:
            try:
                parsed = json.loads(block)
                parse_ok = True
            except Exception:
                parsed = {}
    out = {
        "reasoning": parsed.get("reasoning", ""),
        "verbatim_reaction": parsed.get("verbatim_reaction", ""),
        "key_objection": parsed.get("key_objection", ""),
        "baseline_30d_buy_likelihood": _safe_pct(parsed.get("baseline_30d_buy_likelihood")),
        "stimulus_30d_buy_likelihood": _safe_pct(parsed.get("stimulus_30d_buy_likelihood")),
        "friction_score": _safe_pct(parsed.get("friction_score")),
        "confidence": _safe_pct(parsed.get("confidence")),
        "raw_text": text,
        "parse_ok": parse_ok,
    }
    val = out["stimulus_30d_buy_likelihood"]
    if val is None:
        val = out["baseline_30d_buy_likelihood"]
    out["stated_intent_prob"] = (val / 100.0) if val is not None else 0.5
    return out


def _safe_pct(v) -> float | None:
    """Coerce v into a 0..100 percent. AUDIT FIX (Agent C APPLY NOW):
    previously, the value `1` was interpreted as `1.0 on a 0..1 scale → 100%`.
    But the prompt asks for 0..100, so an int `1` likely means 1%. We now
    treat integer-typed values >=1 as percent literals, and only strictly
    fractional <1.0 values as 0..1 scale."""
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    # If v is exactly an integer (no fractional part) and >= 1, treat as percent.
    is_int_like = isinstance(v, (int, bool)) or (isinstance(v, float) and x.is_integer())
    if is_int_like and x >= 1.0:
        return max(0.0, min(100.0, x))
    if 0.0 <= x < 1.0:
        return x * 100.0  # 0..1 scale
    return max(0.0, min(100.0, x))


def deliberate(trace: dict, attention: dict, memories: list[dict], affect: dict,
               include_base_rate: bool, model: str = "gemini-2.5-flash") -> dict:
    """One LLM call. Returns parsed JSON + the prompt used (for audit)."""
    prompt = build_prompt(trace, attention, memories, affect, include_base_rate)
    resp = call_llm(model, prompt, system=SYSTEM_PROMPT_BASE, max_tokens=400)
    parsed = parse_response(resp["text"])
    parsed["_prompt"] = prompt
    parsed["_cost"] = resp.get("cost", 0.0)
    return parsed
