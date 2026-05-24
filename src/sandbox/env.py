"""Behavioral sandbox environment for LLM digital twins (v3 paper).

Per preregistration_v3 §4: a 30-day window with 4 weekly cycles. Each cycle has
3 decision points (DP1 stimulus, DP2 menu, DP3 purchase). Stimuli are exogenous,
stochastic, and 3-candidate menus drawn from H&M articles. Attention budget is
depletable (3 units across 4 weeks). Funnel history is persistent across cycles.

Transitions are deterministic given the agent's action — the sandbox is a
decision-elicitation protocol under commitment pressure, not a world model.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from hashlib import sha256
import random
import polars as pl

from ..data import load_articles, load_transactions
from .. import T_TEST_CUTOFF


SECTIONS_POOL = [
    "Womens Everyday Collection",
    "Mens Underwear & Basics",
    "Kids Boy Bottoms",
    "Womens Premium Outerwear",
    "Ladies Sport",
    "Mens Sport",
    "Baby Boy",
    "Womens Lingerie",
    "Mens Casual",
    "Womens Tops",
]


@dataclass
class Candidate:
    article_id: str
    prod_name: str
    product_type: str
    garment_group: str
    colour: str
    section: str
    label: str  # "in-section" | "cross-section" | "OOD"


@dataclass
class SandboxState:
    customer_id: str
    week_t: int                # 0..3
    funnel_history: list[dict] = field(default_factory=list)  # [(week, dp, action, item?, note?)]
    attention_budget: int = 3
    weekly_purchases: list[str] = field(default_factory=list)
    declared_max_purchases: int | None = None  # for S4 commitment device
    declared_max_spend_units: int | None = None
    # Snapshot of pre-cutoff trace (constant across sandbox session)
    trace_snapshot: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Stimulus generation ----------

_ARTICLES_DF: pl.DataFrame | None = None
_POPULAR_BY_SECTION: dict[str, list[dict]] | None = None
_GLOBAL_POPULAR: list[dict] | None = None


def _load_articles_cached() -> pl.DataFrame:
    global _ARTICLES_DF
    if _ARTICLES_DF is None:
        # Articles enriched with pre-cutoff popularity from transactions
        art = load_articles().select([
            "article_id", "prod_name", "product_type_name",
            "garment_group_name", "colour_group_name", "section_name",
            "index_group_name", "department_name",
        ]).collect()
        cutoff = date.fromisoformat(T_TEST_CUTOFF)
        # Rank by pre-cutoff popularity within section
        tx = (
            load_transactions()
            .filter(pl.col("t_dat") < cutoff)
            .group_by("article_id").len()
            .rename({"len": "popularity"})
            .collect()
        )
        art = art.join(tx, on="article_id", how="left").with_columns(
            pl.col("popularity").fill_null(0)
        )
        _ARTICLES_DF = art
    return _ARTICLES_DF


def _ensure_popularity_indices():
    global _POPULAR_BY_SECTION, _GLOBAL_POPULAR
    if _POPULAR_BY_SECTION is not None:
        return
    art = _load_articles_cached()
    pop_per_section: dict[str, list[dict]] = {}
    for sec, sub in art.filter(pl.col("section_name").is_not_null()).group_by("section_name"):
        sec_name = sec[0] if isinstance(sec, tuple) else sec
        top = sub.sort("popularity", descending=True).head(200)
        pop_per_section[sec_name] = top.to_dicts()
    _POPULAR_BY_SECTION = pop_per_section
    _GLOBAL_POPULAR = art.sort("popularity", descending=True).head(500).to_dicts()


def _seed_for(customer_id: str, week: int) -> int:
    h = sha256(f"{customer_id}|{week}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def generate_stimulus_menu(customer_id: str, week: int, trace: dict) -> list[Candidate]:
    """Three-candidate menu for (customer, week). Deterministic given inputs.

    A: in-section (customer's top section)
    B: cross-section (a section the customer has visited but is not their top)
    C: OOD (popular item from a section the customer has never visited)
    """
    _ensure_popularity_indices()
    rng = random.Random(_seed_for(customer_id, week))
    top_section = trace["product_summary"].get("top_section")
    seen_sections = {
        rp["section"] for rp in trace.get("recent_purchases", []) if rp.get("section")
    }
    # Fallback if no top section
    if top_section is None:
        top_section = rng.choice([s for s in SECTIONS_POOL if s in _POPULAR_BY_SECTION] or list(_POPULAR_BY_SECTION.keys()))

    # In-section candidate
    in_pool = _POPULAR_BY_SECTION.get(top_section, _GLOBAL_POPULAR or [])
    a = rng.choice(in_pool[:50]) if in_pool else rng.choice(_GLOBAL_POPULAR)

    # Cross-section candidate
    cross_options = [s for s in seen_sections if s != top_section and s in _POPULAR_BY_SECTION]
    if not cross_options:
        cross_options = [s for s in _POPULAR_BY_SECTION.keys() if s != top_section][:20]
    cross_section = rng.choice(cross_options)
    b = rng.choice(_POPULAR_BY_SECTION[cross_section][:50])

    # OOD candidate
    unseen = [s for s in _POPULAR_BY_SECTION.keys() if s not in seen_sections and s != top_section]
    if not unseen:
        unseen = list(_POPULAR_BY_SECTION.keys())
    ood_section = rng.choice(unseen)
    c = rng.choice(_POPULAR_BY_SECTION[ood_section][:50])

    def _to_cand(d: dict, label: str) -> Candidate:
        return Candidate(
            article_id=str(d["article_id"]),
            prod_name=str(d.get("prod_name") or "Article"),
            product_type=str(d.get("product_type_name") or "Unknown"),
            garment_group=str(d.get("garment_group_name") or "Unknown"),
            colour=str(d.get("colour_group_name") or "Unknown"),
            section=str(d.get("section_name") or "Unknown"),
            label=label,
        )

    return [_to_cand(a, "in-section"), _to_cand(b, "cross-section"), _to_cand(c, "OOD")]


# ---------- Funnel run ----------

@dataclass
class FunnelOutcome:
    customer_id: str
    purchased: bool  # f_hat_LLM
    weekly_actions: list[dict]
    weekly_purchases: list[str]  # article_ids
    final_state: SandboxState
    n_dp_calls: int = 0  # LLM call count for cost accounting


def render_state_for_prompt(state: SandboxState, trace: dict) -> str:
    """Compact JSON-ish text rendering of state passed to agents."""
    rp = trace.get("recent_purchases", [])[-5:]
    ps = trace["purchase_stats"]
    out = [
        f"CUSTOMER PROFILE (pre-cutoff snapshot, no PII):",
        f"  age: {trace['identity'].get('age')}",
        f"  total_orders: {ps['total_orders']}",
        f"  recency_days: {ps['recency_days']}",
        f"  tenure_days: {ps['tenure_days']}",
        f"  aov: {ps.get('aov')}",
        f"  channel2_share: {ps.get('channel2_share')}",
        f"  top_section: {trace['product_summary'].get('top_section')}",
        f"  top_garment_group: {trace['product_summary'].get('top_garment_group')}",
        f"  top_colour: {trace['product_summary'].get('top_color')}",
        f"  personality: {trace.get('personality')}",
        "",
        "RECENT PURCHASES (most-recent 5):",
    ]
    for r in rp:
        out.append(
            f"  {r['days_ago']}d ago: {r['product_type']} / {r['garment_group']} / {r['color']} / {r['section']} (${r.get('price','?')})"
        )
    out.append("")
    out.append(f"SANDBOX WINDOW STATE: week={state.week_t+1}/4, attention_budget_remaining={state.attention_budget}, weekly_purchases_so_far={len(state.weekly_purchases)}")
    if state.funnel_history:
        out.append("FUNNEL HISTORY (this 30-day window):")
        for h in state.funnel_history[-12:]:
            out.append(f"  week{h['week']+1}/{h['dp']}: {h['action']}{(' item=' + h['item']) if h.get('item') else ''}{(' note=' + h['note']) if h.get('note') else ''}")
    if state.declared_max_purchases is not None:
        out.append(f"COMMITMENT (you declared at the start of this window): max_purchases={state.declared_max_purchases}")
    return "\n".join(out)


def render_menu_for_prompt(menu: list[Candidate]) -> str:
    lines = ["MENU (three options this week):"]
    for i, c in enumerate(menu):
        lines.append(f"  [{['A','B','C'][i]}] {c.product_type} / {c.garment_group} / {c.colour} / section: {c.section}")
    return "\n".join(lines)
