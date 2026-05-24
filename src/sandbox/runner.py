"""Run one (customer, method) full 30-day sandbox session.

Yields a FunnelOutcome with the agent's actions, item picks, and a
funnel_realized_purchase_rate signal.
"""
from __future__ import annotations
from datetime import date

from .env import (
    SandboxState, Candidate, FunnelOutcome,
    generate_stimulus_menu,
)
from .methods import Method


def run_session(customer_id: str, trace: dict, method: Method,
                n_weeks: int = 4, attention_budget: int = 3,
                neighbours: dict | None = None,
                history_pool: list[dict] | None = None) -> FunnelOutcome:
    """Run one customer's 30-day window through `method`. Returns FunnelOutcome."""
    state = SandboxState(
        customer_id=customer_id, week_t=0,
        attention_budget=attention_budget, trace_snapshot=trace,
    )

    # Optional pre-window setup (S2 plan, S4 commitment)
    n_calls = 0
    try:
        n_calls += method.setup(state, trace, neighbours=neighbours, history_pool=history_pool) or 0
    except NotImplementedError:
        pass

    weekly_actions = []
    weekly_purchases: list[str] = []

    for w in range(n_weeks):
        state.week_t = w
        menu = generate_stimulus_menu(customer_id, w, trace)

        # If no attention budget, force-skip the week (saves cost).
        if state.attention_budget <= 0:
            entry = {"week": w, "dp": "DP1", "action": "FORCE_SKIP", "item": None}
            state.funnel_history.append(entry)
            weekly_actions.append({"week": w, "dp1": entry, "dp2": None, "dp3": None})
            continue

        # DP1
        try:
            dp1 = method.step_dp1(state, menu, trace, neighbours=neighbours)
        except Exception as e:
            dp1 = {"action": "SKIP", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        entry1 = {"week": w, "dp": "DP1", "action": dp1["action"], "item": None,
                  "note": dp1.get("note")}
        state.funnel_history.append(entry1)

        if dp1["action"] == "ENGAGE":
            state.attention_budget -= 1
        if dp1["action"] != "ENGAGE":
            weekly_actions.append({"week": w, "dp1": dp1, "dp2": None, "dp3": None})
            continue

        # DP2
        try:
            dp2 = method.step_dp2(state, menu, trace, neighbours=neighbours)
        except Exception as e:
            dp2 = {"action": "EXIT", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        entry2 = {"week": w, "dp": "DP2", "action": dp2["action"],
                  "item": dp2.get("item"), "note": dp2.get("note")}
        state.funnel_history.append(entry2)

        if dp2["action"] != "CONSIDER" or not dp2.get("item"):
            weekly_actions.append({"week": w, "dp1": dp1, "dp2": dp2, "dp3": None})
            continue

        # Pick the menu candidate corresponding to dp2's choice
        ch = (dp2.get("choice") or "A").upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(ch, 0)
        chosen = menu[idx]
        state.attention_budget -= 1

        # DP3 — short-circuit if budget gone
        if state.attention_budget < 0:
            weekly_actions.append({"week": w, "dp1": dp1, "dp2": dp2, "dp3": None})
            continue

        try:
            dp3 = method.step_dp3(state, chosen, trace, neighbours=neighbours)
        except Exception as e:
            dp3 = {"action": "ABANDON", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        entry3 = {"week": w, "dp": "DP3", "action": dp3["action"],
                  "item": chosen.article_id, "note": dp3.get("note")}
        state.funnel_history.append(entry3)

        if dp3["action"] == "PURCHASE":
            state.weekly_purchases.append(chosen.article_id)
            weekly_purchases.append(chosen.article_id)
            state.attention_budget -= 1

        weekly_actions.append({"week": w, "dp1": dp1, "dp2": dp2, "dp3": dp3,
                               "chosen_candidate": {
                                   "article_id": chosen.article_id,
                                   "product_type": chosen.product_type,
                                   "garment_group": chosen.garment_group,
                                   "colour": chosen.colour,
                                   "section": chosen.section,
                                   "label": chosen.label,
                               }})

    purchased = len(weekly_purchases) > 0
    return FunnelOutcome(
        customer_id=customer_id, purchased=purchased,
        weekly_actions=weekly_actions, weekly_purchases=weekly_purchases,
        final_state=state, n_dp_calls=n_calls,
    )
