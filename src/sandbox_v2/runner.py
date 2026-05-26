"""Sandbox v2 runner — runs one (customer, agent_policy) session through the
real-world environment with stochastic dynamics.

Differences from sandbox_v1.runner:
- ENGAGE doesn't deterministically produce a browse — bernoulli draw with
  p_stimulus_arrives(recency, dow, week-of-month, fatigue).
- DP2 CONSIDER may fail at the environment if inventory is depleted (shared
  inventory model).
- After PURCHASE, satisfaction signal is computed and fed back to next DP.
- Recency rolls forward as the simulation progresses.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

from .world import (
    WorldState, SharedInventory, simulate_stimulus_arrival,
    post_purchase_reward, advance_week, world_seed,
)
from ..sandbox.env import generate_stimulus_menu, Candidate


@dataclass
class WorldOutcome:
    customer_id: str
    purchased: bool
    weekly_actions: list[dict]
    weekly_purchases: list[str]
    final_state: WorldState
    n_dp_calls: int


def _articles_words(c: Candidate) -> str:
    return f"{c.product_type} {c.garment_group} {c.colour} {c.section}".lower()


def run_world_session(customer_id: str, trace: dict, method,
                      inventory: SharedInventory,
                      n_weeks: int = 4, attention_budget: int = 3,
                      neighbours: dict | None = None,
                      run_id: str = "v2_default") -> WorldOutcome:
    """Run a stochastic sandbox session. Returns a WorldOutcome.

    The agent's per-DP decisions are made via `method` (which has the same
    interface as v1 methods). The environment then applies stochastic
    dynamics on top of the agent's intent.
    """
    rng_world = random.Random(world_seed(customer_id, 0, run_id))
    recency = trace["purchase_stats"]["recency_days"]
    # Map physical week-in-month to a starting "day of week"
    state = WorldState(
        customer_id=customer_id,
        week_t=0, day_of_week=2, week_in_month=1,
        recency_days=recency,
        attention_budget=attention_budget,
        trace_snapshot=trace,
    )

    # Optional pre-window setup (S2 plan, S4 commitment)
    n_calls = 0
    try:
        n_calls += method.setup(state, trace, neighbours=neighbours) or 0
    except (NotImplementedError, TypeError):
        pass

    weekly_actions = []
    weekly_purchases: list[str] = []

    for w in range(n_weeks):
        state.week_t = w
        state.week_in_month = min(4, w + 1)
        menu = generate_stimulus_menu(customer_id, w, trace)

        if state.attention_budget <= 0:
            entry = {"week": w, "dp1_intent": "FORCE_SKIP",
                     "stimulus_arrived": False, "p_arrive": 0.0}
            state.funnel_history.append(entry)
            weekly_actions.append(entry)
            advance_week(state, purchased_this_week=False)
            continue

        # DP1 — agent INTENT
        try:
            dp1 = method.step_dp1(state, menu, trace, neighbours=neighbours)
        except Exception as e:
            dp1 = {"action": "SKIP", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        intent = dp1.get("action", "SKIP")

        # World layer: even if intent=ENGAGE, the stimulus may not arrive.
        stim = simulate_stimulus_arrival(state, rng_world) if intent == "ENGAGE" \
               else type("X", (), {"arrived": False, "p_arrived": 0.0})()

        entry1 = {"week": w, "dp1_intent": intent,
                  "stimulus_arrived": bool(stim.arrived),
                  "p_arrive": float(stim.p_arrived)}
        state.funnel_history.append({"week": w, "dp": "DP1", "action": intent,
                                      "arrived": stim.arrived})

        if intent == "ENGAGE":
            state.attention_budget -= 1

        if intent != "ENGAGE" or not stim.arrived:
            weekly_actions.append(entry1)
            advance_week(state, purchased_this_week=False)
            continue

        # DP2 — agent's browse decision (because the email arrived)
        try:
            dp2 = method.step_dp2(state, menu, trace, neighbours=neighbours)
        except Exception as e:
            dp2 = {"action": "EXIT", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        dp2_action = dp2.get("action", "EXIT")
        dp2_choice = (dp2.get("choice") or "A").upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(dp2_choice, 0)

        entry2 = {"dp2_intent": dp2_action, "dp2_choice": dp2_choice}
        state.funnel_history.append({"week": w, "dp": "DP2",
                                      "action": dp2_action, "choice": dp2_choice})

        if dp2_action != "CONSIDER":
            weekly_actions.append({**entry1, **entry2})
            advance_week(state, purchased_this_week=False)
            continue

        chosen = menu[idx]
        state.attention_budget -= 1

        # World layer: inventory check. If sold out -> force ABANDON.
        if not inventory.in_stock(chosen.article_id):
            entry3 = {"dp3_intent": "BLOCKED_NO_STOCK", "purchased": False,
                      "satisfaction": None}
            weekly_actions.append({**entry1, **entry2, **entry3,
                                    "chosen_article_id": chosen.article_id})
            advance_week(state, purchased_this_week=False)
            continue

        # DP3 — purchase decision
        if state.attention_budget < 0:
            weekly_actions.append({**entry1, **entry2, "dp3_intent": "FORCE_BUDGET"})
            advance_week(state, purchased_this_week=False)
            continue

        try:
            dp3 = method.step_dp3(state, chosen, trace, neighbours=neighbours)
        except Exception as e:
            dp3 = {"action": "ABANDON", "error": str(e), "scalar_prob": 0.0}
        n_calls += 1
        dp3_action = dp3.get("action", "ABANDON")

        # S4 commitment cap — short-circuit if saturated
        if state.declared_max_purchases is not None and \
           len(weekly_purchases) >= state.declared_max_purchases:
            dp3_action = "ABANDON"

        sat = None
        purch_this_week = False
        if dp3_action == "PURCHASE":
            # Try to claim from shared inventory
            ok = inventory.try_purchase(customer_id, chosen.article_id)
            if ok:
                state.weekly_purchases.append(chosen.article_id)
                weekly_purchases.append(chosen.article_id)
                state.attention_budget -= 1
                sat = post_purchase_reward(state, _articles_words(chosen), trace)
                state.last_satisfaction_signal = sat
                purch_this_week = True
            else:
                dp3_action = "BLOCKED_NO_STOCK"

        entry3 = {"dp3_intent": dp3_action, "purchased": purch_this_week,
                  "satisfaction": sat}
        weekly_actions.append({**entry1, **entry2, **entry3,
                                "chosen_article_id": chosen.article_id,
                                "chosen_in_or_out": chosen.label})
        advance_week(state, purchased_this_week=purch_this_week)

    return WorldOutcome(
        customer_id=customer_id, purchased=len(weekly_purchases) > 0,
        weekly_actions=weekly_actions, weekly_purchases=weekly_purchases,
        final_state=state, n_dp_calls=n_calls,
    )
