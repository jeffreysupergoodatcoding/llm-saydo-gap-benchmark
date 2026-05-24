"""Eight method-policies tested as digital-twin agents in the sandbox.

Per preregistration_v3 §5:
  Literature baselines: M1 zero-shot, M3 few-shot k-NN, M8 RAG with outcome
  labels, M9 implementation-intentions.
  Sandbox-native: S1 Reflexion-in-funnel, S2 outcome-conditioned planning,
  S3 tree-of-thoughts, S4 commitment device.

Each method exposes a uniform interface:
  step_dp(state, dp, menu, trace, neighbours=None, history_pool=None) -> action_dict
  setup(state, trace, neighbours=None, history_pool=None) -> None  # for S2, S4

Action dict shape:
  {"action": <action_str>, "item": <article_id or None>, "raw": <raw_llm_text>,
   "note": <self-critique or plan text or None>, "scalar_prob": <0..1 if step is DP1>}
"""

from __future__ import annotations
import json
import re
from typing import Optional

from ..llm_client import call_llm
from .env import SandboxState, Candidate, render_state_for_prompt, render_menu_for_prompt


MODEL_DEFAULT = "gemini-2.5-flash"
JSON_SCHEMA_HINT = "Respond with a single JSON object only, no prose."


# ============================ utilities ============================

def _safe_json(text: str) -> dict | None:
    if not text:
        return None
    # Strip code fences if present
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    # Find first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _coerce_action(raw: dict | None, allowed: list[str], default: str) -> str:
    if raw is None:
        return default
    a = str(raw.get("action", "")).upper().strip()
    for a_allowed in allowed:
        if a_allowed in a:
            return a_allowed
    return default


def _coerce_prob(raw: dict | None, key: str = "p", default: float = 0.5) -> float:
    if raw is None:
        return default
    try:
        v = float(raw.get(key, default))
    except Exception:
        return default
    return max(0.0, min(1.0, v))


# ============================ method base ============================

class Method:
    name: str = "BASE"
    description: str = ""

    def setup(self, state: SandboxState, trace: dict, **kwargs) -> int:
        """Optional pre-funnel setup. Return n_llm_calls used."""
        return 0

    def step_dp1(self, state: SandboxState, menu: list[Candidate], trace: dict, **kwargs) -> dict:
        raise NotImplementedError

    def step_dp2(self, state: SandboxState, menu: list[Candidate], trace: dict, **kwargs) -> dict:
        raise NotImplementedError

    def step_dp3(self, state: SandboxState, chosen: Candidate, trace: dict, **kwargs) -> dict:
        raise NotImplementedError

    def scalar_prob(self, trace: dict, **kwargs) -> dict:
        """Single-shot scalar probability of any 30-day purchase. Matches v2 scalar."""
        prompt = (
            render_state_for_prompt(SandboxState(customer_id=trace.get("_cid", ""), week_t=0, trace_snapshot=trace), trace)
            + "\n\nQUESTION: What is the probability this customer makes any H&M purchase in the next 30 days?\n"
            + "Return JSON: {\"p\": <0..1>, \"reasoning\": <one short sentence>}"
        )
        r = call_llm(MODEL_DEFAULT, prompt, max_tokens=120)
        j = _safe_json(r["text"]) or {"p": 0.5}
        return {"scalar_prob": _coerce_prob(j, "p"), "reasoning": j.get("reasoning", "")}


# ============================ M1: zero-shot ============================

class M1_ZeroShot(Method):
    name = "M1"
    description = "Zero-shot per-DP elicitation, no extra context."

    def _dp_prompt(self, state, menu, trace, dp_question):
        return (
            render_state_for_prompt(state, trace)
            + ("\n\n" + render_menu_for_prompt(menu) if menu else "")
            + "\n\n" + dp_question
            + "\n" + JSON_SCHEMA_HINT
        )

    def step_dp1(self, state, menu, trace, **kw):
        q = (
            "DECISION POINT DP1: A weekly promotional stimulus has arrived. "
            "Decide if this customer ENGAGES with it or SKIPS. Engaging costs 1 attention budget unit; "
            "SKIPPING costs 0. Consider that you have limited remaining attention.\n"
            'Return JSON: {"action": "ENGAGE" | "SKIP", "p": <prob this customer would engage 0..1>}'
        )
        r = call_llm(MODEL_DEFAULT, self._dp_prompt(state, menu, trace, q), max_tokens=80)
        j = _safe_json(r["text"])
        action = _coerce_action(j, ["ENGAGE", "SKIP"], default="SKIP")
        return {"action": action, "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        q = (
            "DECISION POINT DP2: The customer has chosen to ENGAGE and is browsing. "
            "Decide if they EXIT or CONSIDER one of the three menu items. "
            "If CONSIDER, pick exactly one of A/B/C.\n"
            'Return JSON: {"action": "EXIT" | "CONSIDER", "choice": "A"|"B"|"C", "p": <prob 0..1>}'
        )
        r = call_llm(MODEL_DEFAULT, self._dp_prompt(state, menu, trace, q), max_tokens=100)
        j = _safe_json(r["text"])
        action = _coerce_action(j, ["EXIT", "CONSIDER"], default="EXIT")
        choice_letter = str((j or {}).get("choice", "A")).upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(choice_letter, 0)
        item = menu[idx].article_id if action == "CONSIDER" else None
        return {"action": action, "item": item, "choice": choice_letter,
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp3(self, state, chosen, trace, **kw):
        ctx = render_state_for_prompt(state, trace) + (
            f"\n\nCONSIDERING ITEM: {chosen.product_type} / {chosen.garment_group} / {chosen.colour} / section {chosen.section}\n"
        )
        q = (
            "DECISION POINT DP3: The customer is on the product page of the item above. "
            "Decide if they PURCHASE it or ABANDON. PURCHASE costs 1 attention budget unit.\n"
            'Return JSON: {"action": "PURCHASE" | "ABANDON", "p": <prob 0..1>}'
        )
        r = call_llm(MODEL_DEFAULT, ctx + q + "\n" + JSON_SCHEMA_HINT, max_tokens=80)
        j = _safe_json(r["text"])
        action = _coerce_action(j, ["PURCHASE", "ABANDON"], default="ABANDON")
        return {"action": action, "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}


# ============================ M3: few-shot k-NN ICL ============================

class M3_KNN(Method):
    name = "M3"
    description = "Few-shot in-context learning with 5 RFM-nearest customers + their funnel outcomes."

    def __init__(self, neighbours: dict | None = None):
        # neighbours: dict[cid] -> list of 5 example dicts each with summary + outcome
        self.neighbours = neighbours or {}

    def _examples_block(self, cid: str) -> str:
        examples = self.neighbours.get(cid, [])[:5]
        if not examples:
            return ""
        out = ["FIVE NEAREST PRIOR CUSTOMERS (by RFM) WITH THEIR 30-DAY OUTCOMES:"]
        for i, ex in enumerate(examples):
            out.append(
                f"  EX{i+1}: orders={ex['total_orders']}, recency={ex['recency_days']}d, "
                f"top_section={ex['top_section']}, outcome_30d_purchase={'YES' if ex['label']==1 else 'NO'}"
            )
        return "\n".join(out) + "\n"

    def _dp_prompt(self, state, menu, trace, dp_question, cid):
        return (
            self._examples_block(cid)
            + render_state_for_prompt(state, trace)
            + ("\n\n" + render_menu_for_prompt(menu) if menu else "")
            + "\n\n" + dp_question
            + "\n" + JSON_SCHEMA_HINT
        )

    def step_dp1(self, state, menu, trace, **kw):
        q = M1_ZeroShot().step_dp1.__doc__ or ""
        q = (
            "DECISION POINT DP1: Use the 5 prior-customer outcomes above as in-context reference. "
            "Decide if this customer ENGAGES with the weekly stimulus or SKIPS.\n"
            'Return JSON: {"action": "ENGAGE" | "SKIP", "p": <prob 0..1>}'
        )
        r = call_llm(MODEL_DEFAULT, self._dp_prompt(state, menu, trace, q, state.customer_id), max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["ENGAGE", "SKIP"], "SKIP"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        q = (
            "DECISION POINT DP2: Use the 5 prior-customer outcomes for context. EXIT or CONSIDER A/B/C?\n"
            'Return JSON: {"action": "EXIT" | "CONSIDER", "choice": "A"|"B"|"C", "p": <prob 0..1>}'
        )
        r = call_llm(MODEL_DEFAULT, self._dp_prompt(state, menu, trace, q, state.customer_id), max_tokens=100)
        j = _safe_json(r["text"])
        a = _coerce_action(j, ["EXIT", "CONSIDER"], "EXIT")
        ch = str((j or {}).get("choice", "A")).upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(ch, 0)
        return {"action": a, "item": menu[idx].article_id if a == "CONSIDER" else None,
                "choice": ch, "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp3(self, state, chosen, trace, **kw):
        ctx = (
            self._examples_block(state.customer_id)
            + render_state_for_prompt(state, trace)
            + f"\n\nCONSIDERING ITEM: {chosen.product_type} / {chosen.garment_group} / {chosen.colour}\n"
        )
        q = ('DECISION POINT DP3: PURCHASE or ABANDON?\nReturn JSON: {"action": "PURCHASE" | "ABANDON", "p": <0..1>}')
        r = call_llm(MODEL_DEFAULT, ctx + q + "\n" + JSON_SCHEMA_HINT, max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["PURCHASE", "ABANDON"], "ABANDON"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}


# ============================ M8: RAG with outcome labels ============================

class M8_RAG(Method):
    name = "M8"
    description = "Retrieval-augmented: 5 prior trajectories (history + funnel-step actions) with realized outcomes, retrieved per DP."

    def __init__(self, history_pool: list[dict] | None = None):
        # history_pool: list of dicts each {trace_summary, funnel_actions_proxy, outcome_label}
        # We don't have real prior funnel actions for past customers, so we proxy with their actual
        # post-cutoff transaction sequence — a stronger signal than M3.
        self.pool = history_pool or []

    def _retrieve(self, trace: dict, k: int = 5) -> list[dict]:
        if not self.pool:
            return []
        # Simple similarity over RFM: matches v2's neighbour selection
        target = (
            float(trace["purchase_stats"]["total_orders"]),
            float(trace["purchase_stats"]["recency_days"]),
        )
        def dist(e):
            return abs(e["total_orders"] - target[0]) + 0.1 * abs(e["recency_days"] - target[1])
        return sorted(self.pool, key=dist)[:k]

    def _block(self, trace):
        items = self._retrieve(trace)
        if not items:
            return ""
        out = ["RETRIEVED 5 SIMILAR-PROFILE CUSTOMERS WITH THEIR FUNNEL-LIKE 30-DAY ACTIVITY:"]
        for i, e in enumerate(items):
            out.append(
                f"  CASE{i+1}: orders={e['total_orders']}, recency={e['recency_days']}d, "
                f"top_section={e['top_section']}, 30d_realized_purchase_count={e['n_label_window']}, "
                f"outcome_purchased_within_30d={'YES' if e['label']==1 else 'NO'}"
            )
        return "\n".join(out) + "\n"

    def step_dp1(self, state, menu, trace, **kw):
        p = (
            self._block(trace)
            + render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP1: ENGAGE or SKIP weekly stimulus, given the retrieved cases above.\n"
            + 'Return JSON: {"action": "ENGAGE" | "SKIP", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["ENGAGE", "SKIP"], "SKIP"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        p = (
            self._block(trace)
            + render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP2: EXIT or CONSIDER A/B/C?\n"
            + 'Return JSON: {"action": "EXIT" | "CONSIDER", "choice": "A"|"B"|"C", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=100)
        j = _safe_json(r["text"])
        a = _coerce_action(j, ["EXIT", "CONSIDER"], "EXIT")
        ch = str((j or {}).get("choice", "A")).upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(ch, 0)
        return {"action": a, "item": menu[idx].article_id if a == "CONSIDER" else None,
                "choice": ch, "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp3(self, state, chosen, trace, **kw):
        p = (
            self._block(trace)
            + render_state_for_prompt(state, trace)
            + f"\n\nCONSIDERING ITEM: {chosen.product_type} / {chosen.garment_group} / {chosen.colour}\n\n"
            + 'DECISION POINT DP3: PURCHASE or ABANDON?\nReturn JSON: {"action": "PURCHASE" | "ABANDON", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["PURCHASE", "ABANDON"], "ABANDON"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}


# ============================ M9: implementation-intentions ============================

class M9_ImplementationIntentions(Method):
    name = "M9"
    description = "Gollwitzer (1999): force agent to state a structured if-then plan at DP1, then act on it."

    def step_dp1(self, state, menu, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP1: First, generate an IMPLEMENTATION INTENTION (Gollwitzer 1999) of the form: "
            + '"IF I see [type of stimulus] THEN I will [SKIP or ENGAGE]." '
            + "Then decide for THIS stimulus.\n"
            + 'Return JSON: {"if_then_plan": "<one sentence if-then>", "action": "ENGAGE" | "SKIP", "p": <0..1>}\n'
            + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=140)
        j = _safe_json(r["text"]) or {}
        return {"action": _coerce_action(j, ["ENGAGE", "SKIP"], "SKIP"),
                "scalar_prob": _coerce_prob(j, "p", 0.3),
                "note": str(j.get("if_then_plan", ""))[:200], "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        # Reuse M1 logic for DP2 — the if-then is set at DP1.
        return M1_ZeroShot().step_dp2(state, menu, trace)

    def step_dp3(self, state, chosen, trace, **kw):
        return M1_ZeroShot().step_dp3(state, chosen, trace)


# ============================ S1: Reflexion-in-funnel ============================

class S1_Reflexion(Method):
    name = "S1"
    description = "After each DP, agent emits a 1-sentence self-critique that is appended to funnel history."

    def __init__(self):
        self._base = M1_ZeroShot()

    def _critique(self, state, trace, last_action: dict) -> str:
        p = (
            render_state_for_prompt(state, trace)
            + f"\n\nLAST ACTION: {last_action.get('action')} (item={last_action.get('item')})\n"
            + "Critique that action in one sentence — was it consistent with the customer's pre-cutoff pattern? "
            + "Reply with one short sentence only, no JSON, no preamble."
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=60)
        return (r["text"] or "").strip()[:200]

    def step_dp1(self, state, menu, trace, **kw):
        r = self._base.step_dp1(state, menu, trace)
        r["note"] = self._critique(state, trace, r)
        return r

    def step_dp2(self, state, menu, trace, **kw):
        r = self._base.step_dp2(state, menu, trace)
        r["note"] = self._critique(state, trace, r)
        return r

    def step_dp3(self, state, chosen, trace, **kw):
        r = self._base.step_dp3(state, chosen, trace)
        r["note"] = self._critique(state, trace, r)
        return r


# ============================ S2: Outcome-conditioned planning ============================

class S2_OutcomeConditioned(Method):
    name = "S2"
    description = "Before the 30-day window opens, imagine the PURCHASE leaf and write the backward trajectory."

    def __init__(self):
        self._base = M1_ZeroShot()
        self._plans: dict[str, str] = {}  # cid -> plan text

    def setup(self, state, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + "\n\nBEFORE the 30-day window opens, imagine the END state: this customer makes a PURCHASE in week 2. "
            + "Write the backward trajectory in 3 short lines: "
            + "(line 1) what item they purchased and why; "
            + "(line 2) what menu candidate they considered and why; "
            + "(line 3) what triggered them to engage with the weekly stimulus.\n"
            + "Return JSON: {\"backward_plan\": \"<line1>\\n<line2>\\n<line3>\"}"
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=200)
        j = _safe_json(r["text"]) or {}
        self._plans[state.customer_id] = str(j.get("backward_plan", "")).strip()[:600]
        return 1

    def _plan_ctx(self, state):
        plan = self._plans.get(state.customer_id, "")
        if plan:
            return f"OUTCOME-CONDITIONED BACKWARD PLAN (set before the window):\n{plan}\n\n"
        return ""

    def step_dp1(self, state, menu, trace, **kw):
        p = (
            self._plan_ctx(state)
            + render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP1 (CONSISTENT with the backward plan above): ENGAGE or SKIP?\n"
            + 'Return JSON: {"action": "ENGAGE" | "SKIP", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["ENGAGE", "SKIP"], "SKIP"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        p = (
            self._plan_ctx(state)
            + render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP2 (CONSISTENT with the backward plan): EXIT or CONSIDER A/B/C?\n"
            + 'Return JSON: {"action": "EXIT" | "CONSIDER", "choice": "A"|"B"|"C", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=100)
        j = _safe_json(r["text"])
        a = _coerce_action(j, ["EXIT", "CONSIDER"], "EXIT")
        ch = str((j or {}).get("choice", "A")).upper().strip()[:1]
        idx = {"A": 0, "B": 1, "C": 2}.get(ch, 0)
        return {"action": a, "item": menu[idx].article_id if a == "CONSIDER" else None,
                "choice": ch, "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}

    def step_dp3(self, state, chosen, trace, **kw):
        p = (
            self._plan_ctx(state)
            + render_state_for_prompt(state, trace)
            + f"\n\nCONSIDERING ITEM: {chosen.product_type} / {chosen.garment_group} / {chosen.colour}\n\n"
            + 'DECISION POINT DP3 (CONSISTENT with plan): PURCHASE or ABANDON?\nReturn JSON: {"action": "PURCHASE" | "ABANDON", "p": <0..1>}\n' + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=80)
        j = _safe_json(r["text"])
        return {"action": _coerce_action(j, ["PURCHASE", "ABANDON"], "ABANDON"),
                "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}


# ============================ S3: Tree-of-thoughts over funnel branches ============================

class S3_ToT(Method):
    name = "S3"
    description = "At each DP, agent enumerates rollouts to the leaf, self-scores plausibility, picks highest."

    def step_dp1(self, state, menu, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP1 — TREE-OF-THOUGHTS ENUMERATION:\n"
            + "There are 2 rollouts from this DP: (1) SKIP→[end-of-week, no purchase], "
            + "(2) ENGAGE→[goes to DP2 menu→DP3 purchase].\n"
            + "Score each rollout 0–10 for plausibility for THIS customer given their profile, "
            + "then pick the highest-scored.\n"
            + 'Return JSON: {"rollouts": [{"action": "SKIP", "score": <0..10>}, {"action": "ENGAGE", "score": <0..10>}], "picked": "SKIP"|"ENGAGE", "p": <0..1>}\n'
            + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=200)
        j = _safe_json(r["text"]) or {}
        picked = _coerce_action({"action": j.get("picked", "")}, ["ENGAGE", "SKIP"], "SKIP")
        return {"action": picked, "scalar_prob": _coerce_prob(j, "p", 0.3),
                "note": json.dumps(j.get("rollouts", []))[:300], "raw": r["text"]}

    def step_dp2(self, state, menu, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + "\n\n" + render_menu_for_prompt(menu)
            + "\n\nDECISION POINT DP2 — TREE-OF-THOUGHTS ENUMERATION over the 6 rollouts:\n"
            + "EXIT; CONSIDER(A)→[ABANDON or PURCHASE]; CONSIDER(B)→[ABANDON or PURCHASE]; CONSIDER(C)→[ABANDON or PURCHASE].\n"
            + "Score each of {EXIT, A-PURCHASE, A-ABANDON, B-PURCHASE, B-ABANDON, C-PURCHASE, C-ABANDON} 0–10 then pick the highest.\n"
            + 'Return JSON: {"top_pick": {"choice": "A"|"B"|"C"|"EXIT", "next": "PURCHASE"|"ABANDON"|null, "score": <0..10>}, "p": <0..1>}\n'
            + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=200)
        j = _safe_json(r["text"]) or {}
        pick = j.get("top_pick", {})
        choice = str(pick.get("choice", "EXIT")).upper().strip()[:1]
        if pick.get("choice", "").upper().startswith("EXIT"):
            return {"action": "EXIT", "item": None, "choice": None,
                    "scalar_prob": _coerce_prob(j, "p", 0.3), "raw": r["text"]}
        idx = {"A": 0, "B": 1, "C": 2}.get(choice, 0)
        return {"action": "CONSIDER", "item": menu[idx].article_id, "choice": choice,
                "scalar_prob": _coerce_prob(j, "p", 0.3),
                "note": f"tot_pick={pick}", "raw": r["text"]}

    def step_dp3(self, state, chosen, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + f"\n\nCONSIDERING ITEM: {chosen.product_type} / {chosen.garment_group} / {chosen.colour}\n\n"
            + "DECISION POINT DP3 — TREE-OF-THOUGHTS: 2 rollouts (PURCHASE→[done], ABANDON→[done]). "
            + "Score each 0–10 then pick.\n"
            + 'Return JSON: {"rollouts": [{"action": "PURCHASE", "score": <>}, {"action": "ABANDON", "score": <>}], "picked": "PURCHASE"|"ABANDON", "p": <0..1>}\n'
            + JSON_SCHEMA_HINT
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=160)
        j = _safe_json(r["text"]) or {}
        return {"action": _coerce_action({"action": j.get("picked", "")}, ["PURCHASE", "ABANDON"], "ABANDON"),
                "scalar_prob": _coerce_prob(j, "p", 0.3),
                "note": json.dumps(j.get("rollouts", []))[:300], "raw": r["text"]}


# ============================ S4: Commitment device ============================

class S4_Commitment(Method):
    name = "S4"
    description = "Agent declares max purchases (0/1/2/3) at window start; DP3 hard-constrained to that declaration."

    def __init__(self):
        self._base = M1_ZeroShot()

    def setup(self, state, trace, **kw):
        p = (
            render_state_for_prompt(state, trace)
            + "\n\nBEFORE the 30-day window opens, COMMIT to a maximum number of purchases for this window. "
            + "Options: 0, 1, 2, or 3. This commitment will be enforced — once you hit the cap, "
            + "no further PURCHASE actions will be allowed.\n"
            + 'Return JSON: {"max_purchases": 0|1|2|3, "reasoning": "<one short sentence>"}'
        )
        r = call_llm(MODEL_DEFAULT, p, max_tokens=120)
        j = _safe_json(r["text"]) or {}
        try:
            mp = int(j.get("max_purchases", 0))
        except Exception:
            mp = 0
        state.declared_max_purchases = max(0, min(3, mp))
        return 1

    def step_dp1(self, state, menu, trace, **kw):
        return self._base.step_dp1(state, menu, trace)

    def step_dp2(self, state, menu, trace, **kw):
        return self._base.step_dp2(state, menu, trace)

    def step_dp3(self, state, chosen, trace, **kw):
        # If commitment is already saturated, force ABANDON without an LLM call.
        if state.declared_max_purchases is not None and len(state.weekly_purchases) >= state.declared_max_purchases:
            return {"action": "ABANDON", "scalar_prob": 0.0, "raw": "[commitment-cap-saturated]"}
        return self._base.step_dp3(state, chosen, trace)


# ============================ M8a: RAG WITHOUT outcome labels ============================

class M8a_RAG_NoLabel(M8_RAG):
    """Identical to M8 except retrieved cases' outcome labels are redacted.

    Reviewer red flag #1: M8's outcome-label visibility is essentially label-aware
    retrieval. M8a is the matched ablation that retrieves the same neighbours but
    hides their realized outcomes; any gap between M8 and M8a is attributable to
    the label visibility itself, not to the retrieval mechanism.
    """
    name = "M8a"
    description = "M8 with the retrieved cases' 30-day outcomes redacted."

    def _block(self, trace):
        items = self._retrieve(trace)
        if not items:
            return ""
        out = ["RETRIEVED 5 SIMILAR-PROFILE CUSTOMERS (outcomes redacted for this ablation):"]
        for i, e in enumerate(items):
            out.append(
                f"  CASE{i+1}: orders={e['total_orders']}, recency={e['recency_days']}d, "
                f"top_section={e['top_section']}, 30d_outcome=REDACTED"
            )
        return "\n".join(out) + "\n"


# ============================ M2: random few-shot ICL (un-cut for reviewer transparency) ============================

class M2_RandomICL(M3_KNN):
    """5 RANDOMLY-selected val customers prepended as ICL (matched to M3 in structure).

    Reviewer red flag #5: cutting M2 looked like motivated reasoning. Run it. The
    audit verdict — that random ICL is dominated by k-NN ICL in tabular settings
    (Liu 2022) — is reported as an empirical comparison rather than a cut.
    """
    name = "M2"
    description = "5 random val customers as ICL (M3 with random retrieval instead of RFM k-NN)."

    def __init__(self, neighbours_random: dict | None = None):
        # neighbours_random: dict[cid] -> 5 random val customers (NOT k-NN)
        super().__init__(neighbours=neighbours_random or {})


# ============================ M7: hybrid LLM + LightGBM (honest baseline) ============================

class M7_Hybrid(Method):
    """0.5·LLM_scalar + 0.5·LGBM_pred. Scalar-only method.

    Reviewer red flag #3: cutting M7 as 'trivially wins' looked like motivated
    removal. Reinstated as honest baseline. We expect it to win on scalar gap
    via mean-shrinkage toward LGBM's predictions; whether it improves
    within-bucket ρ over M1 is the test that matters.

    Implementation: reuse the same DP-level pipeline as M1 for sandbox actions
    (so f̂_LLM is the M1 sandbox outcome), and produce a HYBRID scalar by
    blending the M1 scalar with a per-customer LightGBM prediction.
    """
    name = "M7"
    description = "Hybrid: M1 sandbox actions + 0.5*LLM_scalar + 0.5*LGBM_pred on scalar arm."

    def __init__(self, lgbm_preds: dict | None = None):
        self._base = M1_ZeroShot()
        self._lgbm = lgbm_preds or {}

    def setup(self, state, trace, **kw):
        return 0

    def step_dp1(self, state, menu, trace, **kw):
        return self._base.step_dp1(state, menu, trace)

    def step_dp2(self, state, menu, trace, **kw):
        return self._base.step_dp2(state, menu, trace)

    def step_dp3(self, state, chosen, trace, **kw):
        return self._base.step_dp3(state, chosen, trace)

    def scalar_prob(self, trace: dict, **kw) -> dict:
        # M1's scalar
        base = super().scalar_prob(trace, **kw)
        cid = trace.get("_cid", "")
        lgbm = float(self._lgbm.get(cid, 0.21))  # fallback to test rate
        blended = 0.5 * base.get("scalar_prob", 0.5) + 0.5 * lgbm
        return {"scalar_prob": float(max(0.0, min(1.0, blended))),
                "llm_scalar": base.get("scalar_prob"),
                "lgbm_scalar": lgbm,
                "reasoning": "M7 hybrid 0.5·LLM + 0.5·LGBM"}


# ============================ registry ============================

METHOD_REGISTRY = {
    "M1": M1_ZeroShot,
    "M2": M2_RandomICL,
    "M3": M3_KNN,
    "M7": M7_Hybrid,
    "M8": M8_RAG,
    "M8a": M8a_RAG_NoLabel,
    "M9": M9_ImplementationIntentions,
    "S1": S1_Reflexion,
    "S2": S2_OutcomeConditioned,
    "S3": S3_ToT,
    "S4": S4_Commitment,
}
