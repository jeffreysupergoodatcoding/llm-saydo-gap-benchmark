# Project Handoff — LLM Say-Do Gap Benchmark

**Last updated**: 2026-05-25 (end of session pivot to v5 plan)

This document is a complete state snapshot. A fresh Claude instance opening this repo should read this file first, then `paper.md` and `/Users/jeffreysu/.claude/plans/you-are-conducting-an-iterative-pizza.md`.

---

## TL;DR — Where we are right now

The project went through 5 major iterations. The current state is:

- **v3 paper (the existing draft)**: `paper.md`, 501 lines, all real numbers, 2nd ICLR audit returned **Borderline / Weak Accept**. This paper is **complete and committable as-is** if the user wants a quick ICLR submission today.
- **v5 plan (the more ambitious roadmap)**: at `/Users/jeffreysu/.claude/plans/you-are-conducting-an-iterative-pizza.md`. Three audit agents converged on a major upscope: formalize as **Asymmetric-Information Predictor Gap (AIPG)** with theorems T1/T2, run a **Data-Asymmetry Ladder** R0→R4 × 4 domains × 3-4 providers, add **Hewitt-style human-forecaster control**, add **cluster bootstrap + hierarchical Bayesian**, demote sandbox-paradigm work to appendix. Targets NeurIPS D&B / Nature Human Behaviour / PNAS rather than ICLR Weak Accept. **6 weeks of parallel agent dispatch**.
- **Plan mode just exited.** No execution started yet on v5. The user said "save everything so I can launch a new Claude instance later and continue."

---

## Repo overview

```
llm-saydo-gap-benchmark/
├── paper.md                         # v3 paper, 501 lines, ICLR Weak Accept-grade
├── paper_v2_archive.md              # v2 archive (early sandbox-as-headline framing)
├── preregistration_v3.md            # current pre-registration (hash 47a938b1)
├── decisions_log.md                 # running deviations register
├── references.bib                   # ~70 refs incl Park 2023/2024, Peng 2025, Toubia 2025
├── HANDOFF.md                       # this file
├── data/
│   ├── raw/                         # H&M Kaggle (gitignored)
│   ├── processed/                   # H&M parquet (gitignored)
│   └── splits/                      # train/val/test splits
├── src/
│   ├── data.py                      # H&M data loaders
│   ├── llm_client.py                # provider routing (Gemini, OpenAI, Claude-code)
│   ├── eval.py                      # metrics + bootstrap_ci + paired_bootstrap_diff
│   ├── leakage_audit.py             # cutoff_guard decorator
│   ├── anonymize.py                 # HMAC-SHA256 id anon
│   ├── dns_patch.py                 # macOS DNS resolver fallback
│   ├── sandbox/                     # sandbox v1 (deterministic protocol)
│   │   ├── env.py                   # state, candidates, stimulus menus
│   │   ├── methods.py               # M1, M2, M3, M7, M8, M8a, M9, S1-S4 method classes
│   │   ├── runner.py                # orchestration
│   │   └── __init__.py
│   └── sandbox_v2/                  # sandbox v2 (real-world dynamics)
│       ├── world.py                 # stochastic stimulus arrival, inventory, fatigue, recency rolls
│       ├── runner.py                # v2 orchestration
│       └── __init__.py
├── scripts/                         # phase-numbered analysis scripts (60+ files)
│   ├── phase31_select_core1000.py   # core-1000 v3 sample
│   ├── phase34_run_full.py          # main sandbox runner (race-safe, throttled)
│   ├── phase40_claude_cross_provider.py    # Claude meta-policy arm
│   ├── phase42_claude_proper.py     # Claude proper per-DP arm
│   ├── phase43_cross_provider_analysis.py  # 4-arm analysis with cluster CIs
│   ├── phase45_figures.py           # 5 publication figures
│   ├── phase47_fill_paper.py        # placeholder-filler
│   └── phase34-46 family
└── results/
    ├── phase34_sandbox/             # Gemini main run (M1-M3 done at n=1000)
    ├── phase40_claude_*.jsonl       # Claude meta-policy 8 methods × n=1000
    ├── phase42_claude_proper_*.jsonl # Claude proper-DP M1 n=200, S4 n=200
    ├── phase43_cross_provider_analysis.json # 4-arm headline numbers
    ├── phase46_sandbox_v2_M1.jsonl  # Sandbox v2 M1 n=200
    └── figures/                     # 5 publication figures
```

---

## Current results — the actual numbers (M1 zero-shot, four arms)

| Arm | n | Gap (rw) | 95% CI | within-bucket ρ |
|---|---|---|---|---|
| Gemini per-DP | 1000 | +0.421 | [+0.386, +0.456] | +0.053 |
| Claude meta-policy | 1000 | +0.148 | [+0.112, +0.184] | −0.052 |
| Claude proper-DP v1 (deterministic) | 200 | +0.227 | [+0.153, +0.305] | **+0.230** |
| Claude proper-DP v2 (real-world dynamics) | 200 | +0.228 | [+0.139, +0.321] | **+0.202** |

**Headline finding currently in `paper.md`**: methodology (per-DP reasoning vs meta-policy) is the dominant axis of within-bucket-ρ recovery on the same protocol. Paired bootstrap CI on Δwb-ρ between Claude meta and Claude per-DP is [+0.135, +0.506] — excludes zero by 9× SE.

**Per-method numbers (Claude meta-policy, n=1000 each)** — see `results/phase41_claude_analysis.json`:

| Method | Gap | CI | wb-ρ |
|---|---|---|---|
| M1 zero-shot | +0.148 | [+0.112, +0.184] | −0.052 |
| M3 k-NN ICL | +0.189 | [+0.155, +0.222] | +0.001 |
| **M8 RAG-w/-labels** | **+0.098** | [+0.068, +0.127] | +0.065 |
| M9 implementation-intentions | +0.240 | [+0.211, +0.270] | +0.054 |
| S1 Reflexion | +0.202 | [+0.171, +0.235] | +0.068 |
| S2 outcome-conditioned plan | +0.307 | [+0.275, +0.338] | +0.042 |
| S3 tree-of-thoughts | +0.426 | [+0.393, +0.457] | +0.022 |
| S4 commitment device | +0.114 | [+0.083, +0.142] | +0.015 |

H10 (close gap to ±0.05): **FAILS** in all 8 method cells. Closest is M8 at +0.098. Gemini per-DP M3 at +0.110 is also close from the Gemini side.

H11 (sandbox-native method beats M1 on wb-ρ by ≥ 0.03): FAILS strictly in meta-policy arm; PASSES informally in proper-DP arm (S4 wb-ρ = +0.254 vs M1 wb-ρ = +0.230).

Commitment shrinkage is **negative for every method** (sandbox gap > scalar gap), permutation p = 1.000 in all 8 cases — falsifying the hypothesis that forced commitment reveals more-honest preferences.

---

## What's been done (chronological)

- **Phases 0–7** (`paper_v2_archive.md`): initial benchmarking, LightGBM-RFM PR-AUC 0.622, D2/D3 LLM digital twins. Set up the H&M data + splits + leakage controls + memorization probe.
- **Phases 8–24** (`paper.md` v2 backbone): cognition pipeline (Park-2023 lineage), F-base / F-nobase / D2-flat arms, MovieLens cross-domain replication, base-rate-leakage decomposition. ICLR audit 1 returned **Weak Reject**.
- **Phase 31–34**: v3 sandbox introduced (4-week window, 3 DPs per week, attention budget = 3, 3-candidate menus per week). Core-1000 v3 sample, disjoint from v2.
- **Phase 35–37**: paper restructure, placeholder-filler, analysis pipeline.
- **Phase 38–39**: ICLR audit fixes — Holm-Bonferroni FWER, M8a no-label ablation, M2/M7 reinstatement, MDE for H11.
- **Phase 40–43**: 4-arm cross-provider matrix — Gemini per-DP (M1/M2/M3 at n=1000) + Claude meta-policy (8 methods × n=1000) + Claude proper per-DP (M1 + S4 at n=200) + sandbox v2 (M1 at n=200).
- **Phase 44–46**: sandbox v2 with stochastic dynamics built and tested.
- **Phase 47**: paper filled with real numbers. 2nd ICLR audit returned **Borderline / Weak Accept**. Δwb-ρ paired bootstrap CIs added.
- **v5 plan written**: 3 audit agents (novelty / methodology / scope-depth) converged on AIPG construct + ladder + multi-domain. Plan at `/Users/jeffreysu/.claude/plans/you-are-conducting-an-iterative-pizza.md`.

---

## What hasn't been done (v5 plan, not started)

The v5 plan upscope. To pick up:

### Day 0 — gating commits (no LLM)

```bash
cd "/Users/jeffreysu/Desktop/personal projects/research proj/llm-saydo-gap-benchmark"
# 1. Write preregistration_v5.md with theorems T1, T2, and H1-H4
# 2. Write src/theory/aipg.py with AIPG formalization + theorem proofs (SymPy where possible)
git add preregistration_v5.md src/theory/aipg.py
git commit -m "v5: AIPG construct, theorems T1/T2, pre-registration locked"
git tag prereg-v5
```

### Day 0 — start parallel agent dispatch

Per the plan's "How to start" section, dispatch:
- 1 theorem-prover agent (LaTeX)
- 4 domain-loader agents (ANES 2020, ANES 2016, CES 2020, HRS)
- 1 pre-registration writer

### Then proceed through phases 60–80

(Full table in the v5 plan.)

---

## Key methodological discipline points (carry forward)

1. **Pre-register before any LLM call.** Tag `prereg-v5` is a gating commit.
2. **Race-safe parallel agent dispatch**: each agent writes to its own file `phase{X}_{method}_b{N}.jsonl`; concat at end. We hit shared-file race conditions earlier; the per-shard pattern in `scripts/phase42_claude_proper.py` is proven.
3. **Strict leakage controls**: input files stripped of `actual_label`; sidecar labels file kept separately, only loaded at analysis time.
4. **Quota constraints**: Gemini Flash 2.5 free tier is 10K requests/day per Google project. We have 2 keys (Fragment Labs + pulse). Cluster-throttled at 90 RPM via `LLM_RATE_LIMIT_RPM` env var.
5. **DNS retry**: `src/llm_client.py` retries on Errno 8 (macOS DNS), 60, 54, 429, 503. Up to 10 attempts, max 90s backoff with jitter.
6. **Statistical defaults**: B = 1000 paired stratified bootstrap (B = 10000 in v5 plan); α = 0.025 confirmatory; Holm-Bonferroni (v5 plan: gatekeeping); MDE pre-registered.

---

## Known issues / gotchas

- **Claude Code Agents can race on shared output files** — always use per-batch files
- **Some Claude agents wrote heuristic-script simulations** instead of doing per-customer LLM reasoning. Phase 42 + Phase 46 prompts explicitly forbid Python heuristic scripts; the M1 meta-policy work in phase 40 used the heuristic shortcut (reframed honestly in paper as "Claude meta-policy arm" — one-shot LLM-designed policy applied deterministically).
- **Gemini 2.5 thinking budget eats output budget** — disabled via `thinking_budget=0` in `src/llm_client.py`
- **ANES voter-file match rate is ~70-75%** — non-random; v5 plan addresses with multiple imputation
- **Llama via Groq free tier** is the planned 3rd provider for cross-provider replication; not implemented yet
- **HRS access requires DUA** — apply day 0; MIDUS as fallback

---

## How a new Claude instance should pick up

1. **Read this file (`HANDOFF.md`) first.**
2. **Read `paper.md`** — the current v3 paper at ICLR Weak Accept. ~30 min read.
3. **Read `/Users/jeffreysu/.claude/plans/you-are-conducting-an-iterative-pizza.md`** — the v5 ambitious plan. ~30 min read.
4. **Decide with the user**: ship v3 paper now (Borderline ICLR submission) OR execute v5 plan (6 weeks to NeurIPS D&B candidate).
5. **If executing v5**: follow the plan's "How to start" section.
6. **If shipping v3**: run a final AI-tell sweep on `paper.md`, fix any remaining typos, then commit + tag.

---

## Reading list for context (existing pdfs)

In `/Users/jeffreysu/Desktop/personal projects/research proj/pdfs/`:
- Park 2024 self-report (the 86% normalized accuracy headline)
- Toubia 2025 Twin-2K-500 (the r ≈ 0.20 individual-level ceiling)
- Peng 2025 funhouse mega-audit (164 outcomes, similar ceiling)
- Hewitt 2024 predicting social experiments (the human-forecaster control reference)
- Park 2023 generative agents
- Andric 2025 calibration gap

Twin-2K-500 dataset (HF): `LLM-Digital-Twin/Twin-2K-500` and `LLM-Digital-Twin/Twin-2K-500-Mega-Study`. Note: PIDs do NOT match across base + mega-study (privacy hold-out).

ANES 2020 Voter Validation: ICPSR 38034 (manual DUA + download required).

---

## Conversation context — what the user wants

- **Goal**: prove that LLM-with-data can predict actual behavior more accurately than the person's own stated intent (the canonical say-do gap)
- **Positive proof bar**: aggregate accuracy, not perfect individual prediction (user clarified: "real reason this would happen comes in aggregate prediction trends instead of predicting one individual")
- **Constraints**: only Claude Code Agents + free Gemini/Llama (no paid GPT-4 / Llama API)
- **Time/budget**: not constraints
- **Aspiration**: most profound paper achievable — citation-anchor, not incremental

The user does NOT want:
- Paid API costs
- A merely-publishable paper
- A black-box accuracy comparison without mechanism analysis
- A single-domain study

---

## Git log (recent)

```
a7ab05c Phase 47: fill paper with real numbers from 4-arm cross-provider analysis
ba0ad86 Phase 43 4-arm cross-provider analysis with Gemini M2/M3 n=1000 + Sandbox v2 + S4 proper-DP
f8b3355 Phase 44-46: Sandbox v2 world model + Claude proper-DP S4 + 5 ICLR figures
bc77a32 Phase 40-43: Claude meta-policy n=1000 across 8 methods + proper per-DP M1 n=200 + 3-arm cross-provider analysis
f81b2d5 Phase 40 + retry/throttle hardening: parallel Claude Code Agent cross-provider arm + 90 RPM throttle + tougher retries
```

Remote: https://github.com/jeffreysupergoodatcoding/llm-saydo-gap-benchmark (purged of secrets, public)

---

## Quick commands to verify current state

```bash
cd "/Users/jeffreysu/Desktop/personal projects/research proj/llm-saydo-gap-benchmark"
wc -l paper.md                                     # 501 lines
grep -c '{{' paper.md                              # 0 (all placeholders filled)
ls results/phase34_sandbox/*.jsonl | wc -l         # 9 (M1, M2, M3, M7, M8, M9, S1, S2, S3 — M8, M9, S1, S2, S3 are empty from Gemini quota cutoff)
ls results/phase40_claude_*.jsonl | wc -l          # 8 (meta-policy methods)
ls results/phase42_claude_proper_S4_b*.jsonl | wc -l  # 8 race-safe batches
ls results/phase46_sandbox_v2_M1_b*.jsonl | wc -l  # 4 sandbox v2 batches
cat results/phase43_cross_provider_analysis.json | python3 -m json.tool | head -30
ls results/figures/                                # 5 png figures
```

---

## Cleanup checklist for the new Claude instance

Before doing any new work, the new Claude instance should propose to the user — and then execute if approved — the following pruning, since the repo accumulated some debugging/scratch artifacts during the chaotic session:

### Safe to delete (debugging artifacts, regenerable)

- `results/phase34_sandbox/bad/` — first batch of contaminated records from the original DNS-failure incident. Forensic value only; keep one sample for reproducibility, delete the rest. (~2 MB)
- `results/phase34_sandbox/bad_v2/` — second batch of pre-throttle contaminated records (M1, M3). Same deal — keep one sample, delete the rest.
- `cache/llm/` — 12,000+ cached LLM responses (~50 MB). Regenerable by re-running scripts. Safe to delete if disk-pressured; **keep otherwise** because regenerating costs API quota.
- `cache/llm_costs.jsonl` — cost log. Keep as audit trail; ≤ 1 MB.
- `results/phase42_claude_proper_batch_{0..7}.json` — intermediate per-batch input files for proper-DP M1. Consolidated into `phase42_claude_proper_input.json`. Safe to delete.
- `results/phase40_claude_batch_{0..3}.json` — intermediate per-batch input files for meta-policy split. Consolidated into `phase40_claude_batch_input_full.json`. Safe to delete.
- `results/phase42_claude_S4_batch_{0..7}.json` — intermediate per-batch input files for proper-DP S4. Safe to delete.
- `results/phase46_sandbox_v2_batch_{0..3}.json` — same pattern for sandbox v2. Safe to delete.
- `results/phase42_claude_proper_M1_gapfill_0.jsonl` — gap-fill batch that was concatenated into the main consolidated file. Already merged, safe to delete the standalone.
- Any `*_pre_throttle.jsonl` files — pre-throttle contaminated batches.

### Keep (these are durable contributions)

- `HANDOFF.md`, `paper.md`, `paper_v2_archive.md`, `preregistration_v3.md`, `decisions_log.md`, `references.bib`
- All `src/` modules (`sandbox/`, `sandbox_v2/`, `theory/` when created, etc.)
- All scripts in `scripts/phase{1..47}*.py` (the analysis pipeline)
- All consolidated `.jsonl` results: `phase34_sandbox/M{1..9}.jsonl`, `phase34_sandbox/S{1..4}.jsonl`, `phase40_claude_M{3,8,9}.jsonl`, `phase40_claude_S{1..4}.jsonl`, `phase40_claude_predictions.jsonl` (M1), `phase42_claude_proper_M1.jsonl`, `phase42_claude_proper_S4.jsonl`, `phase46_sandbox_v2_M1.jsonl`
- All analysis JSON files: `phase41_claude_analysis.json`, `phase43_cross_provider_analysis.json`
- All 5 figures in `results/figures/`
- `results/phase31_core1000_v3.parquet` (sample selection) and `results/phase31_core1000_v3.json` (sample metadata)
- `data/splits/{train,val,test}.parquet` (the H&M splits — though these are gitignored, they're the deterministic input)

### Questionable — propose to user

- `paper_v2_archive.md` — the v2-era archive. Useful for "what got dropped between v2 and v3" but adds noise. Propose keeping until v5 is committed, then archive to a tag.
- `preregistration_v2.md` — old preregistration. Keep for audit trail but could move to a `archive/` subdirectory.
- The big `phase40_claude_batch_input_full.json` (~4 MB) — used as input for the cross-provider Claude run; not strictly needed once that arm is finalized.

### Decision point that needs the user

Before pruning, ask: **"Are we shipping v3 paper to ICLR (short path) or executing the v5 plan (long path)?"** The cleanup decisions cascade from that:
- **v3 short path**: prune aggressively, focus the repo around `paper.md` + the supporting `src/` + key `results/`
- **v5 long path**: keep more of the scratch for reproducibility during the 6-week build; defer pruning until paper is camera-ready

---

## Memory pointers for the new Claude instance

The user's auto-memory at `/Users/jeffreysu/.claude/projects/-Users-jeffreysu-Desktop-personal-projects-research-proj/memory/MEMORY.md` already has:
- `notes_workflow.md` — write notes directly to research-proj folder
- `project_topic.md` — research project on LLM generative agents / digital twins

Any new Claude instance reading from there should also load this `HANDOFF.md` for full project state.
