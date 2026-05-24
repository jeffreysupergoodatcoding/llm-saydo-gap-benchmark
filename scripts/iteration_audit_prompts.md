# Iteration audit prompts (for the 4-round improvement loop after report_v2.md)

## Iteration 1 — Methodology rigor + statistical claims

**Goal**: Catch any flawed claim, missing control, or under-powered statistic in the draft `report_v2.md`. Identify the 5 highest-impact improvements.

**Agent prompt template**:
> You are a tough methods-stat reviewer for the paper at `/Users/jeffreysu/Desktop/personal projects/research proj/study/report_v2.md`. Read it carefully, plus `decisions_log.md`, `preregistration_v2.md`, and every `results/phase1*.json` and `results/phase4b_D2.json` you can find. Your job: identify the 5 most impactful methodological or statistical fixes I should make right now to strengthen the paper. For each: (a) the issue with file:line, (b) the proposed fix, (c) how much rigor it adds (low/med/high). Cite numbers from the JSON files. Be concrete — propose specific scripts to write or stats to compute.

## Iteration 2 — Novelty positioning + literature coverage

**Goal**: Sharpen the contribution. Catch any prior art that weakens claims; identify what makes our results genuinely interesting.

**Agent prompt template**:
> You are a senior reviewer for a top RecSys/CIKM/KDD program committee. Read the paper at `/Users/jeffreysu/Desktop/personal projects/research proj/study/report_v2.md` plus `references.bib`. The paper claims novelty on: (a) first public-benchmark H&M say-do gap quantification, (b) controlled base-rate-leakage decomposition, (c) counterfactual trace perturbation control. Search arxiv/web (max 5 queries) for 2025-2026 papers that overlap. Identify: (1) which novelty claims are still defensible; (2) where the paper should be reframed to sharpen the contribution; (3) what *additional* finding from the data (already collected) could be the strongest sentence in the abstract. Be concrete — quote sentences from the report and propose rewrites.

## Iteration 3 — Writing quality + figure rigor

**Goal**: Tight prose; readable figures; abstract that hooks the reader.

**Agent prompt template**:
> You are an editor for a top conference. Read `/Users/jeffreysu/Desktop/personal projects/research proj/study/report_v2.md`. List 8-10 specific writing issues: (a) abstract sentences that buried the lead; (b) figures that need labels/legends; (c) paragraphs that bury a result; (d) statistical claims phrased as causal when they're correlational. For each: quote the offending sentence, propose a sharper rewrite. Also: rank-order the report's contributions and tell me which one would make the best first sentence of the abstract.

## Iteration 4 — Final blind peer review

**Goal**: Brutal "would you accept this paper?" final check.

**Agent prompt template**:
> You are an independent blind reviewer with no prior context. Read `/Users/jeffreysu/Desktop/personal projects/research proj/study/report_v2.md`. Verify: (1) every numeric claim against `results/*.json` (sample 10 claims); (2) every citation against `references.bib`; (3) pre-registration adherence vs `preregistration_v2.md`; (4) honest framing of negative results; (5) statistical correctness. Verdict: workshop accept / borderline / reject. Top 3 blockers to fix. Be ruthless and cite file:line.

---

## Iteration execution rules

1. Always run each iteration's audit BEFORE making changes.
2. After audit returns findings, classify into:
   - **Apply immediately** (correctness fixes, citation errors, missing analyses I can run)
   - **Defer** (would require new LLM calls beyond cost cap, or new datasets)
   - **Accept as-is** (acknowledged limitations)
3. Implement all "apply immediately" fixes.
4. Re-render `report_v2.md` via `scripts/phase12_report.py`.
5. Commit with a clear iteration tag.
6. Move to next iteration only when previous one is fully applied.

## Stopping rule

Stop iterating when:
- Two consecutive audits return only "Defer" or "Accept" items (no new "Apply immediately"), OR
- Iteration 4's blind reviewer says "workshop accept" or "borderline accept", OR
- We have completed all 4 planned iterations.
