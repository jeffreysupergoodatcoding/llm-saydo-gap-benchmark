#!/usr/bin/env bash
# Run this once D2, D3, and the chain finish to do a final peer-review audit.
# It's separate from the chain so a human (or another agent invocation) can launch it.
echo "Run the following Agent invocation manually:"
cat <<'EOF'
Subagent type: general-purpose

Prompt: You are a final-round peer reviewer for the paper in /Users/jeffreysu/Desktop/personal projects/research proj/study/report.md. Read it carefully, then read:
  - study/preregistration.md
  - study/decisions_log.md
  - study/references.bib
  - study/results/*.json (especially phase4a_metrics, phase4b_D2, phase4b_D3 if exists, phase4c_regime, phase5_metrics, phase6_levers, phase_score_correlation, phase4a_seed_variance)

Conduct these checks and report under 700 words:

1. **Numerical claim verification**: For each numeric claim in report.md (PR-AUC, ROC-AUC, Brier, Wasserstein, under-dispersion ratios, p-values), find the JSON file it sources from. List any number in the report that does NOT match its JSON, with file:line.

2. **Citation completeness**: For each statement that draws from prior work, check that a [citekey] is present. List statements lacking citations.

3. **Pre-registration adherence**: For each of H1-H5, check that the verdict in §5 is consistent with the actual data. Flag mismatches.

4. **Plagiarism check (light)**: For 5-10 distinctive phrases from the paper's abstract and intro, do a Google/Bing search (web_search) to check no exact match exists in prior work. Report any suspicious matches.

5. **Story coherence**: Does the abstract's claim sequence (regime / scaling / under-dispersion / Pareto) actually pay off in §4? Are the contributions defensible against a tough reviewer?

6. **Overall verdict**: Workshop accept? Conference borderline? Reject? Top 3 blockers.

Be ruthless. Cite file:line for every issue.
EOF
