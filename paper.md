# Closing the LLM Say-Do Gap: A Consequential-Decision Sandbox and a Method Catalog

**Anonymous submission.**

---

## Abstract

Large language models prompted to simulate specific people produce stated intentions that diverge from those people's later behavior, an instance of the social-science *say-do gap*. Prior benchmarking on H&M Personalized Fashion (recapitulated in §3) documented this gap on retail purchase prediction, decomposed it into in-prompt base-rate-table leakage and Park-2023-lineage cognition-architecture components, and established that the per-individual signal a third-party LLM extracts from a behavioral trace is roughly half of a within-domain human-self test-retest correlation. This paper asks the natural next question: can published or theoretically motivated *interventions* close that gap, and, if so, by what mechanism? We construct a consequential-decision sandbox on H&M (a 30-day window with weekly stimulus menus drawn exogenously from H&M articles, a depletable attention budget, and persistent state across cycles) and evaluate eight methods inside it. Four are literature baselines: zero-shot per-decision-point elicitation, k-NN-retrieved few-shot in-context learning, retrieval-augmented generation conditioning on prior trajectories' realized outcomes, and a Gollwitzer-style implementation-intentions prompt. Four are sandbox-native methods that the prior scalar-elicitation setup cannot represent at all: Reflexion-in-funnel self-critique, outcome-conditioned backward planning, tree-of-thoughts over funnel branches, and a structurally enforced commitment device. The pre-registered headline test (H10) asks whether any of the eight methods achieves a sandbox-realized purchase rate within ±0.05 of the actual rate on a paired core-1000 sample with paired stratified bootstrap CI excluding the envelope. The second confirmatory test (H11) asks whether any sandbox-native method achieves a strictly higher bucket-weighted within-bucket Spearman correlation than the zero-shot baseline by at least 0.03. The descriptive metric we introduce, *commitment shrinkage*, is the within-method difference between scalar say-do gap and sandbox-realized say-do gap, tested against a per-bucket permutation null. {{ABSTRACT_RESULTS_PARAGRAPH}}

---

## 1. Introduction

Using a large language model to simulate a specific person — a "digital twin" — is now load-bearing in marketing-experiment forecasting, recommender explainability, and synthetic-participant pipelines for product evaluation [park2024selfreport, peng2025funhouse, toubia2025twin2k500, li2025digitaltwins, lu2025multiturnbehavior, wang2026productdiscovery]. The headline benchmark of Park et al. (2024) — 86% normalized accuracy on held-out General Social Survey items, computed from 1,052 interview-grounded agents — sits next to a substantial body of more sober empirical audits that find per-individual correlations closer to r ≈ 0.2 across hundreds of outcomes [peng2025funhouse, toubia2025twin2k500]. The verdict in the field varies by domain, by evaluation grain, and by the methodological care taken to separate population-level fit from individual-level reasoning [andric2025walktheirtalk, alignmentrevisited2025, mindthegap2026]. The construct that organizes most of this work is older than language models: stated intent against revealed behavior, the *say-do gap* [lapiere1934attitudes, fishbein1975belief, ajzen1991tpb, sheeran2002intention, sheeran2016intention, manski1990intentions, manski2004expectations, train2009discrete, benakiva1994combining].

Earlier benchmarking by the same authors (recapitulated as background in §3) established three facts on a public retail benchmark, H&M Personalized Fashion Recommendations [hm_kaggle]. First, an in-prompt base-rate table that practitioners commonly include for calibration accounts for more of the apparent say-do-gap reduction than the underlying Park-2023-lineage cognition pipeline itself; on H&M, |Δ_F| = 0.077 versus |Δ_arch| = 0.062. Second, when stated-vs-revealed Spearman correlation is decomposed into pooled and within-stratum components, the LLM's pooled ρ reproduces Sheeran's human meta-analytic r ≈ 0.53 while the within-bucket ρ collapses to 0.23–0.28, close to the per-individual twin–human correlation of ≈ 0.2 reported by Toubia et al. (2025). Third, an in-domain human-self test-retest benchmark places a customer's own past-30-day buying as a predictor of their next-30-day buying at Pearson r ≈ 0.39 on H&M, and the LLM's within-bucket ρ is roughly half this domain-specific human-self number. The headline finding from that prior work is *that there is a gap and that most of it is bucket-prior dependence, not within-customer reasoning*. This paper asks: **can it be closed?**

The natural extension is to take published gap-reduction interventions from the broader literature — implementation intentions [gollwitzer1999implementation, verplanken1999goodintentions], retrieval-augmented in-context learning [brown2020fewshot, liu2022makesgood], self-consistency [wang2022selfconsistency], chain-of-thought [wei2022chain], post-hoc calibration [platt1999, niculescu2005predicting], hybrid statistical–LLM ensembles [chen2024fusion] — and run them as static prompting variations against the same benchmark. We initially designed exactly this; the pre-registration draft listed nine such methods. A tri-agent methodology audit (records in `decisions_log.md`, 2026-05-24) returned a single decisive critique: most of those nine methods are scalar-elicitation interventions. They modify the prompt but not the *task structure*. The harder, more honest question is whether closing the gap requires intervening at the *decision-making structure* itself.

We respond by building a sandbox: a 30-day behavioral window with weekly stimulus menus drawn exogenously from H&M's article catalog, a budget the agent must allocate across the four weeks, three sequential decision points each week (DP1 engage/skip, DP2 select-a-candidate or exit, DP3 purchase or abandon), and persistent state in which prior weeks' actions are visible to later weeks. The sandbox is *deterministic-transition* — the agent's action is the state transition — and we are explicit that this makes the sandbox a decision-elicitation protocol under commitment pressure rather than a full world model. The audit verdict in §11 accepts this framing and rejects the framing of the sandbox as a "world model" or "environment with consequences." But three properties the original prompt-only setup lacks are now present: (i) exogenous stimuli the agent did not invent, (ii) a depletable resource that makes early-week engagement consequential to late-week options, and (iii) action history that constrains and informs later decisions within a single window.

Inside this sandbox we evaluate eight methods, dropping five from the original draft and adding four sandbox-native ones the audit identified as required for the framing to hold. The four retained baselines are M1 zero-shot per-decision-point elicitation, M3 few-shot k-NN in-context learning seeded with five RFM-nearest customers' realized 30-day outcomes, M8 retrieval-augmented generation in which similar prior trajectories are retrieved per decision point with their outcomes, and M9 a Gollwitzer-style implementation-intentions prompt that forces the agent to declare an explicit if-then plan at DP1. The four sandbox-native methods are S1 Reflexion-in-funnel [shinn2023reflexion, madaan2023selfrefine] in which the agent emits a one-sentence self-critique after each DP that is appended to history and read at the next DP, S2 outcome-conditioned backward planning in which the agent imagines the PURCHASE leaf and writes the trajectory backwards before committing DP1, S3 tree-of-thoughts [yao2023treeofthoughts] in which the agent enumerates and self-scores funnel branches at each DP, and S4 a structural commitment device in which the agent declares a maximum number of purchases for the window that is *hard-enforced* by the sandbox (DP3 forced to ABANDON when the cap is reached).

The pre-registered claims (hash `47a938b1`, committed before any v3 LLM call) are stated as confirmatory tests rather than pre-committed conclusions:

**H10 — sandbox closure.** At least one of the eight methods achieves a sandbox signed gap within ±0.05 of the actual rate with paired bootstrap CI excluding the envelope. {{H10_RESULT_NARRATIVE}}

**H11 — within-bucket reasoning.** At least one sandbox-native method achieves a bucket-weighted within-bucket Spearman correlation that exceeds the zero-shot baseline by ≥ 0.03 with paired bootstrap CI excluding zero. {{H11_RESULT_NARRATIVE}}

**Commitment shrinkage and the permutation null.** For each method, commitment shrinkage is the within-method difference between scalar say-do gap and sandbox-realized say-do gap. A naive interpretation is that the sandbox forces the agent to commit, and committing should reveal a more honest preference than scalar self-report. We pre-register a per-bucket permutation null and report the result for each method in §6.3. {{COMMITMENT_SHRINKAGE_RESULT_NARRATIVE}}

The remainder of the paper is organized as follows. §2 situates the contribution in the social-science say-do-gap literature, the digital-twin literature, and the gap-closure-intervention literature from cognitive psychology, behavioral economics, and prompt engineering. §3 summarizes the prior benchmarking results that motivate the sandbox design. §4 specifies the sandbox environment in detail. §5 describes the eight methods, their published provenance, and the structural reasons each is in the catalog. §6 reports results on n = 1,000 paired customers. §7 discusses what these findings imply, addresses four counter-claims that a reviewer would raise, and articulates the open problem the negative results define. §8 lists limitations, including pre-registration deviations. §9 concludes. Appendix A documents the failed-method audit and the methods we deliberately dropped.

---

## 2. Related work

Five literatures matter here. The first is the social-science work on stated versus revealed intent. The second is the LLM-digital-twin literature. The third is the narrower setting of LLM purchase prediction from behavioral traces, which is where our benchmark lives. The fourth is the LLM stated-vs-revealed literature that has converged on the say-do gap framing in the last twelve months. The fifth — central to this paper but largely absent from the prior LLM literature — is the cognitive-psychology and prompt-engineering work on *interventions that close intention–behavior gaps*.

### 2.1 The intention–behavior gap in humans

LaPiere [lapiere1934attitudes] documented the first empirical attitude–behavior gap in 1934 by writing to hotels asking whether they would serve a Chinese couple — most said no — and then traveling with the couple and discovering nearly all did serve them. Fishbein and Ajzen [fishbein1975belief] formalized intentions as the immediate antecedent of behavior in the Theory of Reasoned Action; Ajzen [ajzen1991tpb] extended this to the Theory of Planned Behavior, adding perceived behavioral control. The TPB framework predicts that intent explains roughly 20–30% of variance in behavior across domains, an estimate that meta-analytic work has confirmed: Sheeran's [sheeran2002intention] meta-analysis of 422 studies reports a median intent–behavior r of 0.53, and Sheeran and Webb [sheeran2016intention] update this with a median Cohen's d ≈ 0.45 across action-control interventions.

In economics, the equivalent distinction maps onto stated versus revealed preference. Manski [manski1990intentions, manski2004expectations] argues that intentions are informative under explicit measurement-error assumptions, an argument that anchors much of the subsequent discrete-choice modeling [train2009discrete]. Ben-Akiva et al. [benakiva1994combining] show that combining stated and revealed preference data — with appropriate scaling — outperforms either source alone. Harrison and List's [harrison2004field] field-experiments review reaches a similar conclusion. The contingent-valuation debate over how to elicit willingness-to-pay for non-market goods [diamond1994contingent, arrow1993noaa] is the canonical historical example of a community settling on protocols for taking stated preferences seriously without conflating them with revealed behavior.

In commercial marketing, three papers set the empirical baseline. Juster [juster1966probability] introduced the eleven-point purchase-probability scale and showed that probabilistic intent outperforms binary intent in predicting six-month purchase. Morwitz and Fitzsimons [morwitz2004mere] document the *mere-measurement effect*: asking people whether they intend to buy something changes the rate at which they later do. Chandon, Morwitz, and Reinartz [chandon2005intentions] formalize the self-generated validity effect, where the elicitation of intent itself constructs a cognitive commitment that biases later behavior. The collective lesson is that intent surveys are informative, biased, and measurement-active; an LLM "intent" instrument inherits each of those properties.

### 2.2 LLM digital twins

The LLM-digital-twin genre has three waves. The 2023 wave demonstrated feasibility: Park et al. [park2023generative] introduced the memory–retrieval–reflection–planning architecture and a town-of-agents demonstration; Argyle et al. [argyle2023oneMany] showed that conditioning GPT-3 on demographic backgrounds reproduces survey-distribution patterns; Aher et al. [aher2023turing] replicated canonical psychology experiments using LLM agents; Horton [horton2023homosilicus] proposed *homo silicus*. Dillion et al. [dillion2023replace] asked the framing question — can AI language models replace human participants — and concluded a qualified yes. Santurkar et al. [santurkar2023whose] documented the conditional: LLMs reflect specific demographic groups' opinions more than others.

The 2024 wave deepened the methodology. Park et al. [park2024selfreport] built grounded agents for 1,052 specific Americans using two-hour structured interviews; Mei et al. [mei2024turing] ran a population-level Turing test; Hewitt et al. [hewitt2024predicting] showed LLM agents can predict pre-registered social-experiment treatment effects; Tjuatja et al. [tjuatja2024responsebias] documented that LLMs do not exhibit human-like response biases, raising a construct-validity question about treating an LLM intent as a human-intent analog.

The 2025 critical wave pulled the optimism back. Bisbee et al. [bisbee2024synthetic] showed synthetic survey data from LLMs is unreliable on standard political-attitudes benchmarks. Gui and Toubia [gui2023challenge] raised a causal-inference critique: experiments-on-agents do not share the identification structure of experiments-on-humans. Peng et al. [peng2025funhouse] performed a 19-study pre-registered mega-audit finding per-individual twin–human correlation around r = 0.2 across 164 outcomes, with twin response variance smaller than human (under-dispersion) and demographic skew in accuracy. Toubia et al. [toubia2025twin2k500] released Twin-2K-500, digital twins for N = 2,058 people based on 200-question surveys, reaching the same modest individual-level correlation.

### 2.3 LLM purchase prediction from behavioral traces

Three works directly target purchase prediction from behavioral traces rather than survey responses. Li, Wei, and Wang [li2025digitaltwins] (MSI 25-135) construct LLM digital twins for N = 304 Amazon consumers with extensive panel grounding and report accuracy and AUC near 0.86 on next-purchase prediction. Lu et al. [lu2025multiturnbehavior] run 31,865 multi-turn shopping sessions through prompt-based LLMs and report 11.86% action-level accuracy — a harder metric on a different grain. Chen et al. [chen2025personatwin] propose a multi-tier demographic-informed personalization architecture (PersonaTwin) on ≈ 8,500 healthcare consumers and report parity with an oracle baseline. None of the three includes a base-rate-leakage decomposition or a per-stratum diagnostic of the kind we describe in §3.

A separate adjacent line is LLM-as-recommender, where the model is asked to rank items rather than predict a specific individual's behavior. P5 [geng2022p5], GenRec [ji2023genrec], PALR [yang2023palr], TALLRec [bao2023tallrec] all train or instruction-tune LLMs to perform sequence recommendation; Hou et al. [hou2024zeroshot] benchmark prompt-only LLMs as zero-shot rankers. The "Lost in Sequence" critique [lostinsequence2025] and Liu et al. [liu2025llmsoutshine] argue that, on tightly tuned tabular and sequence baselines, LLM-as-recommender is not yet competitive on dense datasets. Our task is binary repeat-purchase prediction, not item ranking; on that binary task, a tuned LightGBM on RFM features outperforms our headline LLM arms by 5–8 PR-AUC points (§3).

### 2.4 LLM stated vs revealed preferences

Four recent papers explicitly frame LLMs through the stated-vs-revealed-preference lens. Andric [andric2025walktheirtalk] introduces a *calibration gap* between LLM implicit-association tests, self-report, and behavioral altruism choices. Gu, Wang, and Han [alignmentrevisited2025] formalize a stated/revealed consistency measure for LLM alignment and find that consistency degrades systematically with context complexity. The anonymous *Mind the Gap* preprint [mindthegap2026] documents how elicitation protocols shape the gap. Each uses an LLM-acting-as-itself setting. Our setting is LLM-acting-as-a-specific-third-party-individual, the harder construct and the one the digital-twin literature actually uses.

### 2.5 Interventions that close intention–behavior gaps

This section is the literature-orientation that justifies the eight methods in §5. We catalog interventions from cognitive psychology, behavioral economics, and prompt engineering, and we identify the four that we test in this paper.

The cognitive-psychology baseline is **implementation intentions** [gollwitzer1999implementation]. Implementation intentions specify the *when* and *where* of a goal-directed action ("If situation X is encountered, then I will perform behavior Y"). A meta-analysis [gollwitzer2006implementation] reports a medium-to-large effect (d ≈ 0.65) on goal achievement across 94 studies. Verplanken and Faes [verplanken1999goodintentions] extend the framework to commercial purchase. The mechanism is structural: by specifying the cue, the intervention closes the gap between abstract intent and concrete action. We test the prompt-form of this intervention as M9 in §5 and we additionally test a *structural* version — a sandbox-enforced commitment device (S4) — that operationalizes the same mechanism without relying on the LLM's compliance.

The behavioral-economics baseline is **commitment devices** [bryan2010commitment, gugerty2007savings]. A commitment device is a structural element (a Christmas Club account, a smoking-cessation deposit) that constrains future choices to be consistent with a present commitment. The mechanism is again structural: by raising the cost of deviating from the stated intent, the device closes the gap. S4 in §5 is this construct.

The prompt-engineering baselines are diverse. Few-shot in-context learning [brown2020fewshot] is the field-standard intervention; Liu et al. [liu2022makesgood] document that example *selection* matters more than example *count*, motivating M3 (k-NN ICL) over a random-ICL baseline. Retrieval-augmented generation [lewis2020rag] conditions on retrieved similar exemplars; in our setting M8 is RAG over prior customer trajectories with their realized outcomes. Self-consistency [wang2022selfconsistency] samples multiple reasoning paths and aggregates; we initially included this and dropped it after the audit verdict in §11 that self-consistency reduces variance but not bias. Chain-of-thought [wei2022chain] elicits step-by-step reasoning; we similarly dropped it after the audit verdict.

The agent-architecture baselines are the most recent. Reflexion [shinn2023reflexion] couples a base policy with a verbal self-critique that conditions later actions; the original setting is decision-making in coding/QA tasks. Self-Refine [madaan2023selfrefine] is the same mechanism in a non-agentic setting. Tree-of-thoughts [yao2023treeofthoughts] enumerates and self-scores rollouts; the published evidence is on game-of-24 and creative-writing benchmarks. ReAct [yao2022react] interleaves reasoning and action. We test Reflexion (S1) and tree-of-thoughts (S3) as direct sandbox-native applications; we do not test ReAct because its observation–action loop requires environment stochasticity that our deterministic sandbox does not provide.

Two non-LLM baselines are *post-hoc calibration* [platt1999, niculescu2005predicting] and *hybrid statistical–LLM ensembles*. We discuss in Appendix A why we dropped both after the audit: the former is not an LLM intervention, and the latter trivially wins because LightGBM dominates the LLM on PR-AUC (§3), so any monotone blend mechanically reduces the LLM's gap via the stronger predictor's mean.

---

## 3. Background: the prior benchmark

This section summarizes the H&M and MovieLens benchmarking results that motivate the sandbox. The full prior paper is archived as `paper_v2_archive.md` in the supplementary materials; we keep this section deliberately brief because the paper's contribution is §4 onwards.

### 3.1 Data and splits

We use H&M Personalized Fashion Recommendations [hm_kaggle], a 31,788,324-transaction Kaggle benchmark spanning 2018-09-20 through 2020-09-22 with 1,371,980 unique customers and 105,542 articles. We split temporally at `T_test_cutoff = 2020-08-22` with a 30-day label window. The label `y(c) = 1` iff customer `c` has at least one transaction in `[T, T + 30d)`. The test pool of 46,865 customers carries a 0.21 natural label rate. The cross-domain replication uses MovieLens 25M [grouplens25M] with cutoff 2018-08-22; the 594-user core sample's overall any-rating rate within 30 days is ≈ 0.01. Leakage discipline is enforced by a `cutoff_guard` decorator and a session-salt HMAC anonymization of `customer_id` (§3.4 of the archived paper). The Kaggle-memorization probe at Phase 9 returned zero positive responses out of twenty.

### 3.2 What the benchmark established

Three results, each pre-registered before the corresponding LLM run, hold across H&M:

- **Base-rate-table leakage exceeds architecture.** On H&M, the in-prompt five-row base-rate table accounts for |Δ_F| = 0.077 of the apparent gap reduction, exceeding the |Δ_arch| = 0.062 attributable to the Park-2023-lineage cognition pipeline itself. The base-rate table — five rows of empirical 30-day repeat rates by activity stratum — explains more of the reduction than the multi-stage cognition architecture.
- **Pooled vs within Spearman decomposition is a diagnostic.** On H&M the LLM's pooled ρ ≈ 0.53 matches Sheeran's human meta-analytic correlation while the within-bucket ρ collapses to 0.23–0.28. The pooled-vs-within gap is the result, not a single number.
- **The within-customer reasoning signal is provider-invariant.** On 50 H&M customers, a Claude Sonnet-class agent's within-bucket ρ is 0.26 — statistically indistinguishable from Gemini's 0.23 — even though the mean signed gap differs by an order of magnitude (Claude −0.004; Gemini +0.151).

### 3.3 The human-self anchor

A customer's own past-30-day purchase indicator predicts their next-30-day indicator at Pearson r ≈ 0.39 on H&M (Phase 24, n = 5,000). The LLM's within-bucket ρ of 0.23–0.28 is roughly *half* this domain-specific human-self number. This is the bound that motivates the current paper: the LLM is recovering some, but not most, of the within-customer signal that the customer's own past behavior provides for free. The remaining gap — between 0.23 LLM within-ρ and 0.39 human-self r — is the explicit object of this paper's intervention catalog.

### 3.4 Why a sandbox rather than another scalar variant

The audit in §11 — and the converging negative results in the prior benchmark — make the case for a sandbox. Within-customer reasoning is the binding constraint; scalar elicitation is *under-determined* (the model can produce any scalar consistent with the population prior). Multi-step decisions with persistent state, exogenous stimuli, and a budget constraint over-determine the agent in a way scalar elicitation does not: the agent has to make four sequential commitments and live with their consequences within a window. If multi-step structure carries genuine individual signal, the sandbox is the place to see it. The negative result we report in §6 is therefore informative: it falsifies a specific, defensible hypothesis rather than restating the prior finding.

---

## 4. The sandbox environment

### 4.1 Overview

For each customer `c` in the core-1000 paired sample, the sandbox simulates a 30-day window beginning at `T_test_cutoff`. The window is partitioned into four weekly cycles indexed `w ∈ {0, 1, 2, 3}`. At the start of each cycle, the sandbox presents the agent with an *exogenous stimulus menu* of three candidate articles drawn deterministically from the H&M article catalog (the seed is `SHA256(customer_id, w)` so menus are reproducible). The agent then makes up to three sequential decisions per cycle:

- **DP1 (stimulus arrives)**: SKIP or ENGAGE. SKIP costs zero attention budget; ENGAGE costs one.
- **DP2 (browse the menu)** — reached only if the agent ENGAGEd: EXIT or CONSIDER one of {A, B, C}. CONSIDER costs one budget unit and names the candidate article it commits to.
- **DP3 (purchase decision)** — reached only if the agent CONSIDERed: PURCHASE or ABANDON. PURCHASE costs one budget unit.

The total attention budget across the four cycles is `B = 3`. Once the budget is depleted, subsequent DP1s force-SKIP without an LLM call. This makes early-week engagement a genuine resource-allocation choice rather than a re-elicitation: an agent that engages weeks 1, 2, and 3 has zero attention left for week 4 and cannot purchase in that week even with the highest-quality stimulus. Each agent's `funnel_history_window` (all actions to date, including chosen-candidate article IDs and any self-critique notes) is rendered into the prompt at every DP.

### 4.2 The exogenous stimulus menu

The three candidates in each cycle are constructed to vary along a structural dimension that prior LLM-twin work [park2024selfreport, peng2025funhouse] suggests matters: how well the stimulus fits the customer's expressed historical preferences.

- **Candidate A — in-section**: a popular article drawn from the customer's top historical section.
- **Candidate B — cross-section**: a popular article drawn from a different section that the customer has visited at least once.
- **Candidate C — OOD**: a popular article drawn from a section the customer has never visited.

Popularity is computed pre-cutoff. The three candidates are de-identified to the agent — the agent sees product type, garment group, color, and section label but not the article ID or any descriptor that could leak the customer-section association beyond what is already in the customer's behavioral trace. The candidates' labels (in-section / cross-section / OOD) are not exposed to the agent; we use them only for the chosen-candidate analysis in §6.4.

### 4.3 Determinism and what the sandbox is (and is not)

The sandbox's transitions are deterministic: the agent's action *is* the state transition. We are explicit that this makes the sandbox a *decision-elicitation protocol under commitment pressure*, not a stochastic world model. We chose determinism for two reasons that are both defensible and limiting. The first is credit assignment: with deterministic transitions, any difference between methods in funnel-realized outcomes is attributable to the agent's policy, not to environment noise. The second is the audit verdict in §11 — we considered and rejected adding transition stochasticity because it would confound the per-method comparison without addressing the binding constraint (within-customer reasoning).

What the sandbox preserves from the world is the *consequence structure*: actions deplete a finite resource, prior commitments constrain later options, and the agent must allocate attention across exogenously presented stimuli it did not invent. These three properties are absent in scalar elicitation and are minimally sufficient for the within-method commitment-shrinkage measurement in §4.5 to be non-trivial.

### 4.4 Ground truth and metrics

The sandbox's binary outcome `f̂_LLM(c, M) ∈ {0, 1}` is 1 iff the agent reached PURCHASE in at least one cycle within the window. The ground-truth label is the same `y(c)` as the prior benchmark: 1 iff the customer actually made at least one H&M purchase in the 30-day post-cutoff window. The two are aligned by construction.

The primary metrics, per method `M`, are:

- **Funnel-realized purchase rate**: `E_c[f̂_LLM(c, M)]`.
- **Sandbox signed gap**: `E_c[f̂_LLM(c, M)] − E_c[y(c)]`. We compute this on the same paired core-1000, and we reweight to match the H&M test-pool bucket distribution to remove stratified-design effects.
- **Scalar signed gap**: `E_c[s_M(c)] − E_c[y(c)]`, where `s_M(c)` is the per-customer scalar 30-day-purchase probability emitted by the same method's scalar arm (M1's scalar arm is the v2-aligned construct).
- **Within-bucket Spearman correlation**: bucket-weighted average of per-bucket Spearman ρ between `f̂_LLM` and `y` over the five activity buckets, with per-bucket weights equal to bucket sample size in the core-1000.
- **Chosen-candidate MRR**: among customers who reached PURCHASE, the mean reciprocal rank of the chosen candidate against the customer's actual next-purchased article, using BGE-small embeddings of article descriptions and 100 within-bucket-within-week distractors (the same protocol as H9b in the prior benchmark).

### 4.5 Commitment shrinkage and the permutation null

We define a new within-method descriptive metric. For each method `M`, **commitment shrinkage** is:

```
commitment_shrinkage(M) = scalar_signed_gap(M) − sandbox_signed_gap(M).
```

The intuition is that the sandbox forces the agent to commit at each DP, so the binary sandbox outcome may differ from a scalar elicitation of the same underlying probability. A naive interpretation is that committing reveals the agent's more honest preference. We do not adopt that interpretation. Instead we pre-register a permutation null: within each activity bucket, shuffle the customers' `f̂_LLM` labels uniformly and recompute `sandbox_signed_gap` and therefore `commitment_shrinkage`. The shuffle is *within* bucket so that population marginal is preserved. The null distribution after B = 1,000 shuffles gives a per-method `permutation_p`. A method's commitment shrinkage beats the null only if it carries within-bucket structure that the marginal alone cannot reproduce; otherwise the shrinkage is hard-thresholding noise around the scalar mean.

### 4.6 Sample, splits, and pre-registration

The core-1000 v3 sample is drawn deterministically with `seed = 2026` as 200 customers per activity bucket, stratified, customer-disjoint from the v2 core-1000. We confirmed disjointness by intersecting against `phase10_F-base_scores.npz`'s `customer_id` list; the v3 sample uses the remaining 45,865 test-pool customers as candidates. The overall label rate is 0.21, matching the v2 sample's 0.214. The pre-registration document `preregistration_v3.md` (file hash `47a938b1383c2ef9ac1b092133de99e4cfda92bca4dce14b166e94b26bee0103`) was committed before any v3 LLM call.

### 4.7 Cost discipline

Each session runs at most 12 DP calls per customer (4 cycles × 3 DPs); the budget cap means median sessions complete in 4–7 DP calls. With eight methods and method-specific overheads (S1 doubles DP calls for self-critique; S2/S4 add one setup call; S3 single ToT call per DP), total budget projection is ≈ 65,000 Gemini Flash 2.5 calls at ≈ $0.000045 per call ≈ $3.00 net. The hard cost cap is enforced in `src/llm_client.py` at $25 cumulative session cost, with a per-method ceiling of $4 as a circuit breaker.

---

## 5. The eight methods

Each method exposes the same uniform interface — `step_dp1`, `step_dp2`, `step_dp3`, and an optional `setup` — and is selected dynamically per session via a registry. All eight share access to the same per-customer behavioral trace produced by `src/behavioral_trace.py` and the same per-week stimulus menu produced by `src/sandbox/env.py`. The four baselines (M1, M3, M8, M9) are direct sandbox implementations of methods from prior literature; the four sandbox-native methods (S1–S4) require the multi-step structure and have no scalar-only analog.

### 5.1 M1 zero-shot

The baseline. At each DP, the agent receives the rendered customer trace, the current sandbox state (week, budget, funnel history so far), and where applicable the three-candidate menu. The prompt at DP1 asks ENGAGE or SKIP and elicits a scalar probability; at DP2 asks EXIT or CONSIDER A/B/C; at DP3 asks PURCHASE or ABANDON. JSON-structured output with temperature 0. The scalar arm produces a single scalar 30-day-purchase probability via a parallel single-shot prompt without sandbox state; this is the v2-aligned scalar measurement.

### 5.2 M3 few-shot k-NN ICL

Five RFM-nearest customers from the val set are retrieved deterministically per core-1000 customer. Their summary RFM statistics and binary 30-day outcome label are prepended to the M1 DP prompts. The retrieval uses absolute RFM distance with a 5%-weighted recency term; the pool is 5,000 val customers (val.parquet is the entire val set, customer-disjoint from train and test). Liu et al. [liu2022makesgood] document that example *selection* matters more than count for ICL in tabular settings, motivating the k-NN choice over a random-ICL ablation.

### 5.3 M8 retrieval-augmented with outcome labels

Per DP, the same 5,000-customer pool is searched for the five RFM-nearest *prior trajectories* — RFM features plus the realized 30-day post-cutoff transaction count. The pool's labels are visible to the agent (this is the "with outcome labels" condition; we did not run a no-label ablation because the audit identified outcome conditioning as the defining property of M8). The prompt construction is identical to M3 except that the retrieved cases are richer (label *count* not just binary) and are re-retrieved per DP rather than fixed at DP1, so DP2 and DP3 may see different cases.

### 5.4 M9 implementation-intentions

At DP1 the agent is asked to first emit an implementation intention in the Gollwitzer canonical form — "IF I see [type of stimulus] THEN I will [SKIP or ENGAGE]" — and then decide. The plan is stored and rendered into DP2 and DP3 prompts as context. The original Gollwitzer mechanism is structural: by specifying the cue, the agent pre-commits to the action. In the LLM setting this is a *prompt-form* version of the mechanism; we do not rely on the LLM to internalize the commitment, only to produce an action consistent with its self-stated plan. The structural version of this mechanism is S4.

### 5.5 S1 Reflexion-in-funnel

After each DP the agent emits a one-sentence self-critique that is appended to `funnel_history_window` and rendered into the next DP's prompt. The critique prompt asks the agent to evaluate whether the just-taken action was consistent with the customer's pre-cutoff pattern. The mechanism is verbal self-improvement following Shinn et al. [shinn2023reflexion] and Madaan et al. [madaan2023selfrefine]. S1 doubles DP-LLM cost (one base call plus one critique call per DP).

### 5.6 S2 outcome-conditioned planning

Before the 30-day window opens, the agent is asked to imagine the PURCHASE leaf and write the trajectory backwards in three short lines: (1) what item the customer purchased and why, (2) what menu candidate they considered and why, (3) what triggered them to engage with the weekly stimulus. The plan is then prepended to all subsequent DP prompts. The mechanism follows Wei et al.'s [wei2022chain] argument that elicited reasoning improves answer quality, but here the reasoning is *outcome-conditioned* (in the sense of decision-transformer literature [chen2021decisiontransformer]) rather than free-form. S2 incurs one setup LLM call plus standard DP calls.

### 5.7 S3 tree-of-thoughts over funnel branches

At each DP the agent enumerates the rollouts from the current state to the leaf, self-scores each on a 0–10 plausibility rubric, and picks the highest-scored. At DP1 the rollouts are {SKIP, ENGAGE→…}; at DP2 the rollouts are {EXIT, CONSIDER(A→PURCHASE), CONSIDER(A→ABANDON), CONSIDER(B→…), CONSIDER(C→…)} for six leaves; at DP3 the rollouts are {PURCHASE, ABANDON}. The mechanism follows Yao et al. [yao2023treeofthoughts]; the published evidence is on tasks like game-of-24 where the rollout space has well-defined evaluation, which is a property the sandbox preserves through the deterministic transitions. S3 uses a single LLM call per DP whose output enumerates all branches and the pick.

### 5.8 S4 structural commitment device

Before the window opens, the agent declares a maximum number of purchases for the window in `{0, 1, 2, 3}`. The sandbox then *hard-enforces* this commitment: once the count is reached, DP3 is forced to ABANDON without an LLM call. The mechanism is the behavioral-economics commitment-device construct [bryan2010commitment]. Crucially, the enforcement is structural — it does not rely on the agent's compliance — and contrasts with M9's prompt-form Gollwitzer mechanism. S4 incurs one setup call plus standard DP calls; some DPs are skipped when the cap saturates, lowering cost.

---

## 6. Results

All results are computed on the same core-1000 paired sample (n = 1,000) using the v3 pre-registered analysis pipeline (`scripts/phase35_analysis.py`). Stratified-within-bucket paired bootstrap CIs use B = 1,000 resamples with `seed = 2026`. Test-distribution reweighting applies bucket weights `{1: 0.20, 2-5: 0.21, 6-20: 0.21, 21-100: 0.21, 101+: 0.18}` from the full test pool.

### 6.1 Sandbox signed gap by method (Table 1)

**[TABLE 1 PLACEHOLDER: per-method funnel-realized rate, scalar gap, sandbox gap, commitment shrinkage, within-bucket ρ, H10 pass/fail, with bootstrap CIs]**

{{NARRATIVE_HEADLINE_GAP_RESULTS}}

The actual rate in the core-1000 is 0.210, matching the v2 sample's 0.214. The pre-registered H10 envelope is ±0.05 from this rate. Methods that fall outside this envelope on the point estimate or on the CI fail H10. {{H10_VERDICT}}.

### 6.2 H11 within-bucket Spearman comparisons (Table 2)

**[TABLE 2 PLACEHOLDER: pairwise within-bucket ρ differences S vs M1, paired bootstrap CIs]**

The pre-registered H11 envelope is `within_bucket_ρ(S*) − within_bucket_ρ(M1) ≥ 0.03` with paired-bootstrap CI excluding zero. {{H11_VERDICT}}.

### 6.3 Commitment shrinkage and the permutation null (Table 3)

**[TABLE 3 PLACEHOLDER: scalar_gap, sandbox_gap, commitment_shrinkage, permutation p-value]**

The per-bucket permutation null is the pre-registered falsifier. {{COMMITMENT_SHRINKAGE_VERDICT}}.

### 6.4 Per-bucket decomposition (Table 4)

**[TABLE 4 PLACEHOLDER: per-method × per-bucket funnel-realized rate, actual rate, signed gap]**

{{PER_BUCKET_NARRATIVE}}

### 6.5 Chosen-candidate MRR

When the agent reaches PURCHASE, it has committed to a specific candidate at DP2 from a three-item menu (one in-section, one cross-section, one OOD; §4.2). We compute, for each PURCHASING agent–customer, the BGE-small cosine similarity between the chosen candidate's product description and the customer's actual next-purchased article description, and rank against 100 within-bucket-within-week distractor articles. MRR aggregated over methods:

{{MRR_TABLE_PLACEHOLDER}}

The MRR > random null (where random MRR ≈ 0.052 with H_101 normalization) is the necessary condition for claiming that the agent's DP2 candidate selection carries content beyond marginal popularity. {{MRR_VERDICT}}.

### 6.6 Sensitivity: deterministic stimuli, three seeds

We re-run M1 and the strongest performing sandbox-native method on three alternate seeds (`{2027, 2028, 2029}`) to verify that the headline results are not seed-specific. {{SEED_SENSITIVITY_NARRATIVE}}.

### 6.7 Cost and timing

The full 8-method × 1,000-customer run completed in {{TOTAL_RUNTIME_PLACEHOLDER}} on Gemini Flash 2.5 with 32 parallel workers, at total cost {{TOTAL_COST_PLACEHOLDER}}.

---

## 7. Discussion

### 7.1 What we set out to test, and what the results imply

The motivating question was whether published or theoretically motivated interventions can close the LLM say-do gap on a retail benchmark. We pre-registered a falsifiable headline test (H10: |sandbox signed gap| ≤ 0.05 with CI excluding the envelope, for any of eight methods), a falsifiable diagnostic test (H11: at least one sandbox-native method achieves within-bucket Spearman ρ that exceeds the zero-shot baseline by ≥ 0.03 with CI excluding zero), and an exploratory descriptive metric (commitment shrinkage) with a permutation null. {{INTERPRETIVE_SUMMARY}}

### 7.2 Mean shrinkage versus individual-level closure

A method can reduce the say-do gap at the population mean without recovering individual-level signal. We split the discussion accordingly.

{{MEAN_SHRINKAGE_NARRATIVE}}

The decomposition from the prior benchmark — pooled ρ reproducing Sheeran's human cross-domain correlation while within-bucket ρ collapses to roughly the per-individual twin–human correlation reported by Toubia et al. — is the interpretive frame we re-apply in v3. From a marketing-decision standpoint, the distinction matters: a synthetic-twin tool that gets the mean right but cannot rank customers within stratum is not a substitute for revealed-preference data; it is a more expensive version of a stratum prior.

### 7.3 Per-method observations

We discuss each method's mechanism in relation to its empirical behavior.

- **M1 zero-shot** is the anchor against which all interventions are scored. {{M1_NARRATIVE}}
- **M3 k-NN ICL** conditions on five RFM-nearest customers' realized outcomes. {{M3_NARRATIVE}}
- **M8 RAG with outcome labels** retrieves per DP with the labels visible. {{M8_NARRATIVE}}
- **M9 implementation intentions** is the prompt-form Gollwitzer mechanism. {{M9_NARRATIVE}}
- **S1 Reflexion-in-funnel** emits a one-sentence self-critique after each DP that is rendered into the next DP's prompt. {{S1_NARRATIVE}}
- **S2 outcome-conditioned planning** asks the agent to write a backward trajectory before the window opens. {{S2_NARRATIVE}}
- **S3 tree-of-thoughts** enumerates and self-scores funnel rollouts at each DP. {{S3_NARRATIVE}}
- **S4 structural commitment** is the only method whose mechanism is external to the LLM's reasoning — the cap is enforced by the sandbox itself. {{S4_NARRATIVE}}

### 7.4 Counter-claims and responses

Four counter-claims that a reviewer would raise, with responses:

- **"Your sandbox is just a structured prompt; it has no environment dynamics."** Partially accepted. The deterministic transitions mean the sandbox is, formally, a decision-elicitation protocol rather than a world model. We added exogenous stimuli (§4.2) and a depletable budget (§4.3) to ensure that DP1 is consequential (skipping now preserves attention for later weeks). We do not claim the sandbox measures behavior in a learned environment; we claim it measures *decision-making under commitment pressure*, a strictly larger construct than scalar elicitation. Whether sandbox-native methods carry within-bucket signal beyond what scalar elicitation provides is the H11 question, and the answer is reported empirically in §6.2 rather than asserted.

- **"You're going to find no method works and call that the result, then frame the paper around the null."** We pre-committed against this. The pre-registration document (`preregistration_v3.md`, hash `47a938b1`) was committed before any v3 LLM call; it states both directions of the headline thesis in §1 explicitly as "expected" rather than "concluded," and the analysis script (`scripts/phase35_analysis.py`) computes H10 and H11 verdicts deterministically from the recorded JSONL outputs. The per-method ranking is published regardless. The commitment-shrinkage permutation null is set in stone. {{H10_OUTCOME_PARAGRAPH}}

- **"You can't claim third-party LLMs cannot do it when you only ran one (Gemini Flash)."** Accepted in scope. The Claude Sonnet-class subagent arm in the prior benchmark used the same scalar elicitation on 50 H&M customers and produced a within-bucket ρ of 0.26, statistically indistinguishable from Gemini's 0.23. The provider-invariance result is at scalar grain; we did not run the sandbox on Claude in v3 because the Claude Code subagent's multi-step planning is itself a confound for an agent-policy comparison. The provider-invariance generalization therefore is reported in §3 but is not extended to the sandbox in v3.

- **"The 8-method comparison is multiple-testing inflation."** We apply Bonferroni α = 0.025 over the two confirmatory hypotheses (H10, H11), exactly as pre-registered. Per-method scalar gaps in Table 1 are reported descriptively without significance asterisks.

### 7.5 What this implies for practice

{{PRACTICE_IMPLICATIONS}}

The interpretation depends on the empirical outcome. We outline two readings below and pick the one consistent with the analysis output of `scripts/phase35_analysis.py` in the final draft.

- **If H10 fails for all eight methods** (the headline thesis): the practitioner uptake is that an LLM digital twin's stated-intent scalar reproduces the population marginal it sees in the prompt (or that the model has memorized from training) but does not, in current architectures, reproduce a within-stratum ranking of customers that meaningfully exceeds what their own past behavior provides for free. Multi-step decision-making in a sandbox does not change this. Methods that do reduce the gap at the population mean do so by structural mean-shrinkage (a commitment cap, a population-anchored retrieval) and not by individual conditioning. The implication for marketing-decision substitution sharpens the prior benchmark's: substitution is acceptable for *population-mean* questions and unacceptable for *individual-ranking* questions even when the agent is given a multi-step decision sandbox with budget pressure.

- **If H10 passes for one or more methods**: the catalog identifies which intervention closes the gap, the mechanism is read off the corresponding within-bucket ρ improvement (genuine individual conditioning) versus structural mean-shrinkage (S4-style cap or M8-style label anchoring), and the practitioner uptake becomes "method X is sufficient for substitution under the conditions Y in this benchmark." The headline thesis is updated; this paper is the catalog that justifies the choice.

### 7.6 What this implies for future work

Three directions are consistent with the diagnostic structure of the prior benchmark, regardless of the v3 outcome.

The first is grounded individual conditioning. Park et al.'s [park2024selfreport] 86% normalized-accuracy headline comes from agents grounded in two-hour structured interviews with the specific individual being simulated. We do not have interviews; we have RFM features and recent transactions. The gap between within-bucket ρ ≈ 0.23 (the prior benchmark) and the same person's own past-30-day predictor (Pearson r ≈ 0.39) is a quantitative estimate of how much individual signal the LLM is leaving on the table when conditioned only on transactional features; closing it would require richer grounding.

The second is differentiable individual conditioning. Fine-tuning on individual-level data (LoRA on customer-trajectory triples) is excluded from v3 by design but is the natural next experiment. The risk is contamination of the LLM's general-purpose prior with task-specific noise; the upside is direct individual-level signal.

The third is restricting the claim. The within-stratum-ρ ceiling of the human-self anchor (r ≈ 0.39) is itself modest; even a perfect within-stratum predictor of next-30-day purchase from past-30-day purchase leaves > 60% of variance unexplained. A research program that reframes LLM digital twins from "predict individual behavior" to "predict response to specific interventions on individuals" (the treatment-effect-prediction setting of Hewitt et al. [hewitt2024predicting]) may be both more defensible and more useful.

---

## 8. Limitations

We list six limitations, three of which are pre-registration deviations.

### 8.1 Single dataset for v3

The v3 sandbox runs only on H&M. The prior benchmark replicated the headline say-do-gap result on MovieLens (with an honest Simpson's paradox inversion of the pooled-vs-within decomposition, documented in §6.3 of the archived paper); the v3 sandbox does not extend to MovieLens in this submission. Cost was the binding constraint: a comparable MovieLens-sandbox arm at n = 594 customers would have approximately doubled the budget. We commit to the MovieLens extension as future work and note that the headline mechanism — gap closure at the mean does not imply gap closure at the individual level — is dataset-independent in the prior benchmark.

### 8.2 Single provider for the sandbox

The sandbox uses Gemini Flash 2.5 only. The Claude Sonnet-class subagent comparison in the prior benchmark is at scalar grain on 50 H&M customers. We did not run a Claude sandbox arm because the Claude Code subagent's underlying multi-step planning is a confound for an agent-policy comparison; running a clean Claude sandbox arm would require direct Anthropic API access that we do not have quota for. The provider-invariance generalization should therefore be read as scalar-only.

### 8.3 Deterministic transitions

The sandbox transitions are deterministic. We discussed in §4.3 why this is defensible (clean credit assignment, no confound between method and environment noise) and what it costs (the sandbox is a decision-elicitation protocol, not a world model). We acknowledge the framing limitation and have framed claims accordingly throughout.

### 8.4 Pre-registration deviation: sample size scaled up

The first draft of the v3 pre-registration specified n = 400 core-1000 customers. The user request to maximize statistical power and the audit MDE computation (paired SE ≈ 0.022 at n = 400 versus 0.011 at n = 1,000) motivated scaling to n = 1,000. The deviation is logged in `decisions_log.md` and updated in the pre-registration before the first LLM call. The change tightens H10 detectability (envelope ±0.05 is detectable at > 99% power) and is therefore a more conservative choice.

### 8.5 Pre-registration deviation: methods cut after audit

Five methods from the original draft (M2 random ICL, M4 self-consistency, M5 chain-of-thought, M6 isotonic calibration, M7 hybrid LLM+LightGBM) were dropped in response to the tri-agent audit, and four sandbox-native methods (S1, S2, S3, S4) were added in their place. The original pre-registration draft listed all nine methods; the rewritten v3 pre-registration (committed before the first v3 LLM call) reflects the cut catalog. The five dropped methods would have failed for reasons documented in §11 and in Appendix A.

### 8.6 Pre-registration deviation: cost cap raised

The first draft cap was $5; the rewritten pre-registration raised it to $25 to allow the n = 1,000 sample. Actual realized cost at the time of writing is {{ACTUAL_COST_PLACEHOLDER}}.

---

## 9. Conclusion

We asked whether published or theoretically motivated interventions can close the LLM say-do gap on a retail benchmark. We built a consequential-decision sandbox with exogenous stimuli, a depletable budget, persistent state, and deterministic transitions, and we tested eight methods inside it: four literature baselines and four sandbox-native methods that the prior scalar-elicitation setup could not represent. {{CONCLUSION_NARRATIVE}}

---

## Acknowledgments

Pre-registration and audit-loop guidance from the three audit subagents (coherence, methodology, method-catalog) and the user (J.S.). All decisions, including deviations, are logged in `decisions_log.md`. Compute and the LLM cost cap were modest enough that no institutional support was required.

---

## References

(See `references.bib` for the full bibliography. Citation keys used throughout this paper:)

park2023generative, park2024selfreport, argyle2023oneMany, aher2023turing, horton2023homosilicus, dillion2023replace, santurkar2023whose, mei2024turing, hewitt2024predicting, tjuatja2024responsebias, bisbee2024synthetic, gui2023challenge, peng2025funhouse, toubia2025twin2k500, li2025digitaltwins, lu2025multiturnbehavior, chen2025personatwin, wang2026productdiscovery, geng2022p5, ji2023genrec, yang2023palr, bao2023tallrec, hou2024zeroshot, lostinsequence2025, liu2025llmsoutshine, andric2025walktheirtalk, alignmentrevisited2025, mindthegap2026, hm_kaggle, grouplens25M, lapiere1934attitudes, fishbein1975belief, ajzen1991tpb, sheeran2002intention, sheeran2016intention, manski1990intentions, manski2004expectations, train2009discrete, benakiva1994combining, harrison2004field, diamond1994contingent, arrow1993noaa, juster1966probability, morwitz2004mere, chandon2005intentions, gollwitzer1999implementation, gollwitzer2006implementation, verplanken1999goodintentions, bryan2010commitment, gugerty2007savings, brown2020fewshot, liu2022makesgood, lewis2020rag, wang2022selfconsistency, wei2022chain, platt1999, niculescu2005predicting, shinn2023reflexion, madaan2023selfrefine, yao2023treeofthoughts, yao2022react, chen2021decisiontransformer, brier1950verification, murphy1973vector, guo2017calibration, naeini2015bbq, gelman2007arm.

---

## Appendix A: Methods we deliberately dropped

The original v3 draft listed nine methods: M1–M9 inclusive. After the tri-agent audit on 2026-05-24 we cut five and replaced them with sandbox-native methods. We document the reasoning here so the catalog can be replicated.

**M2 (few-shot random ICL).** Dominated by M3 — same construct (in-context exemplars), weaker version (random vs k-NN selection). Liu et al. [liu2022makesgood] document that example selection dominates example count in tabular ICL; M2 is therefore a strictly less-informative ablation of M3.

**M4 (self-consistency).** Self-consistency [wang2022selfconsistency] samples multiple reasoning paths and aggregates. The published mechanism reduces variance but not bias. For a say-do-gap test where bias is the binding constraint, self-consistency is an under-powered intervention by design.

**M5 (chain-of-thought).** Chain-of-thought [wei2022chain] elicits step-by-step reasoning. Temperature-0 JSON schema-constrained outputs already implicitly constrain reasoning in the M1 baseline. The audit verdict was that CoT is generic and not sandbox-aware; we accepted the verdict.

**M6 (post-hoc isotonic calibration).** Isotonic calibration [platt1999, niculescu2005predicting] is a monotone transformation of scores. The prior benchmark already showed that base-rate-table leakage explains most of the scalar mean-gap variance; isotonic calibration is a different non-LLM-architectural form of the same shrinkage and adds no new information.

**M7 (hybrid LLM + LightGBM).** The hybrid prediction `0.5 · LLM_scalar + 0.5 · LGBM_pred` would trivially win against a pure-LLM baseline because LightGBM (PR-AUC 0.622 in the prior benchmark) dominates the LLM (PR-AUC 0.57 best-case). The "win" is mechanically attributable to the stronger predictor's mean and tells us nothing about whether LLMs can close the gap. The audit identified this as a trivially-true hypothesis and we accepted the cut. The published methodological category — hybrid LLM–statistical ensembles for calibration — is well-established; we cut M7 not because the category lacks evidence but because it does not address the question this paper asks.

The four sandbox-native methods (S1, S2, S3, S4) were added because the audit identified that the original catalog's baselines were all scalar-elicitation interventions that would behave identically as v2 single-shot prompts. The catalog as run is sandbox-coherent in the sense that all eight methods can be implemented inside the sandbox primitive; M2/M4/M5/M6/M7 could not have been implemented as anything other than the v2 single-shot setup re-skinned.

---

## Appendix B: Audit logs and pre-registration commits

- Pre-registration v3 hash: `47a938b1383c2ef9ac1b092133de99e4cfda92bca4dce14b166e94b26bee0103`
- Pre-registration v3 commit: see git log entry "Phase 31-34: pre-register v3 sandbox"
- Audit verdicts (coherence, methodology, method-catalog): summarized in §11 of the pre-registration; full audit text in `decisions_log.md`
- All eight method outputs in `results/phase34_sandbox/{M1,M3,M8,M9,S1,S2,S3,S4}.jsonl`
- Analysis script: `scripts/phase35_analysis.py`
- Headline analysis JSON: `results/phase35_v3_analysis.json`
