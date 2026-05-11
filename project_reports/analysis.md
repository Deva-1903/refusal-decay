# CS602 Project — Report-by-Report Analysis

Project: **Identifying the Internal Causal Mechanism of Shallow Safety Alignment Decay in Llama-3.1-8B-Instruct under Prefilling Attacks**

Across six reports, the project narrows from a broad mechanistic question about shallow safety alignment to a focused, well-instrumented study of the refusal-direction signal under prefilling. The arc is healthy: each report demonstrably tightens the previous one based on either feedback or experimental surprise.

---

## Report 1 — Three proposals (full marks)

**What it is.** Three independent proposals, each ~1 page: (1) causal mechanism of shallow safety alignment, (2) feature geometry of associative recall across Transformers vs. SSMs, (3) reverse-engineering ReFT interventions.

**Strengths.**
- Each proposal has a clear phenomenon, system, task, and feasibility check.
- Tooling (pyvene, pyreft, tinylang) is correctly identified and matched to the question.
- Compute budget is realistic for a semester (inference-only, single GPU).

**Weaknesses / friction.**
- All three are interpretability-flavored and share an author's preference for mechanistic interventions — Jensen's note implicitly says "pick on feasibility + interest," not "diversify."
- Proposal 1's "Interest" sentence ("top-choice... AI safety research agenda I plan to pursue in my doctoral work") is more of a personal-statement framing than is needed in a research proposal.

**Outcome.** Proposal 1 (shallow alignment) is selected for the remainder of the semester.

---

## Report 2 — Project specification + frontier (full marks)

**What it is.** Locks in the shallow-alignment project. Describes system (Llama-3.1-8B-Instruct), task (safety-constrained generation), environment (prefilling + adversarial suffix attacks on AdvBench/JailbreakBench), variables, and a literature-driven "Frontier" section.

**Strengths.**
- The Frontier section *synthesizes* rather than enumerates: Qi et al. (behavioral shallowness) + Chen et al. (safety neurons) + Arditi et al. (refusal direction) + Zhou et al. (safety heads) → four named gaps. This is exactly what the assignment asked for.
- Variables are organized into system / task-environment / dependent — a structure that pays off later when Report 3 uses them as experimental axes.
- The four gaps at the end ("no causal account of positional decay," "untested restoration hypothesis," "interaction between attack type and decay mechanism," "layer-position interaction") become the through-line for Reports 3–6.

**Weaknesses / friction.**
- Scope is still wide: two attack types, multiple component types (neurons / heads / direction), layer × position factorial. The report doesn't yet say which gap will be tackled first. Report 3 effectively makes that choice for you.
- "We will use the methodology of Arditi et al." appears alongside both pyvene-based interchange interventions and Chen-style neuron patching — three different mechanistic toolkits that would each take real time to wire up. The eventual project uses only Arditi-style projection + simple activation patching, which is the right call but unstated here.

**Feedback.** "The phenomenon is well-explained." → This is the section that pays for everything downstream; the positional-decay framing established here is the same phenomenon Report 6 ends up measuring.

---

## Report 3 — Exploratory experiments (95/100)

**What it is.** First experimental contact with the system. Three prefilling conditions (k ∈ {0, 3, 10}), 25 AdvBench prompts, 8 probed layers, behavioral refusal classifier + refusal-direction projection, plus a 5-prompt patching pilot.

**Strengths — the report does what exploratory research is supposed to do.**
- **Expectations are written down *before* the run** (Section 1.1, five numbered predictions). This is what makes the surprises actually count as surprises.
- **Two genuine surprises, both reported honestly.**
  - *Plateau effect:* k=3 (0.32) ≈ k=10 (0.36) — Expectation 3 (monotonic decay) explicitly falsified.
  - *Late-layer negative shift:* projection at layers 24/27 doesn't decay to zero, it inverts to strongly negative (−3.33 to −5.92). Expectation 4 partially falsified.
- The patching pilot (0/5 recoveries at layer 27) is reported as a small negative result rather than ignored.
- Conclusions section explicitly says "this refines our research questions" and lists three concrete refinements (focus on layers 24–27, use k=3 as cheap representative, redesign the inversion test) — these *do* show up in Reports 4–6.

**Weaknesses / friction.**
- Sample size (n=25) is small enough that the 0.32 vs. 0.36 "plateau" claim is genuinely uncertain (you acknowledge this). Better here would be one or two intermediate k values (k=1, k=2) which the report mentions as "future work."
- Layer 27 in baseline k=0 is already mildly negative (−0.03), which weakens "sign flip" framing for that layer — Report 6 catches this and reframes around layer 24 as the cleanest sign flip.
- Section 3.2 conflates "inversion hypothesis" with "active suppression" — these are separable. Report 5 cleans this up by listing them as separate hypotheses (H1 information-loss vs. H2 active-suppression).

**Feedback.** "Excellent work. Both surprising findings and great initial work on digging into deeper mechanisms." → Earned. The pre-registered expectations + honest negative pilot is the methodologically strongest element of the whole project.

---

## Report 4 — RQs and hypotheses (95/100)

**What it is.** Three research questions, each with 2 hypotheses + falsification criteria. Anonymous, double-blind format. RQ1 asks how the refusal signal changes; RQ2 asks whether patching restores; RQ3 (exploratory) compares prefilling vs. suffix attacks.

**Strengths.**
- Hypotheses are concretely falsifiable, with thresholds ("layers 15–25 will show faster decay than early layers," "patching at t=1 → t≥5 will produce a statistically significant increase in refusal rate").
- RQ2's H2a/H2b cleanly separate "is restoration possible at all?" from "does restoration depend on layer?" — these are independent claims and the report keeps them independent.
- H3a/H3b are explicitly labeled as *competing predictions* — the rare case of a report saying "I don't know which is true; here's how I'll find out."

**Weaknesses / friction (this is where the grader's note lands).**
- You called RQ1 "mechanistic." Jensen's feedback: it isn't. RQ1 measures *how one factor (refusal-direction projection) changes in response to another factor (token position under prefilling)* — that is structurally a **causal** question, even though answering it informs mechanism. A truly mechanistic RQ would be of the form "what computation, when removed/replaced, produces this change?" — which is closer to RQ3.
- The framing slip matters because it changes the bar of evidence. Causal questions need controlled variation; mechanistic questions additionally need a model of *which internal piece does the work*. The two need different experiments.
- Report 6 internalizes this exactly: "This is a *causal* question whose purpose is to inform a mechanistic explanation."

**Feedback.** "You're thinking about mechanism and finding good causal questions to answer that will inform you about mechanism. Nice work." → The substance is right; only the label was wrong.

---

## Report 5 — Research design (not graded)

**What it is.** A single research question with four working hypotheses (information-loss, active-suppression, layer-specificity, threshold-prefilling), a mixed observational-plus-interventional design, an explicit small factorial (layer ∈ {16, 24, 27} × source position ∈ {−1, early gen} × target position ∈ {1, 3, 5}), and a hypothesis→analysis mapping at the end.

**Strengths.**
- **Four working hypotheses, not one.** "The purpose is not to prove one mechanism correct in advance, but to design evidence that can distinguish among plausible alternatives." This is the right epistemic stance, and it directly addresses the Report 4 feedback by avoiding a single "mechanistic" framing.
- Sample sizes (25 harmful + 25 benign for tracing; 10 harmful for patching) are explicitly justified: "the main goal of Report 6 is not a giant benchmark, but a careful first study of one mechanism." Refreshingly disciplined.
- Settings-held-constant list (model weights, tokenizer, chat template, decoding config, etc.) shows real awareness that the study already varies several factors.
- Section 4 maps each hypothesis to a specific analysis + a specific success criterion. Almost everything in Report 6 traces back to a row in this section.

**Weaknesses / friction.**
- **The patching source-position choice is the single biggest weakness, and Report 6 confirms it.** Source = "last input/prefill token of the same k=3 forward pass" means the source activation is *itself attacked*. At layer 27, the source projection turns out to be −0.458 (anti-refusal-aligned), so injecting it into early generated positions cannot meaningfully test restoration. The standard activation-patching design uses a *clean* (k=0) source from the same prompt. This is flagged explicitly in Report 6's threats-to-validity and priorities-for-Report-7.
- "Bootstrap or exact binomial intervals... McNemar if discordant cases are sufficient" is the right plan, but Report 6 ends up reporting raw counts only — would have been good to lock in the uncertainty-quantification method here.
- H2 (active suppression) and H1 (information loss) are framed as opposing, but a null patching result is consistent with *either* "the source we patched was bad" or "active suppression is real." The design doesn't yet contain the discriminator (which would be: patch a clean source, then either (a) refusal returns → information loss, (b) refusal stays low → active suppression). Report 7 needs to add this.

---

## Report 6 — Execution + initial results (not graded)

**What it is.** Runs the Report-5 design. Five hypotheses (H1 attack effect, H2 late-layer specificity, H3 condition-level association, H4 restoration via patching, H5 layer specificity of patching), behavioral table for k ∈ {0, 3} (k=10 dropped due to GPU compatibility error), tracing table for k ∈ {0, 3, 10} at layers {16, 20, 24, 27}, and a 9-cell patching grid (3 layers × 3 target positions, 10 prompts).

**Strengths.**
- **Cleanest framing of the project so far.** Phenomenon, task, environment, and hypotheses are stated tightly in 1 page. Five hypotheses with explicit update rules per piece of evidence.
- **The H2 result is unambiguous and quantified.** |Δ(k=3 − k=0)| grows monotonically with depth: 1.10 → 1.99 → 3.78 → 4.91 at layers 16/20/24/27. This is the strongest empirical contribution of the entire project — the late-layer-specific effect was a Report-3 hunch and is now a measured, monotone gradient.
- **The patching null result is interpreted correctly.** Rather than concluding "refusal-direction interventions don't work," Report 6 reports the mean source projection per cell. At layer 27, source = −0.458, meaning the design was replacing an attacked state with another attacked state. The null is *diagnostic*, not *conclusive*. This is the right way to read a negative result.
- **Threats to validity (Section 7.3) are unusually candid.**
  - Phrase classifier limits ("I cannot stress enough..." as false positive).
  - Direction not held out (the 25 traced prompts overlap with the 50 used for direction extraction).
  - Layer-specific projection magnitudes not comparable across layers.
  - n=10 patching subset means single-prompt flips are noise.
- Priorities for Report 7 (Section 7.2) are specific and actionable: use k=0 source activations, run a prompt-level association test within k=3, concentrate the grid on layers 24/27 and add target position 0, re-run k=10 generation on a compatible GPU.

**Weaknesses / friction.**
- The k=10 row in the behavioral table being borrowed from the Report 3 pilot is a real limitation; the report handles it honestly (separate row, flagged as not-in-main-comparison) but the cleanest fix is to just re-run.
- H3 is "directionally consistent at the condition level" but the prompt-level test (within k=3, do refusing prompts have less-negative late-layer projection?) wasn't run despite the data being available. This is the no-cost win flagged for Report 7.
- The patching grid uses 10 prompts because of compute, but the trace block already has 25 — picking 10 means three of the nine cells flipping a single prompt is the difference between "no effect" and "−10%," and at this n the report cannot say which.
- The active-suppression vs. information-loss disambiguation is not yet resolved — by design (the source position was wrong) — but Report 7 is the right place to actually settle it.

---

## Project arc — what the six reports show as a whole

| Report | Move | What it added | What it set up |
| --- | --- | --- | --- |
| R1 | Pick a project | Three viable proposals | Choice of shallow alignment |
| R2 | Locate at frontier | 4 named gaps in literature | Variable list + dependent measures |
| R3 | Look at the data | **Late-layer negative-shift surprise** + plateau effect | Focus on layers 24–27, use k=3 |
| R4 | State RQs | 3 RQs, falsifiable hypotheses | (Slightly misframed RQ1 as mechanistic) |
| R5 | Design study | 4 working hypotheses, mixed obs+intervention design | Locked in patching grid (with the bad source choice) |
| R6 | Run study | **H2 confirmed with a monotone depth gradient**; H4 null with a clean diagnosis of *why* | Report 7 fixes: clean k=0 source, prompt-level H3 test, k=10 re-run |

**Highest-value finding so far.** The H2 monotone gradient (layer-wise Δ projection: 1.10 → 1.99 → 3.78 → 4.91) is the project's strongest empirical contribution. It upgrades a Report-3 anecdote ("layer 27 looks weirdly negative") into a measured, ordered relationship across four layers under controlled conditions.

**Methodologically strongest move.** Writing down expectations before running experiments in Report 3 — that's what made the late-layer inversion *count* as a surprise rather than a post-hoc observation.

**Single biggest unresolved threat.** The patching design uses an attacked source. Until Report 7 swaps in a k=0 baseline source, H4 (restoration) and H5 (layer specificity of patching) are not fairly evaluable, and the information-loss vs. active-suppression question stays open.

**Recurring lesson.** Causal ≠ mechanistic. Most of this project's research questions are causal questions whose answers inform mechanism — Report 4's grader was right, and Report 6 absorbs that distinction correctly in its framing.
