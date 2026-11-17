# Proposed condensation plan

## Assessment of the current draft

The draft already has the right two-contribution structure: problem setup →
matrix-free rSVD of the forward operator and weights → DLRA-CG, restarts and
DLRA-PCG → experiments. It still reads like the thesis: seven numbered sections
in six body files, roughly 13,800 whitespace-delimited words in those files
(including TeX), 32 figure wrappers, nine tables and six included algorithms.
Discussion and Conclusion are separate sections in `paper/sections/6_conclusion.tex`.

The largest opportunities are the textbook material in Problem setup, repeated
KLS steps in three algorithms, repeated cost derivations in Methods and
Discussion, and the experiment-by-experiment retelling in Discussion and
Conclusion. Several numerical studies have both a full figure and a table of
selected values. Large multirow figures and explanatory schematics also occupy
substantial space. The rSVD side has many secondary investigations; the DLRA
side needs room for its failures and qualifications as well as its speedups.

Recommended emphasis: retain the existing structure initially, shorten shared
background, make the two methods reproducible, and organize results around
cost versus approximation quality and convergence versus reconstruction quality.
No journal or page limit is assumed. A later venue decision chiefly affects
appendix/supplement availability, the number of secondary studies in the main
text, and figure layout; it need not delay the first prose reductions.

## Status and standing decisions

**Approved scope, not approval of the proposals below:** summarize the thesis;
retain both contributions, restarts and preconditioning; introduce no new
methods or theory; reuse existing experiments; preserve the thesis metric
definitions. Whole-section or result removals require user review. The
transformed formulation is currently out of scope as a solver.

**Initial session:** planning only; the initial plan left paper and code unchanged.
**Current authorization:** remove the rSVD-on-K* variant and related content in
small batches, with user review before every commit. All other unchecked items
below remain proposals, including
items described as “retain.” **REVIEW** marks a proposed whole-subsection,
result, algorithm or figure removal/relocation that needs an explicit decision.
Relocation is not deletion, and an appendix is not yet approved.

**Review basis (17 November 2026):** abstract and all six body files; all
`paper/algorithms/*.tex`, `paper/figures/*.tex`, `paper/tables/*.tex`; visual
inspection of the raster assets; `docs/PROBLEM_SETUP.md`, with its stale scope
overridden by `AGENTS.md`. Thesis checks in `thesis/sections/5_results.tex`
confirm the transformed noisy-data experiment and the local-minimum account;
`thesis/sections/appendix.tex` confirms metric definitions.

| Decision / progress | Status |
|---|---|
| Initial inventory and proposed plan | Complete |
| Removal of rSVD-on-K* variant and related content | Approved in conversation, 17 November 2026; review required before every commit |
| Noisy-data scope exception or removal | Pending D1 below |
| Placement of other secondary studies and illustrations | Pending D2 below; K* removal supersedes its earlier placement proposals |
| First editing batch | Adjoint-variant Discussion/Conclusion cuts reviewed and approved for commit |

Record subsequent approvals here with date, affected items and any conditions;
check off editing items only after the corresponding reviewed edit is complete.

## Active removal map: rSVD on the adjoint

**Decision (17 November 2026):** remove the alternative that approximates K*
first, because its added algorithm/recovery machinery and mixed results do not
justify the space. Keep the definition and application of K* in the primary
matrix-free rSVD of K. This decision supersedes the earlier retain/relocate
proposals for the K* variant below, including the K* part of D2 and batch C.
Keep the K-only singular-value study. No other result removal is authorized.

The batches run from dependent discussion to experiments to method so each
intermediate commit can build without dangling references. Stop for user review
after each batch; commit only after approval, then prepare the next batch.

| Batch | Exact changes | Status |
|---|---|---|
| 1 — Discussion and Conclusion | In `paper/sections/6_conclusion.tex`, remove “Overhead of the adjoint formulation” and “Reconstruction error using the adjoint operator” (including the `true_singular_values` figure input). Remove the alternative algorithm reference beside `eq:rsvd_error_bound`, the final K* comparison sentence in “Estimation of singular values,” and the adjoint-specific limitation and future-work sentences in Conclusion. | Reviewed and approved for commit (17 November 2026) |
| 2 — Experimental comparisons | In `paper/sections/5_experiments.tex`, remove “Computational time of adjoint operator” and “Reconstruction error using the adjoint operator.” In “Approximation of singular values,” retain the K-only findings and 0.969 value, removing K* comparisons and 0.929. Simplify `paper/tables/singular_values_comparison.tex` to K rows, removing the redundant method column and updating its caption. | Not started |
| 3 — Method and unused assets | Remove “Approximating the adjoint operator,” its algorithm input and “Recovering the SVD” from `paper/sections/3_rsvd.tex`. Change “Both algorithms” and the M-factorization/solve alternatives in “Initial matrix factorizations” to the requirements of the K algorithm. Remove “and its adjoint variant” from the Introduction roadmap. Delete the exclusive files listed below once no live inputs/references remain. | Not started |

Exclusive files to remove in batch 3:

- `paper/algorithms/matrix-free-rsvd-Kstar.tex`.
- `paper/tables/computational_time_adjoint.tex` and `rsvd_adjoint_error.tex`.
- `paper/figures/rsvd_adjoint_error.tex`, `clipped_solutions.tex`, and
  `true_singular_values.tex`.
- `paper/images/rsvd_adjoint_error_I.png`, `rsvd_adjoint_error_II.png`,
  `rsvd_adjoint_error_III.png`, `clipped_solution_K.png`,
  `clipped_solution_K_star.png`, and `true_singular_values.png`.

Preserve `paper/sections/2_problem_setup.tex`'s adjoint definition/verification,
`paper/algorithms/matrix-free-rsvd.tex`, the primary forward/adjoint solve cost
analysis and breakdown, all DLRA material, and all thesis/code files. The
unincluded `en-weights.tex` is unrelated to this compiled-paper cut.

Batch 1 validation: `latexmk main.tex` passed (48-page PDF); no undefined
references/citations or BibTeX warnings. Caption-class and layout warnings remain.
`git diff --check` passed. Only the Discussion/Conclusion source and this plan
were edited; asset deletion is deferred to batch 3.

Validation per batch: build with `latexmk main.tex` in `paper/`; inspect warnings
and the diff. At the final batch, search live TeX for retired labels
`alg:discrete_rsvd_Kstar`, `sec:reconvering_the_svd`,
`sec:reconstruction_error_using_the_adjoint_operator`, and removed figure/table
labels, distinguishing legitimate uses of the adjoint from the retired variant.

## Section-by-section checklist

Paths in the following sections are relative to the repository. In the result
tables, `F/name` means `paper/figures/name.tex`, and `T/name` means
`paper/tables/name.tex`.

### Abstract — `paper/main.tex`

**Purpose:** state the two computational contributions and their demonstrated
benefits, without promising universal sparse recovery.

- [ ] Compress the five paragraphs into one: problem and weighted-regularization
  bottleneck; matrix-free rank-k factorization and weights; DLRA-CG with
  restart/preconditioning variants; principal observed benefits and caveat.
  Remove repeated motivation and the final restatement.
- [ ] Qualify “large-scale sparse source identification,” “no significant loss,”
  and “low-rank solutions [are] more natural.” Evidence is from simple 2D
  examples, with DLRA on square-grid Problems I–II. Sparsity alone does not
  establish the useful small matrix rank demonstrated by these sources.
- [ ] Link any numerical headline to the retained result: rSVD timing and
  rank-quality study; DLRA threshold counts and rank-quality comparisons.
  Preserve the EMD exception and the need for restarting in the paper, even
  if the abstract only gives a short qualification. Do not imply that large-N
  rSVD timings establish large-N end-to-end DLRA performance.

### 1. Introduction — `paper/sections/1_introduction.tex` (`sec:intro`)

**Purpose:** motivate operator compression and low-rank reconstruction as
distinct, connected contributions; identify what is adapted from prior work.

- [ ] Shorten broad PDE applications and the disease-specific EEG discussion.
  Keep one focused source-identification motivation and the relevant inverse
  problem references. Combine the repeated boundary-bias/smoothing explanation.
- [ ] Retain the two contributions and independent ranks k and r. Condense the
  following method summaries to one paragraph each; mention restarts and
  preconditioning with their empirical tradeoffs. Retain Elvetun–Nielsen,
  randomized-SVD, DLRA-integrator and CG attribution at their points of use.
- [ ] Propose moving `eq:pde` and `eq:problem` to the start of Problem setup,
  leaving a verbal formulation here. Keep their labels and update all forward
  references. This avoids defining the same problem twice.
- [ ] Narrow the final generality paragraph to applicability assumptions, not
  demonstrated performance on other operators. Evidence comes from §5, not
  new application claims; the roadmap must follow any approved relocations.

### 2. Problem setup — `paper/sections/2_problem_setup.tex` (`sec:setup`)

**Purpose:** supply exactly the operator, inner products, weights and objective
needed by both contributions.

- [ ] Merge “The forward operator” / “Ill-posedness” with a shortened “Finite
  element discretization.” Retain the PDE assumptions, weak form, piecewise
  linear space, N and N_b, `eq:matrix_system`, `eq:discrete_K`, and both mass
  norms. Shorten definitions of Hilbert spaces, hat functions, and repeated
  coefficient expansions. Retain the distinction between nonuniqueness and
  noise sensitivity with the existing regularization references.
- [ ] Keep “The adjoint operator,” especially `eq:K_star_def` and
  `eq:discrete_K_star`, and the short inline verification. Compress the general
  existence discussion. The weighted adjoint is not the Euclidean transpose;
  §3's projection identity depends on these mass matrices.
- [ ] **REVIEW:** absorb the standalone “Tikhonov regularization” subsection
  into “The discrete inverse problem” (`sec:the_discrete_inverse_problem`),
  retaining `eq:discrete_inverse_problem` and the positivity assumptions.
  The generic A-problem and its separate normal equations duplicate §4.
- [ ] **REVIEW:** replace the standalone “SVD and low-rank approximations”
  tutorial and its subheadings with compact notation: thin/truncated SVD,
  pseudoinverse and projection used in `eq:weights`, and the tSVD reference.
  Move “Randomized SVD” into §3; define the condition number briefly where
  preconditioning uses it. Retarget `sec:tsvd`, `sec:the_pseudoinverse` and
  any equation references before retiring headings. Keep appropriate
  Golub–Van Loan, Eckart–Young, Hansen and Halko citations with retained facts.
- [ ] Shorten “Regularization weights” to the localization-bias motivation,
  `eq:weights`, and its truncated version. Preserve M-normalization and explain
  how the approximate right singular vectors supply W. Do not turn the
  orthonormal-basis motivation into a claim that normalized FE hats are
  mutually orthogonal. Keep Elvetun–Nielsen attribution.
- [ ] **REVIEW / D1:** “A transformed formulation”
  (`eq:transformed_discrete_inverse_problem`) cannot be silently reduced to
  motivation while §5 still solves it on noisy data. Decide that experiment's
  status first; preserve its definition wherever the experiment is retained.
- [ ] **REVIEW:** remove the pedagogical `F/forward_operator` and
  `F/triangulation` from the paper; the equations and test-problem figure cover
  their purpose. Retain `F/standard_tikhonov_solution` initially as evidence
  of the bias being corrected; later consider a shared motivation panel,
  without losing its parameter setting or confusing it with weighted results.

### 3. Matrix-free rSVD — `paper/sections/3_rsvd.tex` (`sec:rsvd`)

**Purpose:** show how PDE solves replace explicit K construction and deliver
the factorization used for W and downstream optimization.

- [ ] Retain “A matrix-free rSVD scheme” (`sec:a_matrix_free_rsvd_scheme`),
  `eq:rsvd_relation`, k/p/l, dimensions, truncation and
  `paper/algorithms/matrix-free-rsvd.tex`. Shorten the column-by-column
  narrative and repeated range explanation. Make the connection to
  `eq:weights` explicit rather than leaving the reader to assemble it.
- [ ] **REVIEW:** absorb `paper/algorithms/rsvd.tex` into a brief standard-rSVD
  introduction; keep the matrix-free algorithm as the full specification.
  Retarget `alg:rsvd` references. Standard rSVD can use operator actions;
  describe the PDE-specific realization rather than claiming B inherently
  requires storing K. Do not add the currently unincluded
  `paper/algorithms/en-weights.tex`: it is a different weight-estimation scheme.
- [ ] Retain “Approximating the adjoint operator” and “Recovering the SVD”
  (`sec:reconvering_the_svd`) in compressed form initially. The QR/core-SVD
  recovery is necessary: mass-scaled factors are not automatically an SVD.
  **REVIEW:** express `matrix-free-rsvd-Kstar.tex` as substitutions in the
  primary algorithm, or relocate the full variant with its secondary study.
  Do not remove the K* comparison while retaining claims of its inferiority.
- [ ] Merge “Initial matrix factorizations” with Discussion's “Asymptotic
  analysis” and “Overhead of the adjoint formulation.” Keep `eq:rsvd_cost`
  once here: setup, solves, small SVD; separate 2D asymptotics from observed
  near-linear timing. State which mass factorizations each variant needs.
  Retain sparse-factorization and dense-linear-algebra cost citations.
- [ ] Retain spectral-decay evidence (`F/singular_values`) initially.
  **REVIEW:** omit `F/rsvd`'s geometric tutorial. Timing evidence must retain
  setup and k-dependence; approximation evidence must include small-k failure,
  oversampling variability, and the larger EMD discrepancy on Problem III.

### 4. DLRA-CG — `paper/sections/4_dlra_cg.tex` (`sec:dlracg`)

**Purpose:** specify the low-rank iteration, distinguish its CG recurrence
from the actual iterate gradient, and explain restarts and preconditioning.

- [ ] Move “Application to the inverse problem”
  (`sec:application_to_inverse_problem`) before step-size rules: define Phi,
  gradient, H and normal equations before using them. Retain lambda > 0 and
  positive-weight conditions for SPD, vec/mat ordering and k versus r.
  State the square-grid scope of the source matrix; Problem III supports rSVD,
  not this DLRA representation. Avoid implying all search directions are
  stored in factored form or that a new fully factored solver was tested.
- [ ] Shorten “The low-rank manifold” history to the projection idea and the
  basis-update/Galerkin integrator actually used. Keep Koch–Lubich,
  Ceruti–Lubich and Schotthöfer attribution. Distinguish the motivating tangent
  update from the implemented augmented KLS step and truncation.
- [ ] **REVIEW:** consolidate `paper/algorithms/DLRA.tex`, `DLRA-CG.tex`, and
  `DLRA-PCG.tex`: specify KLS once, then a CG/PCG algorithm calling it, with
  P=I giving CG and explicit restart branches. Preserve the baseline descent
  rule, recurrence, truncation and initialization. Update all algorithm and
  line-number references (including the current “line 10” rank-adaptation
  reference). This removes repetition, not a method.
- [ ] Keep “Step-size selection and rank adaptation” brief: define the three
  tested baselines, the steepest-descent formula and fixed maximum rank.
  Keep adaptive truncation as an available but untested option, not a result.
  Move reproducibility settings to §5; shorten neural-network optimizer history.
- [ ] Retain “Restarted CG” (`sec:restarted_cg`) and `eq:full_rank_iterate` as
  central. Merge Discussion's repeated drift explanation here. Specify true
  gradient recomputation and direction reset, including Z for PCG. Check the
  current inference “G tends to zero, so alpha tends to zero” before retaining
  it; it is not justified by that fact alone. No new convergence theorem is
  proposed. A full-CG line search is not a guarantee after KLS/truncation.
- [ ] Retain all four choices in “Choice of preconditioner”
  (`sec:choice_of_preconditioner`) in a compact formula table: Jacobi, IC of
  S, IC–Woodbury and exact S–Woodbury. Keep H=F+S and the Woodbury formula,
  but shorten the rank/SPD derivation. Perfect is a diagnostic reference;
  its one-step full-rank property does not guarantee a good truncated iterate.
- [ ] **REVIEW:** combine or omit `F/low_rank_source`, `F/low_rank_manifold`
  and Discussion's `F/cg_direction_disagreement`; favor the last if one
  schematic is kept because it explains the distinctive restart issue.
  Retain empirical support in the convergence and rank studies. Do not equate
  number of sources with matrix rank in general.

### 5. Numerical experiments — `paper/sections/5_experiments.tex` (`sec:experiments`)

**Purpose:** establish both benefits and their limits, with enough settings to
interpret existing evidence. Keep the rSVD and DLRA experiment groups.

- [ ] Merge “Experimental setup,” “Computational setup” and “Performance
  metrics” into a compact setup subsection. Keep `F/problem_illustration`,
  domains, coefficients, mesh sizes, lambda, k/p, initialization, hardware,
  software and source-code reference. Preserve experiment-specific repeat
  counts and mean/standard-deviation versus median conventions in captions.
- [ ] Compress metric explanations without changing definitions: absolute
  coefficient normalization and physical-node EMD distances; AUC-IoU on [0,1].
  Explicitly distinguish raw Euclidean error, relative error, metric-relative
  differences and e^(i), which is normalized by initial reconstruction error.
  AUC-IoU is higher-is-better. The known metric-code mismatch and unregenerated
  thesis results remain a limitation; no recalculation or code change is
  authorized. Do not present these inherited values as newly validated.
- [ ] Recover missing settings from thesis records when editing (e.g. fixed
  step/Adam parameters, rank/initialization treatment, stopping conditions,
  residual definition and timing scope). If unavailable, mark the gap rather
  than invent it. The reported error thresholds use the known true source;
  they are evaluation measures, not deployable stopping criteria.

#### rSVD results: proposed disposition

| Current subsection / label | Proposed action and evidence to preserve |
|---|---|
| Computational time; Computational breakdown (`sec:computational_time`, `sec:computational_breakdown`) | Merge into one cost study. Retain `F/rsvd_computational_time_comp` and `F/rsvd_computational_breakdown`; **REVIEW:** replace the same-named tables with selected ratios in prose. Preserve dense assembly plus SVD comparator, medians/repetitions, setup cost and increasing small-SVD share at large k. Near-linear behavior is measured over the tested range, not the asymptotic total cost. |
| Target rank effect on reconstruction quality (`sec:target_rank_reconstruction`) | Retain `F/test_problems_selected`, `F/rsvd_over_k` and a compact selection from `T/rsvd_over_k`; shorten the panel-by-panel narration. **REVIEW:** reduce repeated panels/rows only after choosing the retained comparisons. Keep all three problems represented, small-k smoothing/shifts, p-dependence and uncertainty. At k=50, Euclidean metric differences are 0.56–2.92%, AUC-IoU 1.97–4.07%, but EMD reaches 39.13%; these are differences in error metrics, not norms of differences between solutions. |
| Test vector distribution effect on error (`sec:test_vector_distribution_effect_on_error`) | **REVIEW:** move this secondary study to an appendix, or retain one short comparison with `F/random_vectors_error`. `F/random_test_vectors` and `F/clipped_random_vectors` can be omitted only with approval. Preserve Gaussian/Rademacher similarity and Peak's EMD/variance benefit; do not promote the smooth-singular-vector explanation to a proved mechanism. |
| Mesh resolution effect (`sec:mesh_resolution_effect_on_reconstruction_error`) | Retain `F/mesh_resolution_effect` and a short paragraph: fixed-k behavior is observed for Problem I and the plotted mesh range. It does not establish universal rank independence or that refinement never adds information. This evidence supports qualified coarse-to-fine rank reuse. |
| Approximation of singular values (`sec:approximation_of_singular_values`) | **REVIEW:** merge into the adjoint comparison or relocate with it. Preserve `T/singular_values_comparison`'s dependence on p and singular-value index, including worse high-index estimates. Singular-value accuracy alone does not certify W or reconstruction accuracy. |
| Computational time of adjoint operator; Reconstruction error using the adjoint operator (`sec:reconstruction_error_using_the_adjoint_operator`) | Merge into one comparison: `T/computational_time_adjoint`, `F/rsvd_adjoint_error`, `T/rsvd_adjoint_error`, `F/clipped_solutions`. **REVIEW:** keep a compact table/figure selection in the main text and relocate remaining diagnostics. Preserve approximately 30% overhead including SVD recovery, mixed quality, and low-k EMD reversals. Table values at k=10 span roughly 6.5–19%, so avoid repeating the prose's narrower 10–19% range. |
| Accuracy on noisy data (`sec:accuracy_on_noisy_data`) | **REVIEW / D1:** recommend excluding this solver experiment under current scope, only after approval. The thesis confirms it uses the transformed objective. Alternative: explicitly approve its retention as a separate supplementary study. Dependencies are `T/noisy_problem`, `F/noisy_problem`, `F/reconstruction_noisy_problem`, the transformed objective, tSVD definitions, Discussion's noise analysis and Conclusion's noise claim. If retained, keep 1% noise, discrepancy safety factor 10, method/rank-specific lambda, 50 runs, metric-specific optimal ranks and III's optimum outside the tested range. It supplies no DLRA noise-robustness evidence. |

#### DLRA results: proposed disposition

| Current subsection / label | Proposed action and evidence to preserve |
|---|---|
| Rate of convergence (`sec:rate_of_convergence`) | Retain `F/number_of_iterations` and `T/convergence_iterations`; curve and thresholds have distinct roles. Shorten narration. Keep baseline failures, semi-convergence and restart-period tradeoff. The approximately 210-fold claim is 20,916 versus 100 iterations to e<0.85 on Problem I; qualify runtime using the reported per-step times rather than calling it a universal speedup. |
| Quality of the low-rank solutions (`sec:quality_of_the_low_rank_solutions`) | Retain comparison to full-rank weighted Tikhonov, `F/low_rank_convergence` and `F/low_rank_restarted_cg`. **REVIEW:** combine selected panels, retaining early/minimum/final distinction and restarted outcome. Keep amplitude deficits, shifted final iterates and checker-like artifacts. The minimum-error iterate uses ground truth; it is not an automatic output-selection rule. |
| Effect of rank on reconstructions (`sec:effect_of_rank_on_reconstructions`) | Retain `F/low_rank_different_r` and `F/low_rank_solution_error`, with shorter prose. Preserve I's EMD disadvantage, II's useful rank 2 despite true rank 3, false sources at too-small rank, and return toward full-rank behavior at r=5. Keep exact-K and rank-k full-rank baselines distinct and the 30-run uncertainty. |
| Residuals of low-rank solutions (`sec:residuals_of_low_rank_solutions`) | Merge with rank sensitivity, retaining `F/low_rank_residual` initially. The corner helps on II but not I; do not describe it as an established rank-selection algorithm. **REVIEW:** optional appendix relocation must retain the negative finding in the main discussion. |
| Rate of convergence using preconditioners (`sec:rate_of_convergence_using_preconditioners`) | Retain `F/number_of_iterations_preconditioners` and `T/convergence_iterations_precond`, shortening repeated count lists. Keep setup/per-step costs, strong-preconditioner stalling, Jacobi's limited benefit, IC plus restarting, and IC–Woodbury instability. The roughly eightfold runtime gain concerns the reported comparison at e<0.75; distinguish that from other thresholds and state whether setup is included. |
| Local minima (`sec:local_minima`) | Merge into preconditioning failures, retaining `F/low_rank_diverged_solutions` initially. Describe observed suboptimal/stagnating or unstable iterates, not a proved local minimum or unbounded divergence. The text says Problem II's “left panel,” but the caption places II on the right; the thesis repeats this mismatch. Clarify which restart experiment resolved trapping versus which caused instability before compressing the account. Do not erase the failure because its mechanism is unresolved. |

### 6. Discussion — `paper/sections/6_conclusion.tex` (`sec:conclusions`)

**Purpose:** interpret tradeoffs across the two contributions, separate evidence
from explanation, and state limitations without repeating the results catalogue.

- [ ] Consolidate the eight rSVD subsubsections. Move “Asymptotic analysis” and
  “Overhead of the adjoint formulation” to §3; combine “Analysis of
  reconstruction error,” “Independence from mesh resolution” and “Estimation
  of singular values” into a short qualified interpretation. Keep the
  distinction between operator approximation and reconstruction quality.
- [ ] Before retaining `eq:rsvd_error_bound`, check the cited Halko theorem's
  assumptions (including p>1), projection versus final rank-k truncation,
  and whether the same bound applies after mass-scaled K* recovery. It is not
  a reconstruction-error or weight-error theorem. **REVIEW:** replace the
  partitioned-SVD derivation (`eq:rsvd_partitioned_1`) with a precise citation
  if it adds little beyond §3; do not introduce a new proof here.
- [ ] Merge “Reconstruction error using the adjoint operator” with the §5
  comparison, retaining `F/true_singular_values` only if its spectral
  explanation remains. **REVIEW:** move “Effect of the test vector
  distribution” and `F/singular_vectors_of_K` with the optional sampling study.
  “Reconstructions under noisy data,” its Picard expansion and Hansen citation
  follow D1; do not leave a noise-performance claim after its evidence is cut.
- [ ] Condense “Convergence of DLRA-CG” (`sec:convergence_of_DLRA_CG`) to the
  curvature motivation and explicit lack of a convergence theory under
  truncation. **REVIEW:** replace the full-CG bound `eq:convergence_of_CG`
  with a citation or short comparison. It bounds unconstrained optimization
  error in H-norm, not e^(i) against the true source. Keep any condition-number
  examples only with their problem/parameter context.
- [ ] Move the duplicated trajectory mechanism in “Semi-convergence and
  restarting” to §4; retain interpretation of the observed restart tradeoff.
  Merge “Rank as a regularization parameter”
  (`sec:rank_as_a_regularization_parameter`) with a concise discussion of rank
  sensitivity. **REVIEW:** omit `F/outer_product_II` if the rank-3 source/rank-2
  reconstruction distinction is explained in text; its SVD components should
  not be conflated with individual localized sources.
- [ ] Keep “Effect of preconditioning” as a limitation: faster full-rank
  recurrence need not improve the truncated reconstruction. Preserve the
  unresolved IC–Woodbury behavior; avoid a causal local-minimum claim.
  Bring general limitations here from Conclusion: simple 2D source geometries,
  untested broader applicability, manual k/r/restart choices, unavailable
  practical stopping rule, and inherited metric-evidence caveat.

### 7. Conclusion — `paper/sections/6_conclusion.tex` (`sec:conclusion`)

**Purpose:** close with the two demonstrated contributions and their limits.

- [ ] Reduce to two or three short paragraphs: matrix-free factorization and
  tested cost/quality tradeoff; DLRA-CG benefit with restart/preconditioning
  qualifications; a small set of existing open questions.
- [ ] Remove repeated PDE motivation, metric-by-metric adjoint numbers and
  final slogan. Move detailed limitations to Discussion. Any retained headline
  must match §5's settings and exceptions; change “semi-convergence ... solved”
  to the tested restart behavior. Noise wording depends on D1.
- [ ] Shorten existing future-work lists to rank/stopping/restart selection,
  convergence under truncation/preconditioning and broader validation. Power
  iteration, adaptive rank and 3D applications remain possibilities, not
  contributions or authorized work. Keep both top-level Discussion and
  Conclusion initially; **REVIEW:** merging/removing either whole section is
  an optional later venue-dependent decision.

## Cross-file dependencies and completion checks

- [ ] For every approved move/cut, update prose, labels, algorithm numbers,
  captions, introductory roadmap and conclusion claims together. Keep
  citations supporting retained facts; do not prune `references.bib` merely
  because a citation disappears. Preserve BibTeX/`siamplain` and template files.
- [ ] Use existing figure assets; preserve axes, color scales, ground-truth
  outlines and uncertainty when proposing smaller layouts. Do not compress
  multipanel plots until legends become unreadable. All paper assets must
  remain self-contained under `paper/`, including any future appendix assets.
- [ ] Unincluded `en-weights.tex`, `images/forward_illustration.png` and
  `images/low_rank_local_minima.png` are not sources of compiled-paper length.
  Do not add or delete them as part of condensation by default.
- [ ] After each approved TeX batch, build with `cd paper && latexmk main.tex`,
  check references/citations and inspect affected pages. Fine typography comes
  after content decisions, using `MIGRATION.md`. No build is needed for this
  plan-only change.

## Decisions to settle first and editing sequence

1. **D1 — Noisy-data scope:** approve removal of the transformed solver study
   and dependent claims, or explicitly allow it as a separately identified
   supplementary experiment. It cannot stand as evidence for the ordinary
   objective or DLRA-CG.
2. **D2 — Secondary evidence:** prefer short main-text summaries plus an
   appendix for sampling distributions and the expanded K* diagnostics, or
   keep them in the main text for now? Decide individual result removals
   separately; no blanket approval is implied. Venue rules can settle the
   final destination later.
3. **D3 — First batch:** review the Problem setup checklist together, especially
   the proposed merging of generic Tikhonov/SVD background and the two
   pedagogical figure cuts. Keep the weights, mass norms and adjoint identity
   explicit. This offers a substantial reduction without selecting winners
   among the experimental results.

Recommended small, separately reviewable edits after those decisions:

- [ ] A. Condense Problem setup's PDE/FE exposition; move the opening equations
  if approved. Review the mathematical definitions before proceeding.
- [ ] B. Condense its Tikhonov/SVD/weight background and enact only approved
  figure/heading cuts; handle the transformed formulation according to D1.
- [ ] C. Consolidate §3's rSVD description and cost analysis, then separately
  review treatment of the K* variant and its algorithm.
- [ ] D. Reorder §4's objective and baseline description; review the shared KLS
  specification. In a separate batch consolidate CG/PCG and explicit restarts.
- [ ] E. Condense experimental setup and the core rSVD cost/quality study;
  review secondary studies one at a time under D1/D2.
- [ ] F. Condense DLRA convergence/quality/rank results; separately review
  preconditioning and failure evidence, preserving all qualifications.
- [ ] G. Condense Discussion and then Conclusion against the retained evidence.
  Rewrite Introduction and Abstract last so their claims match the final body.
- [ ] H. Check cross-references, citations, float readability and claim/evidence
  consistency; update this decision/progress log. Leave all edits uncommitted
  until explicitly asked to commit.
