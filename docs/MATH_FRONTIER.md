# Math frontier  -  what we could revolutionize via primitives

This is the **deep-research memo** companion to `INNOVATION_SWEEP.md`.
Where `INNOVATION_SWEEP.md` is a rolling product-feature ledger, this
file is a research catalog of frontier math that could become
vyre primitives  -  graded by impact, lego-fit, plausible consumers, and
novelty. It is not a release checklist.

## Architecture thesis (the load-bearing idea)

Every primitive we ship is **dual-use**: it must clear the lego-block
discovery checklist (≥ 2 user-dialect consumers, no domain glue) AND
identify ≥ 1 *vyre-self consumer*  -  somewhere `vyre-foundation::transform`,
`vyre-driver-*`, or `xtask` will dispatch the same Program against vyre's
own IR / dispatch graph / cost model. The recursion compounds the moat:
math we ship as ops becomes math we use to compile ops. Primitives that
fail the self-consumer test stay in `vyre-libs` as Tier-3 dialects until
a self-use materializes.

This document is graded on that bar. Items that ship to user dialects
without a clear self-use are flagged.

## Round 1 recap (tasks #1-#18, already on the list)

Captured in TaskList #1-#18 + dual-use substrate uplifts in #19-#23, #26-#30.
Headline picks: semiring_gemm (#1), sinkhorn (#2), randomized_svd (#3),
ntt (#4), chebyshev_filter (#5), tensor_train (#6), differentiable_argmax/sort (#7),
clifford_product (#8), homotopy_continuation (#9), sum_product_circuit (#10),
planar_rewrite (#11), effects-handler lowering (#12), VSA (#13), SOS (#14),
persistent homology (#15), KFAC/Shampoo (#16), spectral_shape (#17),
linear-logic typed BufferAccess (#18).

Round 2 below extends to deeper frontier candidates not covered in the
round-1 sweep. Each item is tagged:

- **Impact**: workspace-wide, a-class workload, niche
- **Lego-fit**: clean / glue-y / IR-construct
- **Self-consumer**: which vyre-internal pass uses the same Program
- **Novelty**: not-yet-packaged / known-but-unpackaged / actively explored
- **Dispatch**: shipped / source-change required / research-only

---

## ROUND 2  -  frontier candidates

### Group I  -  Geometric deep learning beyond Clifford

#### #31 sheaf::laplacian + sheaf::diffusion  -  sheaf neural networks

Sheaf NNs (Bodnar-Di Giovanni 2022, Hansen-Gebhart 2023) generalize
GNNs from "all nodes share one feature space" to "each node carries
its own vector space + restriction maps to neighbors." Captures
heterophilic graph structure that vanilla GNNs fail on. Sheaf
Laplacian = block matrix where the (i,j) block is the restriction
composition F_{i→ij}^T F_{j→ij}.

- **Impact**: a-class (heterophilic graphs are everywhere  -  code,
  social, biology). Could replace GNN as the substrate for security
  call-graph analysis in vyre-libs::security.
- **Lego-fit**: clean. Sheaf Laplacian = sparse block matmul
  (composes from #1 semiring_gemm with block-aware variant).
- **Self-consumer**: vyre's own dispatch graph is heterophilic
  (compute-heavy node next to memory-bound node next to control-flow
  node)  -  sheaf diffusion on the dispatch graph predicts where
  fusion fails.
- **Novelty**: actively explored, no GPU primitive shipped.
- **Dispatch**: source-change required; blocked on block-sparse semiring_gemm.

#### #32 simplicial::message_passing  -  SCNN over simplicial complexes

Simplicial neural networks (Bodnar-Frasca 2021, Yang-Sala 2023)
generalize GNNs from edges to higher-order simplices (triangles,
tetrahedra, …). The boundary operator ∂ becomes the substrate.
Substrate for: hypergraph learning, mesh processing, topological
signatures of code (basic blocks → loops → loop-of-loops).

- **Impact**: a-class for mesh / topology workloads.
- **Lego-fit**: clean. Boundary operator = sparse matrix; messages
  = semiring matmul. Composes from #1 + #15 persistent homology.
- **Self-consumer**: same dispatch-graph-as-simplicial-complex view
  as #31; higher-order interactions captured (3-way conflict between
  sibling Programs sharing a buffer).
- **Novelty**: not-yet-packaged.
- **Dispatch**: source-change required.

#### #33 equivariant::tfn_block  -  tensor field networks for SE(3) data

Tensor field networks (Thomas 2018, Geiger 2022 e3nn) are the
SE(3)-equivariant building block for molecules + 3D vision. Wigner
D-matrices + Clebsch-Gordan tensor products. Each layer = sparse
contraction over CG coefficients  -  pure GPU shape but shipped
nowhere as a Tier-2.5 primitive.

- **Impact**: a-class (molecular dynamics, cryo-EM, robotics).
- **Lego-fit**: clean after #8 clifford_product source work; Wigner D and
  CG products are special cases.
- **Self-consumer**: weak  -  equivariance is workload-specific. Likely
  user-dialect-only.
- **Novelty**: known-but-unpackaged (e3nn is Python; no GPU-IR
  primitive).
- **Dispatch**: source-change required, blocked on #8.

### Group II  -  Quantum-inspired classical algorithms

#### #34 qsvt::block_encoding + qsvt::polynomial_transform

Quantum singular value transform (Gilyen-Su-Low-Wiebe 2018) generalizes
quantum phase estimation, Hamiltonian simulation, matrix inversion
(HHL), and amplitude amplification into one framework. Classical
simulation of QSVT on classical hardware (Tang 2019, "dequantizing")
showed many quantum advantages were illusory  -  but the *algorithm
structure* (block-encoding + Chebyshev polynomial of singular values)
is GPU-shaped and underexploited classically.

- **Impact**: a-class. Block encoding + Chebyshev-of-SVD gives a
  unified primitive for matrix functions: matrix inverse, matrix sqrt,
  matrix exp, all without eigendecomposition.
- **Lego-fit**: clean. Composes from #5 chebyshev_filter on the
  singular-value spectrum + #3 randomized_svd for the spectrum
  estimate.
- **Self-consumer**: vyre's optimizer needs matrix exp / inverse for
  transport-based fusion analyses (Wasserstein over dispatch graphs).
- **Novelty**: not-yet-packaged classically as a GPU primitive.
- **Dispatch**: source-change required, blocked on #3 + #5.

#### #35 tensor_network::contract  -  string-diagram tensor network compiler

PEPS / MPS / MERA tensor networks compress functions over high-dim
exponentially. Recent work (Khoromskij 2018, Glasser 2020) makes
GPU contraction tractable. Contraction order is the hard part  - 
solved via tropical-semiring shortest-path (#1 again).

- **Impact**: a-class. Tensor networks are the substrate for everything
  from quantum chemistry to graphical models to compressed transformers.
- **Lego-fit**: clean. Same shape as #6 tensor_train, generalized.
- **Self-consumer**: vyre's IR analysis as tensor-network contraction
  problem (each Region is a tensor; wires are buffer dependencies;
  optimal fusion = optimal contraction order). Self-applied this is
  the megakernel scheduler #22 in disguise.
- **Novelty**: known-but-unpackaged at primitive level.
- **Dispatch**: source-change required, blocked on #6.

### Group III  -  Causal + counterfactual inference

#### #36 causal::do_calculus  -  Pearl do-calculus rule application

Pearl's three rules (insertion/deletion of observations, action/observation
exchange) reduce a do-query to an observable-query when the causal
graph admits it (Shpitser 2008 ID algorithm). Recent advances
(Correa-Bareinboim 2020) extend to identifiability under multiple
treatments. GPU primitive: rule-application as graph-rewrite over
adjacency.

- **Impact**: workspace-wide for any system that needs counterfactuals
  (security: "would this finding still fire under fix X?", ML training:
  "what if we removed example Y?").
- **Lego-fit**: clean. Composes from sparse graph rewriting +
  semiring closure (#1).
- **Self-consumer**: vyre's own change-impact analysis (do(rule_X)
  on the rule dependency graph predicts which downstream Programs
  invalidate). Replaces ad-hoc "what got cached and what didn't."
- **Novelty**: not-yet-packaged.
- **Dispatch**: source-change required.

#### #37 causal::front_door + causal::back_door  -  adjustment set finder

Front-door / back-door adjustment finds the set of variables to
condition on for valid causal estimation. Algorithmic problem
(Tian-Pearl 2002, modified bayes-ball). GPU primitive: parallel
search over candidate adjustment sets, scored by a graph-cut
condition.

- **Impact**: a-class for observational ML.
- **Lego-fit**: clean. Composes from BFS over the causal graph + cut
  scoring.
- **Self-consumer**: weak. Mostly user-dialect.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: source-change required.

### Group IV  -  Probabilistic programming + neuro-symbolic

#### #38 probabilistic::knowledge_compilation

Compile a probabilistic logic program into a tractable circuit
(d-DNNF, sentential decision diagrams). Recent work (Vergari 2021
PROUST, Choi 2024) shows GPU-tractable compilation. The compiled
circuit is then evaluated with #10 sum_product_circuit.

- **Impact**: workspace-wide. Substrate for neuro-symbolic systems.
- **Lego-fit**: glue-y (the compilation step is host-side; the
  evaluation step is the primitive). Split: KC compiler stays in
  vyre-libs, SPN evaluator stays in vyre-primitives.
- **Self-consumer**: vyre's own rule-conflict resolution (when two
  rules disagree, which fires?) as probabilistic logic.
- **Novelty**: actively explored, no GPU substrate.
- **Dispatch**: source-change required, blocked on #10.

#### #39 probabilistic::scallop_join  -  probabilistic Datalog operator

Scallop (Huang 2021) compiles probabilistic Datalog into provenance
semirings. Recent work makes Scallop joins GPU-shaped: each round
of fixpoint iteration = one semiring matmul (#1 semiring_gemm again,
provenance semiring).

- **Impact**: a-class. Substrate for neuro-symbolic reasoning.
- **Lego-fit**: clean. Just #1 with a different semiring.
- **Self-consumer**: vyre's rule provenance tracking (which input
  rule produced which output finding) is a Datalog query.
- **Novelty**: not-yet-packaged on GPU.
- **Dispatch**: ship-now after #1 (it's a semiring choice).

### Group V  -  Privacy + uncertainty

#### #40 privacy::dp_accountant  -  Rényi DP / Gaussian DP accountant

Differential privacy budget tracking with Rényi DP (Mironov 2017) or
Gaussian DP (Dong-Roth-Su 2022) gives composition bounds tighter than
naive ε,δ-composition. Per-step accounting reduces to elementary
function evaluation; GPU vector-friendly.

- **Impact**: a-class for privacy-preserving compute.
- **Lego-fit**: clean. Pure elementwise math.
- **Self-consumer**: weak; mostly user dialect.
- **Novelty**: known-but-unpackaged as a primitive.
- **Dispatch**: ship-now standalone.

#### #41 conformal::quantile + conformal::predict

Conformal prediction (Vovk 2005, Angelopoulos 2023) gives
finite-sample uncertainty intervals with NO distributional assumptions.
Per-prediction operation: split conformal = compute non-conformity
scores, take quantile, expand prediction. Embarrassingly parallel.

- **Impact**: a-class. Adds calibrated uncertainty to ANY model
  without retraining. Should be the default uncertainty primitive.
- **Lego-fit**: clean. Composes from #7 differentiable_argmax (for
  smoothed quantiles) + reduce::sum.
- **Self-consumer**: vyre's dispatch cost model uncertainty (#28) gets
  conformal intervals, not point estimates. Tighter scheduling.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: ship-now after #7.

#### #42 dp::moments_accountant  -  privacy-preserving SGD primitive

Per-sample gradient clipping + Gaussian noise, with Rényi/Gaussian
DP accounting (#40). The primitive is the per-sample-clip itself  - 
naive implementations destroy GPU throughput; recent work (Li-Tramer
2022, GhostClip) shows how to amortize clipping across the batch via
power-iteration approximation.

- **Impact**: a-class for any DP-trained model.
- **Lego-fit**: clean. Per-sample gradient norm via power-iteration on
  the Jacobian-vector product.
- **Self-consumer**: weak.
- **Novelty**: actively explored, no Tier-2.5 primitive.
- **Dispatch**: source-change required.

### Group VI  -  Continuous-time + flow-based ML

#### #43 ode::integrator_step  -  adaptive Runge-Kutta as primitive

Neural ODE (Chen 2018), neural CDE (Kidger 2020), normalizing flows
all reduce to "adaptive ODE step." Dormand-Prince 5(4) is the standard,
but it's a serial loop. Recent work (Massaroli 2020) shows
embarrassingly-parallel multi-shooting variants where each segment is
GPU-independent.

- **Impact**: a-class for continuous-time ML, physics simulation,
  control.
- **Lego-fit**: clean. Composes from #5 chebyshev_filter (for the
  RK stages) + reduce::max (error estimate).
- **Self-consumer**: vyre's homotopy_continuation (#9) IS an ODE
  integrator  -  same primitive, two consumers from day 1.
- **Novelty**: known-but-unpackaged at IR level.
- **Dispatch**: ship-now alongside #9.

#### #44 score::denoise_step  -  score-based generative modeling primitive

Diffusion / flow-matching (Song-Ermon 2020, Lipman 2023) reduce to
"one denoise step." Substrate for: image / audio / 3D generation.
Each step = single network forward + small predictor-corrector.

- **Impact**: a-class for generative.
- **Lego-fit**: clean. Composes from existing nn primitives.
- **Self-consumer**: weak.
- **Novelty**: known, no Tier-2.5 primitive.
- **Dispatch**: ship-now after nn-substrate matures.

### Group VII  -  Combinatorial / discrete-optimization

#### #45 submodular::greedy_max  -  submodular maximization with curvature bounds

Submodular function maximization gives constant-factor approximation
guarantees (Nemhauser 1978). Recent work (Mirzasoleiman 2015 stochastic
greedy, Buchbinder 2020 continuous extensions) makes it GPU-friendly:
sample candidate elements, compute marginal gains in parallel, pick
the max.

- **Impact**: a-class. Substrate for: feature selection, sensor
  placement, summarization, active learning, coreset construction.
- **Lego-fit**: clean. Composes from #7 differentiable_argmax (over
  candidate set) + #1 semiring_gemm (for marginal gain via
  set-function adjacency).
- **Self-consumer**: vyre's compile-cache eviction policy as
  submodular coverage (which K Programs to keep cached to maximize
  expected hit rate).
- **Novelty**: known-but-unpackaged at primitive level.
- **Dispatch**: ship-now.

#### #46 matroid::intersect  -  matroid intersection on GPU

Matroid intersection (Edmonds 1970) finds max independent set in
intersection of two matroids  -  generalizes bipartite matching,
spanning-tree variants, scheduling. Recent breakthroughs (Chakrabarty-
Lee-Sidford 2021) reduce to O(n^2) iterations of solving sparse
linear systems, GPU-amenable.

- **Impact**: a-class for combinatorial scheduling.
- **Lego-fit**: clean after linear-system solver primitive source work.
- **Self-consumer**: vyre's megakernel scheduler (#22)  -  selecting
  which Programs to fuse subject to memory + sync constraints IS a
  matroid intersection problem.
- **Novelty**: not-yet-packaged on GPU.
- **Dispatch**: source-change required.

### Group VIII  -  Compressed / sparse / sketched compute

#### #47 sketch::count_sketch + sketch::leverage_score

Count-sketch (Charikar 2002) and leverage-score sampling (Drineas
2012) give compressed estimators for matrix products, norms,
eigenvalues. Underexploited because deep-learning ate the attention
budget.

- **Impact**: a-class. Substrate for streaming / approximate.
- **Lego-fit**: clean. Composes from existing hash primitives + #3
  randomized_svd.
- **Self-consumer**: vyre's profiler  -  measure dispatch latency
  distribution via count-sketch instead of storing every sample.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: ship-now.

#### #48 sparse::omp_step + sparse::iht_step  -  sparse recovery primitives

Orthogonal matching pursuit + iterative hard thresholding for sparse
recovery (compressed sensing). Each iteration = one matmul + top-k.
Substrate for: signal recovery, model sparsification, pruning.

- **Impact**: a-class.
- **Lego-fit**: clean. Composes from #1 + reduce::top_k (which doesn't
  exist yet  -  promote when ≥ 2 callers want it).
- **Self-consumer**: vyre's sparse-buffer compaction (when a Region's
  output is mostly zero, ship sparse  -  IHT picks the threshold).
- **Novelty**: known-but-unpackaged.
- **Dispatch**: source-change required.

#### #49 sparse_fft::sft  -  sparse FFT (Hassanieh 2012)

Recover a k-sparse signal in n-dim Fourier domain in O(k log n) vs
FFT's O(n log n). Massive speedup for sparse signals; underexploited
because the algorithm is fiddly. Recent work (Indyk 2019) makes the
constants reasonable.

- **Impact**: niche but high. Audio / radio / scientific compute.
- **Lego-fit**: clean if FFT primitive source work is implemented first.
- **Self-consumer**: weak.
- **Novelty**: known, hard to package.
- **Dispatch**: research-only.

### Group IX  -  Physical / numerical analysis

#### #50 amg::v_cycle  -  algebraic multigrid V-cycle

AMG solves elliptic PDEs in O(n) instead of O(n log²n)  -  100×
speedup at scale. Recursive structure historically hard to GPU-fy;
recent work (BoomerAMG 2020, Smoothed Aggregation 2022) cracks it.

- **Impact**: a-class for any PDE / sparse-system solver.
- **Lego-fit**: glue-y. The hierarchy construction is host-side; each
  level's smoothing is GPU-shaped. Split.
- **Self-consumer**: vyre's IR-graph contraction (Region-tree levels
  match V-cycle levels  -  apply AMG smoothing as the dispatch
  scheduling smoother).
- **Novelty**: known, recently unblocked.
- **Dispatch**: source-change required.

#### #51 fmm::p2m + fmm::m2l + fmm::l2p  -  fast multipole evaluation

FMM evaluates n-body sums in O(n log n) or O(n) via hierarchical
expansions. Each level is GPU-parallel. Substrate for: kernel methods
at scale, computational physics, dense Gaussian process inference.

- **Impact**: a-class for kernel-method ML at large n.
- **Lego-fit**: 3 primitives (P2M particle-to-multipole, M2L multipole-
  to-local, L2P local-to-particle). Each is clean.
- **Self-consumer**: vyre's all-pairs dispatch dependency analysis
  (#19 polyhedral fusion considers all pairs of Regions; FMM-style
  hierarchical compression keeps it tractable at workspace scale).
- **Novelty**: known-but-unpackaged at primitive level on GPU.
- **Dispatch**: ship-now (3 small primitives).

### Group X  -  Type theory / category theory in IR

#### #52 functorial::data_migration  -  categorical-database operations as IR rewrites

Functorial data migration (Spivak 2012) treats data migrations between
schemas as functors between categories. Recent work (Patterson 2022
Catlab.jl, Brown 2023) shows the underlying graph rewrites compile to
sparse matrix ops.

- **Impact**: a-class for any data-pipeline / ETL workload + as an
  IR rewrite engine.
- **Lego-fit**: glue-y at user-dialect level; clean as a SUBSTRATE
  primitive (Region tree as category, transform passes as functors).
- **Self-consumer**: vyre's transform passes ARE functors. Ship the
  primitive and the substrate adopts it.
- **Novelty**: not-yet-packaged.
- **Dispatch**: source-change required + research.

#### #53 string_diagram::compile  -  string-diagram tensor compiler

String diagrams (Selinger 2010, Coecke-Kissinger ZX calculus) are
the visual language of monoidal categories  -  already used informally
in tensor networks (#35), quantum circuits, optical fibers, even neural
nets (Hinton's slot diagrams). Recent work (DisCoPy 2022) compiles them
to numeric tensor contractions.

- **Impact**: workspace-wide if adopted as IR view. The Region-tree IR
  IS a string diagram  -  making this explicit unlocks decades of
  monoidal-category math.
- **Lego-fit**: IR-construct OR primitive depending on adoption.
- **Self-consumer**: every IR transform pass becomes a string-diagram
  rewrite; equational reasoning replaces ad-hoc rewrites.
- **Novelty**: actively explored, no GPU primitive.
- **Dispatch**: research-only; possible IR pivot requires source work.

### Group XI  -  Numerics + alternative number systems

#### #54 padic::digit_norm + padic::hensel_lift

p-adic numerical analysis (Krasner 1986) gives stable arithmetic for
problems ill-conditioned in real numbers. Recent ML work (Robin 2024)
uses p-adic embeddings for stable training of deep networks. Hensel
lifting is the algorithmic core  -  GPU-friendly via parallel digit
evaluation.

- **Impact**: niche but unique. Substrate for stable-training research
  + cryptographic compute.
- **Lego-fit**: clean.
- **Self-consumer**: weak.
- **Novelty**: not-yet-packaged.
- **Dispatch**: research-only.

#### #55 fractional::caputo_derivative  -  fractional calculus kernels

Fractional derivatives (Caputo, Riemann-Liouville) generalize d/dx
to arbitrary order. Substrate for: anomalous diffusion, viscoelastic
materials, recent ML work (FractalNet 2017, fractional GD 2023) uses
them for better training dynamics. Discrete approximation = sparse
convolution with specific kernel  -  composes from existing conv1d.

- **Impact**: niche, but a wedge into scientific-compute markets.
- **Lego-fit**: clean. Just conv1d with a precomputed kernel.
- **Self-consumer**: weak.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: ship-now (trivial after conv1d is hardened).

### Group XII  -  Information geometry + statistical manifolds

#### #56 ig::natural_gradient_block  -  Fisher block-diagonal natural gradient

Generalizes #16 KFAC. Natural gradient = preconditioner is the
Fisher matrix; for exponential families it has closed forms. Recent
unification (Martens 2020, Pascanu 2024) shows the common substrate.

- **Impact**: a-class.
- **Lego-fit**: clean. Composes from #16.
- **Self-consumer**: weak.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: ship-now after #16.

#### #57 ig::fisher_rao_distance + ig::amari_alpha_connection

Fisher-Rao distance = Riemannian distance on the statistical manifold;
α-connection generalizes between exponential and mixture families.
Substrate for: distribution-aware loss functions, MoE routing,
calibration.

- **Impact**: niche but rising (MoE).
- **Lego-fit**: clean.
- **Self-consumer**: weak.
- **Novelty**: known-but-unpackaged.
- **Dispatch**: source-change required.

### Group XIII  -  Coarse-graining / model-reduction

#### #58 mori_zwanzig::projection  -  closed-form coarse-graining

Mori-Zwanzig projection (1965) gives an EXACT reduction of a high-dim
dynamical system to a low-dim system + memory kernel. Recent ML work
(Stinis 2020, Lin 2024) makes the projection learnable. Substrate
for: scientific ML, climate modeling, chemistry.

- **Impact**: a-class for science workloads.
- **Lego-fit**: clean. Composes from #3 randomized_svd + #43
  ode::integrator_step.
- **Self-consumer**: vyre's coarse view of its own dispatch graph
  (treat groups of Regions as macro-nodes; M-Z gives the optimal
  reduction).
- **Novelty**: known-but-unpackaged.
- **Dispatch**: source-change required.

### Group XIV  -  Hybrid analog-digital primitives

#### #59 stochastic::bitstream_op  -  stochastic computing primitives

Stochastic computing (Gaines 1969, recent revival via Alaghi 2018)
represents numbers as bitstreams; multiplication = AND, addition = MUX.
Trades precision for power. Recent NN inference work (Tehrani 2023)
uses it on GPU as bitset ops. Composes from existing bitset primitives.

- **Impact**: niche, but power-efficient inference is becoming a
  market.
- **Lego-fit**: clean. Bitset ops we already have.
- **Self-consumer**: weak.
- **Novelty**: known, recently unblocked.
- **Dispatch**: research.

### Group XV  -  Dependent / refinement-typed compute

#### #60 refinement::liquid_check  -  liquid types as buffer constraints

Liquid types (Vazou 2014) are refinement types decidable by SMT.
Recent work compiles them to runtime-checkable predicates. Substrate
for: prove "this buffer is sorted and within bounds" at compile time.

- **Impact**: workspace-wide if adopted as IR construct.
- **Lego-fit**: IR-enrichment, not a primitive (BufferDecl gets a
  predicate field).
- **Self-consumer**: vyre's #21 linear-logic types extended with
  liquid refinements becomes a unified buffer typing layer.
- **Novelty**: known but never compiled to GPU IR.
- **Dispatch**: research, tied to #21.

---

## Roll-up

### Bucket A primitives + dual-use uplift

Already on the list as #1-#7, #10, #13, #16. Land semiring_gemm
first because it unlocks #19 (polyhedral fusion uses it on the
fusion-graph), #23 (spectral analysis uses chebyshev_filter which
composes from it), #26 (dataflow fixpoint uses it directly), #39
(Datalog joins use it), and is the substrate for #36-#37 causal
inference. **One primitive, six self-consumers locked.**

### Bucket B primitives + research source work

#8 clifford_product, #11 planar_rewrite, #14 SOS, #15 persistent
homology, plus from this round: #34 QSVT, #35 tensor networks,
#36 do-calculus, #43 ODE integrator, #45 submodular max, #51 FMM.

### Substrate completes the recursion

All of #19-#23 + #26-#30 + this round's #52 functorial migration,
#53 string diagrams, #60 liquid types. By this point vyre is
self-improving  -  every primitive ships into both user dialects and
vyre's own compilation pipeline.

### Research-only (no dispatch yet)

#49 sparse FFT, #54 p-adic, #59 stochastic, #60 liquid types as
Research items. Either too niche or blocked on substrate that is not
named yet.

---

## The decisive question for every primitive

> When this primitive ships, what part of vyre itself uses it?

If the answer is "nothing  -  it's just for users," it stays in
`vyre-libs` as a Tier-3 dialect. The Tier-2.5 promotion bar is now:
≥ 2 user-dialect consumers AND ≥ 1 vyre-self consumer. The list
above is graded on that bar.

The recursion is the moat: math we ship as ops becomes math we
use to compile ops. Every shipped primitive simultaneously expands
the user-workload surface AND lets vyre self-improve. Compounding.
