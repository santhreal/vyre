# Vyre Paradigm Shift Plan: 100 Concrete Release-Critical Tasks

This file defines the minimum concrete work needed before Vyre can honestly be called a paradigm-shift release. The bar is not cleanup, warning reduction, or docs polish. The bar is measurable new capability: GPU-first execution where physically possible, structural software innovation, deep optimization, contributor-clear architecture, complete Weir dataflow integration, credible Vyrec beta positioning, and release gates backed by adversarial validation.

## Non-negotiable release invariants

1. No silent CPU fallback in GPU-owned paths. A CPU path may exist only for explicit parity testing, host orchestration, or cases where hardware architecture physically requires it.
2. Every hot path must have a named performance contract, a benchmark, and a regression threshold.
3. Every module boundary must explain one responsibility. If a file orchestrates, it must not implement. If a file implements, it must not own pipeline policy.
4. Weir must be a real dataflow substrate, not a side experiment.
5. Vyrec ships only as an explicitly beta C compiler frontend under active development unless semantic parity and corpus results prove otherwise.
6. Release claims must be proven by tests, benchmarks, adversarial corpuses, and reproducible command lines.

## 100 tasks

1. Define the Vyre release contract as executable gates: GPU residency, dataflow correctness, megakernel scale, parser beta status, organization limits, and publication readiness.
2. Split every pipeline file so orchestration, state, kernel dispatch, diagnostics, validation, and configuration each live in separate files.
3. Add an architecture boundary map that names every crate, module, and cross-crate contract used by Vyre, Weir, and Vyrec.
4. Create a contributor path map showing where to add a new optimization pass, a new Weir operator, a new parser rule, and a new benchmark.
5. Enforce a 500-line soft cap and one-duty rule for all files in Vyre, Weir, and Vyrec, with explicit exceptions requiring written architectural justification.
6. Replace duplicated pipeline state structs with one canonical state model and thin per-stage views.
7. Replace duplicated graph shape logic with one canonical shape descriptor shared across CPU parity tests and GPU execution.
8. Replace duplicated buffer packing routines with one typed packing layer that records layout, alignment, element width, and ownership.
9. Replace stringly typed kernel resource bindings with typed binding descriptors and validation at construction time.
10. Add a kernel manifest model so every GPU program declares inputs, outputs, mutability, grid shape, and expected residency.
11. Add a pass manifest model so every optimizer pass declares phase, dependencies, invalidations, cost model, and correctness invariants.
12. Implement pass scheduling from manifests instead of hand-ordered opaque vectors.
13. Add a pass invalidation engine that prevents stale analysis reuse after IR mutations.
14. Add a pass fusion planner that groups compatible passes into fewer traversals when dependencies allow.
15. Add a pass profitability model using IR size, graph density, memory pressure, and target backend.
16. Add a zero-copy IR view layer so passes can inspect common structures without cloning nodes or expressions.
17. Add arena-backed transient allocation for optimizer scratch data.
18. Add stable node identifiers and compact side tables to avoid hashmap-heavy pass metadata.
19. Add a canonical IR fingerprint for regression tracking and incremental caching.
20. Add structural hash-consing for repeated expression fragments in optimization-heavy workloads.
21. Add a GPU-resident worklist abstraction shared by Weir, graph propagation, and future parser acceleration.
22. Add a GPU-resident bitset abstraction with explicit word count, lane strategy, and atomic policy.
23. Add a GPU-resident queue abstraction for frontier-style algorithms that must avoid host round trips.
24. Add a GPU-resident reduction abstraction for changed flags, counters, and convergence signals.
25. Add a GPU-resident sparse relation abstraction for IFDS, grammar edges, and parser state transitions.
26. Replace per-iteration host downloads in dataflow loops with resident convergence flags unless explicitly testing parity.
27. Replace full-frontier uploads with GPU seed scatter for batched IFDS and related frontier algorithms.
28. Add GPU-side frontier clearing kernels so scratch reuse never requires host memset uploads.
29. Add batched query execution for every dataflow problem where independent queries share the same graph.
30. Add global convergence batching so N dataflow queries use one convergence flag when correctness permits.
31. Add per-query convergence only when it reduces total work more than it increases synchronization overhead.
32. Add scale-aware batch partitioning for frontier matrices that exceed resident scratch capacity.
33. Add persistent scratch pools keyed by graph shape and query capacity.
34. Add resident prepared-graph caches keyed by graph fingerprint and device identity.
35. Add explicit GPU memory budget accounting for all resident resources.
36. Add resource lifetime scopes so temporary GPU buffers are freed on every error path.
37. Add a no-hidden-copy audit for every GPU dispatch path.
38. Add transfer counters to benchmarks: host-to-device bytes, device-to-host bytes, dispatch count, and resident allocation count.
39. Add a dataflow megakernel prototype that executes multiple compatible Weir operators in one dispatch.
40. Add a megakernel planner that decides when fusion beats separate kernels based on graph size and frontier density.
41. Add a megakernel register-pressure estimator to prevent fusion from destroying occupancy.
42. Add a megakernel shared-memory budget model for architectures with different SM limits.
43. Add a megakernel fallback split plan that remains GPU-resident instead of falling back to CPU.
44. Add warp-level bitset propagation kernels for dense local neighborhoods.
45. Add block-level cooperative propagation kernels for high-degree nodes.
46. Add hybrid sparse/dense frontier switching based on active word density.
47. Add degree-bucketed traversal so low-degree and high-degree nodes do not share one inefficient kernel shape.
48. Add graph reordering for locality: node renumbering by procedure, block, fact, and edge locality.
49. Add CSR plus CSC dual layout when reverse traversals are required.
50. Add compressed edge labels for dataflow graphs where labels fit in narrower integer widths.
51. Add edge deduplication during graph preparation with deterministic diagnostics for duplicates.
52. Add graph invariant validation: sorted offsets, bounded destinations, no malformed ranges, and no overflow.
53. Add adversarial graph tests: empty graphs, single-node cycles, huge sparse graphs, dense graphs, duplicate edges, and malformed CSR.
54. Add property tests proving GPU dataflow results match CPU parity for random valid IFDS graphs.
55. Add hostile seed tests: empty seed sets, duplicate seeds, out-of-range seeds, huge seed batches, and uneven batch sizes.
56. Add deterministic replay for every randomized dataflow test using recorded seeds.
57. Complete Weir operator boundaries: relation construction, propagation, convergence, projection, join, filter, and materialization.
58. Add Weir operator manifests with input shape, output shape, algebraic properties, and GPU support level.
59. Add Weir fusion rules for associative, commutative, distributive, and idempotent operators.
60. Add Weir correctness contracts for monotonicity, fixed-point convergence, and lattice bounds.
61. Add Weir schedule visualization for contributors so dataflow execution is understandable without reading kernels.
62. Add Weir benchmark suites for IFDS, graph reachability, transitive closure, sparse joins, dense joins, and mixed workloads.
63. Add Weir regression thresholds for dispatch count, transfer bytes, allocations, and throughput.
64. Add Weir CPU parity executors only under explicit parity-test naming and feature boundaries.
65. Remove production-facing APIs that imply CPU fallback is normal for GPU-owned Weir paths.
66. Add Vyre optimization inventory: every current pass, its duty, its input invariants, output invariants, and benchmark target.
67. Add at least 100 optimizer opportunities as tracked pass candidates, grouped by IR canonicalization, memory layout, graph scheduling, fusion, CSE, DCE, bounds, vectorization, and GPU lowering preparation.
68. Implement the first wave of high-confidence optimization passes where correctness is local and measurable.
69. Add optimization profitability tests that prove passes do not run blindly when they increase work.
70. Add optimizer fixed-point protection so repeated pass cycles cannot oscillate.
71. Add optimizer trace mode that records why each pass ran, skipped, fused, or invalidated analysis.
72. Add benchmark fixtures that compare optimized and unoptimized IR on representative graph and parser workloads.
73. Add kernel dispatch coalescing so adjacent compatible Vyre GPU programs can share setup and resident resources.
74. Add buffer alias analysis so kernels can safely reuse memory without accidental overwrite.
75. Add explicit memory-layout transforms for SoA versus AoS choices in hot paths.
76. Add narrow integer specialization for shapes that fit in u16 or u32 instead of always using larger widths.
77. Add constant-shape specialization to remove dynamic bounds checks from generated kernels where safe.
78. Add runtime shape guards that make specialized kernels safe without weakening correctness.
79. Add device capability detection that loudly fails on misconfiguration instead of silently degrading.
80. Add GPU architecture profiles for RTX 5090, RTX 4090, and common CI GPU targets.
81. Add performance dashboards generated from checked-in benchmark baselines and local run outputs.
82. Add parser corpus management for Vyrec: tiny smoke corpus, Linux subsystem corpus, adversarial C corpus, preprocessor corpus, and semantic corpus.
83. Add a Vyrec beta status document that states exactly what is implemented, what is missing, and what is not claimed.
84. Implement full C preprocessing plan: macro expansion, conditional inclusion, include resolution, token pasting, stringification, builtin macros, and diagnostics.
85. Add GPU-tokenization experiments for C preprocessing where parallel lexical structure gives real advantage.
86. Add parser parity tests against Clang for tokens, AST shape, diagnostics, and semantic facts on supported inputs.
87. Add semantic analysis parity targets: typedef resolution, scopes, declarations, expressions, integer conversions, lvalues, storage classes, and control-flow legality.
88. Add differential testing that runs Vyrec and Clang on the same corpus and classifies every mismatch.
89. Add mismatch minimization so failing C files shrink to useful reproducers.
90. Add explicit beta gates for Vyrec so incomplete compiler work cannot block the broader Vyre release.
91. Add end-to-end examples showing Vyre library use, Vyre CLI use if present, Weir dataflow use, and Vyrec beta use.
92. Add crate metadata audits for publish readiness: license, repository, README, categories, keywords, versioning, and feature flags.
93. Add CI that runs formatting, lints, unit tests, property tests, GPU tests, benchmark smoke, and docs for the release crates.
94. Add deep local validation scripts using `cargo_full` that reproduce the release gate sequence without raw cargo.
95. Add fuzz targets for IR parsing, graph preparation, Weir relation loading, Vyrec tokenization, and C parser entry points.
96. Add sanitizer-compatible test modes for memory, panic, overflow, and concurrency bug discovery.
97. Add release-blocking issue templates for performance regression, GPU fallback regression, parser parity regression, and architecture boundary regression.
98. Add final release review checklist requiring every public API, every file over the line limit, every unsafe block, every GPU fallback, and every benchmark claim to be personally reviewed.
99. Run the final release gate: deep tests, GPU tests, fuzz smoke, benchmark comparison, corpus comparison, docs build, package dry run, and crate metadata review.
100. Publish only after the gate is clean: tag the release, run `cargo_full publish` for approved crates, push the public launch branch, and publish the GitHub release notes with measured claims only.

## First execution focus

The first implementation target should be Weir GPU-resident dataflow completion because it is both a software-level structural innovation and a measurable performance lever. The immediate proof is a resident batched IFDS path with GPU seed scatter, GPU frontier clearing, global convergence, no full-frontier per-iteration download, reusable scratch, and benchmarks that report transfer bytes and dispatch count at scale.
