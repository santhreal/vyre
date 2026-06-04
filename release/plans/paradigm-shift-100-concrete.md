# GPU-First Dataflow/Compiler Platform Paradigm-Shift Launch Plan

Scope: platform, dataflow, and compiler frontend only. The goal is not cleanup. The goal is a release-worthy GPU-first dataflow/compiler platform with concrete software-level innovation, measurable performance wins, deep validation, contributor-grade organization, and no half-migrations.

Release definition: this plan is complete only when the code, tests, benchmarks, docs, packaging, and public launch path are all finished. The final steps intentionally end in cargo_full publish and git push after deep review.

## A. Non-negotiable architecture invariants

1. Remove every silent CPU fallback in platform, dataflow, and compiler frontend; any non-GPU path must be an explicit parity-test adapter or an architecture-impossible boundary.
2. Replace every skip-on-no-GPU test with a loud probe failure that reports adapter/device discovery details.
3. Establish a repo-wide GPU execution contract: device discovery, adapter selection, feature requirements, and failure semantics are centralized in one module per crate.
4. Define a single host/device memory ownership model for Vyre buffers so no subsystem invents its own staging convention.
5. Define an explicit boundary map for platform, dataflow, and compiler frontend showing which crate owns parsing, graph formation, lowering, scheduling, dispatch, and validation.
6. Split every orchestration file that contains execution logic into one-duty modules; pipeline files orchestrate only.
7. Enforce a one-duty-per-file rule for new work: parser stage, scheduler stage, kernel generation, cache, diagnostics, benchmark harness, and validation stay separate.
8. Create contributor-facing module maps for platform, dataflow, and compiler frontend that explain where new kernels, analyses, and parser phases belong.
9. Create public API boundaries that prevent consumers from depending on internal staging buffers, temporary graph encodings, or pipeline internals.
10. Add crate-level architecture tests that fail if forbidden modules import CPU execution helpers outside approved parity-test directories.

## B. Vyre runtime and megakernel innovation

11. Implement resident output-slot reuse across all dispatch paths so repeated runs preserve allocation capacity instead of rebuilding Vec<Vec<u8>> shells.
12. Implement resident input-slot refresh for grid-sync intermediate data so segment transitions reuse device-facing staging buffers.
13. Replace megakernel readback reallocation with caller-owned scratch for retry and recovery paths.
14. Build a scale-aware megakernel scheduler that chooses dispatch topology from graph shape, frontier density, and memory pressure.
15. Add persistent device-resident graph state for repeated dataflow evaluation so dependency graph structure is uploaded once per session.
16. Add kernel fusion for adjacent dataflow stages with compatible memory layouts, eliminating intermediate host-visible materialization.
17. Add a megakernel barrier planner that minimizes global synchronization by grouping independent dataflow waves.
18. Add a warp-specialized execution mode for sparse frontier expansion, avoiding block-wide waste on low-density workloads.
19. Add a block-specialized execution mode for dense frontier propagation, using coalesced scans and shared memory where profitable.
20. Add a runtime cost model that selects sparse, dense, hybrid, or fused evaluation based on measured frontier and graph statistics.
21. Add megakernel telemetry counters for bytes moved, allocations, kernel launches, sync points, occupancy proxy, frontier density, and readback volume.
22. Add golden benchmark cases that prove megakernel evaluation scales sublinearly with repeated fixed graph execution.
23. Add a device-side work queue for dataflow-dependent execution to reduce CPU orchestration overhead on large graphs.
24. Add batched multi-query execution for many analyses over the same graph, sharing graph residency and traversal work.
25. Add device-side convergence detection for iterative analyses so the host does not poll or coordinate every iteration.
26. Add a megakernel plan cache keyed by graph layout, analysis kind, and device features.
27. Add a zero-copy result compaction format for small outputs so readback transfers only meaningful bytes.
28. Add explicit failure diagnostics when a kernel cannot run on the selected GPU, including missing features and required limits.
29. Add a production benchmark proving the megakernel path is 100x to 1000x faster than naive host-orchestrated evaluation at the right scale.
30. Remove or quarantine every benchmark that accidentally measures setup, parsing, or allocation instead of steady-state kernel performance.

## C. dataflow consumer dataflow completion and performance

31. Convert all dataflow consumer analyses to caller-owned output scratch APIs with compatibility wrappers only where public API stability requires them.
32. Make reaching definitions reuse fixed-point iteration buffers across invocations.
33. Make liveness reuse fixed-point iteration buffers across invocations.
34. Make points-to analysis reuse graph, frontier, and result buffers across invocations.
35. Make slicing reuse traversal and output buffers across invocations.
36. Make IFDS solve reuse exploded-supergraph CSR buffers and iteration outputs.
37. Add resident CSR construction for dataflow consumer graphs so graph topology is not rebuilt for every analysis.
38. Add a shared graph-layout module for dominators, callgraph, IFDS, slicing, and range propagation.
39. Implement a dataflow batch API that runs multiple analyses against one graph residency handle.
40. Add graph-layout compatibility checks to prevent analyses from silently interpreting wrong edge encodings.
41. Add property tests for monotonicity, convergence, idempotence, and lattice join associativity for every dataflow consumer analysis.
42. Add adversarial graph tests: empty graph, single node, huge fan-out, huge fan-in, cycles, irreducible CFG, disconnected components, and degenerate callgraph.
43. Add differential tests against a simple reference implementation that is compiled only for parity testing, not production fallback.
44. Add GPU residency tests proving repeated analysis does not allocate new host output slots.
45. Add benchmark gates for small, medium, large, and pathological graphs with explicit throughput targets.
46. Add device-side bitset operations for dense dataflow sets.
47. Add sparse set kernels for low-density facts.
48. Add automatic sparse/dense switching based on fact density.
49. Add canonical graph normalization so equivalent graphs produce stable layout hashes and cache hits.
50. Document dataflow consumer as the dataflow substrate for Vyre and Vyrec with exact extension points.

## D. Vyrec C frontend and clang-parity path

51. Mark Vyrec as beta in public release docs while stating it is the active C compiler frontend built on Vyre.
52. Define the current supported C dialect matrix against clang: preprocessing, lexing, parsing, semantic analysis, diagnostics, and unsupported lower steps.
53. Finish full GPU-first preprocessing: macro expansion, conditional inclusion, include graph tracking, token provenance, line markers, stringification, token pasting, variadics, and builtin macro handling.
54. Add complete token provenance from source bytes through macro-expanded tokens to diagnostics.
55. Remove redundant macro invocation rescans by using live macro prefiltering from classified token streams.
56. Add function-like and object-like macro prefilter tests that distinguish identifier mention from actual invocation.
57. Add include cache residency so repeated includes do not re-tokenize or re-upload unchanged files.
58. Add GPU token classification for comments, identifiers, literals, punctuation, whitespace, directives, and string/char states.
59. Add a parser recovery model that keeps diagnostics precise without pretending malformed code parsed successfully.
60. Add semantic analysis parity for declarations, typedefs, tag namespaces, scopes, linkage, storage class, qualifiers, integer promotions, usual arithmetic conversions, and lvalue rules.
61. Add clang differential tests over real Linux subsystem headers and source slices for every supported pre-lowering phase.
62. Add corpus minimization for clang mismatches so failures become small reproducible fixtures.
63. Add structured diagnostic comparison against clang: location, severity, category, and primary message class.
64. Add semantic invariant tests that reject impossible AST states before lowering.
65. Add GPU preprocessing benchmarks against clang preprocessing on realistic corpora, measuring throughput and latency separately.
66. Add benchmarks that isolate preprocessing, lexing, parsing, semantic analysis, and end-to-end frontend time.
67. Add a public beta limitation statement only for lower steps intentionally not shipped in this version; do not hide parser or semantic gaps as limitations.
68. Add a parity dashboard artifact showing clang-compatible, partially compatible, and failing C features with test links.
69. Add fuzzing for preprocessor directives, macro recursion, malformed literals, include cycles, and unicode/encoding edge cases.
70. Add OOM and huge-file adversarial tests for preprocessing and parsing without silent truncation.

## E. Compiler/dataflow structural innovations

71. Add a unified device-resident token/fact graph format so parsing and dataflow can share layout and dispatch strategy.
72. Add incremental invalidation for source edits: only changed token spans, macro regions, semantic scopes, and dependent facts recompute.
73. Add multi-corpus batching so thousands of translation units share preprocessing, include cache, and semantic graph execution.
74. Add a frontier-typed IR that represents parser, semantic, and dataflow work as explicit dependency waves.
75. Add dependency-aware megakernel execution for the frontier-typed IR.
76. Add graph coloring or partitioning to reduce contention when many facts update shared nodes.
77. Add device-side diagnostic aggregation so errors are compacted on GPU before host readback.
78. Add a pluggable optimization registry for Vyre that keeps 100 plus optimizations discoverable by owner, phase, invariant, and benchmark.
79. Add optimization passes for allocation reuse, launch fusion, layout normalization, branch compaction, frontier density switching, bitset compression, and readback minimization.
80. Add pass-order validation so optimization composition cannot violate parser or dataflow invariants.
81. Add benchmark-driven pass selection so expensive optimizations only run when graph statistics justify them.
82. Add a stable pass explanation format that tells contributors why a pass fired and what invariant it preserved.
83. Add regression tests for pass idempotence and pass commutativity where mathematically required.
84. Add a device-memory budget planner that bounds peak allocations and fails loudly with actionable diagnostics.
85. Add cross-crate perf contracts so a change in Vyrec cannot accidentally disable a Vyre runtime optimization.

## F. Testing, validation, and release proof

86. Add a deep test command matrix for platform, dataflow, and compiler frontend using cargo_full and GPU probes.
87. Add unit, adversarial, property, benchmark, fuzz, and gap tests for every major module.
88. Add tests that intentionally prove missing clang parity as tracked gap findings rather than silent green dashboards.
89. Add tests that fail if CPU fallback modules are reachable from production dispatch paths.
90. Add allocation-regression tests for hot loops and repeated dispatch.
91. Add benchmark baselines and thresholds committed as release artifacts, not vague performance claims.
92. Add real corpus tests for Linux subsystem slices with reproducible fixture provenance.
93. Add sanitizer-style hostile input tests for invalid bytes, extreme nesting, include cycles, recursive macros, massive graphs, and corrupted cache metadata.
94. Add documentation tests for public library APIs so examples cannot drift from real code.
95. Add contributor onboarding docs that explain the architecture in under 10 minutes and direct each duty to one file/module family.
96. Add release docs that distinguish stable Vyre/dataflow consumer capabilities from Vyrec beta status without overselling unsupported lower steps.
97. Run deep personal review of every public crate file touched by the release and fix every finding before publishing.
98. Run the full GPU validation and benchmark suite on local RTX 5090 and record exact hardware, driver, and command outputs in release artifacts.
99. Publish only after the release checklist is green: cargo_full checks/tests, GPU tests, fuzz/gap findings triaged, benchmarks, docs, crate metadata, and public API review.
100. Final launch step: cargo_full publish the approved crates, make the repositories public, and git push the release branch and tags.
