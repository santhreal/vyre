# Optimization proof

This artifact backs optimization release requirements.

Evidence sources:

Required generated evidence:

- `release/evidence/optimization/optimization-corpus.json`
- `release/evidence/optimization/optimization-corpus-contracts.json`
- `release/evidence/optimization/optimization-family-manifest.json`
- `release/evidence/optimization/optimization-analysis-fixtures.json`
- `release/evidence/optimization/optimization-case-manifest.json`
- `release/evidence/optimization/optimization-integration-matrix.json`
- `release/evidence/optimization/alias-aware-dse.json`
- `release/evidence/optimization/alias-aware-stlf.json`
- `release/evidence/optimization/alias-aware-licm.json`
- `release/evidence/optimization/alias-aware-fusion-fission.json`
- `release/evidence/optimization/weir-facts-pass-firing.json`

Release contract:

- The generated corpus must contain at least `4096` verified cases.
- The optimization family manifest must cover at least `14` required release families, list `required_families`, report empty `missing_required_families`, and ensure each required family contributes at least `128` generated cases.
- The analysis fixture manifest must prove A13-A16 trigger the intended substrate-neutral facts for every generated fixture case: coalesced/strided/broadcast access, shared-memory promotion candidates, critical bank conflicts, and vec-pack chains.
- `optimization-analysis-fixtures.json` must expose empty `missing_required_families`, `total_fixture_cases >= 512`, and `total_triggered_cases == total_fixture_cases`.
- The optimization case manifest must list every generated pass instance with a unique id, family attribution, operation count, child-body count, binding count, and literal count.
- `optimization-case-manifest.json` must expose nonzero `cases_with_child_bodies`, `cases_with_bindings`, and `cases_with_literals` so the 4096-case proof cannot collapse to flat no-op descriptors.
- The pass-family benchmark manifest must map every required optimization family to concrete CUDA benchmark evidence.
- Each pass-family benchmark manifest case must have empty `missing_custom_metrics`, empty `non_positive_required_metrics`, empty `non_winning_cases`, empty `blockers`, at least `30` wall/baseline samples, positive `p50`/`p95`/`p99` latency percentiles for speed-proof wall/baseline metrics, and `min_wall_speedup_x1000 > 1000` proving optimized `wall_ns.p50` beats `baseline_wall_ns.p50`.
- The egraph benchmark must prove arithmetic/bitwise saturation and boolean predicate-chain saturation through positive `egraph_bitwise_case_count`, `egraph_boolean_case_count`, `egraph_equality_classes`, and `egraph_applied_rewrites` metrics.
- The Weir-dataflow DSE family must exist and must fire under Weir facts.
- Alias/reaching-def-aware DSE, STLF, LICM, loop fusion, and loop fission must be wired into the canonical Weir-aware optimization pipeline or explicit loop transform APIs.
- `weir-facts-pass-firing.json` must include non-comment code markers proving that Weir facts actually fire DSE, STLF, LICM, loop fusion, and loop fission transformations, plus the reaching-def Copy-chain canonicalization used by those legality checks.
- Cross-binding memory accesses are conservative by default; Weir `NoAliasFact` evidence must be required to recover DSE, STLF, LICM, loop fusion, and loop fission across distinct global bindings.
- Store-to-load forwarding must keep Global, Shared, and Constant memory-space cache entries separate even when numeric binding slots match.
- Benchmark artifacts must prove pass-family wins, not only correctness.
- Alias-aware benchmark artifacts must include positive `alias_cross_binding_fact_count`, positive `reaching_def_fact_count`, and per-family before/after metrics so the proof cannot shrink back to same-slot-only cases or alias-only legality checks.
