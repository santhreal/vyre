# Test architecture evidence

This artifact backs `modular-test-architecture`.

Evidence sources:

Required generated evidence:

- `release/evidence/tests/test-matrix.json`
- `release/evidence/tests/modularization-map.json`
- `release/evidence/tests/oversized-test-closure.json`
- `release/evidence/tests/release-surface-suite-coverage.json`

Release contract:

- No test file may exceed the `500` line modularity threshold.
- Test architecture must expose fixtures, contracts, properties, backend tests, corpus tests, benchmarks, and regressions.
- Test matrix evidence must scan platform, dataflow-analysis, and parser-CLI release surfaces.
- Test matrix evidence must report nonzero platform, dataflow-analysis, and parser-CLI release-surface test file counts.
- Release-surface coverage evidence must show platform, dataflow-analysis, and parser-CLI surfaces each have unit, integration, property, adversarial, corpus, benchmark, conformance, gap, and fuzz coverage.
- Oversized-test closure must report `closed = true`, `total_oversized_files = 0`, `total_god_test_candidates = 0`, and an empty `god_test_candidates` array.
- Structural evidence generation must use `cargo_full run --bin xtask -- test-matrix --output release/evidence/tests/test-matrix.json`.
- Modular-directory evidence must cover the Vyre workspace, a standalone dataflow-analysis crate, and a parser CLI; parser CLI tests are not allowed to remain monolithic while release evidence only checks library crates.
