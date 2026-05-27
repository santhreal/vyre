# Exhaustive verification evidence

This artifact backs `exhaustive-verification`.

Evidence sources:

Required generated evidence:

- `release/evidence/tests/test-matrix.json`
- `release/evidence/tests/release-surface-suite-coverage.json`
- `release/evidence/tests/unit-suite.json`
- `release/evidence/tests/adversarial-suite.json`
- `release/evidence/tests/property-suite.json`
- `release/evidence/tests/conformance-suite.json`
- `release/evidence/tests/corpus-suite.json`
- `release/evidence/tests/benchmark-suite.json`
- `release/evidence/tests/gap-suite.json`
- `release/evidence/tests/fuzz-suite.json`

Release contract:

- Every suite artifact must contain at least one Vyre-side file, one Weir-side file, and one `tools/vyrec` file.
- `test-matrix.json` must report nonzero Vyre, Weir, and `tools/vyrec` release-surface test file counts.
- `release-surface-suite-coverage.json` must report zero missing layers for all three release surfaces.
- `modularization-map.json` must include modular test directories for Vyre, Weir, and `tools/vyrec` so all release surfaces follow the same fixture/contract/property/backend/corpus/bench/regression layout.
- Suite artifacts must record concrete test files, entrypoints, and assertions or benchmark bodies.
- Suite artifacts must include counts for Vyre, Weir, and `vyrec`.
- Fuzz and gap coverage are release requirements, not optional follow-up work.
