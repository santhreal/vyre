# Weir integration proof

This artifact backs `weir-analysis-integration`.

Evidence sources:

Required generated evidence:

- `release/evidence/weir/weir-analysis-api-matrix.json`
- `release/evidence/weir/weir-vyre-integration-tests.json`
- `release/evidence/weir/weir-readme-contracts.json`
- `release/evidence/optimization/weir-facts-pass-firing.json`
- `release/evidence/benchmarks/weir-dataflow-release.json`

Release contract:

- Weir must expose real dataflow APIs for SSA, reaching-def, points-to, may-alias, IFDS, callgraph, slicing, summaries, loops, and fixpoint-style analyses.
- `weir-analysis-api-matrix.json` must list required API items for the shipped public primitives and report zero missing API items, not merely prove that each source file contains some public symbol.
- `weir-analysis-api-matrix.json` must report `required_api_item_count >= 100` and `missing_api_item_count == 0`, forcing all release-listed Weir modules to keep explicit public API contracts instead of silently dropping contract rows.
- Vyre must consume Weir facts where they unlock safe optimization.
- Weir evidence must include property, parity, adversarial, perf, fuzz, and gap test families.
- `weir-analysis-api-matrix.json` must prove at least two standalone `examples/*.rs` programs outside tests so Weir is demonstrably usable as a normal library; each example must be non-empty, expose a runnable `fn main`, import or reference the `weir` crate, reference at least two dataflow API tokens, and report zero unresolved markers.
- Standalone Weir examples must include default-feature `serde` evidence for release-facing witness and soundness API types, proving the crate can persist real Weir outputs without bespoke mirror structs; the matrix also rejects serde evidence examples that omit `required-features = ["serde"]` in `Cargo.toml`.
- Weir README evidence must prove the standalone `weir 0.1.0` API story, relation to Vyre, soundness vocabulary, and user examples.
