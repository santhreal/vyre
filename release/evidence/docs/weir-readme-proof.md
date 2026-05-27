# Weir README proof

This artifact backs `docs-evidence-linked`.

Evidence sources:

Required generated evidence:

- `release/evidence/docs/docs-matrix.json`
- `release/evidence/weir/weir-analysis-api-matrix.json`
- `release/evidence/weir/weir-readme-contracts.json`

Release contract:

- Weir docs must state the `0.1.0` API surface honestly.
- Weir docs must reference concrete release evidence artifacts.
- Weir docs must distinguish standalone Weir APIs from Vyre integration.
- `weir-readme-contracts.json` must prove required API/version tokens, the default `serde` feature story for release-facing witness and soundness evidence, the `serde_evidence` feature guard story, and at least one Rust or TOML example block.
