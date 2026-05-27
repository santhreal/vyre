# Dataflow consumer README proof

This artifact backs `docs-evidence-linked`.

Evidence sources:

Required generated evidence:

- `release/evidence/docs/docs-matrix.json`
- `release/evidence/dataflow-consumer/dataflow-consumer-analysis-api-matrix.json`
- `release/evidence/dataflow-consumer/dataflow-consumer-readme-contracts.json`

Release contract:

- Dataflow consumer docs must state the `0.1.0` API surface honestly.
- Dataflow consumer docs must reference concrete release evidence artifacts.
- Dataflow consumer docs must distinguish standalone Dataflow consumer APIs from Vyre integration.
- `dataflow-consumer-readme-contracts.json` must prove required API/version tokens, the default `serde` feature story for release-facing witness and soundness evidence, the `serde_evidence` feature guard story, and at least one Rust or TOML example block.
