# Distributed parser coherence proof

This artifact backs `distributed-parser-coherence`.

Evidence sources:

Required generated evidence:

- `release/evidence/parser/distributed-parser-map.json`
- `release/evidence/parser/vyre-frontend-c-contracts.json`
- `release/evidence/parser/vyrec-cli-contracts.json`
- `release/evidence/parser/weir-contracts.json`
- `release/evidence/parser/surgec-contracts.json`
- `release/evidence/parser/surgec-grammar-gen-contracts.json`

Release contract:

- `vyre-frontend-c`, `vyrec`, Weir, SurgeC, and SurgeC grammar generation must all have explicit parser ownership contracts.
- Parser boundaries must be coherent even though the parser implementation is distributed.
- Every parser contract artifact must report zero blockers, empty `unresolved_ownership_markers`, zero missing contract topics, full unit/integration/property/adversarial/corpus/benchmark/conformance/gap/fuzz required test categories, and zero missing test categories.
- `vyre-frontend-c` contract evidence must cover syntax, AST, diagnostics, spans, preprocessor behavior, GNU extensions, and unsupported-feature handling.
- `vyrec-cli-contracts.json` must prove the CLI owns include paths, macro flags, CUDA-first dispatch linkage, actionable `Fix:` diagnostics, and adversarial/property/corpus/benchmark/conformance/gap/fuzz CLI-contract tests.
- `distributed-parser-map.json` must expose required test-category coverage for `vyre-frontend-c`, `vyrec`, Weir, SurgeC, and SurgeC grammar generation so distributed parser ownership cannot pass on README terms alone; category evidence must include component `tests/`, `benches/`, and `fuzz/` trees.
- Each component contract artifact must list non-empty `required_evidence_trees` entries for `tests`, `benches`, and `fuzz`; the release gate rejects missing or zero-byte trees even when category names appear elsewhere in docs.
