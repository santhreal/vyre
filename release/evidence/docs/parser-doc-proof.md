# Parser documentation proof

This artifact backs parser release documentation requirements.

Evidence sources:

Required generated evidence:

- `release/evidence/parser/c-parser-linux-subsystem.json`
- `release/evidence/parser/distributed-parser-boundary-map.json`
- `release/evidence/parser/vyre-frontend-c-contracts.json`

Release contract:

- Parser docs must distinguish parsing, object emission, VAST, semantic graph, and future compiler lowering.
- C parser release proof is parsing and semantic artifact evidence, not a full C compiler claim.
- Distributed parser docs must name the parser contract topics they own: syntax, AST fidelity, diagnostics, spans, preprocessor behavior, GNU extensions, unsupported-feature handling, CLI include/macro handling, Weir dataflow facts, SurgeC consumer boundaries, and grammar/token generation.
- Distributed ownership between Vyre C, Weir, and SurgeC must be documented with concrete artifacts.
