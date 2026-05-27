# Documentation evidence map

This artifact backs `docs-evidence-linked`.

Evidence sources:

Required generated evidence:

- `release/evidence/docs/docs-matrix.json`

Required source documents:

- `README.md`
- `docs/RELEASE.md`
- `docs/TESTING_PROGRAM.md`
- `docs/optimization/AGENT_CONTRACT.md`
- `conform/README.md`
- `vyre-bench/README.md`
- `vyre-frontend-c/README.md`
- `libs/dataflow/weir/README.md`
- `libs/dataflow/weir/VISION.md`

Release contract:

- Every required document must contain concrete `release/evidence/...` artifact references.
- Documentation must not contain unresolved release markers.
- The docs matrix must report zero blockers and zero missing topics.

