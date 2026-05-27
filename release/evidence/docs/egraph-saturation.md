# E-graph saturation proof

This artifact backs `egraph-saturation`.

Evidence sources:

Required generated evidence:

- `release/evidence/optimization/egraph-saturation-matrix.json`
- `release/evidence/optimization/egraph-semantic-contracts.json`
- `release/evidence/optimization/egraph-before-after.json`

Release contract:

- Bounded e-graph saturation must be an implemented rewrite entry point, not only a hand-written optimization shell.
- Canonical `run_all` rewrite pipelines must invoke the non-recursive e-graph saturation entrypoint.
- Marker evidence must prove both algebraic reassociation and bitwise condition-chain reassociation are implemented.
- Benchmark evidence must include a positive `egraph_bitwise_case_count` metric with the release corpus floor for bitwise chain cases.
- Semantic contracts must prove output equivalence.
- Before/after evidence must show either wall-time improvement or a concrete rewrite-quality win with applied rewrites.
