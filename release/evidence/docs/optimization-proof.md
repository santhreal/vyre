# Optimization proof

This artifact backs optimization release requirements.

Evidence sources:

Required generated evidence:

- `release/evidence/optimization/optimization-corpus.json`
- `release/evidence/optimization/optimization-corpus-contracts.json`
- `release/evidence/optimization/optimization-family-manifest.json`
- `release/evidence/optimization/optimization-case-manifest.json`
- `release/evidence/optimization/optimizer-pass-manifest.json`
- `release/evidence/optimization/optimization-integration-matrix.json`
- `release/evidence/optimization/optimizer-impact-cuda.json`
- `release/evidence/optimization/pass-family-benchmark-manifest.json`

Release contract:

- The generated corpus contains at least `4096` valid semantic `Program` cases.
- Eight source-owned families contribute at least `512` unique cases each.
- Every case runs through the registered release scheduler and records a canonical Program fingerprint.
- Optimized output validates, scheduler convergence is complete, and blockers remain empty.
- The optimizer pass manifest and integration matrix derive from the same live registration and catalog sources.
- Every catalog row records one owner, semantic Program input/output, phase, boundary, required facts, invalidations, invariant, proof, and benchmark or reviewed non-benchmark disposition.
- Verified lowering performs no semantic rewrite after the registered Program optimizer.
- `vyre-lower` analyses and emitter audits are read-only. Concrete target strategy remains driver-owned.
- CUDA before/after evidence covers all eight semantic optimizer families without lowering-layer rewrite cases.
