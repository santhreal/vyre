# Vyre README proof

This artifact backs `docs-evidence-linked`.

Evidence sources:

Required generated evidence:

- `release/evidence/docs/docs-matrix.json`
- `release/evidence/docs/vyre-readme-contracts.json`

Release contract:

- `README.md` must describe the current CUDA-first/WGPU-fallback release path.
- `README.md` must reference concrete release evidence artifacts.
- `docs-matrix.json` must prove those referenced non-generated evidence artifacts exist; generated docs under `release/evidence/docs/` are exempt from existence checks during docs-matrix generation order.
- `README.md` must avoid unsupported claims that are not backed by benchmark, conformance, or parser evidence.
- `vyre-readme-contracts.json` must prove release-specific tokens for `0.6.1`, CUDA, WGPU, GPU requirements, bytecode conditions, `vyre::Program`, concrete evidence paths, and at least one example block.
