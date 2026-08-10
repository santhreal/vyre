# Vyre README proof

This artifact backs `docs-evidence-linked`.

Evidence sources:

Required documentation authority:

- `docs/DOCS.toml`
- `docs/SUMMARY.md`
- `release/evidence/docs/vyre-readme-contracts.json`

Release contract:

- `README.md` must describe the current CUDA-first/WGPU-fallback release path.
- `README.md` must reference concrete release evidence artifacts.
- `python3 scripts/docs_manifest.py --check` must verify lifecycle, navigation, and generated provenance.
- `README.md` must avoid unsupported claims that are not backed by benchmark, conformance, or parser evidence.
- `vyre-readme-contracts.json` must prove release-specific tokens for `0.6.1`, CUDA, WGPU, GPU requirements, bytecode conditions, `vyre::Program`, concrete evidence paths, and at least one example block.
