# Crate metadata proof

This artifact backs `crate-metadata`.

Evidence sources:

Required generated evidence:

- `release/evidence/metadata/metadata-matrix.json`
- `release/evidence/metadata/feature-matrix.json`

Release contract:

- Publishable Vyre crates must be version `0.4.1`.
- Publishable Weir crates must be version `0.1.0`.
- `vyre-frontend-c` must be included as a versioned `0.4.1` non-publishable release-surface crate with `README.md`; `publish=false` is intentional for this release and does not waive metadata quality.
- `metadata-matrix.json` must report a positive `non_publishable_release_surface_count` so intentional release-surface crates cannot disappear silently.
- `metadata-matrix.json` must report `parser_release_surface_count >= 2` and an empty `missing_required_release_surfaces` array, proving `vyre`, `vyre-driver-cuda`, `vyre-driver-wgpu`, `weir`, `vyrec`, and `vyre-frontend-c` are present with the expected versions, release kinds, release surfaces, and README metadata.
- The root `tools/vyrec` package must be included in metadata and feature evidence as a `0.4.1` `parser-cli` release surface with `README.md`.
- CUDA and WGPU driver crates must be classified as required `0.4.1` publishable backend release surfaces (`cuda-backend` and `wgpu-backend`), not as generic internal Vyre crates.
- `feature-matrix.json` must report an empty `missing_required_release_packages` array covering `vyre`, `weir`, `vyrec`, CUDA, WGPU, and `vyre-frontend-c`.
- Every publishable package must expose at least one runnable `examples/*.rs` program, and at least one example must reference the crate API or crate identity; README Rust/TOML/shell usage blocks are additional evidence but do not replace a runnable example for publishable crates.
- `feature-matrix.json` must prove explicit release feature surfaces: `vyre` has `cuda` and `wgpu`, `vyre-driver-cuda` has `cuda`, `vyre-driver-wgpu` has `wgpu`, and `weir` has `default` plus `serde`.
- Internal tooling must not masquerade as publishable release crates.
- Package metadata and features must be coherent for crates.io release.
