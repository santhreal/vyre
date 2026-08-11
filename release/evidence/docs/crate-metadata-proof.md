# Crate metadata proof

This artifact backs `crate-metadata`.

Evidence sources:

Required generated evidence:

- `release/evidence/metadata/metadata-matrix.json`
- `release/evidence/metadata/feature-matrix.json`

Release contract:

- Publishable Vyre crates must be version `0.7.2`.
- `vyre-frontend-c` must be included as a versioned `0.7.2` non-publishable release-surface crate with `README.md`; `publish=false` is intentional for this release and does not waive metadata quality.
- `metadata-matrix.json` must report a positive `non_publishable_release_surface_count` so intentional release-surface crates cannot disappear silently.
- `metadata-matrix.json` must report an empty `missing_required_release_surfaces` array, proving `vyre`, `vyre-driver-cuda`, `vyre-driver-wgpu`, and `vyre-frontend-c` are present with the expected versions, release kinds, release surfaces, and README metadata.
- CUDA and WGPU driver crates must be classified as required `0.7.2` publishable backend release surfaces (`cuda-backend` and `wgpu-backend`), not as generic internal Vyre crates.
- `feature-matrix.json` must report an empty `missing_required_release_packages` array covering `vyre`, CUDA, WGPU, and `vyre-frontend-c`.
- `feature-matrix.json` must prove explicit release feature surfaces: `vyre` has `cuda` and `wgpu`, `vyre-driver-cuda` has `cuda`, and `vyre-driver-wgpu` has `wgpu`.
- `publish-readiness.json` must contain one successful `cargo package --list` content check for every package in publish order, including a BLAKE3 file-list digest, required metadata and licenses, Rust source, no internal instruction files, and no blockers.
- Internal tooling must not masquerade as publishable release crates.
- Package metadata and features must be coherent for crates.io release.
