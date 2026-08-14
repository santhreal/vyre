# Contributing to Vyre

Vyre is a GPU-first execution substrate. Contributions are reviewed as changes to a compiler and runtime contract, not as isolated patches.

## Required Local Context

Read these files before changing architecture, public APIs, op definitions, or backend behavior:

- `docs/ARCHITECTURE.md`
- `docs/ARCHITECTURE.md`
- `docs/THESIS.md`
- `.github/CI_REQUIRED.md`
- `docs/LEGO_BLOCK_PHILOSOPHY.md`
- `docs/gpu_parity.md`

If a change conflicts with those documents, fix the architecture or update the contract in the same pull request. Do not add a workaround that leaves the conflict unresolved.

## GPU Requirement

Vyre assumes a real GPU is present on Santh development machines and self-hosted CI.

Before claiming a backend test failure is environmental, run:

```bash
nvidia-smi
cargo test -p vyre-driver-wgpu --test capability_contract -- --nocapture
```

Tests must fail loudly when a GPU probe is broken. Do not add silent CPU fallbacks or `skipped: no GPU` behavior for GPU-required lanes.

## Build Commands

Use the workspace gate wrapper when available:

```bash
cargo_full(workspace)
```

If the wrapper is unavailable in your shell, use bounded Cargo parallelism:

```bash
CARGO_BUILD_JOBS=1 cargo test --workspace
```

For targeted work, run the smallest meaningful gate first, then the broader gate that owns the contract you touched.

### Several checkouts, one target directory

Cargo hashes a workspace member by its path relative to the workspace root and
decides freshness by mtime, so two checkouts of this repository that share a
target directory address the same artifacts. The checkout whose files are older
than the last build receives the other one's compiled binary, with no warning
and no rebuild.

`.cargo/config.toml` declares `VYRE_CHECKOUT_ROOT`, the absolute path of the
checkout. `structure-gate`, `xtask`, `xtask-registry`, `xtask-evidence` and
`vyre-lints` read it, cargo records the value in each crate's dep-info, and a
gate therefore reports the tree it ran in.
`structure-gate/tests/checkout_provenance.rs` fails when that stops holding.

Two consequences. A checkout that predates that declaration overwrites the
shared record with one that names no root, and a gate binary built there is then
accepted as fresh; run `cargo clean -p <crate>` before trusting a number
measured beside such a checkout. Crates outside that list still share artifacts
across checkouts, which is what makes the shared directory worth having; the
same `cargo clean -p <crate>` is the cure when a test result looks like another
tree's.

Running cargo from outside the checkout root skips `.cargo/config.toml`, and the
gate crates then fail to compile with a `Fix:` message instead of baking a
foreign root.

### SCCache Support

Vyre uses `sccache` to speed up compilation. It is enabled by default in `.cargo/config.toml` (via `rustc-wrapper = "sccache"`).

To install `sccache`:
- **Linux (Debian/Ubuntu)**: `cargo install sccache --locked` (or download prebuilt binaries from the GitHub releases page)
- **macOS**: `brew install sccache`
- **Windows**: `choco install sccache` or `scoop install sccache`

Ensure `sccache` is in your `PATH` so Cargo can locate it during compilation.

## Required Gates by Change Type

Public API or crate boundary:

```bash
CARGO_BUILD_JOBS=1 cargo xtask release-gate
CARGO_BUILD_JOBS=1 cargo test --workspace
```

LEGO primitive, composite op, or registry behavior:

```bash
CARGO_BUILD_JOBS=1 cargo xtask gate1
CARGO_BUILD_JOBS=1 cargo xtask lego-audit
CARGO_BUILD_JOBS=1 cargo test -p vyre-primitives --all-features
```

WGPU backend or dispatch behavior:

```bash
nvidia-smi
CARGO_BUILD_JOBS=1 cargo test -p vyre-driver-wgpu --test capability_contract --test async_dispatch_contract -- --nocapture
```

C parser, VAST, program graph, or object sections:

```bash
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs --features c-parser --test c11_parser_integration --test c11_build_vast_nodes --test c_lower_ast_to_pg_nodes --test c_lower_ast_to_pg_nodes_gpu_parity --test c11_sema_scope
```

Repository discipline, CI, review metadata, or community files:

```bash
bash scripts/check_repo_hygiene.sh
```

Oracle-matrix sweep, or a `[[test]]` entry's `required-features`:

```bash
bash scripts/run_sweep_oracle_matrix.sh
bash scripts/run_volume_sweep_shard.sh 0 1
```

Both are merge gates in the `feature-gated-sweeps` job of
`.github/workflows/gates.yml`. `ci.yml` tests the workspace with default
features, so a sweep whose `required-features` name a non-default feature runs
nowhere else. The roster and the features come from
`scripts/lib/sweep_targets.py`, which reads tracked `<crate>/tests/sweep_*.rs`
and each crate's own `[[test]]` entries, so a new sweep is picked up without an
edit to either script. Pass a shard index and count to split the volume waves
locally; `0 1` runs all of them, which is what CI does.

## Manual Tools

These are run by hand. No workflow invokes them, and none is a merge gate.

- `bash scripts/bench_smoke.sh` runs the canonical vyre-bench smoke suite and prints JSON.
- `bash scripts/install_wire_precommit_hook.sh` installs `scripts/wire_ci_local.sh` as the pre-push hook, which blocks a push when the wire-surface fmt, clippy, check, or test steps fail.

## LEGO Block Rules

Vyre code should be built from small reusable primitives:

- Put reusable kernels in `vyre-primitives`.
- Compose domain features in `vyre-libs`.
- Keep backend dispatch in driver crates.
- Keep conformance logic in conform crates.
- Do not duplicate a primitive under a library feature because it is convenient.
- Do not introduce a composite op when the existing primitive chain can express the behavior cleanly.

Every new op needs a registry entry, reference behavior, GPU behavior when applicable, meaningful tests, and catalog coverage.

## Review Standard

A pull request is not ready until it has:

- A precise contract statement.
- Tests that would fail on the previous behavior.
- GPU proof for GPU-owned code.
- No new stubs, TODOs, FIXMEs, placeholder branches, or silent default returns.
- No hidden allocations or avoidable copies on hot paths.
- No public API break without an explicit migration.
- Updated docs when the public contract changes.

## Security

Report vulnerabilities through `SECURITY.md`. Do not put exploit details, credentials, or private test targets in public issues or pull requests.
