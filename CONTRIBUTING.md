# Contributing

## Build

Toolchain version is pinned in `rust-toolchain.toml`. Do not override the host Cargo target directory; rely on the host Cargo configuration.

Scope builds to the packages you edit:

```sh
cargo check -p vyre-ir -p vyre-libs -p vyre-compiler
cargo build -p vyre --features cuda,wgpu
```

Format modified Rust files with `rustfmt`:

```sh
rustfmt --edition 2021 crates/vyre-ir/src/lib.rs
```

## Test

Run unit and integration tests for the affected packages:

```sh
cargo test -p vyre-ir -p vyre-compiler
cargo test -p vyre-test-support
```

Testing requirements:

- Every GPU backend change verifies output parity against `vyre-reference`.
- Every bug fix includes a regression test covering the invariant and sibling variants; verify the test fails when the defect is reintroduced.
- Workspace line coverage floor is 85% via `cargo llvm-cov`.
- Run structural checks with `cargo run -p xtask -- check`.

## Measure

Run the end-to-end megakernel workload suite in `vyre-bench` before and after performance changes:

```sh
cargo run --release -p vyre-bench -- --suite megakernel
```

Measure steady-state host heap traffic with `vyre_bench::alloc_probe` (`ThreadAlloc` and `Region`) to verify zero host copies and zero allocations on dispatch paths.

## Release

1. Verify `cargo run -p xtask -- check`, scoped GPU parity suites, and `cargo llvm-cov` pass on the target commit.
2. Update crate versions in `Cargo.toml` and add concise single-sentence declarative entries under the release heading in `CHANGELOG.md`.
3. Commit, push to `main`, confirm CI passes, and push an annotated tag `vyre-v<version>`.
4. Publish crates in dependency order: `vyre-ir`, `vyre-macros`, `vyre-libs`, `vyre-compiler`, `vyre-emit`, `vyre-runtime`, `vyre-reference`, `vyre`.
