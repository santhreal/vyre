# Build Performance and Optimization

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

Runtime benchmark claims come from `docs/optimization/BENCH_TARGETS.toml` and
the generated evidence under `release/evidence/benchmarks/`. Build settings do
not substitute for workload evidence.

## Release profile

The workspace release profile uses Thin LTO and one codegen unit:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
```

These settings live in the root `Cargo.toml`.

## Local build configuration

The checked-in `.cargo/config.toml` sets the build job count, disables proptest
failure persistence, and defines workspace aliases. It does not configure a
compiler cache. You may configure `sccache` locally, but release instructions
must not assume it is installed.

The workspace root's `cargo_full` wrapper owns every build-affecting variable
that is not in that file, including the serialized job count it applies when
diagnosing link pressure. Run gates through it:

```bash
./cargo_full test -p <crate>
```

A command that sets `CARGO_BUILD_JOBS`, `CARGO_TARGET_DIR`, `RUSTFLAGS`, or
`--target-dir` itself is wrong: each such override makes one reader's build a
different build, and the setting stops being reviewable in one place.

## Profile-guided optimization

PGO is optional local experimentation. If you use `cargo-pgo`, collect profiles
from the exact release workload and backend route you intend to optimize:

```bash
cargo pgo build
cargo pgo test
cargo pgo optimize
```

Record the toolchain, workload identity, backend, device, and before/after
measurements. Do not publish a PGO result as a general Vyre claim unless the
release benchmark evidence reproduces it.

## Runtime performance claims

| Claim type | Authority |
| --- | --- |
| Target budgets | `docs/optimization/BENCH_TARGETS.toml` |
| Measured samples | `release/evidence/benchmarks/` |
| Backend availability | `release/evidence/backends/backend-matrix.json` |
| Operation support | `docs/optimization/OP_MATRIX.toml` + generated OP schema |

A faster local laptop run is not release evidence. Prefer `vyre-bench` and the
xtask benchmark/release commands documented in [`CLI.md`](CLI.md).

## Common slow paths

- Full workspace builds while other cargo jobs hold the shared target dir
- Enabling every `vyre-libs` feature when you only need one domain
- Rebuilding emitters after naga/toolchain bumps (pipeline caches must invalidate)
- Treating reference-oracle runs as throughput benchmarks

## Related docs

- [`optimization/README.md`](optimization/README.md)
- [`optimization/START_HERE.md`](optimization/START_HERE.md)
- [`vyre-bench` testing guide](testing/vyre-bench.md)
