# Testing `vyre-bench`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-bench
```

Own reproducible workload benchmarks against the best available native baseline for each class, not against vyre's own unfused output.

The crate lives at `vyre-bench` and owns the `workload-benchmarks` seam in the `tooling` layer.

## Commands

```console
./cargo_full test -p vyre-bench
```

```console
./cargo_full test -p vyre-bench --all-features
```

```console
./cargo_full run --bin xtask -- release-benchmarks --backend cuda --measured-samples 30 --write
```

```console
./cargo_full run --bin xtask -- release-benchmarks --backend wgpu --measured-samples 30 --write
```

## Feature sets

- Default feature members: `cli`
- Available manifest features: `cli`, `default`, `device-tests`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `release` | `vyre-bench/benches/release.rs` | None | `./cargo_full test -p vyre-bench --bench release` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | None | `./cargo_full test -p vyre-bench --bin vyre-bench` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | `cli` | `./cargo_full test -p vyre-bench --bin vyre-bench` |
| `lib` | `vyre_bench` | `vyre-bench/src/lib.rs` | None | `./cargo_full test -p vyre-bench` |
| `test` | `all_tests` | `vyre-bench/tests/all_tests.rs` | None | `./cargo_full test -p vyre-bench --test all_tests` |
| `test` | `determinism_gate` | `vyre-bench/tests/determinism_gate.rs` | None | `./cargo_full test -p vyre-bench --test determinism_gate` |
| `test` | `min_samples_gate` | `vyre-bench/tests/min_samples_gate.rs` | None | `./cargo_full test -p vyre-bench --test min_samples_gate` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

Benchmark unit tests are host-capable. Release measurements require the backend and device named by the benchmark command on the execution host; probe failure invalidates the run.

## Evidence outputs

- `release/evidence/benchmarks/`
- Raw per-sample benchmark records
- Generated suite summaries and source-tree fingerprints

## Skips and failures

Ignored physical benchmarks are absent from the default test command. A release benchmark command must execute its requested device on the execution host and preserve raw samples; it cannot report a synthetic or skipped result.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
