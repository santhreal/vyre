# Testing `vyre-driver-cuda`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-cuda
```

Own pure PTX target compilation, native device acquisition, materialization, dispatch, graphs, and release-path evidence.

The crate lives at `vyre-driver-cuda` and owns the `cuda-driver` seam in the `concrete-backend` layer.

## Commands

```console
./cargo_full test -p vyre-driver-cuda
```

```console
./cargo_full test -p vyre-driver-cuda --all-features
```

```console
./cargo_full test -p vyre-driver-cuda -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `cuda`, `default`, `device-tests`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `cuda_release_surface` | `vyre-driver-cuda/examples/cuda_release_surface.rs` | None | `./cargo_full test -p vyre-driver-cuda --example cuda_release_surface` |
| `lib` | `vyre_driver_cuda` | `vyre-driver-cuda/src/lib.rs` | None | `./cargo_full test -p vyre-driver-cuda` |
| `test` | `all_tests` | `vyre-driver-cuda/tests/all_tests.rs` | None | `./cargo_full test -p vyre-driver-cuda --test all_tests` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

You need an NVIDIA GPU and a working CUDA driver on the execution host for device, dispatch, graph, and ignored physical-adapter tests. Probe failure is a configuration failure, not a skip.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- Command status and exact backend parity assertions

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run the ignored-test command on the designated GPU host. An ignored hardware test must fail if CUDA was requested but cannot be acquired.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
