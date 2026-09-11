# Testing `vyre-driver-wgpu`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-wgpu
```

Own pure WGSL target compilation, portable GPU acquisition, materialization, dispatch, graph execution, and backend evidence.

The crate lives at `vyre-driver-wgpu` and owns the `portable-driver` seam in the `concrete-backend` layer.

## Commands

```console
./cargo_full test -p vyre-driver-wgpu
```

```console
./cargo_full test -p vyre-driver-wgpu --all-features
```

```console
./cargo_full test -p vyre-driver-wgpu -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `device-tests`, `parity-testing`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `./cargo_full test -p vyre-driver-wgpu --bin vyre` |
| `bin` | `vyre-wgpu` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `./cargo_full test -p vyre-driver-wgpu --bin vyre-wgpu` |
| `example` | `wgpu_release_surface` | `vyre-driver-wgpu/examples/wgpu_release_surface.rs` | None | `./cargo_full test -p vyre-driver-wgpu --example wgpu_release_surface` |
| `lib` | `vyre_driver_wgpu` | `vyre-driver-wgpu/src/lib.rs` | None | `./cargo_full test -p vyre-driver-wgpu` |
| `test` | `all_tests` | `vyre-driver-wgpu/tests/all_tests.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test all_tests` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

You need a supported physical GPU adapter on the execution host for device dispatch and ignored physical-adapter tests. A requested adapter that cannot initialize is an error.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- Command status and exact portable-backend parity assertions

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run the ignored-test command on a configured GPU host. Backend initialization failures must remain visible. The bitwise transcendental tests in `gap_transcendentals_parity.rs` stay ignored on that host too: they wait on strict-IEEE lowering that blocks multiply-add contraction and on f32/u32 bitcast ops in the IR, and the enforced contract is the bounded envelope in `transcendentals_parity.rs`.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
