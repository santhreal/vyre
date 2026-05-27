# vyre - Crate Ownership Boundaries

Single source of truth for what each workspace crate may and may not depend on. Cross-layer imports go down only. Every direction is enforced by `scripts/check_layering.sh`, `scripts/check_ownership_boundaries.sh`, and `scripts/check_consumers.sh`.

The ownership rules are dependency and source-reference rules, not folder rules. The folder layout may evolve; concrete backend leakage may not.

## Concrete Driver Isolation Law

Concrete backend names, crates, APIs, features, and implementation types are legal only inside the crate that implements that backend. `vyre-driver-wgpu` may mention `wgpu`, `WgpuBackend`, and `vyre_driver_wgpu`; `vyre-driver-cuda` may mention CUDA and cudarc; `vyre-driver-spirv` may mention SPIR-V-specific lowering. Shared crates, runtime crates, domain libraries, references, facades, and harness crates must talk through `vyre-driver` traits, registries, capabilities, artifacts, or opaque backend IDs.

This law covers source, tests, manifests, feature names, docs, and scripts. A runtime test importing `vyre_driver_wgpu`, a facade feature named after a concrete driver, or a shared doc comment that names `wgpu::Limits` is a boundary violation. If shared code needs a capability, add it to `vyre-driver` as a backend-neutral contract and let each concrete driver implement it locally.

## Layer DAG

```text
                   vyre-spec  vyre-macros
                       ^          ^
                       |          |
                  vyre-foundation
                       ^
            +----------+----------+
            |          |          |
   vyre-reference  vyre-driver  vyre-primitives
                       ^          ^
                       |          |
                 vyre-runtime  vyre-libs
                       ^          ^
                       |          |
              vyre-aot, vyre-frontend-c, vyre-intrinsics
                       ^
                       |
                  vyre-harness
                       ^
                       |
                    conform/*

Concrete driver crates hang off `vyre-driver` and own all target-specific code:

            vyre-driver
          +------+------+
          |      |      |
 vyre-driver-wgpu  vyre-driver-spirv  vyre-driver-cuda
```

`vyre-core` is the meta crate (`vyre`) that re-exports the user-facing neutral surface. Backend selection happens in the consumer or concrete driver crate, not in shared crates.

## Per-Crate Ownership

### `vyre-spec`
- **May depend on**: nothing in the workspace.
- **May not depend on**: any other workspace crate.

### `vyre-macros`
- **May depend on**: nothing in the workspace.
- **May not depend on**: any other workspace crate.

### `vyre-foundation`
- **May depend on**: `vyre-spec`, `vyre-macros`, lightweight data crates.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-driver`, `vyre-runtime`, `vyre-libs`, `vyre-reference`, `vyre-intrinsics`, `vyre-aot`, `vyre-frontend-c`, `wgpu`, `cudarc`, or target-specific backend types.

### `vyre-reference`
- **May depend on**: `vyre-foundation`, `vyre-spec`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-runtime`, `vyre-libs`, `wgpu`, `cudarc`, or target-specific backend types.

### `vyre-driver`
- **May depend on**: `vyre-foundation`, `vyre-spec`, `vyre-macros`, `vyre-primitives`, shared lowering IR dependencies.
- **May not depend on or mention implementation APIs from**: concrete driver crates. Examples and docs may describe backend classes only as abstract backends, never as imports from concrete crates.

### `vyre-driver-wgpu`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-runtime`, `vyre-spec`, `vyre-primitives`, `wgpu`, `naga`.
- **May not depend on**: peer concrete drivers.
- **Why**: this crate owns WGPU lowering, dispatch, buffers, and WGPU megakernel execution.

### `vyre-driver-spirv`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-spec`, SPIR-V lowering dependencies.
- **May not depend on**: peer concrete drivers or WGPU/CUDA APIs.

### `vyre-driver-cuda`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-spec`, CUDA driver dependencies.
- **May not depend on**: peer concrete drivers or WGPU/SPIR-V implementation APIs.

### `vyre-primitives`
- **May depend on**: `vyre-foundation`, `vyre-spec`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-libs`, `vyre-runtime`, `vyre-intrinsics`, `wgpu`, `cudarc`.

### `vyre-intrinsics`
- **May depend on**: `vyre-foundation`, `vyre-spec`, `vyre-driver`, `vyre-reference`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-runtime`, `vyre-libs`, `vyre-frontend-c`.

### `vyre-libs`
- **May depend on**: `vyre-foundation`, `vyre-spec`, `vyre-primitives`, `vyre-intrinsics`, `vyre-driver`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-runtime`, `vyre-frontend-c`, `vyre-aot`, `wgpu`, `cudarc`.

### `vyre-runtime`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-spec`, `vyre-primitives`.
- **May not depend on or mention**: concrete driver crates/APIs, `wgpu`, `cudarc`, `vyre-libs`, `vyre-frontend-c`, `vyre-aot`, `conform/*`.
- **Why**: runtime owns backend-neutral megakernel ABI, planning, telemetry, pipeline cache traits, and IO substrate. Concrete dispatch and buffer handles live in each concrete driver crate.

### `vyre-aot`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-spec`, `vyre-primitives`.
- **May not depend on**: live runtime, domain libraries, compiler tools, or concrete live-driver APIs.

### `vyre-frontend-c`
- **May depend on**: `vyre`, `vyre-libs`, `vyre-driver`, `vyre-primitives`, `vyre-runtime`, compiler-support crates.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-aot`, `wgpu`, `cudarc`.

### `vyre-harness`
- **May depend on**: `vyre`, `vyre-foundation`, `vyre-spec`.
- **May depend on (dev)**: pure IR/domain crates.
- **May not depend on or mention**: concrete driver crates/APIs or runtime at production scope.

### `vyre-core`
- **May depend on**: `vyre-foundation`, `vyre-driver`, `vyre-spec`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-runtime`, `vyre-libs`, `vyre-frontend-c`.

### `conform/*`
- **May depend on**: `vyre`, `vyre-foundation`, `vyre-libs`, `vyre-primitives`, `vyre-intrinsics`, `vyre-spec`, `vyre-driver`.
- **May not depend on or mention**: concrete driver crates/APIs, `vyre-frontend-c`, `vyre-aot`, `vyre-runtime`, `wgpu`, `cudarc`.
- **Why**: conformance proves parity through backend-neutral registry/capability surfaces. Concrete backend self-tests live in the concrete backend crates.

### `xtask`
- **May depend on**: anything needed for workspace tooling.
- **May not depend on**: external network services at build time.

### `benches/*`
- **May depend on**: anything they benchmark.
- **May not depend on**: implementation crates they do not benchmark.

## Forbidden Patterns

- No concrete driver reference outside the concrete driver crate.
- No internal path dependency without a matching version for publishable crates.
- No `default-features = true` pull on feature-gated mega crates.
- No `unsafe` outside backend/FFI boundaries.
- No string shader assets in production code.
- No raw `unwrap()` in production paths.

## Changing A Boundary

1. Move the backend-neutral contract into `vyre-driver`, `vyre-foundation`, or `vyre-runtime`.
2. Keep concrete implementation inside the concrete driver crate.
3. Update this file and the enforcement scripts in the same patch.
4. Run the boundary scripts before landing.
