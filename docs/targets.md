# vyre Targets

A target is a substrate that can execute vyre IR. Each target lives in its own
crate, implements the `VyreBackend` trait, and registers with the global backend
registry through `inventory::submit!`.

You can list what is registered in your build:

```rust
let ids: Vec<&str> = vyre::backend::registered_backends()
    .iter()
    .map(|b| b.id)
    .collect();
```

A backend only appears in that list if its crate is linked into your binary.
Linking `vyre-driver-cuda` and `vyre-driver-wgpu` on a Linux host with an NVIDIA
GPU gives `["cuda", "wgpu"]`.

## Target matrix

| Target | Crate | Status | Execution path |
|--------|-------|--------|----------------|
| `cuda` | `vyre-driver-cuda` | Release path on NVIDIA systems | vyre IR → `vyre-emit-ptx` → PTX → CUDA Driver API |
| `wgpu` | `vyre-driver-wgpu` | Portable path | vyre IR → `vyre-emit-naga` → naga Module → wgpu → Vulkan, DX12, Metal, WebGPU |
| `metal` | `vyre-driver-metal` | Native Apple runtime backend. Registers only on Apple targets, and never fabricates a backend on non-Apple builds | vyre IR → `vyre-lower` → `vyre-emit-metal` → MSL → Metal.framework |
| `spirv` | `vyre-driver-spirv` | Emission and validation target | vyre IR → naga Module → `naga::back::spv` → Vulkan direct |
| `native_module` | `vyre-emit-metal` | Artifact emission only. Runtime dispatch needs `vyre-driver-metal` | vyre IR → `vyre-lower` → `vyre-emit-naga` → `naga::back::msl` → structured Metal artifact |
| `cpu` | `vyre-reference` | Oracle, not a production target | Pure-Rust structural interpreter, the conformance reference |

There is no `backends/` directory and no `vyre-wgpu` crate. The driver crates
are top-level workspace members named `vyre-driver-*`.

## Capabilities

Each target reports `Capabilities`:

```rust
pub struct Capabilities {
    pub supports_dispatch: bool,
    pub supports_storage_buffers: bool,
    pub supports_uniform_buffers: bool,
    pub supports_push_constants: bool,
    pub supports_workgroup_atomics: bool,
    pub supports_subgroup_ops: bool,
    pub max_invocations_per_workgroup: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_storage_buffer_bytes: u64,
    pub max_push_constant_bytes: u32,
    pub datatype_support: DatatypeSupport,
}
```

Frontends query capabilities before dispatch. Programs exceeding a target's limits return `BE_E200_CAPABILITY` at compile time, not a runtime panic.

## Registration

```rust
inventory::submit! {
    vyre::BackendRegistration {
        id: "wgpu",
        factory: || Box::new(WgpuBackend::new()?),
        supported_ops: vyre::core_supported_ops,
    }
}
```

`vyre::backend::registered_backends()` returns the registration list, and
`vyre::backend::acquire(id)` constructs an instance. Both live in the
`vyre::backend` module, not at the crate root. There is no manual global
registration and no init function: link the crate and the backend is visible.

Asking for a backend whose crate is not linked is an error, not a silent
fallback:

```text
backend `photonic` is not linked into this binary. Fix: link the concrete
driver crate that registers this backend or choose one of the registered
backend ids.
```

## The IR extension forcing function

Adding an IR node, an op, or a wire-format field must not require editing any
existing backend. That property is what keeps the substrate list open. The way
to check it is to add the construct and confirm the other driver crates still
compile untouched.

A non-dispatching contract-check backend used to exist to enforce this
mechanically. It is gone. No such crate is in the workspace today, so treat this
as a review obligation rather than an automated gate.

## Adapter selection (wgpu target)

The `wgpu` backend exposes:

- `enumerate_adapters()` returns every adapter the wgpu instance can see.
- `AdapterCriteria` is the policy struct: vendor preference, discrete versus
  integrated, required limits, required features.
- `select_adapter(criteria)` chooses one adapter.
- `init_device_for_adapter(adapter)` produces a `Device` and `Queue` pair.
- `VYRE_ADAPTER_INDEX` is an environment variable that overrides selection
  manually, for diagnostics.

The default dispatch path uses a cached singleton adapter chosen by
`AdapterCriteria::default()`. A multi-GPU frontend builds its own adapter list
and dispatches per adapter.

## Target coverage

Coverage is not maintained by hand in this file, because a hand-written matrix
goes stale the moment an op lands. Two generated sources hold the real answer:

- `docs/generated/OP_INVENTORY.md` lists every registered op.
- `cargo run -p xtask --bin xtask -- conformance-matrix` reports per-engine
  support. The engines it requires a verdict for are `cpu_ref`, `cuda`, `wgpu`,
  `metal`, `rust_regex`, `hyperscan`, and `vectorscan`, defined in
  `xtask/src/conformance_matrix.rs`. An engine with no device on the host records
  `unsupported` rather than being dropped from the report.

Every op added to `vyre` must enter that matrix. `scripts/check_dialect_coverage.sh`
blocks a merge that declares an op without at least one concrete target lowering
(`primary_text`, `primary_binary`, `secondary_text`, or `metal_ir`).

## Adding a new target

1. Create the crate: `backends/<name>/`.
2. Implement `VyreBackend`. Validate capabilities at compile time, not at dispatch time.
3. Register via `inventory::submit! { BackendRegistration { … } }`.
4. Run the conform suite: `cargo test -p vyre-conform-runner -- --backend <name>`. Every witness case must match the reference.
5. Emit a certificate. Two backends with byte-identical certificates (modulo backend-id field) are exchangeable.

No step in this flow touches `vyre-core`, `vyre-reference`, `vyre-conform-spec`, or any other existing target. That is the test of whether the design is right.
