# vyre Targets

A **target** is a substrate that can execute vyre IR. Each target lives in its own crate, implements the `VyreBackend` trait, and registers with the global backend registry via `inventory::submit!`.

## Target matrix

| Target | Crate | Status | Execution path |
|--------|-------|--------|----------------|
| `wgpu` | `vyre-wgpu` | Primary, production | vyre IR → naga Module → wgpu → Vulkan / DX12 / Metal / WebGPU |
| `spirv` | `backends/spirv` | Emission/validation target | vyre IR → naga Module → `naga::back::spv` → Vulkan direct |
| `photonic` | `backends/photonic` | Contract-check target, no dispatch | registers, `supports_dispatch = false`, listed by `registered_backends()` |
| `cuda` | `backends/cuda` | Registered backend work only if the crate is present in the workspace build | vyre IR → PTX emitter → CUDA Driver API |
| `metal` | `vyre-driver-metal` | Native Apple runtime backend; registers only on Apple targets and never fabricates a backend on non-Apple builds | vyre IR → `vyre-lower` → `vyre-emit-metal` → MSL → Metal.framework |
| `native_module` | `vyre-emit-metal` | Artifact emission implemented; runtime dispatch requires the native `metal` backend crate | vyre IR → `vyre-lower` → `vyre-emit-naga` → `naga::back::msl` → structured Metal artifact |
| `cpu` | `vyre-reference` | Oracle | Pure-Rust structural interpreter  -  the conformance reference, not a production target |

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

`vyre::registered_backends()` returns the id list; `vyre::backend(id)` constructs an instance. No manual global registration, no init function. Link the crate; the backend is visible.

## The photonic forcing function

`backends/photonic/` is intentionally minimal. It registers, reports
`supports_dispatch = false`, and every conform cycle confirms it's
listed in `registered_backends()`.

The contract-check target exists so that **every IR extension, every new
op, every new wire-format field must compile photonic without changes**.
A CI test asserts this. If adding `Node::Speculate` breaks photonic, the
IR extension story is broken  -  merge blocked.

## Adapter selection (wgpu target)

The `wgpu` backend exposes:

- `enumerate_adapters()`  -  returns every adapter the wgpu instance can see.
- `AdapterCriteria`  -  policy struct (vendor preference, discrete-vs-integrated, required limits, required features).
- `select_adapter(criteria)`  -  chooses one adapter.
- `init_device_for_adapter(adapter)`  -  produces a `Device + Queue` pair.
- `VYRE_ADAPTER_INDEX` env var  -  manual override for diagnostics.

The default dispatch path uses a cached singleton adapter chosen by `AdapterCriteria::default()`. Multi-GPU frontends construct their own adapter list and dispatch per adapter.

## Target cross-matrix (what the conform gate runs)

```
             wgpu   spirv emission   photonic registry   cpu (reference)
primitive       ✓          ✓                ✓*              ✓
hash            ✓          ✓                ✓*              ✓
decode          ✓          ✓                ✓*              ✓
graph           ✓          ✓                ✓*              ✓
…
* Photonic is the forcing function, not a real execution target.
  "✓" means "compiles, registers, passes the cert check that it can
  see the op declared" - not that it executes on hardware.
```

Every op added to `vyre-core` must enter this matrix. The
dialect-coverage CI script (`scripts/check_dialect_coverage.sh`) blocks
merges that declare ops without at least one concrete target lowering
(`primary_text | primary_binary | secondary_text | metal_ir`).

## Adding a new target

1. Create the crate: `backends/<name>/`.
2. Implement `VyreBackend`. Validate capabilities at compile time, not at dispatch time.
3. Register via `inventory::submit! { BackendRegistration { … } }`.
4. Run the conform suite: `cargo test -p vyre-conform-runner -- --backend <name>`. Every witness case must match the reference.
5. Emit a certificate. Two backends with byte-identical certificates (modulo backend-id field) are exchangeable.

No step in this flow touches `vyre-core`, `vyre-reference`, `vyre-conform-spec`, or any other existing target. That is the test of whether the design is right.
