# tests/SKILL.md  -  vyre-driver-wgpu

## Purpose

`vyre-driver-wgpu` is the wgpu backend: device acquisition, buffer pool,
pipeline cache, shader compilation, dispatch, IR-to-naga lowering, and the async
readback and resident-buffer engines. Every wgpu execution path in vyre lives
here.

## Critical invariants

- Parity with the CPU reference. Every op with both a reference implementation
  and a wgpu lowering produces byte-identical output on every witnessed input.
  A divergence is a backend bug, never a tolerance to widen.
- `validate_with_cache` is a single atomic load on the hit path. It re-hashes
  nothing and probes no map.
- The pipeline cache key covers every dimension that changes the lowering
  outcome: workgroup size, every binding attribute, and the feature flags. A
  missing dimension is a silent miscache.
- Capability queries never over-promise. Reporting `supports_subgroup_ops` as
  available requires both that the lowering emits subgroup intrinsics and that
  the adapter supports them.
- Deadline enforcement is honest. A `DispatchConfig` timeout overrun surfaces a
  structured error.
- Dispatch never falls back to the host. A program this backend cannot execute
  on the device fails closed.

## Adversarial surface

- `Program` with workgroup `[0, 0, 0]`. Rejected, no panic.
- `Program` with ten thousand buffers. Bounded, structured error past the cap.
- Concurrent `WgpuBackend::dispatch` from eight threads. No data race, no
  poisoned mutex, stats still consistent.
- An adapter that advertises subgroup support but fails to compile a subgroup
  shader. The capability report must flip to unavailable after the first
  failure instead of staying available.
- Readback with `Maintain::Wait` on a dropped queue. Structured error.
- Artifact materialization racing device-loss recovery. Stale generations fail
  closed.

## Cross-crate contracts

- Implements `vyre_driver::VyreBackend`. The trait's defaults and overrides are
  owned by `vyre-driver/tests/backend_trait_contract.rs`; this crate tests what
  its own overrides do on a device.
- Implements `vyre_driver::CompiledPipeline`. Dispatch through a compiled
  pipeline must be bit-identical to `VyreBackend::dispatch`.
- Consumes `vyre_foundation::Program`. A wire round trip must produce
  bit-identical GPU output.

## What NOT to test here

- Wire format. That is `vyre-foundation/tests`.
- IR semantics and the CPU reference. Those are `vyre-reference/tests`.
- Operation metadata. That is `vyre-spec/tests`.
- Driver-tier trait contracts. Those are `vyre-driver/tests/backend_trait_contract.rs`
  and `vyre-driver/tests/backend_registry.rs`.

## Running

```bash
./cargo_full test -p vyre-driver-wgpu
./cargo_full test -p vyre-driver-wgpu --test capability_contract
./cargo_full test -p vyre-driver-wgpu --test dispatch_never_cpu_fallback
./cargo_full test -p vyre-driver-wgpu --test async_dispatch_contract
```
