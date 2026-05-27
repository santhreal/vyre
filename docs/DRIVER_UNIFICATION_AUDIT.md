# Supersession notice

This audit is evidence. Active shared-driver vs concrete-backend placement is
controlled by `docs/optimization/START_HERE.md`,
`docs/optimization/README.md`, and `docs/optimization/OWNERSHIP.toml`.

# Driver unification audit (P-UNIFY-1)

Maps every concern across `vyre-driver-{wgpu,cuda,spirv,megakernel}` and
identifies which lifts to `vyre-driver` (the shared crate) versus which
must remain backend-private.

The rule: **a backend defines only what is irreducible to its hardware
shape**. Everything else lives once, in `vyre-driver`.

## Irreducible per-backend (must stay private)

| Concern | Why backend-private |
|---|---|
| Shader emission (WGSL / PTX / SPIR-V bytes) | The IR-to-shader lowering is the backend's reason to exist. |
| Device + queue construction | Backend SDK is type-incompatible (`wgpu::Device` vs `CudaContext` vs raw Vulkan). |
| Buffer alloc / submit / readback primitives | Hardware-side allocator is opaque. |
| Backend-format validation | naga/NVVM/SPIRV-Tools have backend-specific rules. |
| Adapter info → fingerprint bytes | The adapter struct is backend-typed. The bytes-out is the unifiable surface. |

Each backend exposes these via a small trait (~5 methods) and inherits
everything else from `vyre-driver`.

## Duplicated today (lift to vyre-driver)

### Pipeline cache (in-memory)
- `vyre-driver-wgpu/src/pipeline.rs`  -  `WgpuPipeline`, in-memory map, eviction
- `vyre-driver-cuda/src/backend.rs:148`  -  `module_cache: HashMap<[u8; 32], CachedModule>`
- (spirv)  -  requires the P-SPIRV-1 source change

**Lift target:** generic `vyre_driver::pipeline_cache::PipelineCache<B: BackendCodegen>` keyed by
`PipelineCacheKey` (already in `vyre-driver`). Backend provides
`compile_blob(program, config) -> B::Pipeline` and `Drop` on the entry.
Eviction policy lives in driver-core (P-DRIVER-6 = submodular).

### Pipeline cache (on-disk)
- `vyre-driver-wgpu/src/pipeline_disk_cache.rs`  -  1092 LOC of atomic writes,
  metadata, env-var gating, key derivation, bounded reads, normalized digests
- (cuda/spirv)  -  requires P-CUDA-2 / P-SPIRV-2 before duplication can be audited

**Lift target:** `vyre_driver::disk_cache` (~700 LOC). Backend trait:
```rust
trait BackendDiskCache {
    fn fingerprint_adapter(&self, hasher: &mut blake3::Hasher);
    fn lower_to_blob(&self, program: &Program, config: &DispatchConfig) -> Result<Vec<u8>, BackendError>;
    fn load_blob(&self, blob: &[u8]) -> Result<Self::Pipeline, BackendError>;
}
```
Driver-core owns: atomic write, key derivation (`normalized_cache_digest`), env-var gating, bounded reads, metadata serialization, `disk_pipeline_cache_dir`, `cache_entry_path`.

### Validation cache
- `vyre-driver-wgpu/src/lib.rs:106,331,605`  -  `validation_cache: DashSet<ProgramHash>`
- (cuda/spirv)  -  same cache shape should be audited when those backends expose validation caching

**Lift target:** `vyre_driver::validation_cache::ValidationCache`. Backend
provides `validate_native(program) -> Result<(), BackendError>`. Driver-core
owns the dashset + hash key (P-WGPU-2 = VSA fingerprint key).

### Persistent buffer residency
- `vyre-driver-wgpu/src/buffer.rs`  -  `BufferPool`, persistent `wgpu::Buffer` recycling
- `vyre-driver-cuda/src/backend.rs:49,149,150`  -  `resident_buffers + inflight_resident_handles` reference counting
- (spirv)  -  to be added

**Lift target:** `vyre_driver::residency::ResidencyManager<B>`. Generic over
backend's buffer type. Driver-core owns: handle allocation (`AtomicU64`),
inflight ref-counting (`HashMap<u64, usize>`), recycling pool. Backend
provides `alloc_buffer(bytes) -> B::Buffer` and `free_buffer(B::Buffer)`.

P-DRIVER-11 wires this into every backend at once, not three times.

### Specialization cache
- `vyre-driver-wgpu/src/pipeline_compound.rs:249`  -  workgroup-size specialization keys
- (cuda) PTX has its own constant-replacement specialization

**Lift target:** `vyre_driver::specialization::Specialization` (P-WGPU-4 ⇒
applies to all backends, not just wgpu).

### Bind-group / parameter-binding cache
- `vyre-driver-wgpu/src/pipeline_binding.rs:115` + `pipeline_bindings.rs:69`
- `vyre-driver-cuda/src/binding.rs:253`

**Lift target:** `vyre_driver::binding::BindingPlan` keyed by `(program_hash,
buffer_count)`. Backend provides `materialize(plan) -> B::BindingHandle`.

### Persistent megakernel
- `vyre-driver-wgpu/src/pipeline_persistent.rs:465`
- `vyre-runtime/src/megakernel/planner.rs:541`  -  generic megakernel scheduling

**Lift target:** generic part already lives in `vyre-runtime megakernel`. The
wgpu-specific `pipeline_persistent.rs` should call into megakernel's
scheduler and provide only the wgpu submission glue. Today the two are
parallel implementations.

### Indirect dispatch resolver
- `vyre-driver-wgpu/src/pipeline.rs:603,617`  -  `find_indirect_dispatch` walks Program
- (cuda)  -  source-change required when CUDA exposes indirect launch

**Lift target:** `vyre_driver::indirect::resolve(program) -> Option<IndirectDispatch>`.
Driver-core walks the IR; backend interprets the result.

### Output-budget enforcement
- `vyre-driver-wgpu/src/pipeline.rs:905`  -  `enforce_actual_output_budget`
- (cuda)  -  source-change required

**Lift target:** `vyre_driver::output_budget::enforce(program, actual_outputs)`.
Pure IR analysis, not backend-specific.

### Output layout
- `vyre-driver-wgpu/src/pipeline.rs:954,973`  -  `OutputLayout`, `output_layout_from_program`

**Lift target:** `vyre_driver::output_layout`. Pure Program walk.

### Async submission protocol
- `vyre-driver-wgpu/src/async_dispatch.rs` (260 LOC)
- `vyre-driver-cuda/src/stream.rs` (216 LOC)

**Status:** Likely irreducible  -  wgpu uses `wgpu::Submit + Maintain` while
CUDA uses `cuStreamSynchronize`. Keep private but ensure both implement
`PendingDispatch` (already in `vyre-driver::backend`).

## Lift order (lowest-risk first)

Each step is its own commit; each leaves the workspace green.

1. **disk_cache**  -  pure plumbing (atomic writes, key derivation, env-var
   gating). No backend types in the lifted code. Lowest risk.
2. **output_layout + output_budget + indirect**  -  pure Program walks. No
   backend types involved.
3. **validation_cache**  -  small (~30 LOC of cache logic) + clean trait
   contract.
4. **residency**  -  reference-counting pool, generic over `B::Buffer`.
5. **pipeline_cache (in-memory)**  -  generic over `B::Pipeline`. Touches
   the dispatch hot path; commit + benchmark before moving on.
6. **specialization**  -  depends on P-WGPU-4 wiring; coordinate.
7. **binding**  -  last; touches the most code.

## Outcome

After P-UNIFY-1:
- `vyre-driver-wgpu/src/lib.rs` shrinks ~1072 → ~400 LOC (just adapter +
  device + WGSL emission + the trait impls)
- `vyre-driver-cuda/src/backend.rs` shrinks ~978 → ~350 LOC
- `vyre-driver-spirv` (currently 102 LOC) gets full feature parity for
  free  -  disk cache, validation cache, residency, etc., all inherited
- Every P-DRIVER-* and P-WGPU-* task ships **once**, not three times
- P-CUDA-1, P-CUDA-2, P-SPIRV-1, P-SPIRV-2 reduce to "implement
  BackendCodegen + BackendDiskCache + ResidencyOps" (~3 trait impls each)

## Trait surface (post-unification)

```rust
// vyre-driver/src/backend_codegen.rs (new)
pub trait BackendCodegen: Send + Sync {
    type Pipeline: CompiledPipeline + Send + Sync + 'static;
    type Buffer: Send + Sync + 'static;

    /// Lower a Program to a backend blob (WGSL/PTX/SPIR-V bytes).
    fn lower_to_blob(&self, program: &Program, config: &DispatchConfig)
        -> Result<Vec<u8>, BackendError>;

    /// Load a previously-lowered blob into a runnable pipeline.
    fn load_blob(&self, blob: &[u8]) -> Result<Self::Pipeline, BackendError>;

    /// Validate a Program against backend-specific rules.
    fn validate_native(&self, program: &Program) -> Result<(), BackendError>;

    /// Hash adapter capabilities into a stable fingerprint.
    fn fingerprint_adapter(&self, hasher: &mut blake3::Hasher);

    /// Allocate a backend buffer of the requested size.
    fn alloc_buffer(&self, bytes: usize) -> Result<Self::Buffer, BackendError>;
}
```

Driver-core types (`PipelineCache<B>`, `DiskCache<B>`, `ResidencyManager<B>`,
`ValidationCache<B>`) are generic over `B: BackendCodegen` and need nothing
else from the backend.

The 13 P-DRIVER-* tasks all become driver-core code that takes
`&dyn BackendCodegen` (or `<B: BackendCodegen>`) and works for every backend.
