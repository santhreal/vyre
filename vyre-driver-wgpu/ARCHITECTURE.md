# vyre-driver-wgpu  -  architecture

WebGPU/wgpu backend. The reference VyreBackend implementation;
the conform suite measures every other backend's behaviour
against this one's CPU-reference oracle.

## Modules

### `lib.rs` (OFF-LIMITS  -  validation cache + invalidate_impacted methods active)
Top-level wiring of the backend trait, the public `WgpuBackend`
type, and the registration token.

### `runtime/`
Adapter discovery, device creation, queue management, indirect dispatch, and
the prerecorded command path. Caches the adapter info so the conformance
certificate stays stable across runs.

### `engine/`
The dispatch hot path: command-encoder allocation, dispatch scratch, graph
execution, record-and-readback, and multi-GPU submission.

### `emit/`
vyre IR to naga (WGSL AST) lowering, and the descriptor gate that admits a
lowered program.

### `pipeline/`
Pipeline cache (compiled compute pipelines keyed on program fingerprint).
Parts:
- `binding.rs` and `bindings_reflection.rs`  -  per-binding metadata and
  bind-group layout.
- `compound.rs`  -  multi-output pipelines.
- `disk_cache/` and `disk_cache_entries.rs`  -  on-disk pipeline persistence.
- `persistent.rs` and `persistent_resources.rs`  -  persistent-residency hot
  path.
- `output_slots.rs` and `output_readback.rs`  -  output slot mapping and
  readback.
- `tuning.rs`  -  workgroup selection.

### `buffer/`
Buffer pool, bind-group cache, staging, and GpuBufferHandle lifecycle.

### `async_dispatch.rs` + `resident_dispatch.rs`
Queue submission and pending readback paths. Resident dispatch keeps resource
handles, output maps, trap state, and timestamp queries alive until retirement.

### `capabilities.rs`
Adapter-cap probe  -  returns the adapter's max workgroup size,
storage-buffer count, subgroup support, etc.

### `target_compiler.rs`
Compiles a program into the backend's target payload.

### `ext.rs`
Extension hooks for vendor-specific intrinsics.

### `spirv_backend.rs`
SPIR-V emission shortcut for the wgpu backend's Vulkan path.

### `bin/`
Standalone binaries (debug helpers, conform probes).

## Public types

- **`WgpuBackend`**  -  backend-trait implementation. Acquired via
  `WgpuBackend::acquire()` or `::new()`.
- **`PipelineCache` / `LruPipelineCache`**  -  pipeline cache
  surface.
- **`GpuBufferHandle`**  -  persistent-buffer handle.
- **`OutputBindingLayout`**  -  re-exported from vyre-driver for
  call-site convenience.

## Integration points

- Default portable GPU backend.
- The conform runner uses this backend's CPU reference as the
  oracle.
- Downstream fused-dispatch paths target this backend's standard
  binding layout.
