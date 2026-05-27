# Supersession notice

This is **not the current plan of record**. This document is historical CUDA
context. Active CUDA optimization work uses the `driver_cuda` lane in
`docs/optimization/OWNERSHIP.toml` and the patch proof contract in
`docs/optimization/AGENT_CONTRACT.md`.

Release routing policy is defined by the `0.4.1` release gate and public
README contract: CUDA is the NVIDIA fast path after its conformance,
performance, feature, and metadata evidence closes; WGPU is the portable GPU
fallback. Historical statements below that say CUDA must remain behind WGPU are
superseded by that release evidence model.

# Historical VYRE CUDA Backend Execution Plan

This was the original plan for making `vyre-driver-cuda` worthy of VYRE. The target was not a minimal CUDA backend. The target was a reliable, high-performance, conformant NVIDIA backend that could become the fastest dispatch path for VYRE programs and the substrate for persistent megakernel training workloads.

## Non-Negotiable Bar

`vyre-driver-cuda` must end with the full CUDA capability set implemented and proven. "Truthful" in this plan is not permission to downgrade the goal; it is a safety rule while building. If a capability is valuable for CUDA, the work item is to implement it end-to-end, not to permanently mark it unsupported.

The backend only gets registered as a preferred backend when it is implemented and proven:

- It compiles on the workspace MSRV without forcing an unrelated Rust upgrade.
- It never advertises capabilities that are not implemented and tested.
- Unsupported IR fails loudly with actionable `BackendError`; generated PTX must never contain TODO comments as semantic placeholders.
- Backend routing preference is controlled by the active release evidence gate, not by this historical plan.
- Tests run against the actual NVIDIA GPU path. A missing GPU probe is a configuration failure, not a skip.
- Public docs describe the implemented backend, not the intended backend.
- All new tests live under `vyre-driver-cuda/tests/` unless updating an existing inline contract is unavoidable.

## Current State To Fix First

The existing scaffold is useful but not landable:

- `cargo check -p vyre-driver-cuda` currently fails because `libloading@0.9.0` requires Rust 1.88 while the workspace MSRV is 1.85.1.
- `README.md` claims SPIR-V-to-PTX, tensor cores, async compute, and cooperative persistent megakernel support before those paths are proven.
- `backend.rs` hardcodes compute capability and device memory instead of probing CUDA device attributes.
- `codegen.rs` emits TODO comments for unsupported nodes instead of returning errors.
- Kernel launch only handles a few fixed arities and falls back to first input/output for unsupported signatures.
- The backend registration rank is too aggressive for an unproven backend.
- Unsafe pointer casts exist and must be isolated behind audited, tested helpers.

## Phase 0: Remove False Claims, Then Implement Them

Goal: make the crate safe to build while preserving the final requirement that every strategically important CUDA capability is implemented, tested, and enabled.

Tasks:

- Fix the `cudarc`/`libloading` dependency path so the crate compiles on Rust 1.85.1.
- Remove or gate inventory registration until CUDA passes the conformance subset.
- Set backend precedence behind a feature gate such as `cuda-routing` while the backend is incomplete.
- Create explicit implementation tickets/tests for tensor cores, bf16, f16, subgroup ops, async compute, and persistent megakernel support. These are mandatory deliverables for the CUDA backend, not optional exclusions.
- Replace README capability table with implemented/proven status only.
- Remove all TODO/FIXME markers from code by either implementing the path or returning a hard error.

Exit gate:

- `cargo check -p vyre-driver-cuda` passes.
- `rg -n "TODO|FIXME|For now|fallback|ready|✅|🔜" vyre-driver-cuda` has no misleading implementation or docs hits.
- The plan/test tree contains a failing or pending external contract test for every CUDA capability not yet implemented, so missing work stays visible until closed.

## Phase 1: Device Introspection And Capability Truth

Goal: CUDA capability answers come from the device, not constants.

Tasks:

- Implement CUDA device probing for:
  - device ordinal/name
  - compute capability
  - total memory
  - max threads per block
  - max grid dimensions
  - shared memory per block
  - warp size
  - cooperative launch support
  - concurrent kernel/async engine support
- Store a `CudaDeviceCaps` struct and use it for every `VyreBackend` capability query.
- Add tests that compare CUDA capability fields to direct CUDA probe results.
- Fail loudly when the CUDA driver/runtime cannot be loaded.

Exit gate:

- No hardcoded compute capability or memory values remain.
- `cargo test -p vyre-driver-cuda --test capability_contracts` passes on the RTX 5090 machine.

## Phase 2: Safe Buffer And ABI Layer

Goal: buffer binding is deterministic, validated, and compatible with VYRE `Program` declarations.

Tasks:

- Build an explicit `CudaBindingPlan` from `Program::buffers()`:
  - separates read-only, read-write, write-only, shared, persistent
  - validates input count and byte alignment
  - validates output counts and byte ranges
  - maps VYRE buffer order to CUDA kernel parameter order
- Replace ad hoc `Vec<CudaSlice<u32>>` launch assumptions with typed binding descriptors.
- Add audited helpers for byte/u32 conversion; no scattered raw pointer casts.
- Support exact output byte trimming according to buffer declarations, not `count * 4` assumptions where ranges exist.
- Implement the memory kinds needed by VYRE's CUDA path. Any memory kind not yet implemented must have an external failing/gap contract test and must not be hidden behind silent fallback.

Exit gate:

- Tests cover zero inputs, wrong input count, non-u32 byte lengths, multiple outputs, read-write buffers, shared buffers, and persistent buffers.
- No silent first-input/first-output fallback remains.

## Phase 3: PTX Emission Core

Goal: implement a correct PTX emitter for the VYRE IR subset it claims.

Required initial IR subset:

- `Expr::Load`
- `Node::Store`
- `Node::Let`
- `Node::Assign`
- `Node::If`
- `Node::Block`
- integer arithmetic and bitwise binops
- comparisons and predicates
- `gid.x/y/z`
- `select`
- barriers only when mapped to legal CUDA synchronization

Tasks:

- Replace the current register sketch with a real `PtxFunctionBuilder`:
  - register allocator
  - symbol table for `let` and assigned variables
  - typed expression lowering
  - label allocator
  - predicated branch lowering
- Emit loads and stores according to actual buffer names and bindings.
- For every unsupported `Expr` or `Node`, return a structured error naming the exact unsupported variant.
- Add PTX syntax tests that load the generated module through CUDA, not just string snapshots.

Exit gate:

- External tests prove identity, add, mul, bitwise, comparison/select, and conditional store programs execute correctly on GPU.
- Unsupported node tests assert actionable errors.

## Phase 4: Dynamic Kernel Launch

Goal: launch arbitrary supported VYRE buffer signatures without arity special cases.

Tasks:

- Implement or expose a raw CUDA driver launch path that accepts dynamic kernel parameter arrays.
- Preserve lifetimes for all device buffers and params through kernel completion.
- Validate launch geometry against probed device limits.
- Honor `DispatchConfig::workgroup_override` and `grid_override`.
- Derive default grids from output shapes while preserving 1D/2D/3D program shape rules.

Exit gate:

- Tests cover 1 to 8 inputs, 1 to 4 outputs, and mixed read-write buffers.
- Oversized block/grid requests fail before launch with actionable errors.

## Phase 5: Conformance Harness Integration

Goal: CUDA becomes a real VYRE backend, not a sidecar experiment.

Tasks:

- Add `vyre-driver-cuda/tests/conformance_subset.rs`.
- Run the same byte-exact CPU/reference comparisons used by wgpu for the supported IR subset.
- Add capability tests for f16, bf16, subgroup, tensor-core, and async features as implementation gates; these tests should drive the implementation rather than being postponed until after it.
- Wire CUDA into conform runner under an explicit feature.
- Keep CUDA out of default backend routing until the conformance subset is green.

Exit gate:

- `cargo test -p vyre-driver-cuda` passes on NVIDIA hardware.
- CUDA conformance subset is documented with exact supported ops.

## Phase 6: Performance Architecture

Goal: make CUDA materially faster than wgpu for the workloads it owns.

Tasks:

- Add a content-addressed module cache keyed by:
  - program fingerprint
  - CUDA device UUID or name
  - compute capability
  - PTX target
  - workgroup/grid policy
- Avoid recompiling PTX for repeated dispatches.
- Add persistent device buffer pools:
  - input staging pool
  - output pool
  - params pool
  - optional pinned-host transfer pool
- Add CUDA stream support:
  - one default stream for synchronous dispatch
  - stream pool for async dispatch
  - event-based readiness for `PendingDispatch`
- Add benchmark tests comparing:
  - cold compile
  - warm dispatch
  - small elementwise kernels
  - multi-output kernels
  - megakernel launch path

Exit gate:

- Warm CUDA dispatch has no PTX regeneration.
- Buffer allocations are reused in a benchmark-visible way.
- CUDA beats wgpu on at least small elementwise and repeated-dispatch microbenchmarks before routing preference is enabled.

## Phase 7: Hardware Intrinsics

Goal: expose NVIDIA hardware because it is backed by real lowering.

Tasks:

- Implement warp/subgroup lowering:
  - shuffle
  - ballot
  - subgroup add/reduce
- Implement f16/bf16 lowering with explicit type support and tests.
- Implement tensor-core/MMA lowering only through a dedicated VYRE primitive contract, not generic fake capability.
- Add negative tests proving capability probes fail loudly when the device truly cannot support a feature. On supported NVIDIA hardware, the expected outcome is implemented-and-passing, not disabled.

Exit gate:

- Capability methods are feature-and-device truthful.
- Hardware intrinsic tests pass on GPU and fail loudly when a device lacks support.

## Phase 8: Persistent Megakernel CUDA Path

Goal: CUDA can run VYRE persistent workloads without pretending ordinary launches are cooperative megakernels.

Tasks:

- Implement a CUDA-specific megakernel launch path:
  - persistent ring buffer mapping
  - control buffer
  - debug log
  - IO queue
  - stream/event synchronization
- Decide and document whether cooperative launch is required or whether ordinary persistent kernels are sufficient for the first CUDA path.
- Add tests for:
  - publish slot
  - packed slot
  - done count
  - metrics
  - shutdown
  - timeout
  - malformed ring buffers
- Benchmark launch amortization versus wgpu megakernel dispatch.

Exit gate:

- `vyre-runtime` megakernel contract tests pass against CUDA where applicable.
- CUDA persistent mode is advertised only when these tests pass, and this phase is not complete until they do.

## Phase 9: Parameter Golf Trainer Prototype

Goal: build the competition proof on top of a reliable backend.

This is not part of `vyre-driver-cuda` itself. It belongs in a wrapper or experiment crate that consumes VYRE.

Tasks:

- Define a tiny model ABI:
  - parameter buffer under 16MB
  - optimizer state buffer
  - token batch ring
  - metrics buffer
- Implement persistent CUDA/VYRE training loop:
  - forward
  - loss
  - backward
  - optimizer update
  - periodic metric readback
- Add a baseline comparison:
  - PyTorch or tinygrad baseline
  - same model math
  - same parameter budget
  - same 10-minute wall clock
- Report:
  - tokens/sec
  - steps/sec
  - BPB after fixed wall time
  - compile overhead
  - warm steady-state overhead

Exit gate:

- A reproducible script runs the baseline and VYRE trainer under the same wall-clock limit.
- The VYRE trainer shows a systems-speed advantage before any architecture claims are made.

## Test Matrix

Every phase that touches code must add external tests under `vyre-driver-cuda/tests/`.

Required suites:

- `capability_contracts.rs`
- `binding_plan_contracts.rs`
- `ptx_codegen_contracts.rs`
- `gpu_elementwise_conformance.rs`
- `launch_geometry_contracts.rs`
- `unsupported_ir_errors.rs`
- `module_cache_contracts.rs`
- `stream_async_contracts.rs`
- `megakernel_cuda_contracts.rs`
- `performance_smoke.rs`

Tests must assert behavior, not just compile. No `let _ = ...` tests.

## Routing Rule

CUDA can be linked during development without becoming the default backend. This is a temporary safety gate, not a final product mode. The completed CUDA backend must move above wgpu on NVIDIA hardware when all of these are true:

- CUDA crate compiles on MSRV.
- Supported IR subset has conformance tests.
- Capability claims are device-probed and lowering-backed.
- Warm dispatch benchmarks beat wgpu on representative workloads.
- Persistent megakernel contracts pass if persistent support is advertised.

Before that point, CUDA may be manually selected by tests and experiments only. After that point, failure to route NVIDIA workloads to CUDA is itself a bug.

## Handoff Work Packages

Use these as independent execution chunks:

1. Dependency/MSRV and false-claim removal.
2. Device capability probing.
3. Binding plan and safe buffer ABI.
4. PTX emitter core and unsupported-IR errors.
5. Dynamic launch path.
6. External conformance tests.
7. Cache, pools, streams, and benchmarks.
8. Hardware intrinsics.
9. CUDA megakernel path.
10. Parameter Golf trainer wrapper.

Do not merge later packages before earlier correctness gates are green.
