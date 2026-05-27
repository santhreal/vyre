# Optimization taxonomy

This file names the optimization classes Vyre recognizes and where each class
must live. Add to this taxonomy before creating a new optimization subsystem.

## Layer 1: IR-pure optimizer work

These optimizations transform Vyre IR or shared optimizer facts. Every backend
inherits them.

| Class | Home | Examples | Required proof |
|---|---|---|---|
| Arithmetic strength reduction | `vyre-foundation/src/optimizer/passes/strength_reduce/` | power-of-two div/mod, exact division, shift-add decomposition, constant reciprocal division | before/after IR test and reference parity |
| Algebraic canonicalization | `vyre-foundation/src/optimizer/passes/canonicalize*` | commutative operand ordering, identity elimination, normalized compare forms | canonical fingerprint or wire-byte test |
| FMA and expression synthesis | `vyre-foundation/src/optimizer/passes/strength_reduce/` or dedicated pass | `a*b+c`, `a*b-c`, negated multiply-add chains | IR-shape test and float tolerance/parity test |
| Loop unroll | `vyre-foundation/src/optimizer/passes/loop_unroll.rs` | fixed trip count loops, bounded small loops | node-count guard, scope correctness test |
| Vectorization | `vyre-foundation/src/optimizer/passes/vectorization.rs` | vec2/vec4 load-store widening from shape facts | shape fact test and backend lowering smoke |
| Fusion | `vyre-foundation/src/optimizer/passes/fusion.rs` and shared facts | producer/consumer fusion, decode-scan fusion | alias/effect safety test and cost gate |
| Shared fact graph | `vyre-foundation/src/optimizer/fact_substrate.rs` | var-use counts, buffer use facts, type facts, shape facts | single-walk proof and invalidation tests |
| Compile-time complexity reduction | same module as hotspot | O(n^2) scans, repeated hash construction, clone storms | adversarial size test or allocation/count proof |

Layer 1 must not inspect concrete backend names or emit target code.

## Shared driver work

These optimizations are backend-neutral runtime/driver policy.

| Class | Home | Examples | Required proof |
|---|---|---|---|
| Launch planning | `vyre-driver/src/launch.rs` | grid inference, workgroup validation, param words | shared launch tests |
| Binding layout | `vyre-driver/src/binding.rs` | input/output binding plan, output ordering | cross-backend ABI test |
| Validation cache | `vyre-driver/src/validation.rs` or cache module | fingerprint keyed validation skip | cache hit/miss tests |
| Pipeline identity | `vyre-driver/src/pipeline.rs` | canonical cache digest, feature flags | cross-backend fingerprint test |
| Residency policy | `vyre-driver/src/` when backend-neutral | handle lifecycle, inflight tracking | race/free tests |

Shared driver code may define neutral capability bits. It must not import
concrete backend SDK types.

## Layer 2: backend-specific lowering strategy

These optimizations are legal only inside the owning backend crate.

| Class | Home | Examples | Required proof |
|---|---|---|---|
| CUDA lowering | `vyre-driver-cuda/src/codegen/` | PTX instruction selection, tensor-core emission, warp ops | PTX smoke and CUDA conformance |
| CUDA launch/runtime | `vyre-driver-cuda/src/` | streams, events, module cache, resident buffers | live CUDA tests |
| wgpu/naga lowering | `vyre-driver-wgpu/src/lowering/` | naga IR, WGSL/SPIR-V emission details | naga/wgpu tests |
| wgpu runtime | `vyre-driver-wgpu/src/engine/` | buffer pool, readback ring, command submission | wgpu dispatch tests |
| SPIR-V lowering | `vyre-driver-spirv/src/` | SPIR-V binary emission and validation | SPIR-V tests; experimental unless matrix says otherwise |

Backend code must not duplicate a Layer-1 rewrite. If the backend needs a
safety net for a fold/rewrite, lift the evaluator to `vyre-foundation` and call
it from both places.

## Runtime megakernel work

| Class | Home | Examples | Required proof |
|---|---|---|---|
| Persistent queue protocol | `vyre-runtime/src/megakernel/protocol*` | ring status, slot publish/claim/done | protocol edge tests |
| Scheduler | `vyre-runtime/src/megakernel/scheduler.rs` | priority scans, tenant fairness, strided probing | scheduler model tests |
| IO | `vyre-runtime/src/megakernel/io.rs` | DMA request/completion queue | IO protocol tests |
| Resident runtime | `vyre-runtime/src/megakernel/resident.rs` | mirrors, update path, dispatch handles | resident contract tests |

Drivers call the megakernel runtime. They do not reimplement its scheduling
policy.
