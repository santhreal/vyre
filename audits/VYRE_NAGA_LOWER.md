# VYRE_NAGA_LOWER  -  Audit Report

**Scope:** `vyre-driver-wgpu/src/lowering/naga_emit/*`, `vyre-driver-wgpu/src/lowering/mod.rs`, `vyre-driver-wgpu/src/pipeline.rs`, `vyre-driver-wgpu/src/pipeline_disk_cache.rs`, `vyre-driver-wgpu/src/spirv_backend.rs`, `vyre-driver-spirv/src/backend.rs`, `vyre-driver/src/backend/lowering.rs`.

**Date:** 2026-04-24

**Focus:** Expr/Node variant coverage, validation frequency, caching, wire round-trips, function-call lowering, clippy pedantic casts, dispatch geometry correctness.

---

## Findings

| SEVERITY | file:line | defect | fix |
|---|---|---|---|
| CRITICAL | `vyre-driver-wgpu/src/pipeline.rs:653` | Dispatch geometry ignores `workgroup_size[1]` and `[2]`; host divides `output_word_count` by `workgroup_size[0]` only. A shader with `workgroup_size(8,8,1)` gets 8× too many X workgroups (64× total thread oversubscription), causing out-of-bounds writes. | Compute total threads per workgroup as `workgroup_size[0] * workgroup_size[1] * workgroup_size[2]` and divide output size by that product. |
| CRITICAL | `vyre-driver-wgpu/src/pipeline.rs:223` | `compile_with_device_queue` calls `load_or_compile_disk_wgsl` **before** checking the in-memory pipeline cache. Every compile pays program clone + wire serialization + disk read even when the compiled pipeline is already hot in memory. | Add an in-memory `DashMap<(program_fingerprint, config_hash), Arc<str>>` WGSL cache tier, or restructure the pipeline cache check to precede any lowering/disk work. |
| HIGH | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:999` | `unreachable!()` guards the fallback of a subgroup `BinOp` match. If a new variant is added and the outer `matches!` is updated without the inner `match`, the shader compiler panics at runtime. | Replace `unreachable!()` with `Err(LoweringError::unsupported_op(...))` so the backend returns a structured error instead of aborting the process. |
| HIGH | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:377` | `buffer.element.size_bytes().ok_or_else(...)? as u32` silently truncates strides larger than 4 GiB. A future `DataType` with `size_bytes() > u32::MAX` would produce a corrupted shader stride that still passes naga validation. | Use `try_into().map_err(|_| LoweringError::invalid("stride overflows u32"))?` instead of `as u32`. |
| HIGH | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:107` + `vyre-driver-wgpu/src/lowering/mod.rs:147` | Naga validation runs **twice** on the WGSL path: once inside `emit_module`, again inside `write_wgsl`. Validation is O(n) on the IR; doubling it on every compile is pure overhead. | Remove validation from `emit_module`; validate only at the writer boundary and thread `ValidationInfo` through `WgpuProgram`. |
| MEDIUM | `vyre-driver-wgpu/src/pipeline_disk_cache.rs:346` | `normalized_compile_wire` clones the full `Program`, normalizes buffer counts, and serializes to wire bytes **solely** to hash a disk cache key. This is O(n) allocation + serialization on every compile call. | Compute the cache key directly from `program.fingerprint()` plus a lightweight normalized signature hash (buffer names + kinds, ignoring runtime counts) without cloning or serializing the full program. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:1283` | `(*b as u32) % 32` in `fold_binary_literal` loses sign on negative `i32` shift counts. The wrapped value differs from CPU-side `i32::wrapping_shl` semantics for negative counts. | Reject negative shift counts at fold time, or fold with semantics that exactly match the shader's `ShiftLeft`/`ShiftRight` behavior. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:1426-1434` | `fold_cast_literal` uses bare `as` casts (`f32 -> u32`, `u32 -> i32`, `f32 -> i32`, etc.). These silently truncate, wrap, or lose precision; folded constants may not match GPU runtime results for out-of-range values. | Use `try_from` with explicit range checks, or use `wrapping_`/`saturating_` variants that exactly match the target GPU semantics. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:108` + `vyre-driver-wgpu/src/spirv_backend.rs:43` + `vyre-driver-spirv/src/backend.rs:31` | Three different `Capabilities` sets are used for the same `naga::Module`: WGSL uses `all()`, wgpu-SPIR-V uses `empty()`, standalone-SPIR-V uses `all()`. A program using subgroups passes WGSL validation but fails wgpu-SPIR-V validation. | Unify capability negotiation in a single helper that derives capabilities from the actual adapter feature set; use the same set for all validators and backends. |
| MEDIUM | `vyre-driver-wgpu/src/pipeline.rs:374` | Pipeline cache eviction uses `pipeline_cache.iter().next()`  -  arbitrary DashMap shard order. Hot pipelines may be evicted while cold ones survive, causing unnecessary recompilation on repeated dispatches. | Migrate to a true LRU cache (e.g., `lru::LruCache`) or the shared `PipelineCacheKey` structure already planned in `vyre-driver/src/pipeline.rs`. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/mod.rs:151` | On naga validation failure, `write_wgsl` prints the full expression tree and function body via `println!`. This leaks potentially sensitive shader constants and buffer layouts into application logs/stdout. | Replace `println!` with `tracing::error!` gated behind a `shader_debug` feature flag, and scrub literal values before logging. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:22` | `fold_expr` recursively calls `left.as_ref().clone()` and `right.as_ref().clone()` for every non-foldable sub-expression. For a large tree with no foldable literals, it allocates a full copy of the entire expression tree before returning `None`. | Return `Cow<Expr>` from `fold_expr` so the non-foldable path borrows instead of clones. |
| MEDIUM | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:1221` | `rotate_width_bits` returns `64` for `U64`/`I64`, implying rotate ops are supported, but `emit_binop` rejects all arithmetic on `U64`/`I64` except bitwise/equality. A `RotateLeft` on `U64` passes the width check and then hits the arithmetic rejection, producing a confusing error sequence. | Move the `U64`/`I64` arithmetic rejection before `rotate_width_bits`, or add `U64`/`I64` to the `rotate_width_bits` reject list. |
| LOW | `vyre-driver-wgpu/src/spirv_backend.rs:42` + `vyre-driver-spirv/src/backend.rs:28` | Both SPIR-V emitters create a fresh `Validator` even when the module was already validated by `emit_module`. This is redundant work on the SPIR-V path. | Accept a pre-computed `ValidationInfo` parameter, or store it alongside `WgpuProgram` so emitters reuse it. |
| LOW | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:82` | `emit_module` takes `&vyre::DispatchConfig` but binds it to `_config` and never uses it. Specialization constants, workgroup overrides, or no-opt paths cannot be implemented without changing the signature. | Either use `config` to drive `naga::Override` constants / policy, or remove the parameter to make the dead argument obvious. |
| LOW | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:554` | `Expr::SubgroupAdd` is always lowered to `CollectiveOperation::Reduce`. There is no IR variant for inclusive/exclusive scan, so scan operations cannot be expressed. | Extend the IR with `SubgroupScan { inclusive: bool }` or document that scans must be composed manually from shuffles. |
| LOW | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:84` | `emit_module` unconditionally calls `vyre::ir::inline_calls` and `vyre::ir::optimize`. No `DispatchConfig` flag exists to skip optimization for debugging shader compilation issues. | Add `optimize: bool` to `DispatchConfig` and gate the optimize pass; default to `true` for production. |
| LOW | `vyre-driver-wgpu/src/lowering/mod.rs:208` | `static_workgroups` returns `[x, 1, 1]`, ignoring multi-dimensional workgroup sizes. While the runtime dispatch path computes workgroups dynamically, consumers of `WgpuProgram::dispatch_geometry` receive incorrect metadata. | Compute workgroups across all three dimensions or document that the field is 1D-only and deprecate it if unused. |
| LOW | `vyre-driver-spirv/src/backend.rs:28` | `SpirvBackend::emit_spv` returns `Result<Vec<u32>, String>` instead of a structured error type. Consumers must parse strings to distinguish validation failures from writer failures. | Return `Result<Vec<u32>, BackendError>` (or `LoweringError`) with actionable `Fix:` messages. |
| LOW | `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:529` | `Expr::Call` is rejected with a hard error rather than emitted as a `naga::Function`. The architecture correctly inlines upstream, but any optimizer bug that misses a call produces a user-visible error rather than a slower-but-correct shader. | (Optional degradation) Emit `naga::Function` + `Expression::Call` as a debug-build fallback so programs still run even when the inliner misses an edge case. |

---

## Focus-Area Analysis

### 1. Expr Variant Coverage
All `Expr` variants (`LitU32`, `LitI32`, `LitF32`, `LitBool`, `Var`, `Load`, `BufLen`, `InvocationId`, `WorkgroupId`, `LocalId`, `BinOp`, `UnOp`, `Call`, `Select`, `Cast`, `Fma`, `Atomic`, `SubgroupBallot`, `SubgroupShuffle`, `SubgroupAdd`, `Opaque`) have explicit arms in `emit_expr`. No `todo!()` or `panic!` exists. The single `unreachable!()` at `expr.rs:999` is a defensive time-bomb (Finding 3). `Expr::Call` is explicitly rejected (Finding 20) because `inline_calls` is required upstream.

### 2. WGSL Caching
There is **no** in-memory WGSL cache. The only cache tier is disk (`pipeline_disk_cache.rs`). The in-memory `pipeline_cache` (`DashMap<[u8; 32], Arc<CachedPipelineArtifact>>`) caches compiled `wgpu::ComputePipeline` objects, but it is checked **after** the disk read / WGSL lowering (Finding 2). The cache key includes adapter fingerprint + WGSL hash + Naga version, so it is technically per-(WGSL, backend), not per-(Program, backend_id). A program that lowers to identical WGSL shares the pipeline  -  correct, but the lack of an in-memory WGSL cache means every new process repeats the lowering.

### 3. Program-to-Wire Round-Trip
`normalized_compile_wire` (Finding 6) clones the program and calls `to_wire()` to produce cache-key bytes. There is no `from_wire()` round-trip in the lowering hot path  -  the wire bytes are only hashed and discarded. The redundant work is the serialization itself, not a round-trip.

### 4. Naga Validation Frequency
Validation runs:
- Once in `emit_module` (WGSL path)
- Once in `write_wgsl` (WGSL path)
- Once in `SpirvEmitter::emit` (wgpu-SPIR-V path, `Capabilities::empty()`)
- Once in `SpirvBackend::emit_spv` (standalone-SPIR-V path, `Capabilities::all()`)

That is **2× on WGSL**, **1× on wgpu-SPIR-V**, and **1× on standalone-SPIR-V** for the same module. Findings 5 and 14.

### 5. Function-Call Lowering
Calls are **inlined** before emission (`vyre::ir::inline_calls(program)` at `naga_emit/mod.rs:84`). No `naga::Function` is emitted for user calls. This is the correct architectural choice  -  compute shaders rarely benefit from function calls, and wgpu/Naga function call overhead is non-trivial. The hard error on residual `Expr::Call` (Finding 20) is a safety net.

### 6. Clippy Pedantic Casts
Running `cargo clippy -p vyre-driver-wgpu -- -W clippy::cast_possible_truncation -W clippy::cast_sign_loss -W clippy::cast_possible_wrap` surfaces **confirmed truncations** at:
- `mod.rs:377`  -  `usize -> u32` stride (Finding 4)
- `expr.rs:1283-1284`  -  `i32 -> u32` sign loss (Finding 7)
- `expr.rs:1298-1299`  -  `i32 -> u32` sign loss (Finding 7)
- `expr.rs:1355-1357`  -  `u32 -> i32` wrap (Finding 8)
- `expr.rs:1426-1434`  -  mixed `as` casts in constant folding (Finding 8)
- `utils.rs:14`  -  `usize -> u32` argument index (minor, WGSL arg limits make it practically safe)

---

## Competitor Comparison (Naga / WGSL Ecosystem)

| Competitor | Caching Strategy | Validation | Call Lowering |
|---|---|---|---|
| **vyre (current)** | Disk-only WGSL cache; in-memory pipeline cache checked **after** lowering | 2× on WGSL path | Inline before emission |
| **wgsl-analyzer / tint** | In-memory shader module cache keyed by source hash | 1× at parse/validation boundary | Inlined by default |
| **naga (standalone)** | No cache; validate-then-write once | 1× | Function calls emitted as functions |

**Gaps:** vyre is the only path that serializes the full program to wire bytes on every compile just for cache-key hashing. It is also the only path that validates the same module twice before dispatch.

---

## Recommendations

1. **Fix Finding 1 (CRITICAL) immediately**  -  multi-dimensional workgroup sizes are a correctness bug that can corrupt GPU memory.
2. **Fix Finding 2 (CRITICAL) immediately**  -  the cache architecture is upside-down; the in-memory check must precede disk I/O.
3. **Consolidate validation**  -  run `Validator` exactly once per module, store `ValidationInfo`, and pass it to all writers (WGSL, SPIR-V, etc.).
4. **Add an in-memory WGSL cache**  -  a `DashMap<(blake3(program.fingerprint), config_hash), Arc<str>>` in `WgpuBackend` eliminates disk I/O and redundant lowering for repeated programs.
5. **Replace all bare `as` casts in the lowering** with `try_into` or `From` conversions that fail fast with actionable errors.

---

*End of report. 20 findings. All files reviewed against LAWS 0–8.*
