# NVIDIA driver JIT miscompiles f32 cross-lane `shfl` on Blackwell (sm_120)

## Summary
On an **RTX 5090 (compute_cap 12.0 / sm_120), driver 570.211.01, CUDA 12.8**,
the NVIDIA driver's PTX→SASS JIT (`cuModuleLoadData`) **silently miscompiles
cross-lane warp operations on a globally-loaded value**: every lane receives
*its own* value instead of the full cross-lane result. Confirmed for BOTH:
- f32 `shfl.sync` (value moved through an `f32`↔`b32` `mov.b32` bitcast) —
  breaks f32 `subgroup_shuffle`/`subgroup_broadcast`/`subgroup_reduce`;
- integer `redux.sync.{add,min,max,…}` — breaks u32 `subgroup_reduce`.

This is an **NVIDIA driver JIT bug, NOT a vyre bug.** vyre emits correct PTX:
the *same* PTX compiled ahead-of-time by `ptxas` 12.8 — and the equivalent
`__shfl_sync` / `__reduce_add_sync` built by `nvcc -arch=sm_120` — produce the
correct cross-lane result. The one case that JITs correctly is a `shfl` on a
**register-computed** value (`subgroup_local_id`), which is what first masked the
bug; every failing case operates on a value just **loaded from global memory**,
so the trigger looks like a load→cross-lane ordering/scheduling miscompile in
the JIT (exact driver-internal cause is theirs to fix; AOT sidesteps it).

## Evidence (all on the 5090, reproduced 2026-06-19)
All probes launch one full 32-lane warp (`grid=[1,1,1]`, `workgroup=[32,1,1]`);
geometry was verified — laneids `[0..31]`, `subgroup_size=32`, `ballot(true) =
0xffffffff` on every lane, so the warp is full and converged.

| probe (one warp, lane i input `i+1` unless noted) | result | verdict |
|---|---|---|
| `subgroup_shuffle(laneid_u32, 5)` (u32, no bitcast) | `[5,5,…,5]` | ✅ shfl works |
| `subgroup_shuffle(in_f32, 5)` (f32, via `mov.b32`) | `[1,2,…,32]` (own) | ❌ returns own |
| `in_f32 + subgroup_shuffle(in_f32, 5)` | `[1,2,…,32]` (= in+0) | ❌ shuffle≡0 here |
| `subgroup_add(in_f32)` (xor butterfly) | lane0 `1` (≠ 528) | ❌ acc never combined |
| `subgroup_max(in_f32)` | lane0 `-3` (≠ 12.5) | ❌ acc never combined |
| `redux.sync` u32 `subgroup_add(in)` | lane0 `1` (≠ 528) | ❌ returns own |
| `redux.sync` u32 `subgroup_max(in)` | lane0 `1` (≠ 63) | ❌ returns own |

Ground-truth comparison via AOT (`nvcc -arch=sm_120`, ptxas, **no JIT**):
- `__shfl_sync(0xffffffff, v, 5)`: `6 6 6 … 6` ✅ (vyre JIT: `[1,2,…,32]` ❌)
- `__reduce_add_sync(0xffffffff, in[g])`: lane0 `528`, lane31 `528` ✅
  (vyre JIT: lane0 `1` ❌)
And vyre's own emitted u32-reduce PTX (`redux.sync.add.u32 %d, %a, %mask`) is
accepted by offline `ptxas -arch sm_120` — so the PTX is well-formed; only the
driver JIT corrupts it.

The emitted PTX is textbook-correct (verified by `cuobjdump -sass` of an
offline `ptxas -arch sm_120` build): `mov.b32 %r,%f; shfl.sync.idx.b32 %r2,%r,
0x5,0x1f,%mask; mov.b32 %f2,%r2` — the offline SASS contains the correct
`SHFL.IDX` reading lane 5. Only the **driver JIT** path is wrong.

The contradiction that pinpointed it: f32-shuffle-alone returns the *own* value
while f32-shuffle-inside-an-add contributes *zero* — i.e. the driver JIT drops
the SHFL's data movement entirely, leaving the destination as either the source
register (store-alone) or an effectively-neutral operand (in arithmetic). The
u32 path (no `mov.b32` around the shfl) is immune.

## vyre status (what is correct today)
- **PTX emit is correct.** `vyre-emit-ptx` lowers f32 `subgroup_reduce` to an
  XOR all-reduce using `shfl.sync.idx` with an explicit `laneid ^ offset`
  source (every lane ends with the full reduction). Unit tests assert the
  instruction shape; the reference oracle asserts the all-lane semantics.
  (The earlier `shfl.sync.bfly` form was also correct PTX; switching to `.idx`
  did not change the runtime symptom, confirming the bug is below the PTX.)
- **All four `subgroup_reduce` GPU parity tests** (f32 add/max + u32 add/max) in
  `vyre-driver-cuda/tests/subgroup_reduce_gpu_parity.rs` are `#[ignore]`d with a
  reason pointing here; they are the rerunnable reproducer once the fix lands.
  vyre's u32-reduce PTX is well-formed (offline `ptxas` accepts it; AOT
  `__reduce_add_sync` of the same instruction returns the full reduction), so
  the failure is purely the driver JIT.

## The fix: AOT cubin compilation via `ptxas` (also a perf win)
`vyre-driver-cuda` loads kernels with `cuModuleLoadData(ptx_text)`
(`backend/module_cache.rs::load_module`), i.e. it relies on the driver JIT.
The fix is to compile PTX → cubin **ahead of time with `ptxas`** (the toolkit
compiler, which is correct) and load the cubin, falling back to driver JIT only
when `ptxas` is unavailable — and then **loudly** (Law 10), since JIT may
miscompile f32 cross-lane ops on affected drivers.

Benefits beyond correctness: removes the per-kernel **runtime JIT cost** (Law 7)
and makes SASS reproducible/inspectable. Design constraints:
1. Locate `ptxas` (CUDA toolkit): `VYRE_PTXAS`, then `CUDA_HOME`/`CUDA_PATH`,
   then `PATH`, then known install roots. Cross-platform (`ptxas.exe` on
   Windows).
2. Invoke `ptxas --gpu-name sm_<cc> -O3 -o <tmp.cubin> <tmp.ptx>` for the
   probed `select_loadable_ptx_target_sm`; surface stderr on failure.
3. Add a cubin image loader (cubin is binary ELF — it does NOT use the PTX
   NUL-termination convention `load_cuda_module_data` enforces). Keep the
   single-FFI-source contract (`cuda_module_lifecycle_ffi_is_single_sourced`).
4. Cache cubins on disk keyed by (PTX source key, compute capability) — extend
   the existing PTX disk cache machinery; cubins are smaller than PTX text.
5. Choose AOT-vs-JIT via config/env, default **AOT when `ptxas` is present**.
6. Re-enable the two f32 GPU parity tests; add a differential test that the AOT
   and JIT cubins agree on the integer path and that AOT is correct on f32.

## Repro commands
```
# vyre runtime (driver JIT) — currently wrong on f32 (ignored tests):
cargo test -p vyre-driver-cuda --features cuda --test subgroup_reduce_gpu_parity -- --ignored --nocapture
# AOT ground truth — correct:
nvcc -arch=sm_120 shfltest.cu -o shfltest && ./shfltest   # prints 6 6 ... 6
```
