# CODEX-IMPL Report

## Implemented

- DO 1: `WgpuBackend` now owns a backend-local `DispatchArena` and routes `dispatch_borrowed` through pooled size-classed buffers. Borrowed inputs are written to GPU buffers with `queue.write_buffer`; aligned hot-path chunks avoid intermediate input vectors, and readback remains the only owned output allocation.
- DO 2: `execute_chain` accepts `&[&[u8]]`, threads borrowed input through backend dispatch, and retains only per-step owned outputs needed to feed the next chain step.
- DO 3: `vyre-reference` expression evaluation linearizes expressions into flat opcodes and evaluates them with a `SmallVec` operand stack. The previous frame evaluator is retained as a test oracle, and a proptest cross-check compares both implementations.
- DO 4: `WgslDispatchExt::dispatch_wgsl` uses the shared `record_and_readback` command recording and readback path instead of duplicating bind group, dispatch, and map logic.
- DO 5: `WgpuBackend::acquire()` requests its own device and queue per backend instance. The old global `OnceLock<(Device, Queue)>` device singleton is gone; legacy `cached_device()` returns fresh devices and only records them for shared API compatibility.

## Verification

- `cargo check -p vyre-wgpu -p vyre-reference`: passed after implementation fixes.
- `cargo test -p vyre-reference prop_flat_evaluator_matches_frame_oracle -- --nocapture`: passed.
- `cargo check -p vyre-wgpu -p vyre-conform -p vyre-reference`: blocked by existing `vyre-conform-generate` unresolved `vyre_reference` errors before conform could complete.
- `cargo check --workspace`: not green in the current tree for the same `vyre-conform-generate` dependency errors; those are outside the CODEX-IMPL wgpu/reference/conform execution changes.

## Notes

- `dispatch_wgsl` remains an extension API on `vyre_wgpu::WgslDispatchExt`, matching the current architecture where raw WGSL dispatch is no longer part of the substrate-neutral `VyreBackend` trait.
- The branch was already at `origin/main` after the latest automated commit. Earlier `git pull --rebase` was blocked by unrelated dirty worktree changes, so implementation edits were kept scoped to the requested files and follow-up fixes.
