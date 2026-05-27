# Direct-WGSL Emission Sweep  -  2026-04-21

Audit question: **do any vyre ops emit WGSL directly, bypassing the
vyre IR layer?**

Expected answer: **no.** Ops must emit `fn(...) -> vyre::ir::Program`
compositions; only `vyre-driver-wgpu::lowering::naga_emit::emit_module`
(and the equivalent SPIR-V backend) may produce shader source, and
only by lowering IR through Naga.

## Scope

Searched every `.rs` and `.wgsl` file under
`libs/performance/matching/vyre/` for:

- `naga::back::wgsl::write_string`
- `naga::back::spv::{write_vec, Writer}`
- `@compute`, `@workgroup_size(`, `atomicAdd(`, `atomicCompareExchange`,
  `var<storage`
- `include_str!(".*\.wgsl")`, `include_wgsl!(` in non-backend crates
- `wgsl_source` functions
- `inventory::submit!(CompilerPrimitiveShader { ... })` consumers

## Findings

### FINDING-WGSL-1 (BLOCKER)  -  vyre-foundation ships six WGSL-only compiler primitives

**Where:** `vyre-foundation/src/transform/compiler/`

Six primitives (`dataflow_fixpoint`, `dominator_tree`,
`recursive_descent`, `string_interner`, `typed_arena`, `visitor_walk`)
each expose:

```rust
pub const fn source() -> &'static str {
    include_str!("wgsl/<op>.wgsl")
}
```

The six `.wgsl` assets live at
`vyre-foundation/src/transform/compiler/wgsl/*.wgsl`, baked into the
crate via `include_str!`.

**What the sibling `shader_provider.rs` module doc explicitly forbids
(file:line):**

> `vyre-foundation/src/transform/compiler/shader_provider.rs:3-6`
>
> "Law B forbids `.wgsl` asset files under `vyre-foundation`. The six
> compiler primitives (dataflow_fixpoint, dominator_tree,
> recursive_descent, string_interner, typed_arena, visitor_walk) still
> have GPU kernels, but those kernel assets live in
> `vyre-driver-wgpu`."

The `CompilerPrimitiveShader` inventory registry + `wgsl_source(op) ->
Option<&'static str>` resolver (`shader_provider.rs:20-56`) is the
already-designed replacement: driver crates should register the WGSL
strings via `inventory::submit!` and the primitive ops should fetch
them through the resolver. **Today the registry is never populated  - **
`grep -rn "inventory::submit!\s*\{\s*CompilerPrimitiveShader"` finds
zero call sites.

**Why it matters (architectural impact):**

1. **Cross-backend parity is impossible.** These six primitives have
   no `fn(...) -> Program` vyre IR path. `vyre-driver-spirv` cannot
   run them (no IR to lower), nor can `vyre-driver-photonic`. The
   parity matrix Codex is building **cannot include these ops**.
2. **Bypasses vyre-foundation's own optimize + validate passes.** The
   IR-level CSE / DCE / region_inline never see these primitives'
   bodies; they are opaque WGSL to everything above the wgpu
   backend.
3. **Region chain invariant unenforceable.** There's no `Node::Region`
   around a raw WGSL string, so `cargo xtask print-composition
   <op_id>` cannot walk from a deep node back to the rule of origin.
4. **Shadow execution only half-works.** Each primitive has a CPU
   `compute_*` fn (good  -  LAW 5), so vyre-reference can still match
   outputs; but the CPU path and the GPU path are two independent
   implementations with no IR between them. Any drift between the
   two is a correctness hole the existing tests cannot detect by
   structural means.
5. **Violates the four-tier rule.** Tier 1 (`vyre-foundation`)
   explicitly defines "IR model, wire format, frozen contracts. No
   ops." These primitives ARE ops, and they live in Tier 1.

**Concrete file list (each needs remediation):**

- `vyre-foundation/src/transform/compiler/dataflow_fixpoint.rs:17`
- `vyre-foundation/src/transform/compiler/dominator_tree.rs:19`
- `vyre-foundation/src/transform/compiler/recursive_descent.rs:17`
- `vyre-foundation/src/transform/compiler/string_interner.rs:20`
- `vyre-foundation/src/transform/compiler/typed_arena.rs:17`
- `vyre-foundation/src/transform/compiler/visitor_walk.rs:17`
- plus their six `wgsl/*.wgsl` assets.

**Remediation (non-exhaustive, one of three paths):**

- **Path A (IR-first, preferred):** Rewrite each primitive as
  `fn(...) -> vyre::ir::Program` producing proper Node/Expr trees. CPU
  ref stays; vyre-driver-wgpu lowers via `naga_emit`. SPIR-V +
  photonic gain support "for free." This is the pattern the rest of
  vyre-libs uses (`c11_lexer` in the parser smoke test is the
  reference shape  -  see `vyre-libs/src/parsing/c11/lexer.rs:9-175`).
  Cost: six rewrites, bounded. Benefits: parity + optimization.
- **Path B (registry-driven, honors the existing doc):** Move the
  `.wgsl` files to `vyre-driver-wgpu/src/shaders/compiler/`. In
  `vyre-driver-wgpu` add six
  `inventory::submit!(CompilerPrimitiveShader { op: "...",
  wgsl_source: || include_str!("...") })` registrations. Delete the
  `source()` fns and `include_str!` in vyre-foundation. Primitives
  that need the WGSL call `shader_provider::wgsl_source(op_id)`. This
  is what the doc already promises; it's a mechanical move. Cost: one
  PR across two crates. Benefits: Tier 1 stays clean; SPIR-V /
  photonic still blocked until Path A lands but the boundary
  violation is gone.
- **Path C (explicit opt-out):** Mark the six primitives as
  wgpu-exclusive in a documented feature gate + `#[cfg(feature =
  "wgpu-compiler-primitives")]`, and declare SPIR-V / photonic
  explicitly don't support them. Worst path but worth naming  -  the
  current state is a silent version of this with no flag and no doc.

**Recommendation:** Path B immediately (cheap, unblocks the Tier 1
contract audit), Path A as a follow-up Phase.

### FINDING-WGSL-2 (MINOR)  -  dead WGSL file in vyre-driver-wgpu/src/shaders/

**Where:** `vyre-driver-wgpu/src/shaders/aho_corasick_scan.wgsl`

No Rust code references this file. `grep -rn
"aho_corasick_scan\|shaders/aho_corasick"` inside `vyre-driver-wgpu/`
finds only the file itself. The canonical aho-corasick matcher is a
vyre IR op in `vyre-libs::matching` (proper path). This WGSL is a
pre-IR artifact.

**Remediation:** Delete the file + any empty `shaders/` directory.
Verify with `cargo build -p vyre-driver-wgpu --all-features` that
nothing breaks.

## Non-findings (false positives vetted and cleared)

- **`vyre-driver-wgpu/src/lib.rs:395-410`**  -  raw WGSL inside
  `WgpuBackend::probe_op`. Gated behind `#[cfg(feature =
  "parity-testing")]`. Feature docs: "Enables `WgpuBackend::probe_op`  - 
  a parity-testing shortcut that bypasses vyre IR + validation + conform.
  Off by default so production builds cannot accidentally route through
  the unvalidated path." Intentional, documented, feature-gated. Clear.
- **`vyre-driver-wgpu/src/lowering/mod.rs:160`**  -  sole legitimate
  `naga::back::wgsl::write_string` call: it's the backend emitter that
  consumes a Naga module produced by `naga_emit::emit_module(Program)`.
  Correct layer-cake.
- **`vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs`**  - 
  takes `wgsl_source: &str`; it's the internal glue between
  `naga::back::wgsl::write_string` output and
  `wgpu::ComputePipelineDescriptor`. Expected.
- **`vyre-driver-spirv/src/backend.rs`**  -  uses
  `naga::back::spv::{Writer, write_vec}` to emit SPIR-V from a Naga
  module. Mirror of the wgsl backend. Correct.
- **`vyre-libs/tests/c11_parser_integration.rs`** +
  **`vyre-driver-wgpu/tests/megakernel_emit.rs`**  -  grep for
  `@compute` / `atomicAdd` in emitted WGSL output to *verify the
  backend emitted the expected shape*. Tests are allowed to inspect
  backend output. Correct.
- **`scripts/check_no_string_wgsl.sh`**  -  the existing CI gate that
  tries to detect exactly this class of violation. It currently misses
  the six compiler primitives (they hide behind `include_str!` rather
  than inline `@compute` strings); FINDING-WGSL-1 should land
  alongside a gate enhancement that also flags `include_str!(".*\.wgsl")`
  outside `vyre-driver-*`.

## Summary

| Severity | Count | Description |
| --- | --- | --- |
| BLOCKER | 1 | Six Tier-1 ops emit direct WGSL via `include_str!` in `vyre-foundation` |
| MINOR | 1 | Orphan `aho_corasick_scan.wgsl` in `vyre-driver-wgpu/src/shaders/` |

Neither was produced by this session's work (Phase 0-3, 8, 1, parser
smoke). Both predate the warpscan perf plan. Both should block the
next claim of "vyre is release" until remediated.
