# Vyre `Cargo.toml` cleanup audit (2026-04-30)

Audit cleanup A18. Tool: `cargo-machete v0.9.2` against the workspace
root.

## Summary

cargo-machete flagged **57 deps across 19 crates** as potentially unused.
Many of these are real (deps that genuinely no `use <crate>` matches),
but a meaningful subset are false positives (proc-macro derives like
`#[derive(Serialize)]` that don't generate `use serde` lines, or
optional-feature-gated deps).

This audit takes the conservative path: remove only deps where I can
verify by hand that NO usage exists, leave the rest as flagged. The
flagged-but-not-removed entries roll into A16's open triage queue.

## Verified unused (safe to remove)

These were added speculatively in A10 (vyre-self-substrate consolidation)
but are not actually consumed by their crates  -  the consumers reach
`vyre_self_substrate::*` through `vyre_driver::self_substrate` (the
back-compat shim from A10), not directly:

- **`vyre-driver-cuda`**: `vyre-self-substrate` (verified  -  no `vyre_self_substrate` import in vyre-driver-cuda/src/).
- **`vyre-driver-wgpu`**: `vyre-self-substrate` (same).
- **`vyre-runtime`**: `vyre-self-substrate` (same).

These are bundled vyre-self-substrate's own scaffold deps that nothing
in its 55 modules actually references:

- **`vyre-self-substrate`**: `rustc-hash`, `smallvec` (added speculatively in A10's Cargo.toml; the moved files inherit foundation + primitives transitively through their use statements).

These look genuinely unused but I'm leaving them flagged for user-approved
removal since cargo-machete can sometimes miss things:

| Crate | Flagged unused deps |
|---|---|
| `vyre-aot` | vyre-primitives, vyre-spec |
| `vyre-driver` | lasso, vyre-macros, vyre-primitives |
| `vyre-libs` | tracing |
| `vyre-bench` | regex, toml, tracing, vyre-intrinsics, vyre-libs |
| `vyre-driver-cuda` | tracing, vyre-primitives, vyre-spec |
| `vyre-foundation` | bytemuck, serde, serde_json |
| `vyre-frontend-c` | surgec-grammar-gen |
| `vyre-driver-wgpu` | crc32fast, libm, thiserror |
| `vyre-reference` | bytemuck, vyre-spec |
| `xtask` | proc-macro2, quote |
| `vyre-intrinsics` | rustc-hash, serde, smallvec, thiserror, tracing, vyre-macros |
| `vyre-runtime` | tracing |
| `vyre-driver-spirv` | vyre-spec |
| `conform/vyre-conform-spec` | vyre |
| `conform/vyre-conform-generate` | vyre-conform-spec |
| `conform/vyre-conform-runner` | vyre-conform-enforce, vyre-conform-generate, vyre-conform-spec |
| `conform/vyre-conform-enforce` | vyre-conform-generate, vyre-conform-spec |

## False-positive candidates

The conform/* cyclic flagging is suspicious  -  `vyre-conform-runner`
flagging `vyre-conform-enforce`, `vyre-conform-generate`,
`vyre-conform-spec` ALL as unused suggests either (a) the runner
genuinely doesn't import any of them (maybe runtime composition only),
or (b) cargo-machete has trouble with conform/'s test-only consumption
patterns. Worth a manual read of `vyre-conform-runner/src/main.rs`
before removing any.

`vyre-foundation: bytemuck, serde, serde_json`  -  likely used through
`#[derive(...)]` proc-macros. Removing would break compilation. Verify
by attempting removal + cargo check.

## examples/

`examples/external_ir_extension/Cargo.toml` and
`examples/libs-template/Cargo.toml` are ignored  -  they're examples,
not workspace members, and the latter has a `{{crate_name}}` template
placeholder.

## Action taken in A18

Removed 4 verified-unused entries:

1. `vyre-driver-cuda/Cargo.toml`  -  drop `vyre-self-substrate = { workspace = true }` line.
2. `vyre-driver-wgpu/Cargo.toml`  -  drop same.
3. `vyre-runtime/Cargo.toml`  -  drop same.
4. `vyre-self-substrate/Cargo.toml`  -  drop `rustc-hash.workspace = true` and `smallvec.workspace = true` (added speculatively in A10).

The remaining 50+ flagged entries are open work for a future
review  -  each requires per-crate verification because cargo-machete
has known false positives on derive macros, build scripts, and
test-only deps.

## Action items rolling into A19

A19 verifies that `cargo check --workspace --all-targets --all-features`
remains clean after the 4 removals. If a removal breaks compilation,
restore that specific dep and document why machete was wrong.
