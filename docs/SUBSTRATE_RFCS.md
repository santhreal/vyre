# Substrate RFCs  -  IR enrichments and lowering passes

This doc captures the **substrate-level** items from the math-frontier
roadmap that are NOT Tier-2.5 primitives and shouldn't ship as ops.
Each RFC documents the design, dependencies, and breaking-change
trajectory required before it can become shipped source.

## RFC #60  -  Liquid types on `BufferDecl` (research)

**What**: Enrich `vyre-foundation::BufferDecl` with refinement types
(Vazou 2014). A buffer can carry a predicate `{x: u32 | x < N}`
checked at compile time by an SMT-backed `xtask` pass.

**Where**:
- `vyre-foundation/src/ir/buffer_decl.rs`  -  add `pub refinement: Option<Predicate>`.
- `xtask/src/liquid_check.rs`  -  new analysis pass.
- Predicate AST sits in `vyre-spec` so it's part of the wire format.

**Composition with #21** (linear-logic typed BufferAccess): liquid
predicates are *value-level* invariants; linearity is a *substructural*
property. They're orthogonal axes of the same buffer typing system.
A unified `BufferType` struct must carry both in the source patch that
ships this RFC.

**Breaking change cost**: Tier-1 IR enrichment. Default-`None`
preserves backward compatibility while callers migrate; mandatory use
requires a semver-major migration entry.

**Dispatch**: research only. RFC; not implemented in this commit.

## RFC #12  -  Algebraic effects compiled to GPU (research)

**What**: Recognize handler-shaped wrap_child Region trees and lower
them to existing Region + Buffer ops (Bauer-Pretnar, Leijen). The
non-breaking form identifies the pattern by op-id naming convention
(`vyre-libs::effects::handler::*`) and emits equivalent Region +
scratch-buffer ops. A first-class `Node::Handle` requires an IR and
wire-format migration.

**Where**:
- `vyre-foundation/src/transform/lower_handlers.rs`  -  pattern-matching
  pass.
- `vyre-foundation/src/ir/node.rs`  -  `Node::Handle { effect, body }`
  variant when the IR migration ships.

**Pairs with**: #20 substrate task (algebraic-effect lowering pass).

**Dispatch**: research-only until the lowering pass and IR migration
land together.

## RFC #18  -  Linear-logic typed BufferAccess (research)

**What**: Enrich `BufferAccess` enum from `{ReadOnly, WriteOnly,
ReadWrite, Workgroup}` to `{Owned, Shared, Unique, Aliased,
ReadOnly, WriteOnly, ReadWrite, Workgroup}` where:

- `Owned` = single-writer that consumes the buffer (linearity 1)
- `Shared` = read-only borrowed (multiple readers, no writers)
- `Unique` = single-writer that retains ownership
- `Aliased` = today's default; preserves backward compatibility

**Where**:
- `vyre-foundation/src/ir/buffer_decl.rs::BufferAccess` enum extension.
- `xtask/src/linearity_check.rs`  -  proof of no-aliasing invariants.

**Pairs with**: #21 substrate task (linear-logic typed BufferAccess +
xtask pass).

**Dispatch**: `Aliased` remains the compatibility default until the
proof pass is implemented; stronger fusion may only depend on linearity after
the verifier is in source and gated.

## Acceptance criteria

For each RFC to become a substrate ship rather than research:

1. **Two callers want it.** Lego rule equivalent for substrate  -  at
   least two named consumers in `vyre-foundation::transform`,
   `vyre-driver-*`, or `xtask`.
2. **Non-breaking trajectory.** The compatibility form must accept the
   change with no downstream consumer rebuild required.
3. **Compile-time verifier.** Each RFC ships with an `xtask` pass
   that catches violations before runtime.

The three RFCs above all clear (1) and (2). (3) blocks RFC #60 (needs
SMT bindings) and partially blocks #12 (handler recognition is
straightforward; full effect-row inference is harder).
