# Two-Level Optimization Architecture

vyre employs a two-level optimization pipeline, analogous to the design
used by LLVM (IR passes → MachineInstr passes) or GCC (GIMPLE → RTL).

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Source (C, DSL, ...)                                  │
│        │                                                │
│        ▼                                                │
│   ┌─────────────┐                                       │
│   │  Program IR  │ ← vyre-foundation IR (tree of Nodes) │
│   └─────┬───────┘                                       │
│         │  87 passes in vyre-foundation/optimizer/passes │
│         │  - algebraic (const_fold, strength_reduce, ...) │
│         │  - loops (unroll, fusion, LICM, ...)           │
│         │  - memory (dead_store_elim, store→load fwd)   │
│         │  - fusion/CSE/DCE                              │
│         │  - dataflow, megakernel                        │
│         ▼                                                │
│   ┌──────────────────┐                                   │
│   │ KernelDescriptor │ ← vyre-lower linearized form      │
│   └──────┬───────────┘                                   │
│          │  41 rewrites in vyre-lower/rewrites/           │
│          │  Operates on flat op lists with result IDs    │
│          │  Re-uses foundation eval rules where possible │
│          ▼                                                │
│   ┌─────────────────┐                                    │
│   │ Backend Codegen │ ← PTX / SPIR-V / WGSL / NAGA      │
│   └─────────────────┘                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## IR Representations

### Program (Foundation IR)

Tree-structured. Nodes contain Exprs that may be nested arbitrarily deep.
Buffer declarations, control flow (If/Loop/Region), and expressions
live in a single unified tree.

**Owned by:** `vyre-foundation`

### KernelDescriptor (Lower IR)

Linearized. Every operation has a result ID and a flat operand list
(indices into other results or a literal pool). The op list is a linear
sequence  -  there is no recursive tree structure.

**Owned by:** `vyre-lower`

## Rule Sharing

Despite operating on different IRs, the two levels share algebraic
evaluation rules to guarantee semantic consistency:

| Shared logic | Location | Used by |
|---|---|---|
| `fold_binary_literal` | `foundation::ir::eval` | `lower::descriptor_const_fold` |
| `fold_unary_literal` | `foundation::ir::eval` | `lower::descriptor_const_fold` |
| `fold_fma_literal` | `foundation::ir::eval` | `lower::descriptor_const_fold` |
| `fold_cast_literal` | `foundation::ir::eval` | `lower::descriptor_const_fold` |
| `strength_reduce_power_of_two_shift` | `foundation::optimizer::algebraic_rules` | `lower::strength_reduce` |
| `identity_elim_*` | `foundation::optimizer::algebraic_rules` | `lower::identity_elim` |
| `BinOp`, `UnOp`, `DataType` | `foundation::ir` | All lower rewrites (type imports) |

## Independent Passes

Passes that are structurally tied to their IR representation operate
independently at each level. This is expected and correct:

- **CSE**: Hash-consing a tree (foundation) vs deduplicating flat result
  sequences (lower) requires fundamentally different algorithms.
- **DCE**: Liveness in a tree (walk children) vs liveness in a flat op
  list (scan result references) differs structurally.
- **Loop transformations**: Loop nodes in Program IR have explicit loop
  bounds and bodies; in KernelDescriptor they are represented as op
  sequences with conditional jump patterns.

## Design Invariant

> Foundation passes must produce semantically equivalent Programs.
> Lower rewrites must produce semantically equivalent KernelDescriptors.
> Both levels use the same evaluation rules for literal folding.
> Neither level may introduce new algebraic rules without updating
> the shared `foundation::ir::eval` module.
