# vyre-lower

Verified representation lowering, read-only analysis, and verification for
Vyre's substrate-neutral `KernelDescriptor` IR.

This crate sits between vyre's frontend (which produces a high-level
`vyre::Program`) and the substrate-specific emitters
(`vyre-emit-naga`, `vyre-emit-ptx`, `vyre-emit-spirv`). It owns:

- The `KernelDescriptor` IR: a flat, SSA-shaped, structured-control-
  flow program that every emitter consumes verbatim.
- Bounded representation canonicalization after semantic optimization and
  descriptor construction.
- Read-only analyses that report on the IR (coalescing, bank conflicts,
  shared-memory candidates, def-use chains, and related shape facts).
- A structural verifier (`verify`) that catches dangling refs, duplicate
  result-ids, and out-of-range pool or child-body indices.

Production `Program` callers use `lower_verified`. Pure emitters consume the
verified descriptor without running another rewrite pipeline.

## Quick start

```rust
use vyre_lower::{
    verify_descriptor, BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody,
    KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};
use vyre_foundation::ir::DataType;

let desc = KernelDescriptor {
    id: "store_seven".into(),
    bindings: BindingLayout {
        slots: vec![BindingSlot {
            slot: 0,
            element_type: DataType::U32,
            element_count: None,
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadWrite,
            name: "out".into(),
        }],
    },
    dispatch: Dispatch::new(64, 1, 1),
    body: KernelBody {
        ops: vec![
            KernelOp { kind: KernelOpKind::Literal, operands: vec![0], result: Some(0) },
            KernelOp { kind: KernelOpKind::Literal, operands: vec![1], result: Some(1) },
            KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![0, 0, 1],
                result: None,
            },
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
    },
};

// Hand-built descriptors enter through the shared verification boundary.
let verified = verify_descriptor(&desc).unwrap();
assert_eq!(verified.id, "store_seven");
```

## The IR

A `KernelDescriptor` is:

- `id: String`: diagnostic.
- `bindings: BindingLayout { slots: Vec<BindingSlot> }`: buffers
  bound at the kernel boundary, looked up by `BindingSlot.slot` field.
- `dispatch: Dispatch { workgroup_size }`: thread-group geometry.
- `body: KernelBody`: the program.

A `KernelBody` is:

- `ops: Vec<KernelOp>`: flat op stream, walked linearly.
- `child_bodies: Vec<KernelBody>`: referenced by `If`/`ForLoop`/
  `Block`/`Region` ops via a child-body index in their operands.
- `literals: Vec<LiteralValue>`: pool, referenced by `Literal` ops.

A `KernelOp` is `{ kind: KernelOpKind, operands: Vec<u32>, result: Option<u32> }`.
Operands are typed by position per `KernelOpKind`: some positions
are SSA result-id refs, some are literal-pool indices, some are
binding slot ids, some are child-body indices.

**Per-body id space.** Each `KernelBody` has its own SSA id space.
Result ids in a child body do not exist in the parent body's id space.
`verify_descriptor` performs one bounded dependency-ordering walk for pure
same-body producers. Semantic optimization remains in `vyre-foundation`.


## Analyses

11 substrate-neutral analyses in `vyre_lower::analyses`:

- `coalesce`: memory-access coalescence per warp/workgroup.
- `shared_mem_promote`: global → shared-memory tile candidates.
- `bank_conflict`: shared-memory bank conflict detection.
- `vec_pack`: adjacent-load vectorization candidates (companion
  to `vyre_emit_naga::patterns::vec_pack`).
- `workgroup_uniform`: values uniform across a workgroup.
- `texture_promote`: read-mostly buffer → texture candidates.
- `layout_aos_to_soa`: AoS-to-SoA layout transform candidates.
- `const_buffer_promote`: uniform-buffer promotion candidates.
- `dead_op`: result-producing ops with no users (a less efficient
  cousin of `def_use::dead_by_no_use`).
- `common_subexpr`: equivalence groups for CSE.
- `def_use`: full def-use chains with per-body `UseSite`s.

Each analysis returns a serializable report. Run `audit::audit(desc)` for a
unified read-only `PerfAuditReport` with prioritized recommendations.
Emitter-specific `patterns::audit` functions report concrete target-strategy
opportunities without mutating the descriptor.

## Verifier

`verify(desc) -> Result<(), Vec<VerifyError>>` checks:

- Result-id uniqueness within each body.
- No dangling result-id refs.
- Literal-pool indices in range.
- Child-body indices in range.
- `Literal` ops have ≥1 operand.
- Per-kind minimum operand counts.

Errors are collected so one call reports every violation.
`lower_verified` verifies both the initial descriptor and the descriptor after
bounded representation canonicalization before any pure target emitter
receives it.

## See also

- `vyre-emit-naga` / `vyre-emit-ptx` / `vyre-emit-spirv`: pure target
  emitters that consume a verified `KernelDescriptor`.
- `vyre-foundation`: IR primitives (`BinOp`, `UnOp`, `DataType`,
  `MemoryOrdering`) that `KernelOpKind` embeds.

## License

MIT OR Apache-2.0.

<!-- BEGIN GENERATED CRATE CONTRACT -->
## Crate contract

This section is generated by `xtask crate-readmes --write` from
the crate manifest, release train, ownership registry, and crate-guide metadata.

### Purpose

Consume verified semantic programs and own the single backend-neutral lowering boundary and pre-emission transforms.

### Boundaries

The `lowering` owner maintains this `lowering` crate at `vyre-lower`.
Its allowed internal production dependencies are: `vyre-foundation`.
Any other normal or build dependency requires an ownership-registry change.

### Minimal real example

Run the checked-in behavior from `vyre-lower/tests/affine_access_map_contracts.rs`:

```console
./cargo_full test -p vyre-lower --test affine_access_map_contracts
```

### Features

- Manifest features: `default`, `test-fixtures`
- Default feature members: None

### Errors and unsupported behavior

Invalid IR, unsupported target operations, and failed lowering invariants stop emission with contextual diagnostics.

### Testing

See [`docs/testing/vyre-lower.md`](../docs/testing/vyre-lower.md) for the crate's test command,
hardware contract, expected skips, and failure semantics. It is generated
from `docs/testing/TESTING.toml`, which is authoritative.

### Release status

`vyre-lower@0.8.0` is a publishable crate on the current Vyre release train. Publication still requires the release evidence and user-approval gates.

### Ownership

[`docs/CRATE_OWNERSHIP.toml`](../docs/CRATE_OWNERSHIP.toml) is authoritative for this crate's
responsibility and allowed internal edges.

### License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option. See the workspace `LICENSE-APACHE` and `LICENSE-MIT` files.

<!-- END GENERATED CRATE CONTRACT -->
