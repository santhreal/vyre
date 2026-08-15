# External operation and target extension

This standalone crate registers one semantic operation and one target compiler
without editing a Vyre workspace crate.

## Run

```bash
cargo run --manifest-path examples/external_ir_extension/Cargo.toml
```

The executable resolves the linked semantic operation, builds a validated
`ProgramGraph`, compiles its neutral artifact, attaches the extension-owned
target payload, and verifies the derived operation-to-target facet.

## Semantic operation

Submit one `vyre_foundation::operation::OperationRegistration`. The record owns
the stable operation ID, semantic tier, neutral `Program` builder, fixtures,
laws, and tolerance. `OperationRegistry` is the only semantic lookup.

```rust
inventory::submit! {
    OperationRegistration::new(
        OPERATION_ID,
        OperationTier::External,
        Some(build_operation),
        None,
        None,
    )
}
```

An operation with flat byte-call semantics may separately submit one
`vyre_reference::ReferenceFacet` from a crate that depends on
`vyre-reference`. Missing reference support is an explicit absence. It does not
execute a placeholder.

## Target

Submit one `vyre_driver::BackendRegistration` from the target-owning crate. The
record carries the validated `TargetId`, supported operation IDs, pure
`TargetCompiler`, and optional materializer. The driver derives
`TargetOperationFacet` rows by joining this record with the canonical semantic
registry.

The fixture target emits the compiler-selected module as opaque wire bytes. It
has no device or dispatch path, so its factory and materializer fail explicitly.
A production target provides its own backend and materializer in the same
concrete driver crate.

## Boundaries

- Semantic records contain no target compiler or CPU function.
- Reference facets are owned by `vyre-reference`.
- Target identities, format names, compilers, and materializers are owned by
  concrete target crates.
- Dialect/category data is a derived namespace view, not a second operation
  registry.
- Typed opaque IR extension registrations remain separate from semantic
  operation registration.

See the workspace [`README.md`](../../README.md) and
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).
