# Add an operation from outside the workspace

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

That is the whole registration. No workspace crate is edited, no central
list is appended to, and `OperationRegistry` is the only semantic lookup
either way.

`examples/external_ir_extension` is a standalone crate that does this and
is built by the `external-ir-demo` gate:

```sh
cargo run --manifest-path examples/external_ir_extension/Cargo.toml
```

It resolves the linked semantic operation, builds a validated
`ProgramGraph`, compiles the neutral artifact, attaches an
extension-owned target payload, and verifies the derived
operation-to-target facet.

## Tier

| Tier | Owner |
|---|---|
| `Foundation` | `vyre-foundation` |
| `Intrinsic` | `vyre-primitives`, Category C |
| `Library` | `vyre-libs`, Category A |
| `External` | a crate outside the workspace |
| `Unknown` | unresolved |

An out-of-tree operation is `External`. Inside the workspace, the choice
between `Intrinsic` and `Library` is not free: see
[the placement rule](../lego-block-rule.md).

## What the record owns and what it must not

A registration owns the stable operation id, the semantic tier, the neutral
`Program` builder, fixtures, laws and tolerance.

A semantic record carries no target compiler and no host function. Those
live elsewhere by ownership, not by convention: target identities, format
names, compilers and materializers belong to concrete target crates.

## Reference support is separate and optional

An operation with flat byte-call semantics may submit one
`vyre_reference::ReferenceFacet` from a crate that depends on
`vyre-reference`. This is a second, separate submission.

Missing reference support is an explicit absence. Nothing executes a
placeholder in its place, and nothing reports a fabricated answer because
the oracle was not implemented.

## Fixtures

`test_inputs` and `expected_output` are deterministic fixtures, and the
universal harness discovers an operation through its registration. An
operation that supplies fixtures is tested by the harness without anyone
writing a test for it; an operation that supplies none is untested by the
harness, which is a decision you are making rather than a step you skipped.

`tolerance` defaults to exact byte identity. Widen it with
`TolerancePolicy::f32_ulp(n)` only where the operation's own numerics
require it, because the operation owns that number and every backend is
held to it.

## Dialects are a view

A dialect or category is a derived namespace projection over the one
registry. It is not a second registry, and typed opaque IR extension
registrations stay separate from semantic operation registration.
