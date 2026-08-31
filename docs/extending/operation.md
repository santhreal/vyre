# Add an operation from outside the workspace

```rust
inventory::submit! {
    OperationRegistration::new_unconstrained(
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

`examples/external_ir_extension` is a standalone crate that does this, and the
`example-capability` gate builds it and runs it:

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

A registration contains the stable operation id, semantic tier, neutral
`Program` builder, fixtures, laws, tolerance, and one schedule-constraint
decision. Use an unconstrained constructor only when the canonical program
contains every semantic width, uniformity, scratch, cooperative-launch, and
memory-order requirement. Attach stronger neutral constraints with
`with_geometry_requirements`.

A semantic record contains no target compiler or host function. Target
identities, formats, compilers, and materializers remain in concrete target
crates.

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

`numeric` defaults to exact byte identity. Widen it with
`NumericContract::ieee_f32(n)` only where the operation's own numerics
require it, because the operation owns that bound and every backend is
held to it. The same contract states whether a schedule may reassociate the
combines, what the reduction accumulates in, and whether an approximate
native instruction is admitted, so a schedule that changes any of those
either proves it stays inside the bound or is refused.

## Dialects are a view

A dialect or category is a derived namespace projection over the one
registry. It is not a second registry, and typed opaque IR extension
registrations stay separate from semantic operation registration.

## Starting from the scaffold

`examples/libs-template` is a cargo-generate scaffold for a Category A dialect
crate: one typed builder over `TensorRef`, the validation helpers
`check_tensors`, `check_same_shape` and `checked_element_count`, and an
`examples/libs-template/tests/cat_a_conform.rs` that asserts byte identity
against `vyre_reference::reference_eval`. Its placeholders are
`{{crate_name}}`, `{{crate_name_snake}}` and `{{gh_org}}`.

The `example-capability` gate renders it, patches every dependency this
checkout provides at the checkout, and runs its conformance test, so the
scaffold cannot drift from the surface it is written against.
