# The operation registry

```rust
use vyre_foundation::operation::{OperationRegistration, OperationTier};

inventory::submit! {
    OperationRegistration::library(
        "vyre-libs::math::gelu",
        build_gelu,
        Some(FIXTURE_INPUTS),
        Some(FIXTURE_OUTPUTS),
    )
}
```

An operation exists because a crate submitted one `OperationRegistration`
into the `inventory` collection declared at
`vyre-foundation/src/operation/mod.rs`. There is no central list to append to
and no second place to declare an operation.

## What a registration carries

| Field | Meaning |
|---|---|
| `id` | stable operation identifier |
| `semantic_version` | semantic schema version, 1 by default |
| `signature` | explicit signature; when absent, the built program is authoritative |
| `tier` | `OperationTier::Intrinsic` or `OperationTier::Library` |
| `category` | coarse taxonomy label |
| `build` | neutral `fn() -> Program` builder |
| `test_inputs` | deterministic fixture inputs |
| `expected_output` | deterministic fixture outputs, or a reference-oracle projection |
| `laws` | algebraic or semantic law identifiers |
| `tolerance` | numerical comparison policy |

Three constructors set the tier for you. `OperationRegistration::library`
is Category A, owned by `vyre-libs`.
`OperationRegistration::primitive` is Category C, owned by
`vyre-primitives`. `OperationRegistration::new` takes the tier explicitly.

`tolerance` defaults to `TolerancePolicy::EXACT`, which is byte identity.
`TolerancePolicy::f32_ulp(n)` accepts drift measured in ULPs, and the
operation owns that number rather than a backend deciding it.

## Effects are derived, not declared

`OperationEffects::from_program` reads effects off the program: buffer
access modes decide reads and writes, a non-zero atomic count decides
`atomics`, and a node barrier or a distributed collective decides
`synchronizes`. An operation cannot claim to be side-effect free while
declaring a `ReadWrite` buffer, because nothing asks it to claim anything.

## The inventory

`docs/generated/OP_SCHEMA.json` is the live catalog at schema version 4:
363 operations, 165 intrinsic and 198 library. It is generated from the
registry, so it is the file to read and not a file to edit.

Reading it is how you check whether an operation already exists before
adding one. See [the placement rule](../lego-block-rule.md) for what to do
with the answer.

## Walking one operation

```sh
./cargo_full run --bin xtask -- print-composition --op-id <id>
```

That walks a registered operation's region tree. Each child region names
the operation it composes, so the chain is the composition, and a region
that names no operation carries an `inline::` or `anonymous::` prefix.
