# Operation catalog

This guide is verified against Vyre 0.7.2.

Vyre publishes one generated operation schema at
[`docs/generated/OP_SCHEMA.json`](generated/OP_SCHEMA.json). Use that file when
you need an operation ID, tier, category, buffer signature, Cargo feature route,
reference oracle, backend support state, algebraic law, composition chain, or
operation count.

The schema is generated from linked `OpEntry` registrations, runtime `OpDef`
registrations, built `Program` buffers, and dialect parameter signatures. It
also reads Cargo manifests, linked algebraic-law inventories, and the backend
rows in [`docs/optimization/OP_MATRIX.toml`](optimization/OP_MATRIX.toml).
These live sources are checked together before the schema is written.

## Check the schema

Run this command from the Vyre workspace:

```text
cargo_full run --bin xtask -- operation-schema --check
```

The command fails when the generated JSON differs from the live registrations.
It also fails when an operation has a duplicate or unknown ID, an inconsistent
tier or category, an empty signature or feature route, no reference oracle, a
missing release-backend row, malformed laws, or an inconsistent composition
chain.

To regenerate the JSON after an intentional operation change, run:

```text
cargo_full run --bin xtask -- operation-schema
```

## Read the browsing views

The schema has two Markdown views:

- [`docs/generated/OP_INVENTORY.md`](generated/OP_INVENTORY.md) groups every
  operation by tier.
- [`docs/catalog/README.md`](catalog/README.md) groups every operation by
  subsystem.

Regenerate these views with the same live schema:

```text
cargo_full run --bin xtask -- list-ops --write docs/generated/OP_INVENTORY.md
cargo_full run --bin xtask -- catalog
```

These pages do not define operations or counts. The JSON schema remains the
authority.

## Understand an operation record

Each `operations` row contains the following fields:

- `id` is the stable registered operation identifier.
- `tier` is derived by the canonical `classify_op_id` function.
- `category` is the registration category. A legacy registration without an
  explicit category uses its registered ID namespace.
- `signature` records either built `Program` buffer declarations or runtime
  dialect input and output parameters. Program buffers include binding, name,
  access mode, memory kind, element type, static count, and live-output state.
- `features` names the Cargo feature route that links the registration into the
  complete catalog build.
- `oracle` records reference evaluation, fixture input, expected output, and ULP
  tolerance contracts.
- `backend_support` records the release status and proof paths for each backend.
- `laws` names linked algebraic-law registrations. An empty array means that the
  operation makes no algebraic-law claim. It does not mean that every law holds.
- `composition_chain` is read by walking nested `Region` nodes in the built
  program. A step with `registered: true` resolves to another live operation. A
  step with `registered: false` is an internal named stage.

For example, a composition can contain its own registered region and then a
registered primitive:

```text
vyre-primitives::text::encoding_classify
  vyre-primitives::reduce::range_counts_u32
  vyre-primitives::text::utf8_shape_counts
```

The chain is evidence about the current built IR. It is not a handwritten call
graph.

## Tiers and categories

The schema uses these tiers:

- `intrinsic` identifies Tier 2 hardware operations.
- `primitive` identifies Tier 2.5 reusable operations.
- `libs` identifies Tier 3 library compositions.
- `runtime` identifies driver-owned dialect operations that do not build a
  dispatchable `Program`.

Category is a domain taxonomy such as `hardware`, `math`, `matching`, `parsing`,
or `security`. Tier describes architectural placement. Category describes the
operation's subject. Do not infer either value from a Markdown heading or a
source directory when the generated schema is available.

## Backend support

A backend row reports the status declared by the operation matrix and the test
paths that support that declaration. `supported`, `experimental`,
`unsupported`, and `not_applicable` are distinct states. A missing row is an
invalid catalog state. It is never permission to select another backend
silently.

## Add or change an operation

Register the operation through its owning crate and build a real `Program`.
Declare deterministic fixtures and the category. Update the operation matrix
with the release-backend status and proof paths. Register an algebraic law only
when the law has a proving conformance test.

Then regenerate and check all three artifacts:

```text
cargo_full run --bin xtask -- op-matrix --write
cargo_full run --bin xtask -- operation-schema
cargo_full run --bin xtask -- list-ops --write docs/generated/OP_INVENTORY.md
cargo_full run --bin xtask -- catalog
cargo_full run --bin xtask -- operation-schema --check
cargo_full run --bin xtask -- catalog --check
```
