# vyre-harness architecture

`vyre-harness` provides the registry and conformance helpers shared by Vyre
tests, benchmarks, and library operations. It re-exports the canonical region
composition helpers from `vyre-foundation`.

## Modules

### `lib.rs`

This module defines the Cat-A registry and its deterministic fixtures. It also
re-exports the region composition surface.

### `region.rs`

This module owns the convenience constructors `wrap`, `wrap_anonymous`, and
`wrap_child`. It re-exports `tag_program` and `reparent_program_children` from
`vyre_foundation::composition`.

`tag_program(parent_id, program)` preserves the existing `Program` metadata.
It wraps the entry in a parent region and keeps each primitive generator as a
child whose `source_region` names the parent.

## Public types

- `OpEntry` registers a runnable Cat-A operation and its fixtures.
- `region::tag_program` exposes the foundation-owned composition operation.
- `region::wrap_child` creates a named child composition edge.

## Integration points

- Library wrappers use `tag_program` when they return an existing primitive
  program under a product-facing operation id.
- The conform runner reads region generators when it builds certificate
  evidence.
