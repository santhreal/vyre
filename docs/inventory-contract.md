# Linked Registration Contract

Applies to Vyre 0.7.2.

Link-time registration is authoritative for extensible runtime surfaces. Never
depend on `inventory` iteration order. Freeze a registry into a sorted,
validated view before exposing deterministic behavior.

## Runtime collections

| Registration | Owner | Purpose |
|---|---|---|
| `OperationRegistration` | `vyre-foundation::operation` | Canonical semantic identity, signature, neutral builder, laws, fixtures, and tolerance |
| `ReferenceFacet` | `vyre-reference` | Portable reference implementation keyed by semantic operation ID |
| `BackendRegistration` | concrete driver crate via `vyre-driver` contract | Backend factory, validated `TargetId`, oracle classification, support set, compiler, and materializer facets |
| typed `Extension*Registration` and opaque resolvers | `vyre-foundation::dispatch::extension` | Opaque IR data, operator, expression, and node ownership |

Library, primitive, intrinsic, driver, conformance, and documentation catalogs
are derived views over the canonical semantic registry. The generated
`docs/generated/OP_SCHEMA.json` joins semantics with reference, target,
algebraic-law, Cargo-feature, and composition evidence.

Duplicate backend IDs, target IDs, target facets, and owner-local backend
metadata return `BackendError` during deterministic registry startup before
lookup or dispatch. Invalid linked semantic-operation or reference-facet
inventories abort their owner's first lookup with the conflicting identity.
Each registry freezes one sorted owned view and never selects a provider by
link order.
