# Linked Registration Contract

Applies to Vyre 0.7.2.

Link-time registration is authoritative for extensible runtime surfaces. Never
depend on `inventory` iteration order. Freeze a registry into a sorted,
validated view before exposing deterministic behavior.

## Runtime collections

| Registration | Owner | Purpose |
|---|---|---|
| `OpDefRegistration` | `vyre-driver::registry` | Frozen operation definitions used by drivers |
| `BackendRegistration` | `vyre-driver::backend::registry` | Backend factories and stable backend IDs |
| `ExtensionRegistration` | `vyre-foundation::dispatch::extension` | Opaque IR extension ownership |

Semantic operations enter through the foundation-owned
`OperationRegistration` registry. Library, primitive, intrinsic, and driver
catalogs are filtered views over that authority. The generated
`docs/generated/OP_SCHEMA.json` joins registrations with built programs,
backend evidence, laws, and composition chains.

Duplicate stable IDs, invalid metadata, and registry drift must fail before
dispatch.
