# vyre architecture

The public facade exposes the canonical compiler and artifact workflow. It
keeps frontend IR and product entry points convenient without re-owning driver
implementation contracts.

## Module

`src/lib.rs` is the only file. Each facade export has one authoritative owner
in the workspace.

## Public surface

- **`vyre::ir`** exposes frontend IR such as `Program`, `Node`, `Expr`,
  `BufferDecl`, and `DataType` from `vyre-foundation`.
- **`vyre::compiler`** exposes whole-program compilation, immutable artifacts,
  target payloads, and target compiler facets from `vyre-megakernel`.
- **`vyre::runtime`** exposes artifact compilation, materialization, typed
  submission, completion, and recovery from `vyre-runtime`.

Backend implementation contracts remain under `vyre-driver`. Concrete driver
crates implement `VyreBackend`, `TargetCompiler`, and materializer facets.
Callers that intentionally use lower-level backend dispatch import
`DispatchConfig`, `BackendError`, and backend registry APIs from
`vyre-driver`; the facade does not duplicate those paths.

## Integration points

- `docs/CRATE_OWNERSHIP.toml` defines crate ownership.
- `docs/ARCHITECTURE.md` defines the canonical compiler lifecycle.
- Public facade workflows terminate in `ArtifactInstance` and typed
  `Submission`; the facade does not compile raw `Program` values through
  backend-specific routes.
