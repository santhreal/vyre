# vyre-compiler

Compiles a validated vyre `Program` into a megakernel artifact for one device.

## Purpose

- Lowers `Program` IR to verified `KernelDescriptor` values (`vyre_compiler::lower`).
- Searches megakernel candidates, scores them by cost and objective, and selects one per device.
- Plans buffer allocation and resident memory for the selected candidate.
- Defines the artifact, envelope and target payload schemas and their digests.

## Boundaries

- Depends on `vyre-foundation` and `vyre-spec` only.
- Contains no device code, no driver calls and no target source text. Target code generation is in `vyre-emit`; execution is in `vyre-runtime`.
- Contains no domain algorithms. Reusable program builders are in `vyre-libs`.
- `vyre-emit`, `vyre-runtime`, `vyre-libs` and the facade `vyre` depend on this crate. It depends on none of them.
- An artifact schema change bumps the schema version and its digest domain in the same change.

## Layout

- `src/lower/`: IR to `KernelDescriptor` lowering, analyses and verification.
- `src/` root modules: candidate search, cost, selection, allocation, artifact schemas.
- `rules/`: data files for lowering schedules and kernel family surfaces.
