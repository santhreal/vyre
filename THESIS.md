# Vyre Thesis

Vyre is a substrate-neutral whole-program compiler that turns typed frontend
programs into authenticated executable artifacts without leaking concrete
target details into domain logic.

## Core contract

- `vyre-foundation` owns semantic IR, `ProgramGraph`, validation, diagnostics,
  operation identity, and the fallible semantic optimizer.
- `vyre-megakernel` owns bounded whole-graph planning, canonical ABI, immutable
  artifact identity, and pure target compilation facets.
- `vyre-driver` owns backend-neutral materialization, device generation,
  binding, submission, completion, capability, and lower-level dispatch
  contracts.
- Concrete driver crates own target formats, device acquisition, native
  executable modules, and target-specific execution.
- `vyre-libs` owns every composition. `vyre-primitives` owns only uncomposable
  intrinsics.
- `vyre-runtime` owns sessions, recovery, residency, and readback. It executes
  the artifact's selected persistence. It does not choose that schedule.

## Architectural laws

- One production route: validated `ProgramGraph` to `Artifact`, authenticated
  `TargetPayload`, `ArtifactInstance`, typed `Submission`, and `Completion`.
- One semantic optimizer and one verified lowering boundary feed every target.
- Backend isolation: concrete APIs, target dialects, and hardware policy stay
  inside the owning concrete driver or emitter.
- Fail closed: missing facets, stale device generations, incompatible payloads,
  invalid bindings, and unavailable targets return structured diagnostics.
- Reference interpretation is an oracle, never a silent production fallback.

## Execution thesis

Whole-program composition must expose cross-operation legality, scheduling,
buffer lifetime, and specialization opportunities before target lowering.
Compilation and materialization happen outside submission hot loops. Recovery
reuses authenticated target bytes rather than repeating semantic compilation.

Performance evidence is part of correctness. Extra copies, stale caches, hidden
fallbacks, panics, duplicate planners, and unbounded allocations are production
defects.

## Extension thesis

A new pass, operation, backend, frontend, emitter, or runtime policy must fit
one existing ownership seam. Every capability claim resolves to executable
parity, adversarial, scale, and lifecycle evidence appropriate to its risk.

## Source map

- `README.md` is the crate-placement charter.
- `docs/ARCHITECTURE.md` is the short page the architecture gate checks.
- `docs/CRATE_OWNERSHIP.toml` defines crate ownership and allowed edges.
- `docs/optimization/OWNERSHIP.toml` and `docs/optimization/OP_MATRIX.toml`
  define optimization lanes and operation coverage.
- `docs/DOCS.toml` defines documentation lifecycle and navigation.
