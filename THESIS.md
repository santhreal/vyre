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
- `vyre-runtime` owns compile-to-materialize orchestration, sessions, recovery,
  persistence, residency, and readback.

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

- `docs/ARCHITECTURE.md` defines the executable lifecycle.
- `docs/CRATE_OWNERSHIP.toml` defines crate ownership.
- `docs/optimization/README.md` defines optimization ownership and evidence.
- `docs/DOCS.toml` defines documentation lifecycle and navigation.
- `docs/targets.md` defines target registration and support evidence.
