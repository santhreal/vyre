# Vyre architecture

Last verified: 2026-08-17

Vyre 0.7.2 is a GPU compiler. You build a `Program` from registered
operations, compile the whole graph into one immutable `Artifact`, emit a
target payload, and run it on the device. There is no host execution path
and no bytecode interpreter. `vyre-reference` is the only crate allowed to
compute on the CPU, and only as the oracle.

Downstream crates do not own shadow operation identities. The live catalog
is `docs/generated/OP_SCHEMA.json`, read through
`vyre-foundation::operation::OperationRegistry`.

## Two placement rules

**Composed, not rewritten.** A composition returns a `Program` built from
IR that already exists. It belongs in `vyre-libs`, whoever calls it.

**Intrinsic means uncomposable.** An operation belongs in `vyre-primitives`
only when it needs its own backend emitter arm and its own reference
interpreter arm.

## Layers

- `vyre-spec` is the frozen vocabulary. It does not execute.
- `vyre-foundation` owns IR, validation, the host optimizer, and the
  registry. No application semantics.
- `vyre-libs` owns every composition: consumer dialects and the compiler's
  own solvers, encoding, analysis, scheduling, and reasoning. Equal
  residents.
- `vyre-primitives` owns marker types and hardware intrinsics. A composition
  belongs in `vyre-libs`.
- `vyre-lower` is the last dialect-free stage: `Program` to
  `KernelDescriptor`.
- `vyre-megakernel` owns Cross-program composition: candidate generation,
  fusion legality, the cost model, selection under an explicit
  `SearchBudget`, and target compiler facets. It does not own admission or
  claim a measured winner that no clock produced.
- `vyre-driver` is backend-agnostic machinery. Concrete drivers own names,
  dialects, and device quirks.
- `vyre-runtime` executes the artifact's selected persistence. It does not
  decide whether to be persistent.
- `vyre-pass-engine` runs the optimizer's passes as vyre Programs.
- `vyre-reference` is the oracle, not a backend and not a fallback.

## Production route

```text
frontend Program(s)
  -> validated ProgramGraph
  -> vyre-megakernel Compiler
  -> immutable Artifact + TargetPayload
  -> driver admission and materialization
  -> ArtifactInstance
  -> typed Submission
  -> completion and readback
```

Every production compile emits a megakernel artifact. Persistence is a
schedule inside that artifact, not a second output type. Static and
persistent routes consume the same artifact class and must produce the same
bytes. Hardware enters compile as a fact vector, never as a backend name.
Unmeasured selections are recorded as unmeasured and are never called
autoroute. GPU execution is capability-based on the designated execution host
(`axiomexec`). Every release crate enforces a zero panic budget.

## Chapters

- [Crate boundaries](architecture/crates.md): what each crate owns.
- [The artifact is the output type](architecture/artifact.md): identity,
  persistence, payload admission.
- [Whole-program compile search](architecture/compile-search.md): legality,
  budget, cost model.
- [Parsing](architecture/parsing.md): the language-neutral substrate and
  its frontends.
- [The placement rule](lego-block-rule.md): which crate a new operation
  belongs in.

Machine-readable contracts live next to this file:
`CRATE_OWNERSHIP.toml`, `optimization/OWNERSHIP.toml`,
`optimization/OP_MATRIX.toml`, and `testing/TESTING.toml`.
