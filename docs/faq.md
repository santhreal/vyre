# Vyre FAQ

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

## What is Vyre?

Vyre is a typed `Program` IR, optimizer, backend interface, runtime, and
conformance toolchain for GPU-oriented compute. You build a program, validate
it, optionally freeze it into a megakernel artifact, then dispatch it on an
explicit backend. Read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for crate
boundaries and [`docs/megakernel-wiring.md`](megakernel-wiring.md) for the
persistent-execution ownership matrix.

## Why not use LLVM or MLIR directly?

Vyre owns a smaller domain contract: linked operation registries, typed buffer
signatures, reference fixtures, backend support evidence, and composition
chains. LLVM or MLIR can still participate below a backend boundary. Vyre does
not replace those stacks; it sits above selected targets as a contract layer.

## How do I add an operation?

Choose the operation tier first:

| Tier | Crate | When to use it |
| --- | --- | --- |
| Intrinsic | `vyre-intrinsics` | Needs a dedicated hardware lowering and reference arm |
| Primitive | `vyre-primitives` | Reusable `fn(...) -> Program` used by multiple libraries |
| Library | `vyre-libs` | Product composition over primitives |
| Runtime dialect | `vyre-driver` registry | Driver-owned runtime ops only |

Follow [`docs/ops-catalog.md`](ops-catalog.md) and
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md). Regenerate
`docs/generated/OP_SCHEMA.json` before you submit the change. Prefer
`cargo_full run --bin xtask -- list-ops --check` and related gates after edits.

## Which backends are active?

| Backend | Role |
| --- | --- |
| CUDA | Preferred evidence-backed release path on NVIDIA hosts |
| WGPU | Portable GPU path |
| Metal | Active on Apple targets |
| SPIR-V | Emission and parity coverage |
| CPU reference | Correctness oracle, never a silent GPU fallback |

Read `release/evidence/backends/backend-matrix.json` for the current measured
state. A missing or stale probe is an invalid autoroute state, not permission to
guess.

## Can I run without a GPU?

Yes for build, validation, reference execution, and structural checks. Production
GPU dispatch requires an explicit available backend. A requested GPU path fails
visibly when its device or driver is unavailable. There is no silent CPU
fallback for a GPU request.

## What is `vyre-megakernel`?

It is the **artifact compiler**: validated typed graphs become immutable
`Artifact` values and versioned envelopes. It does **not** own queue
protocol, resident execution, or device topology. Those live in
`vyre-runtime/src/megakernel/**` and driver wave-policy modules. See
[`docs/megakernel-wiring.md`](megakernel-wiring.md).

## How do AOT packages and runtime caches relate?

`vyre-aot` packages envelopes. Runtime authenticates them through
`artifact_admission` (`admit_artifact`, `admit_envelope`,
`admit_cached_artifact`). Blob stores such as `DiskCache` are format-agnostic;
executable hits must still admit envelope bytes before dispatch.

## Is the API stable?

Vyre 0.7.2 is pre-1.0. Stability depends on the surface. Frozen traits and wire
tags have stronger compatibility rules than experimental compiler and runtime
features. Read [`docs/semver-policy.md`](semver-policy.md).

## What does conformance prove?

Conformance compares a backend result with the registered reference oracle for
the witnessed inputs and laws that an operation declares. It does not turn an
untested input domain into a formal proof. Backend and operation claims remain
bounded by their recorded evidence.

## Where do crate ownership and docs authority live?

| Fact | Authority |
| --- | --- |
| Allowed crate edges | `docs/CRATE_OWNERSHIP.toml` |
| Architecture narrative | `docs/ARCHITECTURE.md` |
| Megakernel stage owners | `docs/megakernel-wiring.md` |
| Doc lifecycle | `docs/INDEX.md` |

Generated views (`OWNERSHIP.md`, `CRATE_GRAPH.md`, catalog pages) are derived.
Do not hand-edit them as source of truth.

## Where do errors and support policy live?

Use [`docs/error-codes.md`](error-codes.md) for stable diagnostic codes,
[`docs/ERROR_SURFACE.md`](ERROR_SURFACE.md) for operator workflow, and
[`docs/support.md`](support.md) for contact paths and release-line support.
