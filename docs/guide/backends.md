# Run an artifact on a device

```rust
use vyre_driver::{
    acquire, acquire_preferred_dispatch_backend, registered_backends_by_precedence,
};

// Every backend linked into this binary, highest precedence first.
let linked = registered_backends_by_precedence()?;

// The highest-precedence dispatch-capable backend that initializes on this host.
let backend = acquire_preferred_dispatch_backend()?;

// Or one named backend, which fails closed when it is not linked.
let cuda = acquire("cuda")?;
```

These come from `vyre-driver`, not from the `vyre` facade. The facade
re-exports the artifact and submission types and deliberately does not
re-export `VyreBackend`, `DispatchConfig`, `BackendError`,
`CompiledPipeline` or the backend registry.

## The registry is what is linked

`registered_backends_by_precedence` returns registrations, not names a
table hardcodes. Each concrete driver crate submits one
`BackendRegistration` into an `inventory` collection, and the registry is
frozen once per process behind a fallible `LazyLock`. A backend absent from
the binary is absent from the registry, and `acquire` says so:

```text
backend `cuda` is not linked into this binary. Fix: link the concrete
driver crate that registers this backend or choose one of the registered
backend ids.
```

`acquire_preferred_dispatch_backend` walks the precedence-sorted slice,
skips every registration that does not declare live dispatch, skips every
registration marked as a reference oracle, and takes the first remaining
backend whose factory initializes here. It records the factories that
failed and the reference oracles it skipped, so a host with no usable
device reports what it tried rather than selecting something that is not a
device. A reference oracle stays reachable through `acquire` by id and is
never selected implicitly.

## The concrete backends

| Crate | Target |
|---|---|
| `vyre-driver-cuda` | NVIDIA through the CUDA driver API |
| `vyre-driver-wgpu` | wgpu |
| `vyre-driver-spirv` | SPIR-V |
| `vyre-driver-metal` | Metal |
| `vyre-driver-reference` | the reference interpreter, presented as a backend |

`vyre-driver-reference` exists so parity runs through the same seam as
every other backend. It is the oracle, not a fallback: nothing in vyre
routes a user's answer to it because a device was missing.

## What a registration carries

Four facets, per `vyre-driver/BACKEND_CONTRACT.md`: a lower-level backend
factory, the semantic operation view the backend supports, a
`TargetCompiler` factory, and an `ArtifactMaterializer` factory. A missing
compiler or materializer facet fails closed. Supported operations derive
from the foundation semantic operation registry rather than from a list the
driver keeps.

## The production route

```text
validated ProgramGraph
  -> vyre-megakernel compile
  -> immutable Artifact
  -> registered TargetCompiler
  -> authenticated TargetPayload
  -> ArtifactMaterializer
  -> ArtifactInstance
  -> BindingSet
  -> Submission
  -> Completion
```

`TargetCompiler` is pure and device-independent, and consumes selected
programs from an `Artifact` rather than a caller-owned `Program`.
`ArtifactMaterializer` acquires one device generation, authenticates
payload identity and format, and owns resident allocation for that
generation. `ArtifactInstance` accepts only a `BindingSet` carrying the
same artifact digest, and every submission grid axis must be positive.

Device loss invalidates the native modules and resident handles of that
device generation. Recovery rematerializes the authenticated payload bytes;
it does not lower or optimize the source program again.

## Capability answers are conservative

A positive capability answer is a promise that the live device and the
concrete lowering path both support the thing. A backend that cannot decide
answers no.
