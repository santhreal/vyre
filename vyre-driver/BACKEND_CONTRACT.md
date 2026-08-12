# vyre-driver backend contract

`vyre-driver` owns backend-neutral device, artifact materialization, binding,
submission, completion, capability, and lower-level dispatch contracts.
Concrete driver crates own target formats, device acquisition, native modules,
and execution.

## Production lifecycle

Production callers use one route:

```text
validated ProgramGraph
  -> vyre-megakernel Compiler
  -> immutable Artifact
  -> registered TargetCompiler
  -> authenticated TargetPayload
  -> ArtifactMaterializer
  -> ArtifactInstance
  -> BindingSet
  -> Submission
  -> Completion
```

`TargetCompiler` is pure and device-independent. It consumes selected programs
from an `Artifact`; it does not accept a caller-owned `Program`.

`ArtifactMaterializer` acquires one device generation, authenticates target
payload identity and format, creates native executable modules, and owns
resident resource allocation for that generation.

`ArtifactInstance` is immutable executable state. It accepts only a
`BindingSet` carrying the same artifact digest. Runtime invocation geometry is
typed submission state and every grid axis must be positive.

## Lower-level backend dispatch

`VyreBackend` remains the concrete-driver capability and diagnostic dispatch
contract. Direct `Program` dispatch is reserved for explicit oracle,
capability, and low-level driver tests. It is not a production compiler route.

The facade crate does not re-export `VyreBackend`, `DispatchConfig`,
`BackendError`, `CompiledPipeline`, or backend registry modules. Lower-level
callers import them from `vyre-driver`.

## Registration

Each production backend submits one `BackendRegistration` containing:

- a lower-level backend factory;
- the supported semantic operation view;
- a `TargetCompiler` factory;
- an `ArtifactMaterializer` factory.

Missing compiler or materializer facets fail closed. Registered operation
support derives from the foundation semantic operation registry.

## Capabilities and recovery

Capability queries are conservative by default. A positive capability is a
promise that the live device and concrete lowering path both support it.

Device loss invalidates native modules and resident handles from that device
generation. Runtime recovery rematerializes authenticated target payload bytes;
it does not lower or optimize the source program again.

## Backend extension

A new concrete backend crate implements its target compiler, materializer,
device, executable module, and optional lower-level `VyreBackend` dispatch.
Its tests must prove payload authentication, materialization, typed submission,
completion, capability honesty, device-generation rejection, and recovery.
