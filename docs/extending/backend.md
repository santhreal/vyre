# Add a backend

```rust
inventory::submit! {
    BackendRegistration { /* four facets */ }
}
```

A concrete driver crate submits exactly one `BackendRegistration` carrying
four facets:

1. a lower-level backend factory,
2. the supported semantic operation view,
3. a `TargetCompiler` factory,
4. an `ArtifactMaterializer` factory.

A missing compiler or materializer facet fails closed. Supported operation
coverage derives from the foundation semantic operation registry, so a
backend does not maintain its own list of operations to fall out of date.

## What you implement

| Piece | Contract |
|---|---|
| `TargetCompiler` | pure and device-independent; consumes selected programs from an `Artifact`, never a caller-owned `Program` |
| `ArtifactMaterializer` | acquires one device generation, authenticates payload identity and format, creates native modules, owns resident allocation for that generation |
| device and executable module | target-specific acquisition and native execution |
| `VyreBackend` | optional lower-level capability and diagnostic dispatch |

`vyre-driver` owns the neutral side: device, materialization, binding,
submission, completion, capability and lower-level dispatch contracts. Your
crate owns target formats, device acquisition, native modules and
execution. Nothing in `vyre-driver` needs editing to add a backend.

Direct `Program` dispatch through `VyreBackend` is reserved for oracle,
capability and low-level driver tests. It is not a production route, and a
new backend that only implements it is not a backend.

## Capability answers are promises

A positive capability answer asserts that the live device and the concrete
lowering path both support the thing. A backend that cannot decide answers
no. An optimistic answer that fails at dispatch is worse than a negative
one, because the caller had no way to route around it.

## Device loss

Device loss invalidates the native modules and resident handles of that
device generation. Recovery rematerializes the authenticated target payload
bytes. It does not lower or optimize the source program again, so a
recovered instance runs the same bytes the lost one ran.

`ArtifactInstance` accepts only a `BindingSet` carrying the same artifact
digest, and every submission grid axis must be positive. Both are checked,
not documented.

## What your tests must prove

Seven properties, per `vyre-driver/BACKEND_CONTRACT.md`:

- payload authentication,
- materialization,
- typed submission,
- completion,
- capability honesty,
- device-generation rejection,
- recovery.

A backend whose tests cover dispatch and none of these is not admissible.

## Registering out of tree

`examples/external_ir_extension` submits a `vyre_driver::BackendRegistration`
from outside the workspace. Its record carries a validated `TargetId`,
supported operation ids, a pure `TargetCompiler` and an optional
materializer, and the driver derives `TargetOperationFacet` rows by joining
that record with the canonical semantic registry.

That example's target emits the compiler-selected module as opaque wire
bytes and has no device, so its factory and materializer fail explicitly
rather than returning a stub. A production target supplies its own backend
and materializer in the same crate.
