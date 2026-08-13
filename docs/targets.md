# Vyre Targets

Applies to Vyre 0.7.2.

A production target contributes three facets under one stable backend ID:

1. A pure `TargetCompiler` that consumes a compiler-selected `Artifact` and
   emits an authenticated `TargetPayload`.
2. An `ArtifactMaterializer` that admits the payload on one device generation
   and creates an `ArtifactInstance`.
3. A device registration that exposes capabilities and submits typed
   `BindingSet` values.

The linked concrete driver crate owns all three facets. Shared crates do not
select a concrete target or reconstruct target bytes.

## Discovery

```rust
let registrations = vyre_driver::backend::registered_backends()?;
for registration in registrations {
    println!("{}", registration.id);
}
```

A target appears only when its concrete driver crate is linked. Acquiring an
unavailable target returns a structured `BackendError`; callers must not replace
it with a silent fallback.
Registry discovery is fallible. Duplicate backend IDs, duplicate target IDs,
or conflicting owner-local metadata return a startup `BackendError`; no
provider wins by inventory order.

## Production lifecycle

```text
validated ProgramGraph
  -> vyre-megakernel Compiler
  -> immutable Artifact
  -> registered TargetCompiler
  -> authenticated TargetPayload
  -> registered ArtifactMaterializer
  -> ArtifactInstance
  -> BindingSet
  -> Submission
  -> Completion
```

`vyre_megakernel::attach_target` is the only owner of payload attachment. A
target compiler reads compiler-selected modules from the artifact. It does not
accept a caller-owned raw `Program`, acquire a device, or create native handles.

The materializer validates artifact identity, payload identity, module digests,
ABI slots, entry identity, default geometry, target compatibility, and device
generation before constructing native handles. Submission validates typed
bindings against that admitted instance.

## Registration

Concrete drivers submit one `vyre_driver::BackendRegistration`. The target
identity and payload format are owner-local constants:

```rust,ignore
inventory::submit! {
    vyre_driver::backend::BackendRegistration {
        id: BACKEND_ID,
        target_id: TARGET_ID,
        payload_format: Some(TARGET_PAYLOAD_FORMAT),
        reference_oracle: false,
        factory: backend_factory,
        supported_ops,
        semantic_operations,
        target_compiler: Some(target_compiler_factory),
        materializer: Some(materializer_factory),
    }
}
```

`target_compiler` and `materializer` are required for production execution.
The independent reference backend intentionally omits them because raw
`Program` interpretation is an oracle and conformance path, not a production
compiler target.

## Operation support

Operation support derives from the foundation-owned semantic operation
registry. Each target declares canonical semantic operation IDs through
`BackendRegistration::semantic_operations`. The shared driver joins that set
with `OperationRegistry` to produce read-only `TargetOperationFacet` values.
Conformance compares registered production targets with the independent
reference engine.
Signature-only semantic operations remain in the canonical catalog but are
excluded from executable conformance selection because they have no neutral
`Program` builder.

The generated operation inventory and conformance matrix report coverage:

```console
./cargo_full run -p xtask -- op-matrix --check
./cargo_full run -p xtask -- conformance-matrix
```

An unclassified new operation fails the registry and conformance contracts.
There is no hand-maintained target-operation matrix in this document.

## Adding a target

1. Create one concrete driver crate.
2. Implement a pure `TargetCompiler` over compiler-selected artifact modules.
3. Implement `ArtifactMaterializer` for authenticated payload admission.
4. Register backend, compiler, materializer, device capabilities, and operation
   facets under one backend ID.
5. Add production compile-materialize-submit-readback conformance against the
   independent reference engine.
6. Add device-loss rematerialization and stale-generation rejection coverage.
7. Add cold and warm compiler, materializer, submission, and readback
   benchmarks.

No shared compiler, runtime, facade, or existing concrete driver needs a target
specific match arm.

The isolated `examples/external_backend_extension` crate compile-checks the
published facade and wire boundary. `scripts/check_backend_extension_contract.sh`
proves that linked concrete drivers register through owner-local inventory
submissions and that the shared registry contains no concrete target IDs.
