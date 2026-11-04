# vyre-aot  -  Configurability

`vyre-aot` is a library. Callers provide a validated opaque `TargetId` to
`compile`, `package_artifact`, and `bundle`.

## Target selection

Each linked concrete driver contributes one `BackendRegistration` containing
its target identity, payload format, pure compiler, and materializer. AOT
compilation resolves the supplied `TargetId` through those registrations. An
unknown target or a registration without a compiler fails explicitly.

Target spellings and compilation profiles are concrete-driver contracts.
`vyre-aot` defines no target enum, target-specific default, shader format,
architecture selector, or environment-variable configuration layer.

## Packaging options

`LauncherOpts` controls the generated launcher crate name and optional
target-owned collective or test-time-training integration. `package_artifact`
writes the authenticated envelope, compressed weights, and manifest without a
launcher. `bundle` also invokes the target-owned launcher emitter.

The package manifest serializes the opaque target identity and the digests of
the neutral artifact, authenticated target payload, envelope, and weight
bytes. Deserialization validates target identity syntax before lookup.
