# vyre

vyre is a GPU compiler. You build a program out of registered operations as
IR, it compiles the whole graph into one immutable artifact, emits that
artifact as a target payload in PTX, WGSL, SPIR-V or MSL, and runs it on the
device.

Nothing in vyre computes on the host. The single exception is
`vyre-reference`, an interpreter that exists to be the oracle every backend
is proved byte-identical against. A host path that produces a user's answer
is a product failure, not a fallback.

## Install

```toml
[dependencies]
vyre = { version = "0.7.2", features = ["cuda"] }
```

The default feature set is empty and links no device. `cuda` and `wgpu` are
the facade's backend features. See [install](docs/guide/install.md).

## What it looks like

```rust
let request = CompileRequest::new(graph, facts, budget, ceiling).validate()?;
let artifact = compile(&request)?;
assert_eq!(artifact.digest(), compile(&request)?.digest());
```

The whole runnable version is
[compile a graph to an artifact](docs/guide/first-program.md).

## Documentation

The book starts at [the summary](docs/SUMMARY.md).

- [Install](docs/guide/install.md)
- [Compile a graph to an artifact](docs/guide/first-program.md)
- [Run an artifact on a device](docs/guide/backends.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Crate boundaries](docs/architecture/crates.md)
- [Placement rule](docs/lego-block-rule.md)
- [Add an operation](docs/extending/operation.md)
- [Add a backend](docs/extending/backend.md)
- [Conformance](docs/conformance/program.md)
- [Release](docs/release/process.md)

API documentation: <https://docs.rs/vyre>.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the workflow.
[SECURITY.md](SECURITY.md) covers reporting a vulnerability.
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) applies to every interaction here.

## License

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
