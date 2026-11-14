# vyre

`vyre` is a general, backend-neutral GPU megakernel compiler and runtime. Programs are expressed in one typed intermediate representation, lowered through whole-program fusion and task-graph scheduling into a persistent fused megakernel, emitted as PTX, WGSL, SPIR-V, or MSL, and executed on the GPU without host copies or inter-stage host round trips.

All production computation executes on the GPU. `vyre-reference` is a CPU interpreter used exclusively as the differential testing oracle against which GPU backends verify output parity.

## Install

Add `vyre` with the Cargo features matching your target backends:

```toml
[dependencies]
vyre = { version = "0.8.0", features = ["cuda"] }
```

Available backend features on `vyre`: `cuda`, `wgpu`, `metal`, `spirv`. Default features link no GPU driver so callers compile artifacts without device libraries when needed.

## Example

```rust
use vyre::compiler::{compile, CompileRequest};
use vyre::admit_artifact;

let request = CompileRequest::new(graph, facts, device, budget, objective).validate()?;
let artifact = compile(&request)?;
let session = admit_artifact(&artifact, backend)?;
```

## Documentation

- [Manual (`docs/VYRE.md`)](docs/VYRE.md): goals, non-goals, compiler and runtime architecture, megakernel execution model, workspace structure, extending operations and backends, and workload measurement.
- [Contributing (`CONTRIBUTING.md`)](CONTRIBUTING.md): build, test, benchmark, and release commands.
- [Changelog (`CHANGELOG.md`)](CHANGELOG.md): release history.
- API reference: <https://docs.rs/vyre>

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
