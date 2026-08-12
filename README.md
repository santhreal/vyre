# Vyre

Vyre is an experimental whole-program GPU compiler and runtime for parsing,
matching, graph, dataflow, and numerical workloads.

The production lifecycle is:

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

Raw `Program` execution is reserved for the independent reference and
conformance paths. Production targets compile compiler-selected artifact modules
and admit authenticated payloads before submission.

## Install

```console
cargo add vyre
```

Link the concrete driver crate for each production target you ship:

```console
cargo add vyre-driver-cuda
cargo add vyre-driver-wgpu
```

A missing or incompatible target fails with a structured diagnostic. Vyre does
not silently replace it with CPU execution.

## Compile a graph

```rust
use std::collections::BTreeMap;
use vyre::compiler::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};

let mut graph = ProgramGraph::new();
graph.add_external_value(
    "out",
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(1)],
        access: BufferAccess::WriteOnly,
        lifetime: ValueLifetime::Output,
    },
)?;
graph.add_node(
    "main",
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        Vec::new(),
    ),
    Vec::new(),
    Vec::new(),
)?;

let request = CompileRequest::new(
    graph,
    ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
    SearchBudget::new(1, 1, 1, 0, 1_000_000),
    1_000_000,
)
.validate()?;
let artifact = compile(&request)?;
assert!(!artifact.to_bytes()?.is_empty());
# Ok::<(), Box<dyn std::error::Error>>(())
```

Attach a payload through the selected registered target compiler. Construct an
`ArtifactSession` with the matching registered materializer, bind values through
`BindingSet`, submit, and wait for typed readback.

## Components

- `vyre-foundation` owns semantic IR, validation, diagnostics, and optimization.
- `vyre-megakernel` owns whole-graph planning, artifact identity, canonical ABI,
  and target payload attachment.
- `vyre-driver` owns backend-neutral compiler, materializer, device, binding,
  submission, and completion contracts.
- Concrete driver crates own target compilation, device acquisition,
  materialization, and native submission.
- `vyre-runtime` owns artifact sessions, recovery, persistence, residency, and
  readback.
- `vyre-scan` owns scan compilation, database framing, paging, residency,
  execution, and readback.
- `vyre-aot` packages canonical artifacts without reconstructing them.
- `vyre-conform` compares production artifact execution with the independent
  reference engine.

The machine-readable ownership source is
[`docs/CRATE_OWNERSHIP.toml`](docs/CRATE_OWNERSHIP.toml).

## Documentation

Build the documentation book:

```console
mdbook build
```

Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The documentation
manifest at [`docs/DOCS.toml`](docs/DOCS.toml) classifies current, generated,
superseded, and archived pages.

## Development

Use the repository Cargo wrapper so builds remain bounded:

```console
./cargo_full test -p vyre --test artifact_workflow
./cargo_full test -p vyre-conform --test production_route
python3 scripts/docs_manifest.py --check
```

GPU probe failures are configuration failures. GPU-required gates fail loudly
when the required device or driver is unavailable.

## License

MIT OR Apache-2.0.
