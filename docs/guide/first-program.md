# Compile a graph to an artifact

```rust
use std::collections::BTreeMap;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
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
    ExternalFacts::new(Digest([9; 32]), BTreeMap::new()),
    DeviceFacts::unknown(),
    SearchBudget::new(1, 1, 1, 0, 1_000_000),
    CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
)
.validate()?;

let artifact = compile(&request)?;
assert_eq!(artifact.nodes().len(), 1);
assert_eq!(artifact.abi().entries.len(), 1);
```

That program is `vyre/tests/artifact_workflow.rs`. Run it with
`cargo test -p vyre --test artifact_workflow`.

## The four steps

**Declare the values.** `ProgramGraph::add_external_value` takes a
`ValueContract`: element type, shape, access and lifetime. Those four facts
are what the compiler checks a node against and what it projects into the
artifact ABI.

**Add the nodes.** `add_node` takes a name, a `Program`, its input values
and its output values. A `Program` is IR: buffer declarations, a workgroup
geometry and a node list. Nothing about a device appears in it.

**Validate the request.** `CompileRequest::new` takes the graph, the
external facts the topology does not encode, the target facts, an explicit
`SearchBudget`, and a `CompileObjective`. `validate` is what turns it into a
`ValidatedCompileRequest`; `compile` accepts nothing else. The budget
arguments are `max_candidates`, `max_cpu_work`, `max_target_compilations`,
`max_measurements` and `max_elapsed_ns`, in that order. There is no
implicit budget: the caller states the search it will pay for.

The objective states what the compile optimizes and every hard bound it refuses
to exceed, including the artifact byte ceiling, which is required. There is no
implicit objective either: a plan selected without one cannot state what it was
selected for. `minimize_latency` is the single-submission case; see
[compile search](../architecture/compile-search.md) for the metrics, the
calibrated facts each one needs, and the bounds.

A floating-point graph also states what its result may be. `with_numeric_budget`
takes the `NumericContract` the caller admits; without one the search keeps only
schedules that combine in the order the program states. See
[numeric contracts](../reference/numeric-contracts.md).

**Compile.** `compile` returns one immutable `Artifact`: node records,
resource records, the whole-program ABI, the selected plan, and a
provenance record. Compiling the same validated request twice produces the
same digest, which is what makes an artifact cache sound.

The selected plan states the declared laws the search derived it through.
`selected_plan().law_derivation` is empty for a plan that computes the programs
as written, and otherwise names one law chain per rewritten node. The numeric
contract governs which laws apply: a bit-exact request derives only alternatives
that produce identical bits.

## What this does not do

It does not run. An artifact is device-neutral by construction; running it
means compiling a target payload for a live device and admitting that
payload. See [backends](backends.md).

It does not execute. Execution is a separate seam: `SemanticExecutor::execute`
takes a `SemanticExecutionRequest` built from a validated `LogicalProgramGraph`,
the byte payload for each canonical graph input, and a
`SemanticExecutionPolicy` carrying external facts, target facts, a
`CompileObjective` and a `SearchBudget`. It returns the
artifact identity, the target payload identity, and one byte buffer per retained
graph value. No launch geometry crosses that boundary in either direction.

A single schedule-free `Program` reaches the same seam through
`execute_single_program`, which validates the program as a one-node graph and
returns a `SingleProgramExecutionOutput` with one buffer per written Program
buffer, in declaration order. `writable_graph_values` and
`writable_graph_value_buffers` state that order for one node.
