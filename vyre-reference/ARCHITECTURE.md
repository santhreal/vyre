# vyre-reference  -  architecture

The parity oracle. It evaluates a `vyre_foundation::Program` on the host and
defines the answer every GPU backend is graded against. It is the single place
in vyre where a program is executed on the CPU, and it is unreachable from
production routing: production crates depend on it as a dev-dependency only,
and the conformance harness is the one package that depends on it normally.

There is one evaluator. A second route is a second answer, so no module here
re-implements node or expression semantics.

## Entry point

`ReferenceRequest::execute` in `request.rs` is the typed, versioned submission
surface: a logical graph, its resource declarations, a workload envelope, a
numerical contract, a schedule policy, and a mandatory budget. Every other
entry point in `execution/mod.rs` (`reference_eval`, `reference_eval_with_grid`,
`reference_eval_oob_report` and the rest) builds a request and calls it.

`reference_eval_expr` in `execution/single_expr.rs` evaluates one `Expr`
against a `ReferenceMemory`. It calls the same expression evaluator the graph
route calls; it does not carry a second implementation.

## Modules

### `execution/hashmap/`
The canonical evaluator. `mod.rs` builds the invocation set and runs the
schedule; `invocation.rs` holds per-lane state; `memory.rs` owns buffer
storage and output extraction; `step/` evaluates nodes and expressions;
`sync.rs` implements barriers and grid fences; `subgroup.rs` implements
collectives over the lanes of one workgroup.

### `execution/call.rs`
`Expr::Call` resolution. The op is looked up in `OperationRegistry::global()`
and the host body through `reference_fn(op_id)`. No evaluator matches on an
op-id string.

### `execution/expr_cast.rs`
`Expr::Cast` semantics for every declared `DataType`.

### `execution/typed_ops/`
Width-typed scalar and vector arithmetic, comparison and bitwise semantics.

### `execution/tile.rs`
Tile-shaped collective bodies.

### `execution/node_tree.rs`
Region and control-flow structure of a wrapped program.

### `execution/async_transfer.rs`
`Node::AsyncCopy` and its wait semantics.

### `execution/step_budget.rs`, `execution/op_count.rs`
Work accounting against the request budget. Exceeding the ceiling is a
structured budget-exhaustion error, never a truncated result.

### `atomics.rs`
Every `AtomicOp` variant under the serialization the oracle fixes.

### `oob.rs`
Buffer storage plus the load, store and atomic access paths. An access outside
a declared extent is reported, not silently defaulted.

### `workgroup.rs`
`InvocationIds`, the workgroup frame, and the shared-scratch byte ceiling.

### `ieee754.rs`, `float16.rs`
Canonical f32 and f16 results for the operations a device may approximate.

### `subgroup.rs`
Lane-identified collective results.

### `value.rs`
`Value`, the byte encoding every output is compared through.

### `error.rs`
`ReferenceError`, its eight failure classes, and the per-class constructors.

### `reference_facet.rs`
`ReferenceFacet` and `reference_facets()`: the host bodies registered against
`OperationRegistry::global()` entries.

### `composition_witness/`
Independent mathematical witnesses for the semantic families the evaluator
implements, used to judge the evaluator rather than restate it.

### `interleaving.rs`
Bounded deterministic exploration of the schedules a program admits.

## Integration points

- The conform runner calls `ReferenceRequest::execute` for every probe.
- The oracle executor id is owned by `vyre-driver-reference`. The oracle is not
  a backend and is never registered as one.
- `Value` outputs are consumed by the byte-identity proofs in every backend.

## Bench targets

The crate declares no bench target. Interpreter throughput is not a shipped
property.

## Fuzz targets

The crate declares no fuzz target. The input space is enumerated by the
generated matrices and property tests rather than sampled, because the oracle
has to be right on every point of a dimension.
