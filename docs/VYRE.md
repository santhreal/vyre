# vyre Manual

## Goal

`vyre` compiles general GPU workloads into persistent fused megakernels from one backend-neutral intermediate representation (IR). Supported workload classes span graph traversal, lexing and parsing, multi-pattern byte scanning, iterative dataflow fixpoints, numerical simulation and linear algebra, and dense or quantized neural inference.

Design invariants:

1. Raw execution speed across compilation, kernel emission, submission, and device execution.
2. Zero host buffer copies across wire decoding, upload, resident execution, and readback.
3. Single-launch persistent execution where multi-stage task graphs and convergence loops run on the device without host round trips.
4. One canonical representation per concept and strict one-way crate dependencies.
5. Correctness verified by differential parity tests against `vyre-reference` rather than runtime overhead on hot paths.

## Non-Goals

- Domain-specific product logic, rule catalogs, or application-specific pipelines in compiler or runtime crates. Domain-flavored helpers exist only under `vyre_libs::utilities` when required by an active caller.
- Host-side production fallback execution. `vyre-reference` is a test and verification oracle, never a production execution target.
- Multiple competing representations for the same IR construct, schedule, or dispatch path.

## Architecture

```
ProgramGraph (vyre-ir / vyre-libs)
        |
        v
Whole-Program Optimizer & Megakernel Compiler (vyre-compiler)
        |
        v
Target Payload Emitter: PTX | WGSL | SPIR-V | MSL (vyre-emit)
        |
        v
Persistent Runtime & Device Drivers (vyre-runtime)
```

1. **IR and Wire Format (`vyre-ir`)**: Defines `Program`, `ProgramGraph`, scalar and tile types, memory spaces, zero-copy canonical wire encoding, structural validation, `KernelDescriptor`, and `SemanticExecutor`.
2. **Generic Library (`vyre-libs`)**: Composes `vyre-ir` primitives into reusable GPU algorithms across `graph`, `pattern`, `parsing`, `text`, `math`, `nn`, `hash`, `decode`, `bitset`, `reduce`, `fixpoint`, `visual`, `builder`, and `utilities`.
3. **Compiler (`vyre-compiler`)**: Runs algebraic optimization passes, lowering, static analyses, task-graph extraction, producer-consumer fusion, static and dynamic scheduling, and ahead-of-time (`aot`) bundle assembly.
4. **Code Emission (`vyre-emit`)**: Translates lowered megakernel modules into target shader and assembly payloads (`ptx`, `naga` WGSL, `spirv`, `metal`), conditional on Cargo features.
5. **Runtime (`vyre-runtime`)**: Manages backend registration, artifact admission, resident device memory pools, `io_uring` host-device streaming, and persistent megakernel execution across `cuda`, `wgpu`, `metal`, and `spirv`.

## Megakernel Design

Traditional GPU pipelines launch one kernel per operator, paying host launch latency, global memory traffic between stages, and host synchronization on data-dependent loops. `vyre` compiles an entire `ProgramGraph` into one persistent megakernel:

```rust
use vyre::compiler::{compile, CompileObjective, CompileRequest, SearchBudget};

let request = CompileRequest::new(
    program_graph,
    external_facts,
    device_facts,
    SearchBudget::default(),
    CompileObjective::MinLatency,
)
.validate()?;
let artifact = compile(&request)?;
```

- **Task Graph Construction**: `vyre-compiler` partitions operations into tiles and tasks connected by explicit memory and barrier dependencies.
- **Default Dispatch Fusion**: Producer-consumer chains fuse into shared register or workgroup-memory stages whenever resource bounds (registers per thread, shared memory per block, barrier compatibility) permit.
- **Device-Side Scheduling**: Persistent thread blocks pull ready tasks from resident device queues and update dependency counters atomically in device memory.
- **In-Kernel Convergence Loops**: Fixpoint and frontier workloads iterate inside the persistent kernel using grid barriers or atomic phase counters, eliminating host readback per iteration.
- **Resident Memory and Zero-Copy I/O**: Input slices map directly into pinned or device-visible buffers; intermediate buffers stay resident in workspace memory across submissions.

## Workspace Structure

```
Cargo.toml              workspace manifest
README.md               landing page
docs/VYRE.md            manual
CONTRIBUTING.md         build, test, benchmark, and release guide
CHANGELOG.md            release history
crates/
  vyre/                 public facade re-exporting ir, libs, compiler, runtime
  vyre-ir/              IR, types, memory model, zero-copy wire format, validation,
                        SemanticExecutor, KernelDescriptor
  vyre-macros/          procedural macros
  vyre-libs/            generic GPU algorithm compositions
  vyre-compiler/        optimizer, pass engine, lowering, megakernel compiler, aot
  vyre-emit/            feature-selected emitters: ptx, naga, spirv, metal
  vyre-runtime/         driver registry, persistent megakernel runtime, backends
  vyre-reference/       CPU differential oracle interpreter
  vyre-bench/           megakernel workload benchmarks and allocation probes
  vyre-test-support/    development test fixtures and conformance harness
tools/
  xtask/                workspace verification and maintenance CLI
```

Dependency edges follow one strict order:

```
vyre-ir -> vyre-libs -> vyre-compiler -> vyre-emit -> vyre-runtime -> vyre
vyre-reference -> vyre-ir
```

Development crates (`vyre-bench`, `vyre-test-support`, `tools/xtask`) may depend on workspace crates as required for benchmarking, conformance, and repository checks. No production crate depends upward in the chain.

## Extending

### Adding a Generic Operation

1. Express the operation as a composition of existing `vyre-ir` statements inside the matching module of `vyre-libs` (`graph`, `pattern`, `parsing`, `text`, `math`, `nn`, `hash`, `decode`, `bitset`, `reduce`, `fixpoint`, `visual`). Add an IR primitive to `vyre-ir` only when the operation cannot be expressed from existing instructions without losing hardware instructions or asymptotic speed.
2. Implement reference evaluation in `vyre-reference` if a new `vyre-ir` primitive was added.
3. Add differential parity tests in `vyre-test-support` comparing `vyre-reference` output against emitted GPU execution across boundary inputs.

### Adding a Backend

1. Add a target emitter module under `crates/vyre-emit/src/<backend>/` behind a Cargo feature of the same name, translating `KernelDescriptor` and megakernel task modules into the target format.
2. Add a runtime driver module under `crates/vyre-runtime/src/<backend>/` behind the matching Cargo feature, implementing buffer residency, persistent queue launch, and device timestamp collection.
3. Re-export the feature flag from `crates/vyre/Cargo.toml`.
4. Run the conformance parity suite in `vyre-test-support` against `vyre-reference` across all scalar, tile, atomic, barrier, and megakernel cases.

## Measuring

`vyre-bench` measures end-to-end megakernel compile time, launch latency, device active time, throughput, and host allocation counts across six representative GPU workload classes:

1. **Graph**: breadth-first traversal, reachability, and connected components on CSR graphs.
2. **Parsing**: parallel lexing, bracket matching, and structural index construction.
3. **Pattern Scanning**: multi-literal Aho-Corasick and automaton byte scanning.
4. **Dataflow**: iterative bitset and relation fixpoint convergence in a single persistent launch.
5. **Math and Simulation**: dense and sparse linear algebra, stencil steps, and parallel reductions.
6. **Inference**: fused normalization, projections, attention, and quantized matrix-vector decode stages.

Run the megakernel benchmark suite and verify zero steady-state host allocations:

```sh
cargo run --release -p vyre-bench -- --suite megakernel
cargo test -p vyre-bench
```

Every benchmark run verifies device output against `vyre-reference` before recording timing metrics.
