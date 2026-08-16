# Crate boundaries

Each crate below states what it owns and what it must not hold. Every
boundary is an application of one of the two placement rules in
[the placement rule](../lego-block-rule.md).

`docs/CRATE_GRAPH.md` and `docs/OWNERSHIP.md` are generated from the
manifests and are the machine-readable form of this page.

## Contract

### vyre-spec

The frozen vocabulary: operation definitions, data types, categories,
algebraic laws, invariants, token tables, known-answer vectors. It has no
dependencies and most of the workspace depends on it.

Not here: anything that executes, allocates or decides. A behavioural
change in a crate this widely depended on is a change to every crate at
once.

### vyre-macros

The proc macros that generate pass and AST-registry boilerplate.

Not here: logic. A macro that decides something moves the decision out of
source a reader can grep and into an expansion they cannot.

### vyre-foundation

The IR: program, node and expression, the type system, the memory model,
the wire format, validation, visiting, the optimizer, the execution plan,
program dispatch.

Not here: application semantics. No operation in this crate knows what a
neural network is. Almost every crate depends on this one, so a domain
concept admitted here is one every crate inherits and none can escape.

## Composition

### vyre-libs

Every composition in the workspace. Each public function returns a
`Program` built from existing IR. Consumer domains and compiler-internal
domains are equal residents: linear algebra, neural network layers,
matching, hashing, parsing, decoding, and equally the solvers, encoding,
analysis, scheduling and reasoning the compiler composes for itself.

Not here: anything that names a concrete backend, links an emitter crate,
or reimplements in host Rust what IR expresses. The first two invert the
dependency, because a composition states what it needs and the driver
decides who provides it. The third is the failure the crate exists to
prevent.

### vyre-primitives

Marker types the interpreter and emitters dispatch on, and hardware
intrinsics: operations that need a dedicated emitter arm in every backend
and a dedicated arm in the reference interpreter.

Not here: compositions. Admission is by whether the operation can be
composed, never by how many callers it has.

### vyre-reference

The interpreter for vyre IR, and the only crate permitted to compute on the
host.

Not here: performance work, and any role other than oracle. It is not a
backend and not a fallback. It exists so a backend's answer can be proved
identical to a definition, so speed here buys nothing and complexity costs
the definition's credibility.

## Compilation and emission

### vyre-lower

Neutral lowering from a `Program` to a `KernelDescriptor`. The last stage
where no dialect exists.

Not here: any dialect. The moment one leaks in, every emitter downstream
inherits a decision that was theirs to make, and the neutral descriptor
stops being a shared contract.

### vyre-megakernel

The whole-program compile seam. See
[whole-program compile search](compile-search.md).

Not here: device admission, submission, queues, residency, recovery. Those
consume the artifact and must not alter its identity, because identity is
what makes two routes comparable and a cache sound. Also not here: any
claim of a measured winner that no clock produced.

### vyre-emit-naga, vyre-emit-ptx, vyre-emit-spirv, vyre-emit-metal

Descriptor to dialect. `vyre-emit-naga` produces the WGSL-family module and
`vyre-emit-ptx` produces NVIDIA assembly text. `vyre-emit-spirv` and
`vyre-emit-metal` both depend on `vyre-emit-naga` and route through it
rather than carrying their own lowering.

Not here: lowering decisions, and any second copy of a translation another
emitter already owns. A fork here is two implementations of one language
that drift apart in the direction of whichever backend was debugged last.

## Execution

### vyre-driver

Backend-neutral machinery: the backend trait every concrete driver
implements, plus registry, routing, pipelines, bindings, residency, caching
and eviction, autotune storage, work queues, dispatch policy, diagnostics.
See [add a backend](../extending/backend.md).

Not here: a driver name, a dialect string, or a backend-specific error
message. This crate is what makes the backends interchangeable, and a
concrete detail admitted here is a detail every backend must then pretend
to have.

### vyre-driver-cuda, vyre-driver-wgpu, vyre-driver-spirv, vyre-driver-metal, vyre-driver-reference

The concrete backends: CUDA through the driver API, wgpu, SPIR-V, Metal,
and the adapter that presents the reference interpreter as a backend so
parity runs through the same seam as everything else.

Here and nowhere else: driver names, dialect names, vendor error strings,
device quirks. The size asymmetry is real work rather than neglect:
resident dispatch, target compile caches and megakernel scheduling live
where the device supports them.

Not here: anything another backend would also need. That belongs in
`vyre-driver`, or it becomes five copies that drift.

### vyre-registry-link

The sole owner of the inventory link anchors, and of the per-source floor
every registry read is judged against.

Deliberately substrate-bound: this is the one crate that names every
backend, because linking is the act of naming them. The floor exists
because an inventory that silently comes up short reads exactly like a
smaller workspace.

Not here: anything but linking and the floor.

### vyre-runtime

Persistent execution: the resident work queue, the persistent executor,
artifact admission, resource residency, pipeline cache, tenancy,
scheduling, replay, recovery, and Linux io_uring streaming into
GPU-visible memory. The io_uring surface is compiled out on every other
platform and reports `PipelineError::NotLinux` rather than emulating
itself.

Not here: compilation, and any decision about whether to be persistent.
Persistence is selected during compile, inside the artifact; this crate
executes that selection. A runtime that decides to fuse has taken a
decision away from the search that could have measured it.

### vyre-pass-engine

The optimizer's own passes, executed as vyre Programs through the
dispatcher seam. The compiler running on the device it compiles for.

Not here: host reimplementations of passes that exist as compositions. That
is the composition rule applied to the compiler itself, and this crate is
the proof the rule is livable.

### vyre-aot

Ahead-of-time compilation to target bytes plus a self-contained launcher
bundle, for embedded and competition targets where no compiler ships
alongside. No workspace member depends on it.

Not here: a second compile path. It produces the same artifact class as
every other route, or the bundle is a different compiler with the same
name.

### vyre

The public facade. Re-exports the IR, the driver, the runtime and the
artifact compiler behind backend features. The crate a user adds. See
[install](../guide/install.md).

Not here: logic. A facade with behaviour is a fourth place to look for a
bug.

## Harnesses and tooling

### vyre-test-support

Shared test fixtures and the registry coverage closure gate.

Not here: production code, and a fixture that exists in a crate that
already owns it. Fixture duplication is how two suites end up testing two
different programs under one name.

### vyre-conform and vyre-conform-spec

The conformance engine and its witness sets. See
[conformance](../conformance/program.md).

Not here: a pass a backend can satisfy without producing the reference's
bytes. A soft pass is worse than no suite.

### vyre-bench

The cross-backend benchmark and parity harness.

Not here: a benchmark whose baseline is vyre's own unfused output. Beating
your own slow path is not a result. The baseline is the best available
native implementation for that class.

### vyre-debug

Inspection: descriptor dumps and diffs, naga traces, source assignments,
dangling analysis, carrier inspection.

Not here: anything a user's program depends on. This crate reads what other
crates produced.

### vyre-safetensors

Bounded safetensors metadata and shard identity. No workspace member
depends on it.

Not here: tensor data handling beyond metadata and identity.

### vyre-lints

The workspace lints: production host fallbacks, silent device skip guards,
module forks, consumer coupling, raw IR construction in composition crates.

Each lint exists because the defect it catches shipped once. Not here:
style. A lint that fires on formatting trains everyone to ignore the lints
that catch a host fallback.

### structure-gate

The structural contract: the crate roster, one identity per operation, one
home per concept. It also owns the source-text reader that handles nested
block comments, raw strings, and a character literal that is not a
lifetime.

Not here: a second source scanner. A bad masker desynchronizes a brace
matcher on the first raw string it meets, and a contract built on a bad
masker reports confident nonsense.

### xtask, xtask-registry, xtask-evidence

The gate registry and its runner, the gates that need the live operation
registry linked, and the gates that read benchmark and release evidence.

Every check in the workspace is a gate in one registry. There are no
categories: not evidence, not composite, not tool, not script. A gate
states what it found as findings, the baseline pins that count, growth
fails, and a fall is reported so the pin can drop. The sweep enumerates the
registry at run time, so a gate registered without a baseline row fails and
a row naming no gate fails. A gate that owns a generated artifact checks it
by default and regenerates only when asked.

Not here: an exemption. A gate that is allowed to be red is not a gate.

## Outside the workspace

The root manifest excludes four directories. Each resolves against its own
manifest, so none of them inherits workspace lints, patches or features, and
none is built by a workspace command.

| Directory | What it proves |
|---|---|
| `examples/external_ir_extension` | An out-of-tree crate registers a semantic operation and a compile-only target facet. |
| `examples/external_backend_extension` | An out-of-tree crate implements `vyre_driver::VyreBackend` and is served by `vyre_driver::acquire`. |
| `examples/libs-template` | The Category A author path, from scaffold to a passing conformance test. |
| `vyre-foundation/fuzz` | The libFuzzer targets for the wire decoder, the byte decoder and the registry TOML reader. |

The `example-capability` gate builds each example against the published surface
and runs what it asserts. See [add an operation](../extending/operation.md) and
[add a backend](../extending/backend.md).
