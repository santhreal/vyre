# vyre

> **Note**: Vyre is heavily experimental, rough around the edges, and not ready for production use.

[![Crates.io](https://img.shields.io/crates/v/vyre)](https://crates.io/crates/vyre)
[![Docs.rs](https://docs.rs/vyre/badge.svg)](https://docs.rs/vyre)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Last verified: 2026-08-04

Part of [Santh](https://santh.dev) - open source Rust security and infrastructure tooling.

Vyre is a Rust GPU compute stack for workloads that usually get pulled back to
the CPU: parsing, graph traversal, fixed-point dataflow, and rule-based
reasoning workloads.

The core IR, contracts, CPU reference path, CUDA path, WGPU path, Metal path,
PTX emitter, conformance system, and shared primitives are active. Some
frontends remain beta. That status split is deliberately explicit, so you can
decide what to depend on today.

For `0.7.2`, the unit of release is `vyre::Program`. A program is built as IR,
validated against frozen specs, checked against the CPU reference interpreter,
and then compared against GPU backends for agreement. CUDA is the release path
on NVIDIA systems. WGPU is the portable GPU path. Metal is the native Apple
path, and its device-level evidence is produced only on a host that has a Metal
device.

## Production Surface

Vyre is used by downstream security and analysis tools that need
GPU-accelerated scanning, high-throughput sequential matching, and
reference-checked GPU compute backends.

## Workspace architecture

```mermaid
flowchart TB
    classDef active fill:#0f766e,color:#fff,stroke:#045a55;
    classDef beta fill:#b45309,color:#fff,stroke:#78350f;
    classDef external fill:#6b7280,color:#fff,stroke:#374151;
    classDef planned fill:#4b5563,color:#fff,stroke:#1f2937;

    subgraph S0["Tier-1 foundations"]
      Vcore["vyre-core\npublic facade + re-exports"]
      Fnd["vyre-foundation\nIR, validation, optimizer"]
      Spec["vyre-spec\ncontracts + schemas"]
      Macros["vyre-macros\nregistration helpers"]
      Intr["vyre-intrinsics\nhardware-facing intrinsic surface"]
    end

    subgraph S1["Tier-2.5/3 and reference"]
      Ref["vyre-reference\nCPU reference oracle"]
      Primitives["vyre-primitives\ngraph/matching/math/nn/hash/text/parse"]
      Libs["vyre-libs\ntier-3 composition surface"]
      SelfSS["vyre-self-substrate\nself-consumer + scheduler path"]
    end

    subgraph S2["Backend, lowering, and execution"]
      Drv["vyre-driver\nbackend traits + registry"]
      Cuda["vyre-driver-cuda\nrelease backend"]
      Wgpu["vyre-driver-wgpu\nportable GPU backend"]
      Spirv["vyre-driver-spirv\nSPIR-V surface"]
      Lower["vyre-lower\nIR shaping helpers"]
      EmitPtx["vyre-emit-ptx\nPTX + NVRTC"]
      EmitNaga["vyre-emit-naga\nWGSL/Naga"]
      EmitSpv["vyre-emit-spirv\nSPIR-V emitter"]
      RefDrv["vyre-driver-reference\nreference backend adapter"]
    end

    subgraph S3["Runtime + conformance + evidence"]
      RT["vyre-runtime\nmegakernel + io_uring"]
      Aot["vyre-aot\noffline packaging"]
      Hs["vyre-harness\nruntime harness"]
      Debug["vyre-debug\ntracing + inspection"]
      Bench["vyre-bench\nbenchmarks"]
      Lints["vyre-lints\npolicy checks"]
      Fuzz["fuzz\nfuzzing suites + mutation inputs"]
      XTask["xtask\nCI/audit matrix"]
      ConSpec["vyre-conform-spec\nprogram spec"]
      ConGen["vyre-conform-generate\ncase generation"]
      ConEnf["vyre-conform-enforce\nenforcement gates"]
      ConRun["vyre-conform-runner\nrunner + reporting"]
      TestHarness["vyre-test-harness\nshared fixtures"]
    end

    subgraph S4["Consumer-facing / external roadmap"]
      FC["vyre-frontend-c\nC frontend pipeline"]
      FR["vyre-frontend-rust\nRust frontend pipeline"]
      Intg["External integrations\nconsumer repositories"]
      MT["Metal backend\nactive on Apple targets"]
      DX["DXIL/DirectX\nplanned"]
      WG["Wasm/WebGPU\nplanned"]
    end

    Vcore --> Spec
    Fnd --> Spec
    Intr --> Fnd
    Primitives --> Libs
    Libs --> SelfSS
    Libs --> ConRun
    Vcore --> Drv
    Ref --> ConRun

    Drv --> Cuda
    Drv --> Wgpu
    Drv --> Spirv
    Cuda --> RT
    Wgpu --> RT
    Spirv --> RT
    RefDrv --> RT
    Lower --> EmitPtx
    Lower --> EmitNaga
    Lower --> EmitSpv
    EmitPtx --> RT
    EmitNaga --> RT
    EmitSpv --> RT

    RT --> Hs
    RT --> Aot
    RT --> Debug
    RT --> Bench
    ConSpec --> ConRun
    ConGen --> ConRun
    ConEnf --> ConRun
    TestHarness --> ConRun
    Fuzz --> XTask
    XTask --> ConRun
    XTask --> Lints
    ConRun --> Bench
    ConEnf --> Bench
    Aot --> RT
    XTask --> Bench

    FC --> Libs
    FC --> RT
    FR --> Libs
    FR --> RT
    Intg --> FC

    XTask -.-> MT
    XTask -.-> DX
    XTask -.-> WG

    class Vcore,Fnd,Spec,Macros,Intr,Ref,Primitives,Libs,SelfSS,Drv,Cuda,Wgpu,Spirv,Lower,EmitPtx,EmitNaga,EmitSpv,RefDrv,RT,Aot,Hs,Debug,Bench,Fuzz,Lints,XTask,ConSpec,ConGen,ConEnf,ConRun,TestHarness,MT active
    class FC,FR beta
    class Intg external
    class DX,WG planned
```

The older SVG remains in [docs/architecture.svg](docs/architecture.svg), but
the diagram above is the README source of truth because it names every
workspace crate and release-support status and separates active, beta,
and planned surfaces.

Legend:

- `active`: release-gated surface for the `0.7.2` train.
- `beta`: implemented and usable but currently excluded from release gate status.
- `planned`: target architecture work planned in repo docs and roadmap.
- `external`: separately released integrations and documentation-only references.

## The 10-second pitch

Most GPU frameworks make the simple parallel case comfortable. Vyre focuses on
the awkward cases: workloads with local state, branches, graph edges, parser
state, convergence loops, or rule-engine control flow. It tries to keep those
programs in IR long enough to test them against a reference implementation and
then run them on GPU without rewriting each workload as hand-authored kernels.

The core promise is practical: compose ops, run the reference path, run the
GPU backend, and keep the two results aligned where the contract requires
exactness.

Vyre is not a replacement for CUDA, WGPU, SPIR-V, or domain-specific compilers.
It is a contract layer above them. It is also not finished. The most
contributions right now are concrete: smaller modules, better conformance
coverage, CUDA parity tests, frontend bug fixes, benchmark cases that represent
real workloads, and docs that make rough edges visible instead of hiding them.

<!-- BEGIN GENERATED LANDING CONTRACT -->
## Workspace and release support

Vyre 0.7.2 contains 36 workspace crates. 26 crates are publishable and 10 are repository-internal (`publish = false`). The table is generated from Cargo manifests, `docs/CRATE_OWNERSHIP.toml`, and `docs/CRATE_GUIDES.toml`.

| Crate | Publication | Status | Role |
| --- | --- | --- | --- |
| `vyre` | crates.io | active | Expose the public Vyre API and feature-gated backend selection surface. |
| `vyre-aot` | crates.io | active | Plan and package ahead-of-time artifacts without owning live backend execution. |
| `vyre-bench` | repository-only | repository-internal | Own reproducible workload benchmarks, comparisons, budgets, and raw benchmark evidence. |
| `vyre-conform-enforce` | repository-only | repository-internal | Evaluate conformance results and enforce release certificate policy. |
| `vyre-conform-generate` | repository-only | repository-internal | Generate deterministic conformance cases from the conformance specification. |
| `vyre-conform-runner` | repository-only | repository-internal | Execute generated conformance cases across eligible concrete and reference backends. |
| `vyre-conform-spec` | repository-only | repository-internal | Define conformance case, result, and certificate schemas against the public facade. |
| `vyre-debug` | crates.io | active | Inspect, explain, and diagnose typed programs, lowering, and product-library composition. |
| `vyre-driver` | crates.io | active | Define backend-neutral device, capability, dispatch, evidence, and artifact contracts. |
| `vyre-driver-cuda` | crates.io | active, release path | Own native NVIDIA device acquisition, lowering, dispatch, graphs, and release-path evidence. |
| `vyre-driver-metal` | crates.io | active on Apple targets | Own native Apple device acquisition, lowering integration, dispatch, and backend evidence. |
| `vyre-driver-reference` | crates.io | active | Adapt the reference interpreter to the backend contract for deterministic conformance execution. |
| `vyre-driver-spirv` | crates.io | active | Own SPIR-V backend lowering, dispatch integration, and backend evidence. |
| `vyre-driver-wgpu` | crates.io | active, portable path | Own portable GPU acquisition, lowering, dispatch, graph execution, and backend evidence. |
| `vyre-emit-metal` | crates.io | active | Lower neutral programs into native Apple shader source through the shared emitter path. |
| `vyre-emit-naga` | crates.io | active | Lower neutral programs into the primary text emitter representation and related binary targets. |
| `vyre-emit-ptx` | crates.io | active | Lower neutral programs into the primary binary backend text artifact. |
| `vyre-emit-spirv` | crates.io | active | Lower neutral programs into SPIR-V artifacts through the shared emitter path. |
| `vyre-foundation` | crates.io | active | Own the typed IR, validation, optimizer, serialization, and foundational program contracts. |
| `vyre-frontend-c` | repository-only | beta | Parse C input and lower supported language constructs into typed Vyre programs. |
| `vyre-frontend-rust` | repository-only | beta | Lower the supported Rust frontend subset into typed Vyre programs and execute it through selected backends. |
| `vyre-grammar-gen` | crates.io | active | Generate host-side grammar tables consumed by frontend and parsing crates. |
| `vyre-harness` | crates.io | active | Provide reusable backend-neutral harness utilities for executing and comparing programs. |
| `vyre-intrinsics` | crates.io | active | Own registered hardware-mapped intrinsic contracts and their neutral program builders. |
| `vyre-libs` | crates.io | active | Own product-facing Tier 3 program compositions built from neutral primitives and contracts. |
| `vyre-lints` | crates.io | active | Enforce source-level project policies without depending on runtime crates. |
| `vyre-lower` | crates.io | active | Own backend-neutral lowering helpers and pre-emission transforms. |
| `vyre-macros` | crates.io | active | Provide compile-time registration and declaration macros without depending on runtime crates. |
| `vyre-primitives` | crates.io | active | Own reusable Tier 2.5 program builders shared by higher-level libraries and runtimes. |
| `vyre-reference` | crates.io | active | Execute programs with the canonical host oracle and produce semantic witnesses. |
| `vyre-runtime` | crates.io | active, experimental | Own backend-neutral execution planning, persistent runtime contracts, caches, telemetry, and IO substrate. |
| `vyre-self-substrate` | crates.io | active | Use Vyre primitives to implement scheduler, graph, coverage, and optimization support. |
| `vyre-spec` | crates.io | active, frozen surface | Own stable schemas, operation definitions, and compatibility contracts without runtime dependencies. |
| `vyre-test-harness` | repository-only | repository-internal | Provide shared execution and comparison infrastructure for conformance suites. |
| `vyre-test-support` | repository-only | repository-internal | Provide shared deterministic fixtures and assertions for workspace tests. |
| `xtask` | repository-only | repository-internal | Generate evidence and enforce repository, release, documentation, and architecture contracts. |

### Registered operation surface

The live operation schema contains 365 operations. Counts come from `docs/generated/OP_SCHEMA.json`; this README does not maintain a second inventory.

| Schema tier | Operations | Meaning |
| --- | ---: | --- |
| `intrinsic` | 9 | Tier 2 hardware operations |
| `primitive` | 149 | Tier 2.5 reusable operations |
| `libs` | 202 | Tier 3 library compositions |
| `runtime` | 5 | Driver-owned runtime dialect operations |

### Executable backend evidence

The current backend evidence selects `cuda` as the preferred release backend. The following rows come from `release/evidence/backends/backend-matrix.json`.

| Backend | Precedence | Dispatches | Acquired on evidence host |
| --- | ---: | :---: | :---: |
| `cuda` | 5 | true | true |
| `spirv` | 30 | true | true |
| `wgpu` | 30 | true | true |
| `cpu-ref` | 900 | true | true |

Metal remains active on supported Apple targets. It is absent from this Linux host probe rather than reported as a Linux dispatch backend. DXIL, DirectX, and browser WebGPU packaging remain planned surfaces.
<!-- END GENERATED LANDING CONTRACT -->

## `0.7.2` Release Execution Contract

The release route is explicit: `0.7.2` is a Vyre platform release, not a
production C compiler release.

| Package | Version | Role |
| --- | --- | --- |
| `vyre@0.7.2` | `0.7.2` | Public IR, lowering, optimizer, and backend trait surface |
| `vyre-driver-cuda@0.7.2` | `0.7.2` | NVIDIA/CUDA fast path for release workloads |
| `vyre-driver-wgpu@0.7.2` | `0.7.2` | Portable GPU fallback path for non-CUDA systems |
| `weirflow@0.1.3` | `0.1.3` | Standalone dataflow, witness, and soundness primitives integrated with Vyre |

External integrations exercise the public Vyre surface and provide end-to-end
feedback for contribution direction. `vyre-frontend-c` and `vyre-frontend-rust` are
beta/active-development consumers of Vyre.
They are included to show the intended compiler-front-end direction, but they
are not the release gate for `0.7.2`, are not advertised as clang-parity, and
must not be treated as production-ready C compiler components until their own
corpus, parity, and performance gates are green.

CUDA is the preferred release backend when an NVIDIA GPU is present. WGPU is a GPU fallback backend, not a CPU fallback. A failed CUDA or WGPU probe on a machine that should have a GPU is a configuration error surfaced to the caller with remediation context; it is never silently converted into CPU execution.

Release readiness is checked through backend metadata, feature matrices,
conformance reports, benchmark reports, and documentation checks generated by
the project tooling. C parser corpus reports are tracked as beta validation for
`vyrec`, not as a blocker for the Vyre platform release.

## Operation placement

An operation ID determines its schema tier. Foundation IR is below the
registered operation surface. Hardware intrinsics use the `intrinsic` tier,
reusable primitives use `primitive`, library compositions use `libs`, and
driver-owned dialect operations use `runtime`. External extension packs remain
outside the workspace inventory.

The generated workspace contract above reports the current count for each tier.
The architectural dependency and stability rules are defined in
[`docs/library-tiers.md`](docs/library-tiers.md).

**Op ID tells you the tier**: `vyre-intrinsics::hardware::fma_f32` is T2,
`vyre-primitives::graph::reachable` is T2.5, `vyre-libs::hash::fnv1a32`
is T3, `<community-dialect>::foo` is T4.

**Dependency direction is enforced**: T2 depends on T1 only;
T2.5 depends on T1 plus narrowly-approved intrinsics; T3 depends on
T2.5+T2+T1; T4 depends on T3+T2.5+T2+T1. Never upward. CI gate
`cargo_full run --bin xtask -- check-tier-deps` rejects violations.

**Region chain invariant**: every op at every tier wraps its body
in `Node::Region` and, when built from another registered op,
populates `source_region` so `cargo_full run --bin xtask -- print-composition <op_id>`
can walk the decomposition chain from public surface down to hardware
intrinsics. Spec in [`docs/region-chain.md`](docs/region-chain.md).

**Frontends stay outside core**. vyre is a GPU IR; source-language
frontends live in Tier-3 crates or downstream tools, generate grammar
tables / packed AST buffers, and feed GPU-side ops that walk those
buffers. Full spec + throughput math in
[`docs/parsing-and-frontends.md`](docs/parsing-and-frontends.md).

## How to navigate the docs

Every significant surface in vyre has a canonical doc. When onboarding:

| You want | Read this |
| --- | --- |
| Architecture and layering | `docs/ARCHITECTURE.md`, `docs/THESIS.md`, `docs/VISION.md` |
| **Which tier does my op belong to?** | `docs/library-tiers.md` |
| **Composition chain: how ops stay auditable** | `docs/region-chain.md` |
| **Source parsers: where frontends live** | `docs/parsing-and-frontends.md` |
| Documentation precedence | `docs/DOCUMENTATION_GOVERNANCE.md` |
| Maintainer execution queue | local `BACKLOG.md` (not published) |
| Current release procedure | `docs/RELEASE.md`, `docs/RELEASE_CHECKLIST.md` |
| Historical delivery evidence | `CHANGELOG.md`, `docs/release/`, `audits/` |
| **Ops catalog: full release surface** | `docs/ops-catalog.md` |
| **Execution inventory** | `docs/generated/OP_INVENTORY.md` |
| Writing a new op (contract + review checklist) | `docs/library-tiers.md` + `docs/region-chain.md`: **no raw WGSL ever; the whole contract is here** |
| Wire format + release tag reservations | `docs/wire-format.md` |
| Backend contract (capability queries, lifecycle hooks, sealing) | `vyre-driver/BACKEND_CONTRACT.md` |
| OpDef field audit (primitive / hardware / composite / tensor-core) | `vyre-spec/OPDEF_CONTRACT.md` |
| Frozen trait surfaces (5-year SemVer) | `docs/frozen-traits/*.md` |
| Memory model + ordering | `docs/memory-model.md` |
| Error-code catalog (stable u32 ids) | `docs/error-codes.md` |
| SemVer + API-stability policy | `docs/semver-policy.md` |
| Observability (tracing spans + stats schema) | `docs/observability.md` |
| Security disclosure + threat model | `SECURITY.md` + `docs/threat-model.md` |
| Release playbook (publish order, alpha soak) | `docs/RELEASE.md` |
| Design RFCs (Region inline, autodiff, quantization, collectives, megakernel) | `docs/rfcs/000*.md` |
| Persistent megakernel + `io_uring` NVMe streaming (Linux) | `vyre-runtime/README.md` |
| Testing standard and per-crate commands | `docs/testing/` |
| Per-crate test contract | `<crate>/tests/SKILL.md` |
| In-flight release-bar gap contracts | `contracts/release.md` |
| Benchmark baselines | `docs/optimization/BENCH_TARGETS.toml` + `release/evidence/benchmarks/` |
| Public-API snapshots (diff gate) | `<crate>/PUBLIC_API.md` |

## Quickstart

Add the IR crate and the CPU reference interpreter. Add a driver crate only for
the GPU you actually target:

```sh
cargo add vyre vyre-reference
cargo add vyre-driver-cuda   # NVIDIA
cargo add vyre-driver-wgpu   # portable GPU fallback
```

### Build a program

A program is a `vyre::ir::Program`. It holds three things: the buffers it reads
and writes, the number of threads per workgroup, and a body of IR nodes. This
program computes an element-wise XOR of two `u32` buffers.

```rust
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

fn xor_program(len: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(len),
            BufferDecl::read("b", 1, DataType::U32).with_count(len),
            BufferDecl::output("out", 2, DataType::U32).with_count(len),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::bitxor(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    )
}
```

Five details in that snippet decide how the program runs.

`BufferDecl::read` declares a buffer the kernel only reads. You supply its
contents at dispatch time.

`BufferDecl::output` declares a buffer the backend allocates for you and returns
after the dispatch. You do not pass it in as an input. Use this rather than
`BufferDecl::read_write` when the buffer is a pure result.

The second argument to both is the binding: the small integer the generated
kernel uses to locate that buffer. Bindings start at 0 and must be distinct.

`.with_count(len)` records how many elements the buffer holds. Set it whenever
you know the length. Backends use it to size the launch grid and to size the
output allocation. Backends do not all infer a missing count the same way, so an
explicit count is the portable choice.

`[64, 1, 1]` is the workgroup size, meaning the number of threads in one
workgroup. The number of workgroups is inferred from the element count, so 256
elements at a workgroup size of 64 becomes 4 workgroups of 64 threads.
`Expr::gid_x()` reads the global thread index on the x axis, which gives each
thread one element.

### Run it on the CPU reference

`vyre-reference` is the interpreter that defines correct behavior for a program.
Every GPU backend is checked against it. Run your program here first: if the
reference and a backend disagree, the backend is wrong.

```rust
use vyre_reference::{reference_eval, value::Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len = 256u32;
    let program = xor_program(len);

    let a: Vec<u8> = (0..len).flat_map(|i| (0xF0 + i).to_le_bytes()).collect();
    let b: Vec<u8> = (0..len).flat_map(|_| 0x0Fu32.to_le_bytes()).collect();

    // One `Value` per buffer you supply. `out` is backend-allocated, so it is
    // not in this list.
    let outputs = reference_eval(&program, &[Value::Bytes(a.into()), Value::Bytes(b.into())])?;

    let Value::Bytes(bytes) = &outputs[0] else {
        return Err("expected a byte buffer".into());
    };
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(&words[..4], &[0xFF, 0xFE, 0xFD, 0xFC]);
    Ok(())
}
```

`reference_eval` returns one `Value` per returned buffer, in declaration order.
A buffer is returned when the backend allocates it, which is what
`BufferDecl::output` requests.

### Run it on a GPU

A backend is acquired once and then dispatched against many times. Acquisition
failure means the device or driver is not usable. Treat it as a configuration
error to fix, not a reason to fall back to the CPU.

```rust
use vyre::{DispatchConfig, VyreBackend};

fn run_cuda(
    program: &vyre::ir::Program,
    inputs: &[Vec<u8>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let backend = vyre_driver_cuda::CudaBackend::acquire()?;
    let mut outputs = backend.dispatch(program, inputs, &DispatchConfig::default())?;
    Ok(outputs.remove(0))
}
```

`dispatch` takes one input buffer per buffer you supply, in binding order, and
returns one `Vec<u8>` per backend-allocated output. For the XOR program that is
two inputs in and one output back, matching the reference call above.

`vyre-driver-wgpu` has the same shape. `WgpuBackend::acquire()` is synchronous,
so it needs no async runtime, and `dispatch_borrowed` avoids copying inputs you
already hold:

```rust
use vyre::{DispatchConfig, VyreBackend};

fn run_wgpu(
    program: &vyre::ir::Program,
    inputs: &[&[u8]],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()?;
    let mut outputs = backend.dispatch_borrowed(program, inputs, &DispatchConfig::default())?;
    Ok(outputs.remove(0))
}
```

Both backends produce the same 256 words as the reference run for the program
above.

### Move a program between processes

`Program` has a binary wire format. `to_wire` serializes and `from_wire`
reconstructs, and any program that passes validation survives the round trip
unchanged:

```rust
let wire = program.to_wire()?;
assert_eq!(Program::from_wire(&wire)?, program);
```

This is how a program crosses a process or machine boundary: build the program in
one place and execute it in another. See
[docs/wire-format.md](docs/wire-format.md) for the byte layout and the round-trip
invariant, [ARCHITECTURE.md](ARCHITECTURE.md) for the crate layering, and
[docs/inventory-contract.md](docs/inventory-contract.md) for how to add an op.

## Standard library

The Layer 1 primitives live in `vyre` (core) and are organized into domains:

- **Primitive ops**: bitwise, arithmetic, and logical operations with exhaustive edge-case coverage and algebraic law verification.
- **Byte/text scan ops**: Aho–Corasick, substring find-all, multi-way scanners with real WGSL kernels (one ingredient inside larger programs).
- **Workgroup coordination primitives**: stack, FIFO queue, priority queue, hashmap, state machine, typed arena, string interner, visitor walk, recursive descent, dataflow fixed-point, dominator tree, and union-find.
- **Compiler primitives**: DFA engines, parser combinators, dataflow solvers, and tree-walk abstractions composed from workgroup primitives.

Composition-inlineable helpers live inside `vyre`'s own `ops::` tree alongside their primitives:

- **DFA/regex compilation pipeline**: `regex_to_nfa` (Thompson) → `nfa_to_dfa` (subset construction) → `dfa_minimize` (Hopcroft) → `dfa_pack` (Dense or EquivClass) → `dfa_assemble` (composite entry).
- **Aho-Corasick construction**: CPU reference + WGSL kernel + 5 GOLDEN samples + 20 KAT vectors.
- **Content-addressed compilation cache**: skips the pipeline when the same pattern set has already been compiled.
- **Arithmetic helpers**: ~80 typed compositional ops (saturating, wrapping, clamp, lerp, midpoint, abs_diff, div_ceil/round/floor).

Scan positioning is data-owned in
[docs/optimization/SCAN_POSITIONING_MATRIX.toml](docs/optimization/SCAN_POSITIONING_MATRIX.toml).
That matrix names Vyre, Hyperscan, Vectorscan, Rust regex, Aho-Corasick,
memchr, hardware regex, and FPGA offload by workload class, with each row tied
to benchmark evidence, a semantic exclusion, or an unsupported capability
reason.

## Benchmarks

The benchmark story is moving toward compiler-grade macro workloads on CUDA,
not primitive element-wise crossover tables. Our primary performance claims
are backed by empirical runs recorded in [release/evidence/benchmarks/cuda-release-suite.json](release/evidence/benchmarks/cuda-release-suite.json),
covering 16 macro workload families with explicit CPU-SOTA release contracts on an RTX 5090 with CUDA 12.8,
using repeated wall-time and CPU-baseline samples per artifact.


| workload family | case | input floor | measured CUDA speedup vs CPU-SOTA |
|---|---|---:|---:|
| condition eval | `release.condition_eval.1m` | 12,582,916 bytes | 12,981.60x |
| string bitmap scatter | `release.string_bitmap_scatter.1m` | 8,388,612 bytes | 7,179.83x |
| offset count aggregation | `release.offset_count_aggregation.1m` | 12,582,916 bytes | 14,908.67x |
| metadata conditions | `conditions.yara_like.eval.1m` | 37,945,348 bytes | 1,537.90x |
| entropy window | `release.entropy_window.1m` | 12,582,916 bytes | 14,242.73x |
| quantified condition loops | `release.quantified_condition_loops.1m` | 12,582,916 bytes | 12,546.00x |
| alias reaching-def | `release.alias_reaching_def.1m` | 12,582,916 bytes | 14,302.58x |
| IFDS witness | `release.ifds_witness.1m` | 12,582,916 bytes | 14,181.72x |
| C AST traversal | `release.c_ast_traversal.1m` | 12,582,916 bytes | 4,378.81x |
| megakernel queued batches | `release.megakernel_queue.1m` | 12,582,916 bytes | 15,476.40x |
| e-graph saturation | `release.egraph_saturation.1m` | 12,582,916 bytes | 15,737.86x |
| sparse output compaction | `sparse.compaction.count.1m` | 4,194,308 bytes | 6,436.50x |
| callgraph reachability | `callgraph.reachability.step.262k` | 5,341,180 bytes | 208.84x |

Primitive measurements are smoke and lower-bound telemetry, not the release
claim. Release claims point to generated benchmark evidence for compound
parsing, dataflow, graph, rule-engine, megakernel, or optimizer workloads.

Linked registration uses the three `OpEntry` registries plus
`OpDefRegistration`, `BackendRegistration`, and `ExtensionRegistration`. The
canonical operation schema freezes and validates the joined operation view.

Versioning follows the substrate pattern. `vyre-spec` publishes rarely and every release is an event: new data types, never removals, aggressive `#[non_exhaustive]`. `vyre` publishes patch releases frequently for optimizations and new lowerings. Backend crates publish on their own cadence after passing their parity suites. A community contributor can depend on `vyre-spec` alone without linking any backend.

## The Cat A / Cat B / Cat C discipline

Vyre organizes every operation into exactly one of three categories. This is not metadata decoration; it is an architectural gate that determines what code can exist and what code is forbidden.

**Category A: Pure composition.** A Cat A op is built entirely from existing ops. It introduces no new backend code, no new shader kernel, and no unsafe hardware assumption. Correctness propagates by construction: if the primitives are certified, the composition is certified. Most user programs and high-level library ops live here.

A new Cat A op ships as a focused builder under `vyre-libs/src/<domain>/`
or, when it becomes shared substrate, under `vyre-primitives/src/<domain>/`.
It introduces no backend-specific lowering and no hidden interpreter. The
filesystem is still the registry boundary: one domain, one responsibility,
and no central hand-edited list.

When a Cat A builder returns a registered primitive program, wrap it with
`vyre_foundation::composition::tag_program`. The parent region names the Cat A
operation. Primitive regions keep their generator ids and record the parent in
`source_region`. The helper also preserves the original entry operation,
workgroup geometry, buffer declarations, and self-composition policy.
Do not rebuild the `Program` just to change its public operation id.

Primitive reuse is established by real parent-child composition edges.
Synthetic `consumer_a` or `consumer_b` registrations do not count as callers
and must not be added to satisfy primitive coverage.

The LEGO audit requires a Tier 3 operation with at least 20 IR nodes to place
at least 25% of those nodes under registered child regions. A domain-owned
operation may instead appear in the audit's reviewed pure-IR leaf set when no
lower registered composition unit exists. A leaf still uses only
backend-neutral IR. It does not gain Category C lowering or host evaluation.

**Category B: Forbidden CPU coupling.** Cat B is the immune system's reject list. No general runtime interpretation engine, stack-machine evaluator, or host-dispatch substitute may exist in vyre. The `nfa_scan` micro-interpreter is absent from the `0.7.2` release line: those scans are expressed as composed ops in vyre IR and lower to GPU. Any construct that forces the host CPU to step into the execution loop of a GPU program is a Category B violation and is rewritten or deleted.

CI enforces this with tripwire gates that scan for forbidden patterns: `typetag`, `#[ctor]`, `Any::downcast`, dynamic async futures, pub-use globs, fake functions with `todo!()`, and frozen trait signature edits. These patterns break the black-box invariant, so their absence is load-bearing. `inventory::submit!` is the sanctioned link-time registration mechanism; it is not a runtime dispatch path. GPU programs are expected to run on GPU backends. If a backend lacks a Category C hardware intrinsic, it returns `UnsupportedByBackend`; it never substitutes slow host execution. `vyre-reference` is a test oracle, not a runtime path.

**Category C: Hardware intrinsic with a contract.** A Cat C op declares a
dedicated backend lowering path, a pure-Rust reference oracle, a set of
algebraic laws, and engine invariants such as determinism, atomic
linearizability, barrier safety, and subnormal preservation. It has no
host substitute; unsupported hardware returns an error rather than silently
degrading the execution contract.

Every Cat C op must pass the parity gate before it ships. The gate runs exhaustive edge cases on the u8 domain, property-based witnesses on the u32 domain, adversarial mutations from the mutation catalog, and backend-oracle parity checks across archetypes. The algebraic laws include commutativity, associativity, identity, self-inverse, distributivity, DeMorgan, and op-specific identities. The engine invariants include deterministic output, atomic linearizability, workgroup invariance, subnormal preservation for strict ops, and declared ULP bounds for approximate float ops.

Performance is part of the contract. `docs/optimization/BENCH_TARGETS.toml`
defines the target classes, and `release/evidence/benchmarks/` records matched
GPU and CPU baseline evidence. An operation without current parity and
performance evidence is not presented as release-supported.

Determinism is achieved via restriction, not elimination. Strict IEEE 754 operations remain as two roundings; the backend cannot fuse them into FMA. Reductions are ordered sequentially or as a canonical binary tree. Subnormals are preserved for strict ops. Transcendentals such as `sin` and `cos` are approximate ops today: the reference path uses Rust `f32` math and the WGSL backend uses shader builtins, so their contract is a declared ULP tolerance rather than correctly rounded results. Approximate and strict never mix in the same certificate. You choose per operation, in the IR, visibly.

## Backend Parity

A backend passes only when it reproduces the reference bit-exactly across the entire op matrix, law suite, archetype corpus, adversarial mutation catalog, and enforcement gate battery. The gate battery includes:

- **Atomics safety**: every atomic operation is linearizable and race-free.
- **Barrier correctness**: control flow reconverges safely at every barrier.
- **Out-of-bounds detection**: buffer accesses stay within declared bounds.
- **Determinism enforcement**: identical inputs produce bit-identical outputs.
- **Wire-format validation**: round-trip serialization is lossless.
- **Architectural tripwires**: forbidden patterns are absent from the source tree.

A violation means the backend emitted a finding with an actionable fix hint that starts with `Fix: `. The suite does not rank findings by severity; backend divergence is treated as release-blocking until it is understood and fixed.

The parity suite is designed to catch silent divergence before release. Green
means the checked surface matched the reference for that run; red means stop
and fix the underlying cause.

There are four contributor flows:

- Add a new op by copying the template and filling in the spec, laws, archetypes, and KAT vectors.
- Add a new gate by dropping a file in `enforce/gates/` with a `REGISTERED` const.
- Add a new oracle by dropping a file in `proof/oracles/` with a `REGISTERED` const.
- Add a new backend by implementing `VyreBackend` and running it through the parity suite.

Community knowledge that does not require Rust can be expressed as TOML rules. Drop a file in `rules/{category}/{name}.toml` and the tool auto-loads on the next scan. Every flow is additive. Nothing requires editing a central list. The architecture grows without refactoring.

## Who uses vyre

- **External integrations.** Start with the generic consumer integration guide:
  [Consumer showcase](docs/consumer-showcase.md), then wire a repository through
  the same reference/GPU parity loop.

- **Security and analysis tools.** Rule compilers can lower detector DSLs into
  Vyre programs and drive evaluation through the same reference/GPU parity
  loop. The rough edges are real: each consumer still needs careful corpus
  tests, performance fixtures, and backend-specific failure handling.

- **Research compilers.** Lexer, parser, type-analysis, and dataflow
  experiments can emit Vyre IR instead of hand-writing WGSL or PTX. The C and
  Rust frontend crates are beta because frontend correctness needs broad real
  corpus coverage before it should be called production-ready.

- **GPU-first applications.** Workloads that need local coordination on GPU can
  use the same backend and conformance surface. New backend authors should
  expect to implement the trait, run the conformance suite, and add explicit
  evidence for any operation they claim to support.

## Contributing

Contributions are welcome. If you want a clean first change, pick one of:
- Add a failing contract test with a precise expected result.
- Reduce a rough edge in parser, graph, or GPU runtime behavior with evidence.
- Add or tighten conformance coverage where current parity is weak.
- Improve diagnostics, documentation clarity, or failure-mode handling.

Small, high-signal changes are preferred over broad refactors. We value
correctness, measurable performance, and reusable test evidence over broad
surface edits.

Review boundaries are strict because this project is mostly contracts. Law
declarations, reference semantics, certificate format, and conformance gates
need extra care. Append-only paths such as corpora, regressions, and golden
evidence should grow, not shrink. The project standard is simple: no fake
implementations, no fake returns, no decorative laws, no swallowed errors, no
dead code, and no contribution that only makes the suite quieter without making
it truer.

## Links

- [Architecture](docs/ARCHITECTURE.md): workspace layout, frozen contracts, CI laws
- [Wire format](docs/wire-format.md): the `VYRE` binary serialization spec
- [Inventory contract](docs/inventory-contract.md): link-time registration and extension rules
- [Semver policy](docs/semver-policy.md): normative version contract
- [Error codes](docs/error-codes.md): canonical registry of stable diagnostic codes
- [Vision](docs/VISION.md): the missing abstraction stack, After Effects architecture, network effects
- [Thesis](docs/THESIS.md): technical axioms and where vyre beats existing options
- [crates.io/crates/vyre](https://crates.io/crates/vyre)
- [github.com/santhreal/vyre](https://github.com/santhreal/vyre)
- [License: MIT](LICENSE-MIT) / [Apache-2.0](LICENSE-APACHE)

Parity is required before release.
