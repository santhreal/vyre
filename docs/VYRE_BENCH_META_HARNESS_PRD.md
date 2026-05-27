# Vyre Bench Meta-Harness PRD

## Purpose

`vyre-bench` is the single future performance and evolution harness for Vyre.
It replaces the removed scattered Criterion benches with one measurement
product that can drive CI, release budgets, dashboards, and an AlphaEvolve-style
candidate evaluation loop.

The old benchmark layout is intentionally deleted, not ported. The replacement
must rebuild cases from current Vyre contracts, correctness requirements, and
hot-path instrumentation instead of inheriting stale file names.

## Required Product Shape

- One crate: `vyre-bench`.
- One command surface: `cargo run -p vyre-bench -- ...`.
- One benchmark registry with stable case IDs.
- One result schema, emitted as JSON.
- One budget model keyed by case ID, never by file path.
- One correctness-before-speed runner.
- One environment capture model for CPU, GPU, OS, Rust, feature set, git commit,
  adapter limits, CUDA/SPIR-V availability, and driver versions.
- One candidate-evaluation API for future automated improvement.
- One gate forbidding new `[[bench]]` targets outside `vyre-bench`.

## CLI Contract

```text
cargo run -p vyre-bench -- list --format table
cargo run -p vyre-bench -- run --suite smoke --format json
cargo run -p vyre-bench -- run --suite release --enforce budgets --format json
cargo run -p vyre-bench -- run --suite gpu --adapter auto --format json
cargo run -p vyre-bench -- compare --baseline baseline.json --candidate candidate.json
cargo run -p vyre-bench -- explain foundation.optimizer.scheduler
cargo run -p vyre-bench -- evolve-eval --candidate candidate.toml --suite evolve
```

Criterion may be a reporting adapter later, but it is not the architecture.

## Suites

- `smoke`: short PR-safe contracts.
- `release`: publish-blocking correctness and performance contracts.
- `deep`: high-sample and large-corpus runs.
- `gpu`: live adapter dispatch, cache, upload, readback, and GPU counters.
- `evolve`: fast candidate scoring with strict correctness rejection.
- `adversarial`: hostile inputs, timeout/OOM boundaries, NaN/Inf, cache churn.
- `competition`: external comparison corpus and reports.

Suites select registered cases by tags, requirements, and budget class. They do
not own benchmark logic.

## Crate Layout

```text
vyre-bench/
  Cargo.toml
  src/
    main.rs
    lib.rs
    api/
      case.rs
      candidate.rs
      metric.rs
      score.rs
      suite.rs
    registry/
      ids.rs
      inventory.rs
    runner/
      budget.rs
      execute.rs
      isolate.rs
      sample.rs
      warmup.rs
    corpus/
      generators.rs
      manifest.rs
      programs.rs
      workloads.rs
    probes/
      allocations.rs
      cpu.rs
      environment.rs
      gpu.rs
      tracing.rs
    cases/
      foundation.rs
      optimizer.rs
      wire.rs
      reference.rs
      primitives.rs
      libs.rs
      runtime.rs
      wgpu.rs
      conform.rs
      competition.rs
    evolve/
      boundary.rs
      validation.rs
      patch.rs
    report/
      compare.rs
      json.rs
      markdown.rs
    config/
      budgets.rs
      profiles.rs
  rules/
    budgets.toml
    candidates.toml
    suites.toml
```

## Stable Case IDs

ID format:

```text
<layer>.<subsystem>.<operation>[.<variant>]
```

Examples:

```text
foundation.optimizer.scheduler
foundation.optimizer.cse
foundation.wire.to_wire
foundation.wire.from_wire
reference.eval.expr
runtime.megakernel.plan
runtime.megakernel.protocol.encode_ring
runtime.gpujson.encode
libs.matching.dfa_compile
libs.matching.decode_scan
wgpu.pipeline_cache.hit_rate
wgpu.dispatch.hot_path
conform.parity.f32_transcendental
```

IDs are stable across file/module renames.

## Core API

```rust
pub trait BenchCase: Send + Sync {
    fn id(&self) -> BenchId;
    fn metadata(&self) -> BenchMetadata;
    fn requirements(&self) -> BenchRequirements;
    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError>;
    fn run(&self, ctx: &mut BenchContext, prepared: &mut PreparedCase) -> Result<BenchRun, BenchError>;
    fn verify(&self, ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError>;
}
```

`run` measures work. `verify` proves the output is valid. A case without
verification cannot enter `release` or `evolve`.

## Metrics

Minimum metrics:

- wall time
- CPU time
- validation time
- canonicalization time
- optimizer total time
- optimizer pass time by pass ID
- wire encode/decode time
- lowering time
- pipeline cache lookup/insert time
- compile time
- upload time
- dispatch time
- readback time
- verification time
- allocation count and bytes
- peak RSS
- input/output bytes
- IR node count
- wire byte count
- GPU counters where available

The runner computes min, p50, p90, p95, p99, max, mean, stddev, and sample
count for every numeric metric.

## Correctness

```rust
pub enum Correctness {
    Exact,
    Toleranced { ulp_budget: u32, max_observed_ulp: u32 },
    Certificate { digest: [u8; 32] },
    Invalid { reason: String },
}
```

Incorrect, panicking, OOMing, timing-out, or GPU-misconfigured cases are invalid
before performance is scored.

## Candidate Evaluation

Candidate kinds:

- optimizer rule
- pass order
- fusion policy
- vectorization policy
- workgroup policy
- megakernel batch policy
- cache retention policy
- backend lowering policy

Evaluation sequence:

1. Capture environment.
2. Validate candidate manifest.
3. Build baseline if missing.
4. Build candidate configuration.
5. Run correctness contracts.
6. Run samples.
7. Compute score.
8. Emit JSON.
9. Exit non-zero on correctness, resource, timeout, or budget failure.

## Scoring

Score is multi-objective:

- runtime
- compile time
- allocation count
- allocated bytes
- memory bandwidth
- cache hit rate
- p99 stability
- correctness confidence

Use weighted geometric means so one catastrophic regression cannot be hidden by
many small wins.

## Instrumentation Required In Vyre

Add a shared `vyre_foundation::perf` surface:

```rust
pub struct PerfScope;
pub struct PerfSample;
pub struct PerfCounter;
pub struct PerfContext;
```

Required phase names:

```text
frontend.build
foundation.validate
foundation.canonicalize
foundation.optimize.total
foundation.optimize.pass.<pass_id>
foundation.wire.encode
foundation.wire.decode
driver.lower
driver.cache.lookup
driver.cache.insert
driver.pipeline.compile
driver.upload
driver.dispatch
driver.readback
driver.verify
runtime.megakernel.plan
runtime.megakernel.encode
runtime.megakernel.dispatch
runtime.gpujson.encode
runtime.gpujson.decode
```

Instrumentation must be no-allocation on hot recording paths, backed by
thread-local buffers and cheap atomics. JSON serialization happens only at
report time.

## GPU Policy

- GPU suites require successful GPU probe.
- Probe mismatch is a hard error, not a skip.
- GPU timing splits host prep, validation, lowering, cache lookup, compile,
  upload, dispatch, readback, and verification.
- Dispatch-only cases must not hide compile time inside dispatch time.

## Budgets

Budgets are keyed by case ID:

```toml
[cases.foundation.optimizer.scheduler]
suites = ["smoke", "release", "evolve"]
wall_ns_p95_max = 300000
alloc_count_p95_max = 0
correctness = "exact"

[cases.wgpu.dispatch.hot_path]
suites = ["gpu", "release"]
dispatch_ns_p95_max = 50000
alloc_count_p95_max = 4
requires_gpu = true
```

No path-keyed performance manifests.

## Implementation Order

1. Delete old bench crates, bench directories, `[[bench]]` entries, stale
   budget manifests, and scattered bench scripts.
2. Add `vyre-bench` crate and CLI.
3. Add registry, IDs, suites, and JSON reporting.
4. Add ID-keyed budget parsing and enforcement.
5. Add environment/GPU probes.
6. Add `vyre_foundation::perf`.
7. Rebuild smoke/release/evolve cases from current product contracts.
8. Rebuild foundation/reference/runtime/libs/WGPU cases.
9. Rebuild competition corpus handling as structured corpus data.
10. Add candidate evaluation boundary.
11. Replace release/signoff scripts with `vyre-bench` commands.
12. Add gates forbidding scattered benchmark reintroduction.

## Acceptance Criteria

- `cargo run -p vyre-bench -- list` shows every registered case.
- `cargo run -p vyre-bench -- run --suite smoke --format json` passes.
- `cargo run -p vyre-bench -- run --suite release --enforce budgets` passes on
  the target GPU machine.
- No workspace crate except `vyre-bench` declares `[[bench]]`.
- No release/evolve case lacks correctness verification.
- No budget is keyed by file path.
- Scripts and contracts call `vyre-bench`, not scattered benches.
- The old smoke/deep/competition benchmark crates and per-crate benches are not
  workspace members.

