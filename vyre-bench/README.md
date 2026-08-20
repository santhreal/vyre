# vyre-bench

The cross-backend benchmark and parity harness for the
[Vyre](../README.md) GPU compiler.

Not here: a benchmark whose baseline is vyre's own unfused output.
Beating your own slow path is not a result. The baseline is the best
available native implementation for that class.

## Architecture

```
vyre-bench
├── api/           # Core types: BenchCase trait, metrics, candidates, suites
├── cases/         # Benchmark implementations (one file per workload)
├── runner/        # Execution engine, snapshot diffing, execute_suite()
├── report/        # JSON schema, scorecard generation
├── cli/           # Binary-only report, dashboard, and evolution-server commands
├── probes/        # NVML environment probing, thermal normalization
├── registry/      # inventory-based case collection
├── cli.rs         # CLI: run, list, explain, snapshot-diff, compare, dashboard
└── main.rs        # Entry point
```

## Quick Start

```bash
# List all registered benchmarks
cargo_full run -p vyre-bench -- list

# Run the smoke suite (30 measured samples, GPU required)
cargo_full run -p vyre-bench --release -- run --suite smoke --measured-samples 30

# Run honest workload suite
cargo_full run -p vyre-bench --release -- run --suite honest --measured-samples 30

# Generate CUDA release evidence
cargo_full run -p vyre-bench --release -- run --backend cuda --suite release --measured-samples 30 --warmup-samples 300 --enforce-budgets

# Generate WGPU release evidence
cargo_full run -p vyre-bench --release -- run --backend wgpu --suite release --measured-samples 30 --warmup-samples 300 --enforce-budgets

# Compare two runs
cargo_full run -p vyre-bench -- compare --baseline baseline.json --candidate candidate.json

# Generate HTML dashboard from latest snapshot
cargo_full run -p vyre-bench -- dashboard --output dashboard/
```

## Suite Kinds

| Suite | Purpose | Min Samples |
|---|---|---|
| `smoke` | Fast CI gate, foundation primitives | 30 |
| `release` | Full coverage pre-release | 30 |
| `deep` | Extended analysis, tail latencies | 100 |
| `gpu` | GPU-specific capabilities | 30 |
| `honest` | Real-world workloads with CPU baselines | 30 |
| `sweep` | Workgroup × size parameter grid | 5 |
| `cross-backend` | Same program across CUDA/SPIR-V/WGPU | 30 |
| `evolve` | Evolutionary optimization search | 50 |
| `adversarial` | Pathological inputs, register exhaustion | 30 |
| `competition` | Parameter golf scoring | 30 |

## Honest Workloads

These benchmarks use CPU baselines that run the same algorithm or contract shape, enabling defensible speedup claims. The release suite contains both real algorithm workloads and synthetic contract workloads: synthetic cases are allowed only when they model a named release contract with exact CPU-output parity and an explicit SOTA baseline class.

| Workload | Description | Contract |
|---|---|---|
| `hashtable.openaddr.probe.10m` | Open-addressing hash table: 1M probes against a prebuilt 10M-key table | 10× vs hashbrown |
| `interpreter.bytecode.dispatch.10m` | Bytecode VM: 4096 instances × 2500 instructions | 3× vs interpreted |
| `crypto.aes_ctr.encrypt.10mb` | AES-128-CTR over 10MB | 3× vs OpenSSL EVP AES-NI |
| `regex.backtracking.adversarial` | `(a+)+b` pattern on hostile inputs (4096 instances) | 100× vs PCRE2 |
| `bigint.modexp.4096` | 1024 instances of modular exponentiation | 3× vs rug/GMP |

## Release Workloads

The `release` suite must cover at least 12 workload families before Vyre can ship. CUDA is the preferred release backend; WGPU is the portable GPU fallback. Every row below has exact output parity against a CPU baseline that runs in this process. The enforced speedup threshold and the CPU baseline each row is judged against live in `docs/optimization/BENCH_TARGETS.toml`, and the measured numbers live under `release/evidence/benchmarks/`; a threshold repeated here would be a second owner that drifts.

| Workload family | Case id | Owner crate |
|---|---|---|
| Conditional rule evaluation | `release.condition_eval.1m` | vyre |
| String bitmap scatter | `release.string_bitmap_scatter.1m` | vyre-libs |
| Offset/count aggregation | `release.offset_count_aggregation.1m` | vyre-libs |
| PE/header metadata predicates | `metadata.condition.filesize_header.1m` | vyre-libs |
| Entropy/window predicates | `release.entropy_window.1m` | vyre-libs |
| Quantified condition loops | `release.quantified_condition_loops.1m` | vyre |
| Alias/reaching-definition predicates | `release.alias_reaching_def.1m` | vyre-bench |
| IFDS witness predicates | `release.ifds_witness.1m` | vyre-bench |
| AST motif traversal predicates | `release.ast_motif_traversal.1m` | vyre-libs |
| Persistent megakernel queued batches | `release.megakernel_queue.1m` | vyre-runtime |
| E-graph saturation predicates | `release.egraph_saturation.1m` | vyre-lower |
| Sparse fired-rule readback | `sparse.compaction.count.1m` | vyre-runtime |
| Callgraph reachability | `callgraph.reachability.step.262k` | vyre-primitives |

## Verification Gates

Every benchmark run enforces these quality gates:

- **G1**: CUDA event timing populates `kernel_queue_submit_ns`, `kernel_execute_ns`, `device_sync_ns`
- **G2**: Tail latencies are monotonic: min ≤ p50 ≤ p90 ≤ p95 ≤ p99 ≤ p999 ≤ p9999 ≤ max
- **G3**: Determinism gate: `CV < 0.005` for stable cases across 10 runs
- **G4**: Roofline metrics: `bytes_read`, `bytes_written`, `peak_bandwidth_gb_s`
- **G5**: Pipeline cache hit rate: second-run cache hit > 95%
- **G6**: Per-commit snapshots: `snapshots/<commit>.json` written automatically
- **G7**: Thermal normalization: NVML temperature monitoring, `thermal_unstable` detection
- **G9**: Sweep matrix: workgroup × size parameter grid
- **G10**: Cross-backend: CUDA/SPIR-V/WGPU parity verification
- **G12**: CLI verification: all subcommands produce correct output

## CI Integration

The `bench-regression.yml` workflow runs on every PR and push to `main`:
1. Builds `vyre-bench` in release mode on a self-hosted GPU runner
2. Runs smoke and honest suites with 30 measured samples
3. Compares against the baseline snapshot (if available)
4. Comments the comparison on the PR
5. Fails if any case regresses by > 1σ

## Schema

Result JSON follows the `vyre-bench.result.v1` schema. See [SCHEMA.md](SCHEMA.md) for full documentation.

## Dashboard

`vyre-bench dashboard --output dashboard/` generates:
- `index.html`: interactive scorecard with dark-mode UI
- Per-case SVG bar charts (p50/p99/max)
- `cross-backend.svg`: cross-backend comparison
- `scorecard.md`: markdown summary
- a raw JSON data file under the selected dashboard output directory

## Adding a New Benchmark

1. Create `src/cases/<workload>.rs` implementing `BenchCase`
2. Add `inventory::submit! { &MyWorkload as &'static dyn BenchCase }` at the bottom
3. Register in `src/cases/mod.rs`
4. Run `./cargo_full test -p vyre-bench` to verify integration

## Release evidence

Release readiness is proven through the Vyre evidence manifest and generated artifacts under `release/evidence/`. Claims here must map to concrete gate output, benchmark output, conformance output, or documentation proof files before the release requirement can be closed.

Concrete evidence anchors:

- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- `release/evidence/benchmarks/wgpu-fallback-suite.json`
- `release/evidence/benchmarks/bench-release-axes.json`

## The frontier leaderboard artifact

`release-benchmarks` writes `release/evidence/benchmarks/frontier-leaderboard.json`
alongside the suite artifacts. It is the head-to-head table: for every frontier
baseline the release claims to beat, one row carrying vyre's measurement and the
comparator's.

```bash
./cargo_full run --bin xtask -- release-benchmarks --backend cuda
```

The baselines it must cover come from `docs/optimization/FRONTIER_LEADERBOARD_BASELINES.toml`,
so adding a competitor is a data edit, not a code edit.

Each row in `rows` carries:

- `baseline_id`, `research_key`, and `baseline` - which comparator the row is against
  and the research source that establishes it as a real baseline.
- `workload_family` and `metric_family` - what was measured. The metric family is
  derived from the workload family, so a row cannot claim a metric the workload does
  not produce.
- `cpu_digest` and `gpu_digest` - the comparator parity check. Both arms must digest
  to the same value, which is what makes the comparison a comparison rather than two
  unrelated runs.
- `throughput_gb_s_x1000_p50`, `latency_wall_ns_p50`, `memory_total_mib_p50`, and
  `transfer_bytes_p50` - the p50 measurements, integer-scaled so the artifact has no
  floating-point drift.
- `selected_plan_reason` and `rejected_plan_reasons` - which execution plan ran and
  why the others did not.
- `blockers` - why this row does not count as evidence, when it does not.

At the top level, `missing_baselines` lists required baselines with no row, and
`blockers` is the release-blocking total. An empty `blockers` array is the only
passing state; `release-benchmarks` exits non-zero otherwise, and exits 2 when a
baseline manifest cannot be read.

`source_tree_fingerprint` keys the artifact to the workspace source it measured, so
any source change after the run invalidates it and the gate says so. Run the
benchmarks last.

<!-- BEGIN GENERATED CLI CONTRACT -->
## Command-line interface

This section is generated from `docs/CLI.toml` and executable help output.

### `vyre-bench`

```console
./cargo_full run -p vyre-bench --bin vyre-bench -- --help
```

Commands: `compare`, `dashboard`, `evolve-server`, `explain`, `list`, `release-matrix`, `run`, `snapshot-diff`, `validate-benchmark-bundle`, `validate-comparison`, `validate-report`.

Hardware: Run commands require the explicitly selected backend device. Report validation and comparison are device independent.

Environment: RAYON_NUM_THREADS configures CPU baselines. VYRE_ALLOW_FEW_SAMPLES=1 permits local smoke runs below the release sample floor.

Configuration: Suite, case, backend, sample, budget, report, and output settings are command-line arguments.

Failure behavior: Unavailable backends, invalid suites or reports, benchmark mismatches, timeouts, and budget violations return non-zero.

Exit codes: 0 on success or help, 1 on benchmark or validation failure, 2 on invalid arguments.
<!-- END GENERATED CLI CONTRACT -->

<!-- BEGIN GENERATED CRATE CONTRACT -->
## Crate contract

This section is generated by `xtask crate-readmes --write` from
the crate manifest, release train, ownership registry, and crate-guide metadata.

### Purpose

Own reproducible workload benchmarks against the best available native baseline for each class, not against vyre's own unfused output.

### Boundaries

The `benchmarks` owner maintains this `tooling` crate at `vyre-bench`.
Its allowed internal production dependencies are: `vyre`, `vyre-driver`, `vyre-driver-cuda`, `vyre-driver-reference`, `vyre-driver-wgpu`, `vyre-emit-ptx`, `vyre-foundation`, `vyre-libs`, `vyre-lower`, `vyre-primitives`, `vyre-reference`, `vyre-registry-link`, `vyre-runtime`, `vyre-spec`, `xtask`.
Any other normal or build dependency requires an ownership-registry change.

### Minimal real example

Run the checked-in behavior from `vyre-bench/src/main.rs`:

```console
./cargo_full run -p vyre-bench -- --help
```

### Features

- Manifest features: `cli`, `default`, `device-tests`
- Default feature members: `cli`

### Errors and unsupported behavior

Invalid arguments, stale evidence, violated repository contracts, and failed commands return a nonzero status with a concrete correction.

### Testing

See [`docs/testing/vyre-bench.md`](../docs/testing/vyre-bench.md) for the crate's test command,
hardware contract, expected skips, and failure semantics. It is generated
from `docs/testing/TESTING.toml`, which is authoritative.

### Release status

This crate is internal benchmark tooling for the 0.8.0 train and is not published to crates.io.

### Ownership

[`docs/CRATE_OWNERSHIP.toml`](../docs/CRATE_OWNERSHIP.toml) is authoritative for this crate's
responsibility and allowed internal edges.

### License

Licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option. See the workspace `LICENSE-APACHE` and `LICENSE-MIT` files.

<!-- END GENERATED CRATE CONTRACT -->
