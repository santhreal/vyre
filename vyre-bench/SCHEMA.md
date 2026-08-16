# vyre-bench Result Schema (v1)

> Schema identifier: `vyre-bench.result.v1`

The JSON output schema `vyre-bench run --format json` produces.
Every table below names the fields of the struct that serializes it, so a field
that is absent here is absent from the report.

## Top-Level Object

`ReportSchema` in `src/report/json.rs`.

| Field | Type | Description |
|---|---|---|
| `schema` | `string` | Schema version identifier. Always `"vyre-bench.result.v1"`. |
| `run_id` | `string` | Unique run identifier, e.g. `"vyre-bench.smoke"`. |
| `suite` | `string` | Suite kind: `smoke`, `release`, `deep`, `gpu`, `sweep`, `cross-backend`, `evolve`, `adversarial`, `competition`, `honest`. |
| `selected_backend` | `string \| null` | Backend id the run dispatched through. |
| `backend_profile` | `BackendProfile \| null` | Capabilities of the selected backend at run time. |
| `git` | `Map<string, string>` | Git context at time of run. |
| `source_fingerprint` | `string` | Fingerprint of the benchmarked sources. |
| `source_tree_fingerprint` | `string` | Fingerprint of the whole source tree. |
| `environment` | `Environment` | Hardware/OS environment snapshot. |
| `features` | `string[]` | Active feature flags for this run (e.g. `["backend:cuda"]`). |
| `cases` | `CaseReport[]` | Per-case results. |
| `summary` | `Summary` | Aggregate statistics. |
| `blockers` | `string[]` | Reasons the run does not qualify as evidence. |

## `BackendProfile` Object

`ReportBackendProfile` in `src/report/json.rs`.

| Field | Type | Description |
|---|---|---|
| `backend` | `string` | Backend id. |
| `timing_quality` | `string` | How the timings were obtained. |
| `supports_device_timestamps` | `bool` | Whether device-side timestamps are available. |
| `supports_hardware_counters` | `bool` | Whether hardware counters are available. |
| `supports_subgroup_ops` | `bool` | Whether subgroup operations are available. |
| `supports_indirect_dispatch` | `bool` | Whether indirect dispatch is available. |
| `max_workgroup_size` | `[u32, u32, u32]` | Maximum workgroup size per dimension. |
| `max_invocations_per_workgroup` | `u32` | Maximum invocations in one workgroup. |
| `max_shared_memory_bytes` | `u32` | Shared memory available to one workgroup. |
| `max_storage_buffer_binding_size` | `u64` | Largest bindable storage buffer. |
| `subgroup_size` | `u32` | Subgroup width. |
| `compute_units` | `u32` | Compute unit count. |
| `mem_bw_gbps` | `u32` | Peak memory bandwidth in GB/s. |

## `Environment` Object

`EnvironmentData` in `src/probes/environment.rs`.

| Field | Type | Description |
|---|---|---|
| `os` | `string` | Operating system identifier. |
| `architecture` | `string` | Host architecture. |
| `cpu_model` | `string \| null` | CPU model string, when the host reports one. |
| `cpu_cores` | `usize` | Logical core count. |
| `has_gpu` | `bool` | Whether a device was probed. |
| `gpu_devices` | `GpuDevice[]` | One entry per probed device. |
| `nvidia_driver_version` | `string \| null` | Driver version, when available. |
| `nvidia_cuda_version` | `string \| null` | CUDA toolkit version, when available. |
| `features` | `string[]` | Feature flags the binary was built with. |

## `GpuDevice` Object

`GpuDeviceInfo` in `src/probes/environment.rs`.

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Adapter name. |
| `driver_version` | `string` | Driver version for the adapter. |
| `memory_total_mib` | `u64 \| null` | Total device memory in MiB. |
| `compute_capability_major` | `u32 \| null` | Compute capability major version. |
| `compute_capability_minor` | `u32 \| null` | Compute capability minor version. |

## `CaseReport` Object

`CaseReport` in `src/report/json.rs`.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Stable case identifier (e.g. `"foundation.elementwise.add.1m"`). |
| `workload_fingerprint` | `string` | Fingerprint of the workload the case ran. |
| `name` | `string` | Human-readable case name. |
| `owner_crate` | `string` | Crate that owns the workload. |
| `workload_class` | `string` | Workload class the case belongs to. |
| `tags` | `string[]` | Case tags. |
| `backend_id` | `string \| null` | Backend the case dispatched through. |
| `device_signature` | `string \| null` | Signature of the device profile the case ran on. |
| `held_out_corpus_id` | `string \| null` | Held-out corpus the case measured. |
| `needs_gpu` | `bool` | Whether the case requires a device. |
| `min_vram_bytes` | `u64 \| null` | Device memory the case requires. |
| `min_input_bytes` | `u64 \| null` | Input size the case requires. |
| `required_features` | `string[]` | Features the case requires. |
| `status` | `string` | `"pass"` for a case that qualifies as evidence, otherwise the reason it does not. |
| `wall_ns` | `f64 \| null` | Host-to-host wall-clock time in nanoseconds. |
| `correctness` | `Correctness` | How the output was checked. |
| `contract` | `PerformanceContract \| null` | The performance contract the case declares. |
| `performance` | `PerformanceEvaluation \| null` | Evaluation of that contract. |
| `metrics` | `Map<string, MetricStats>` | All captured metrics keyed by name. |
| `optimization_passes_applied` | `string[]` | Optimizer passes the run applied. |
| `artifacts` | `string[]` | Paths to generated artifacts (SVGs, traces). |

## `MetricStats` Object

`MetricStats` in `src/api/metric.rs`.

| Field | Type | Description |
|---|---|---|
| `min` | `u64` | Minimum observed value. |
| `p50` | `u64` | Median. |
| `p90` | `u64` | 90th percentile. |
| `p95` | `u64` | 95th percentile. |
| `p99` | `u64` | 99th percentile. |
| `p999` | `u64` | 99.9th percentile. |
| `p9999` | `u64` | 99.99th percentile. |
| `max` | `u64` | Maximum observed value. |
| `mean` | `f64` | Arithmetic mean. |
| `stddev` | `f64` | Standard deviation. |
| `samples` | `u32` | Number of measured samples. |
| `determinism_cv` | `f64 \| null` | Cross-run coefficient of variation from the determinism gate. |

## `Correctness` Value

`Correctness` in `src/api/case.rs`. An externally tagged enum: `"Exact"` is a
string, and every other variant is an object with one key.

| Value | Payload | Description |
|---|---|---|
| `"Exact"` | none | Output matched the oracle bit for bit. |
| `{"Toleranced": {…}}` | `ulp_budget: u32`, `max_observed_ulp: u32` | Output matched inside a ULP budget. |
| `{"Certificate": {…}}` | `digest: u8[32]` | Output was checked against a certificate digest. |
| `{"Invalid": {…}}` | `reason: string` | Output did not qualify; the reason blocks the run. |

## `PerformanceContract` Object

`PerformanceContract` in `src/api/case.rs`.

| Field | Type | Description |
|---|---|---|
| `primitive` | `string` | Name of the benchmarked primitive. |
| `baselines` | `BaselineTarget[]` | Every baseline the case must beat. |

## `BaselineTarget` Object

`BaselineTarget` in `src/api/case.rs`.

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Baseline name. |
| `crate_name` | `string` | Crate the baseline implementation comes from. |
| `class` | `string` | Baseline class: `"CpuSota"` or `"GpuSota"`. |
| `min_speedup_x` | `f64` | Speedup the case must reach over this baseline. |
| `backend_ids` | `string[]` | Backends the target applies to. |

## `PerformanceEvaluation` Object

`PerformanceEvaluation` in `src/api/case.rs`.

| Field | Type | Description |
|---|---|---|
| `speedup_x` | `f64 \| null` | Measured speedup over the baseline. |
| `contract_passed` | `bool` | Whether every declared target was reached. |
| `violations` | `string[]` | One entry per target that was not reached. |

## `Summary` Object

`ReportSummary` in `src/report/json.rs`.

| Field | Type | Description |
|---|---|---|
| `total_cases` | `usize` | Total number of cases attempted. |
| `passed` | `usize` | Cases whose status is `"pass"`. |
| `failed` | `usize` | Cases that did not pass. |
| `total_time_ns` | `u64` | Wall-clock time for the entire suite run. |
| `cache_hit_rate` | `f64 \| null` | Fraction of dispatches that hit the pipeline cache. |

## Example

```json
{
  "schema": "vyre-bench.result.v1",
  "run_id": "vyre-bench.smoke",
  "suite": "smoke",
  "selected_backend": "cuda",
  "git": {
    "commit": "44a3d6b0f8977548ef32a2f60c96e3982cccaf4b",
    "branch": "main",
    "dirty": "false"
  },
  "source_fingerprint": "11bccf28",
  "source_tree_fingerprint": "11bccf28",
  "environment": {
    "os": "linux",
    "architecture": "x86_64",
    "cpu_cores": 32,
    "has_gpu": true,
    "gpu_devices": [
      {
        "name": "NVIDIA GeForce RTX 5090",
        "driver_version": "580.65.06",
        "memory_total_mib": 32607
      }
    ],
    "features": ["backend:cuda"]
  },
  "features": ["backend:cuda"],
  "cases": [
    {
      "id": "foundation.elementwise.add.1m",
      "status": "pass",
      "wall_ns": 13500.0,
      "correctness": "Exact",
      "performance": {
        "speedup_x": 155.2,
        "contract_passed": true,
        "violations": []
      },
      "metrics": {},
      "artifacts": []
    }
  ],
  "summary": {
    "total_cases": 1,
    "passed": 1,
    "failed": 0,
    "total_time_ns": 2040000000,
    "cache_hit_rate": null
  },
  "blockers": []
}
```
