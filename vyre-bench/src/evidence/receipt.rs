//! Versioned benchmark evidence protocol and receipt schemas.
//!
//! BACKLOG row 95 requires one versioned benchmark protocol and content-addressed
//! evidence store recording workload and input identity, semantic graph, resource identity,
//! compiler and backend binaries, objective, budgets, target facts, environment,
//! candidate funnel, selected portfolio, emitted resources, native baseline,
//! warm, cold, transfer and cache state, power/energy where available, raw samples,
//! uncertainty model, parity, and failures.

use serde::{Deserialize, Serialize};

/// Version identifier for the benchmark receipt protocol.
pub const BENCHMARK_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Workload and input identity descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadAndInputIdentity {
    /// Unique workload identifier.
    pub workload_id: String,
    /// Content fingerprint of the input dataset.
    pub input_fingerprint: String,
    /// Tensor or buffer dimensional shape.
    pub data_shape: Vec<usize>,
    /// Total element count across all inputs.
    pub element_count: u64,
}

/// Semantic graph identity and structure descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticGraphIdentity {
    /// Cryptographic digest of the neutral semantic graph.
    pub graph_digest: String,
    /// Total node count in the graph.
    pub node_count: usize,
    /// Total edge count in the graph.
    pub edge_count: usize,
    /// Sequence of operation names in topological order.
    pub operations: Vec<String>,
}

/// Resource identity and allocation requirements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceIdentityReceipt {
    /// Bound resource identifiers.
    pub resource_ids: Vec<String>,
    /// Staging memory reserved in bytes.
    pub staging_bytes: u64,
    /// Resident device memory allocated in bytes.
    pub resident_bytes: u64,
    /// Required memory alignment in bytes.
    pub alignment_bytes: usize,
}

/// Compiler and backend driver binary identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BinaryIdentityReceipt {
    /// Compiler semver version.
    pub compiler_version: String,
    /// Compiler source git commit hash.
    pub compiler_git_commit: String,
    /// Backend driver identifier.
    pub backend_name: String,
    /// Backend driver version.
    pub driver_version: String,
    /// Target binary payload format.
    pub target_format: String,
}

/// Optimization and measurement objective.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkObjective {
    /// Target metric name (e.g. latency, throughput, energy).
    pub metric: String,
    /// Optimization direction (e.g. minimize, maximize).
    pub direction: String,
    /// Target threshold value if bounded.
    pub target_value: Option<f64>,
}

/// Resource, compilation, and execution budgets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchmarkBudgets {
    /// Maximum search budget in milliseconds.
    pub search_budget_ms: u64,
    /// Maximum compilation budget in milliseconds.
    pub compile_budget_ms: u64,
    /// Execution deadline in nanoseconds.
    pub execution_deadline_ns: u64,
    /// Hard device memory budget in bytes.
    pub max_device_memory_bytes: u64,
}

/// Physical execution target hardware facts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetFactsReceipt {
    /// Device name reported by driver.
    pub device_name: String,
    /// Hardware architecture family.
    pub architecture: String,
    /// Number of physical compute units.
    pub compute_units: u32,
    /// Native subgroup / warp execution width.
    pub subgroup_size: u32,
    /// Maximum workgroup dimensions.
    pub max_workgroup_size: [u32; 3],
    /// Theoretical peak memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: u32,
}

/// Host execution environment provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvironmentReceipt {
    /// Operating system name.
    pub os: String,
    /// Kernel release version.
    pub kernel: String,
    /// Host CPU model description.
    pub cpu_model: String,
    /// Anonymized hostname hash.
    pub hostname_hash: String,
}

/// Candidate exploration and pruning funnel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateFunnelReceipt {
    /// Total schedule candidates generated.
    pub candidates_explored: usize,
    /// Candidates rejected during legal verification.
    pub candidates_pruned: usize,
    /// Candidates successfully compiled to target bytes.
    pub candidates_compiled: usize,
    /// Candidates evaluated on target device.
    pub candidates_evaluated: usize,
}

/// Selected schedule and optimization portfolio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedPortfolioReceipt {
    /// Portfolio candidate identifier.
    pub portfolio_id: String,
    /// Selected tile dimensions.
    pub tile_sizes: Vec<usize>,
    /// Selected loop unroll factors.
    pub unroll_factors: Vec<usize>,
    /// Subgroup partitioning strategy name.
    pub subgroup_partitioning: Option<String>,
}

/// Emitted device resources and compiled artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmittedResourcesReceipt {
    /// Content digests of emitted kernel binaries.
    pub kernel_digests: Vec<String>,
    /// Total emitted binary size in bytes.
    pub code_size_bytes: usize,
    /// Static shared memory per workgroup in bytes.
    pub shared_memory_bytes: u32,
    /// Register allocation count per thread.
    pub register_count: Option<u32>,
}

/// Native baseline comparator descriptor and relative metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeBaselineReceipt {
    /// Baseline implementation identifier (e.g. cuBLAS, MKL, native CPU).
    pub baseline_name: String,
    /// Baseline median execution time in nanoseconds.
    pub baseline_median_ns: u64,
    /// Measured speedup ratio (baseline_time / vyre_time).
    pub speedup_ratio: f64,
    /// Baseline library version string.
    pub baseline_version: String,
}

/// Warm, cold, transfer, and cache state execution timings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StateReceipt {
    /// Cold execution latency in nanoseconds.
    pub cold_time_ns: u64,
    /// Warm steady-state median latency in nanoseconds.
    pub warm_median_ns: u64,
    /// Host-to-device data transfer latency in nanoseconds.
    pub host_to_device_transfer_ns: u64,
    /// Device-to-host data transfer latency in nanoseconds.
    pub device_to_host_transfer_ns: u64,
    /// Pipeline or prefix cache hits during execution.
    pub cache_hit_count: u64,
    /// Pipeline or prefix cache misses during execution.
    pub cache_miss_count: u64,
}

/// Power and energy measurements where available.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowerAndEnergyReceipt {
    /// Average power consumption in Watts during measurement.
    pub average_watts: Option<f64>,
    /// Peak power consumption in Watts.
    pub peak_watts: Option<f64>,
    /// Total energy consumed in Joules.
    pub total_energy_joules: Option<f64>,
}

/// Uncertainty and statistical distribution model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UncertaintyModelReceipt {
    /// Sample arithmetic mean in nanoseconds.
    pub mean_ns: f64,
    /// Sample median in nanoseconds.
    pub median_ns: u64,
    /// Sample standard deviation in nanoseconds.
    pub stddev_ns: f64,
    /// Median absolute deviation in nanoseconds.
    pub mad_ns: f64,
    /// 95th percentile latency in nanoseconds.
    pub p95_ns: u64,
    /// 99th percentile latency in nanoseconds.
    pub p99_ns: u64,
    /// 95% confidence interval lower bound in nanoseconds.
    pub confidence_95_lower_ns: f64,
    /// 95% confidence interval upper bound in nanoseconds.
    pub confidence_95_upper_ns: f64,
}

/// Parity verification against reference/oracle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParityReceipt {
    /// Whether parity verification passed against oracle.
    pub passed: bool,
    /// Maximum absolute numerical error observed.
    pub max_absolute_error: f64,
    /// Maximum relative numerical error observed.
    pub max_relative_error: f64,
    /// Oracle implementation or backend name.
    pub oracle_backend: String,
}

/// Versioned content-addressed benchmark receipt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkReceipt {
    /// Schema protocol version.
    pub schema_version: u32,
    /// Workload and input identity.
    pub workload_and_input: WorkloadAndInputIdentity,
    /// Semantic graph identity and descriptor.
    pub semantic_graph: SemanticGraphIdentity,
    /// Resource identity and requirements.
    pub resource_identity: ResourceIdentityReceipt,
    /// Compiler and backend binary identity / versions.
    pub compiler_and_backend_binaries: BinaryIdentityReceipt,
    /// Optimization objective.
    pub objective: BenchmarkObjective,
    /// Execution and compilation budgets.
    pub budgets: BenchmarkBudgets,
    /// Physical device target facts.
    pub target_facts: TargetFactsReceipt,
    /// Host execution environment provenance.
    pub environment: EnvironmentReceipt,
    /// Search / compilation candidate funnel metrics.
    pub candidate_funnel: CandidateFunnelReceipt,
    /// Selected optimization / kernel portfolio.
    pub selected_portfolio: SelectedPortfolioReceipt,
    /// Emitted device resources and compiled artifacts.
    pub emitted_resources: EmittedResourcesReceipt,
    /// Native baseline comparator and relative metrics.
    pub native_baseline: NativeBaselineReceipt,
    /// Warm, cold, transfer, and cache state timings.
    pub state: StateReceipt,
    /// Power and energy measurements (if available).
    pub power_and_energy: Option<PowerAndEnergyReceipt>,
    /// Raw sample execution timings in nanoseconds.
    pub raw_samples: Vec<u64>,
    /// Uncertainty and statistical distribution model.
    pub uncertainty_model: UncertaintyModelReceipt,
    /// Parity verification against reference/oracle.
    pub parity: ParityReceipt,
    /// Structured failure reasons, if any.
    pub failures: Vec<String>,
}

impl BenchmarkReceipt {
    /// Compute the cryptographic content address for this benchmark receipt.
    ///
    /// Every field in the receipt contributes to the identity: two receipts
    /// with identical field values produce the exact same content address,
    /// and altering any field changes the address.
    #[must_use]
    pub fn content_address(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-benchmark-receipt-v1:");
        let canonical_bytes = serde_json::to_vec(self).expect("benchmark receipt serialization");
        hasher.update(&canonical_bytes);
        hasher.finalize().to_hex().to_string()
    }
}
