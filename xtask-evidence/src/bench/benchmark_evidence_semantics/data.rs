//! What benchmark release evidence is checked against, and the vocabulary of
//! the findings.
//!
//! The tables here are the contract: which release-surface flags must be
//! scanned, which hygiene patterns each family must cover, which optimization
//! analysis fixture families must exist, which CUDA counters prove which
//! optimization label ran, and which metric names may carry a release axis.
//! The enums are the findings those checks may produce. `logic` interprets
//! all of it, so changing what release evidence must contain is a change to
//! this file alone.

use std::path::PathBuf;

/// Bound on one evidence file read while checking benchmark semantics.
pub(crate) const MAX_BENCHMARK_EVIDENCE_SEMANTIC_TEXT_BYTES: u64 = 16_777_216;

/// Prefix every benchmark schema digest chain carries.
pub(crate) const BENCHMARK_SCHEMA_DIGEST_CHAIN_PREFIX: &str = "benchmark-schema-digest-chain:v1:";

/// Release surfaces the hygiene matrix must report as scanned.
pub(crate) const RELEASE_SURFACE_COVERAGE_FLAGS: &[&str] = &[
    "vyre_workspace",
    "cuda_driver_crate",
    "wgpu_driver_crate",
    "release_scripts",
    "github_workflows",
    "branch_protection_controls",
];

/// Hygiene pattern families and the patterns each one must cover.
///
/// The pattern names come from the scan that emits them, so this table records
/// only which family each list has to appear under.
pub(crate) const RELEASE_SURFACE_REQUIRED_PATTERNS: &[(&str, &[&str])] = &[
    (
        "resource_bound_patterns",
        xtask::gates::hygiene_matrix::RESOURCE_BOUND_PATTERNS,
    ),
    (
        "hidden_fallback_patterns",
        xtask::gates::hygiene_matrix::HIDDEN_FALLBACK_PATTERNS,
    ),
    (
        "release_tooling_patterns",
        xtask::gates::hygiene_matrix::CARGO_WRAPPER_PATTERNS,
    ),
];

/// Required optimization analysis fixture families: fixture id, the A-item it
/// proves, and the per-family counters that must be non-zero.
pub(crate) const OPTIMIZATION_ANALYSIS_FIXTURE_FAMILIES: &[(&str, &str, &[&str])] = &[
    (
        "A13-coalesce-fixture",
        "A13",
        &[
            "coalesced_unit_stride_sites",
            "strided_sites",
            "broadcast_sites",
        ],
    ),
    (
        "A14-shared-mem-promote-fixture",
        "A14",
        &["shared_mem_candidates", "shared_mem_tile_bytes"],
    ),
    (
        "A15-bank-conflict-fixture",
        "A15",
        &["bank_conflict_sites", "bank_conflict_critical_sites"],
    ),
    (
        "A16-vec-pack-fixture",
        "A16",
        &["vec_pack_chains", "vec_pack_ops_eliminated"],
    ),
];

/// CUDA optimization labels and the counters that prove each one ran.
pub(crate) const CUDA_TELEMETRY_CHECKS: &[(&str, &[&str])] = &[
    (
        "cuda-ptx-source-cache",
        &[
            "cuda_ptx_source_cache_entries",
            "cuda_ptx_source_cache_hits",
            "cuda_ptx_source_cache_misses",
        ],
    ),
    ("cuda-graph-replay", &["cuda_graph_launches"]),
    (
        "cuda-graph-materialized-output-cache",
        &["cuda_graph_materialized_cache_hits"],
    ),
    (
        "cuda-transfer-operation-telemetry",
        &[
            "cuda_host_upload_operations",
            "cuda_device_readback_operations",
        ],
    ),
    (
        "cuda-resident-borrowed-escape-hatch",
        &["cuda_resident_borrowed_fallback_dispatches"],
    ),
];

/// Metric names that may carry the cold pipeline build axis, most specific
/// first.
pub(crate) const COLD_PIPELINE_BUILD_METRICS: &[&str] = &[
    "cold_compile_ns",
    "cold_wall_ns",
    "compile_ns",
    "lower_ns",
    "optimize_ns",
];

/// Metric names that may carry the scan throughput axis.
pub(crate) const SCAN_THROUGHPUT_METRICS: &[&str] = &["wall_gb_s_x1000", "device_gb_s_x1000"];

/// Aggregate wall-clock percentile fields an inspection record reports, every
/// one of which a release proof must carry as a positive number.
///
/// Three gates spelled this list out and each one could have drifted from the
/// record it grades. A percentile added to the record is demanded by all of
/// them from here, or by none of them.
pub(crate) const AGGREGATE_WALL_PERCENTILE_FIELDS: &[&str] = &[
    "min_wall_p50",
    "min_wall_p95",
    "min_wall_p99",
    "min_baseline_wall_p50",
    "min_baseline_wall_p95",
    "min_baseline_wall_p99",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BenchmarkArtifactPathIssue {
    AbsolutePath,
    NonReleasePath,
    ParentTraversal,
    Missing {
        artifact_path: PathBuf,
    },
    OutsideWorkspace {
        artifact_path: PathBuf,
        workspace_root: PathBuf,
    },
}

impl BenchmarkArtifactPathIssue {
    pub(crate) fn describe(&self, label: &str, artifact: &str) -> String {
        match self {
            Self::AbsolutePath => {
                format!("{label} `{artifact}` must be a relative release path")
            }
            Self::NonReleasePath => {
                format!("{label} `{artifact}` must start with `release/`")
            }
            Self::ParentTraversal => {
                format!("{label} `{artifact}` must not contain parent directory traversal")
            }
            Self::Missing { .. } => format!(
                "{label} `{artifact}` is not a readable file"
            ),
            Self::OutsideWorkspace { .. } => format!(
                "{label} `{artifact}` resolves outside workspace"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LaunchPlanLabelIssue {
    MissingSingle,
    SingleHasMulti,
    MissingMulti { launch_count: f64 },
    MultiHasSingle { launch_count: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BackendConsistencyIssue {
    MissingCaseId {
        case_index: usize,
    },
    DuplicateCaseId {
        case_id: String,
        count: usize,
    },
    MissingCaseBackend {
        case_id: String,
        expected_backend: String,
    },
    CaseBackendMismatch {
        case_id: String,
        expected_backend: String,
        actual_backend: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContractBackendIssue {
    MissingBaselines { case_id: String, backend_id: String },
    NoApplicableBaseline { case_id: String, backend_id: String },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CudaTelemetryLabelIssue {
    MissingLabel {
        case_id: String,
        label: &'static str,
    },
    LabelWithoutCounters {
        case_id: String,
        label: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CudaForbiddenTelemetryIssue {
    ResidentBorrowedEscapeHatch { case_id: String, observed_p50: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceFingerprintFreshnessIssue {
    Mismatch {
        source_fingerprint: String,
        current_source_fingerprint: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteParityIssue {
    CudaBackendIdentity {
        issue: BackendSuiteBackendIssue,
    },
    WgpuBackendIdentity {
        issue: BackendSuiteBackendIssue,
    },
    MissingCudaPair {
        family_id: String,
        requested_case_id: String,
    },
    MissingWgpuPair {
        family_id: String,
        requested_case_id: String,
    },
    CountMismatch {
        cuda_count: usize,
        wgpu_count: usize,
    },
    SharedArtifactPath {
        path: String,
    },
    DuplicateCudaPair {
        family_id: String,
        requested_case_id: String,
        count: usize,
    },
    DuplicateWgpuPair {
        family_id: String,
        requested_case_id: String,
        count: usize,
    },
    StatusFieldMismatch {
        family_id: String,
        requested_case_id: String,
        field: &'static str,
        cuda_value: Option<u64>,
        wgpu_value: Option<u64>,
    },
    StatusStringFieldMismatch {
        family_id: String,
        requested_case_id: String,
        field: &'static str,
        cuda_value: Option<String>,
        wgpu_value: Option<String>,
    },
    StatusBlockersMismatch {
        family_id: String,
        requested_case_id: String,
        cuda_blockers: Option<Vec<String>>,
        wgpu_blockers: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteInventoryIssue {
    CountMismatch {
        artifact_count: usize,
        status_count: usize,
    },
    DeclaredFamilyArtifactCountMismatch {
        family_count: u64,
        artifact_count: usize,
    },
    DeclaredFamilyStatusCountMismatch {
        family_count: u64,
        status_family_count: usize,
    },
    MissingStatus {
        path: String,
    },
    MissingArtifact {
        path: String,
    },
    DuplicateArtifact {
        path: String,
    },
    DuplicateStatus {
        path: String,
    },
    DuplicateFamily {
        family_id: String,
        count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteMatrixCoverageIssue {
    FamilyCountMismatch {
        matrix_family_count: usize,
        suite_family_count: usize,
    },
    MissingMatrixFamily {
        family_id: String,
    },
    ExtraSuiteFamily {
        family_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteBackendIssue {
    Missing {
        expected_backend: String,
    },
    Mismatch {
        expected_backend: String,
        actual_backend: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteArtifactStatusIssue {
    MissingField {
        path: String,
        field: &'static str,
    },
    SourceFingerprintMismatch {
        path: String,
        status_source_fingerprint: String,
        artifact_source_fingerprint: String,
    },
    SourceTreeFingerprintMismatch {
        path: String,
        status_source_tree_fingerprint: String,
        artifact_source_tree_fingerprint: String,
    },
    SelectedBackendMismatch {
        path: String,
        status_selected_backend: String,
        artifact_selected_backend: String,
    },
    CaseCountMismatch {
        path: String,
        status_case_count: u64,
        artifact_case_count: u64,
    },
    FailedCountMismatch {
        path: String,
        status_failed_count: u64,
        artifact_failed_count: u64,
    },
    NumericFieldMismatch {
        path: String,
        field: &'static str,
        status_value: u64,
        artifact_value: u64,
    },
    StringFieldMismatch {
        path: String,
        field: &'static str,
        status_value: String,
        artifact_value: String,
    },
    CpuSota100xContractCaseCountMismatch {
        path: String,
        status_contract_cases: u64,
        artifact_contract_cases: u64,
    },
    CpuSota100xPassingCaseCountMismatch {
        path: String,
        status_passing_cases: u64,
        artifact_passing_cases: u64,
    },
    MissingRequestedCase {
        path: String,
        requested_case_id: String,
    },
    DuplicateRequestedCase {
        path: String,
        requested_case_id: String,
        count: usize,
    },
}
