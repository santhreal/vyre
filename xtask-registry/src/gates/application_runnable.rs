//! Application runnable gate.
//!
//! A whole application is runnable when the compiler carries it from a closed
//! whole-program graph to a target payload a certified device already executed
//! with verified output. Operation presence, candidate legality and a fast
//! isolated kernel are none of that: each is a fact about a part, and a tree
//! can hold all three while no complete route exists.
//!
//! This gate derives the route from the live registries and settles it against
//! the recorded evidence corpus. One application is derived per registered
//! generic frontend dialect: its operations are staged into one connected
//! multi-node graph, that graph is analyzed, compiled, and lowered through
//! every registered production backend's target compiler, and every staged
//! operation must appear in a conformance certificate that a device produced.
//!
//! Four states therefore fail rather than satisfy the gate. A single-node or
//! disconnected graph is an isolated kernel and is refused by node and edge
//! count. A builder that succeeds without a target payload stops short and is
//! refused by the missing payload. Evidence recorded host-only certifies the
//! reference oracle and never a device, so an application covered by nothing
//! else is refused by name. A schedule feature the artifact record claims and
//! the lowered target record does not carry is refused as a disagreement
//! between the two records.
//!
//! Independently versioned consumers cannot be compiled here, because Vyre
//! never imports a consumer's source, manifests, or fixtures. They prove the
//! same contract through an authenticated domain-neutral evidence receipt whose
//! schema records graph and resource identities and execution facts and carries
//! no model or application name.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use vyre_foundation::dialect::DialectRegistry;
use vyre_foundation::execution_plan::fusion::rename_buffer;
use vyre_foundation::ir::{
    GraphInput, GraphOutput, GraphValueId, Program, ProgramGraph, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest,
    ExecutionTopology, ExternalFacts, ObjectiveMetric, ResidentPartitionMode, SearchBudget,
    TargetModuleBundle,
};

use xtask::artifact_gate::Inspection;
use xtask::gate::{Finding, GateBehavior, GateCtx, GateError, Report};

/// Current schema version for external domain-neutral application evidence receipts.
pub const APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Artifact path where readiness evidence is written.
pub const READINESS_EVIDENCE_PATH: &str = "release/evidence/conformance/application-readiness.json";

/// Current schema version for generated application readiness evidence.
pub const READINESS_SCHEMA_VERSION: u32 = 2;

/// Directory holding the recorded conformance certificate corpus.
pub const CERTIFICATE_DIR: &str = "release/evidence/conformance";

/// Recorded benchmark evidence the release workload roster is read from.
pub const WORKLOAD_MATRIX_PATH: &str = "release/evidence/benchmarks/release-workload-matrix.json";

/// Lowest per-backend conformance certificate shape this gate reads.
pub const MIN_CERTIFICATE_SCHEMA_VERSION: u32 = 4;

/// Product-family terms a domain-neutral receipt may not carry.
///
/// One definition, read by receipt validation and by the closure test over the
/// receipt's own serialized field names, so a receipt field or an accepted
/// value cannot be admitted under a name the other half rejects.
pub const PROHIBITED_DOMAIN_TERMS: &[&str] = &[
    "app",
    "bert",
    "claude",
    "diffusion",
    "gemini",
    "gpt",
    "llama",
    "mistral",
    "model",
    "onnx",
    "pytorch",
    "resnet",
    "tensorrt",
    "tgi",
    "torch",
    "transformer",
    "triton",
    "vllm",
    "whisper",
    "yolo",
];

/// External domain-neutral evidence receipt submitted by downstream consumers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApplicationEvidenceReceipt {
    /// Schema format version.
    pub receipt_version: u32,
    /// Canonical digest of the external schema.
    pub schema_digest: [u8; 32],
    /// Canonical digest of the validated semantic graph.
    pub graph_digest: [u8; 32],
    /// Canonical digest of the resource roster bound for execution.
    pub resource_digest: [u8; 32],
    /// Canonical digest of the selected compiler artifact.
    pub artifact_digest: [u8; 32],
    /// Canonical digest of the materialized target payload.
    pub payload_digest: [u8; 32],
    /// Target payload format identifier.
    pub target_format: String,
    /// Whether target device execution passed with verified output parity.
    pub execution_passed: bool,
    /// Canonical digest of verified output bytes.
    pub output_digest: [u8; 32],
    /// Recorded production benchmark metrics.
    pub benchmark_metrics: BenchmarkEvidenceMetrics,
    /// Authentication tag computed by the authenticated producer.
    pub auth_tag: String,
}

/// Domain-neutral benchmark metrics recorded in an evidence receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkEvidenceMetrics {
    /// Compilation time in nanoseconds.
    pub compile_time_ns: u64,
    /// Artifact materialization / load time in nanoseconds.
    pub load_time_ns: u64,
    /// 50th percentile execution latency in nanoseconds.
    pub p50_latency_ns: u64,
    /// 99th percentile execution latency in nanoseconds.
    pub p99_latency_ns: u64,
    /// Throughput in items / elements per second.
    pub throughput_items_per_sec: f64,
    /// Peak resident memory in bytes.
    pub peak_resident_bytes: u64,
    /// Selected schedule feature tags.
    pub schedule_features: Vec<String>,
}

/// Errors arising during evidence receipt validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptValidationError {
    /// Unsupported receipt schema version.
    UnsupportedVersion {
        /// Found version.
        found: u32,
        /// Expected version.
        expected: u32,
    },
    /// Zero/empty digest in canonical identity field.
    ZeroDigest {
        /// Name of the zero-valued field.
        field: &'static str,
    },
    /// Target format string is empty.
    EmptyTargetFormat,
    /// Execution status is false.
    ExecutionFailed,
    /// Latency metric is zero or invalid.
    InvalidLatency {
        /// P50 latency.
        p50: u64,
        /// P99 latency.
        p99: u64,
    },
    /// Schedule features list is empty.
    EmptyScheduleFeatures,
    /// A product-family name was detected in a receipt value.
    ProhibitedDomainName {
        /// JSON pointer of the field carrying the name.
        field: String,
        /// Offending value.
        value: String,
        /// Term that matched.
        term: &'static str,
    },
    /// Receipt is not authenticated or the authentication tag is invalid.
    UnauthenticatedReceipt,
    /// The receipt could not be serialized for the neutrality scan.
    Unserializable(String),
}

impl fmt::Display for ReceiptValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found, expected } => write!(
                f,
                "unsupported receipt version {found}; expected {expected}. Fix: update evidence generator to schema version {expected}"
            ),
            Self::ZeroDigest { field } => write!(
                f,
                "zero or missing digest in field `{field}`. Fix: supply valid 32-byte cryptographic digest"
            ),
            Self::EmptyTargetFormat => write!(
                f,
                "target format is empty. Fix: specify the target payload format identity the payload was emitted in"
            ),
            Self::ExecutionFailed => write!(
                f,
                "device execution failed in receipt. Fix: resolve execution failure on target backend"
            ),
            Self::InvalidLatency { p50, p99 } => write!(
                f,
                "invalid latency metric: p50={p50}ns, p99={p99}ns (must be > 0 and p50 <= p99). Fix: record non-zero execution timings"
            ),
            Self::EmptyScheduleFeatures => write!(
                f,
                "schedule features roster is empty. Fix: record selected schedule features in the receipt"
            ),
            Self::ProhibitedDomainName { field, value, term } => write!(
                f,
                "receipt field `{field}` carries the product-family term `{term}` in `{value}`. Fix: use domain-neutral descriptors and canonical digests only"
            ),
            Self::UnauthenticatedReceipt => write!(
                f,
                "unauthenticated receipt or invalid authentication tag. Fix: sign the evidence receipt with the shared authentication key"
            ),
            Self::Unserializable(error) => write!(
                f,
                "receipt could not be serialized for the neutrality scan: {error}. Fix: keep every receipt field serializable"
            ),
        }
    }
}

impl std::error::Error for ReceiptValidationError {}

/// Derive a 32-byte keyed hash key for receipt authentication.
#[must_use]
pub fn derive_auth_key(secret: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre.consumer.receipt.auth_key.v1");
    hasher.update(secret);
    *hasher.finalize().as_bytes()
}

impl ApplicationEvidenceReceipt {
    /// Compute the authentication tag over the canonical receipt fields.
    #[must_use]
    pub fn compute_auth_tag(&self, secret: &[u8]) -> String {
        let key = derive_auth_key(secret);
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(b"vyre.consumer.receipt.v1\0");
        hasher.update(&self.receipt_version.to_le_bytes());
        hasher.update(&self.schema_digest);
        hasher.update(&self.graph_digest);
        hasher.update(&self.resource_digest);
        hasher.update(&self.artifact_digest);
        hasher.update(&self.payload_digest);
        hasher.update(&(self.target_format.len() as u64).to_le_bytes());
        hasher.update(self.target_format.as_bytes());
        hasher.update(&[u8::from(self.execution_passed)]);
        hasher.update(&self.output_digest);
        hasher.update(&self.benchmark_metrics.compile_time_ns.to_le_bytes());
        hasher.update(&self.benchmark_metrics.load_time_ns.to_le_bytes());
        hasher.update(&self.benchmark_metrics.p50_latency_ns.to_le_bytes());
        hasher.update(&self.benchmark_metrics.p99_latency_ns.to_le_bytes());
        hasher.update(
            &self
                .benchmark_metrics
                .throughput_items_per_sec
                .to_le_bytes(),
        );
        hasher.update(&self.benchmark_metrics.peak_resident_bytes.to_le_bytes());
        hasher.update(&(self.benchmark_metrics.schedule_features.len() as u64).to_le_bytes());
        for feat in &self.benchmark_metrics.schedule_features {
            hasher.update(&(feat.len() as u64).to_le_bytes());
            hasher.update(feat.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Authenticate this receipt against the shared secret.
    #[must_use]
    pub fn verify_auth_tag(&self, secret: &[u8]) -> bool {
        !self.auth_tag.is_empty() && self.auth_tag == self.compute_auth_tag(secret)
    }
}

/// Name every product-family term carried by a string value in `body`.
///
/// The scan walks the serialized document rather than a chosen pair of fields,
/// so a receipt field added later is covered by the field it becomes and not by
/// a list somebody has to remember to extend.
fn prohibited_names_in(body: &serde_json::Value) -> Vec<(String, String, &'static str)> {
    fn walk(
        pointer: &str,
        value: &serde_json::Value,
        found: &mut Vec<(String, String, &'static str)>,
    ) {
        match value {
            serde_json::Value::String(text) => {
                let lowered = text.to_ascii_lowercase();
                for term in PROHIBITED_DOMAIN_TERMS {
                    if lowered.contains(term) {
                        found.push((pointer.to_string(), text.clone(), term));
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    walk(&format!("{pointer}/{index}"), item, found);
                }
            }
            serde_json::Value::Object(fields) => {
                for (name, field) in fields {
                    let lowered = name.to_ascii_lowercase();
                    for term in PROHIBITED_DOMAIN_TERMS {
                        if lowered.contains(term) {
                            found.push((format!("{pointer}/{name}"), name.clone(), term));
                        }
                    }
                    walk(&format!("{pointer}/{name}"), field, found);
                }
            }
            _ => {}
        }
    }

    let mut found = Vec::new();
    walk("", body, &mut found);
    found
}

/// Validate an external domain-neutral evidence receipt.
///
/// # Errors
///
/// Returns [`ReceiptValidationError`] if the receipt is invalid, unverified, unauthenticated,
/// or carries a product-family name in any field name or string value.
pub fn validate_evidence_receipt(
    receipt: &ApplicationEvidenceReceipt,
    secret: &[u8],
) -> Result<(), ReceiptValidationError> {
    if receipt.receipt_version != APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION {
        return Err(ReceiptValidationError::UnsupportedVersion {
            found: receipt.receipt_version,
            expected: APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        });
    }

    for (field, digest) in [
        ("schema_digest", &receipt.schema_digest),
        ("graph_digest", &receipt.graph_digest),
        ("resource_digest", &receipt.resource_digest),
        ("artifact_digest", &receipt.artifact_digest),
        ("payload_digest", &receipt.payload_digest),
        ("output_digest", &receipt.output_digest),
    ] {
        if *digest == [0; 32] {
            return Err(ReceiptValidationError::ZeroDigest { field });
        }
    }

    if receipt.target_format.trim().is_empty() {
        return Err(ReceiptValidationError::EmptyTargetFormat);
    }

    if !receipt.execution_passed {
        return Err(ReceiptValidationError::ExecutionFailed);
    }

    let p50 = receipt.benchmark_metrics.p50_latency_ns;
    let p99 = receipt.benchmark_metrics.p99_latency_ns;
    if p50 == 0 || p99 == 0 || p50 > p99 {
        return Err(ReceiptValidationError::InvalidLatency { p50, p99 });
    }

    if receipt.benchmark_metrics.schedule_features.is_empty() {
        return Err(ReceiptValidationError::EmptyScheduleFeatures);
    }

    if !receipt.verify_auth_tag(secret) {
        return Err(ReceiptValidationError::UnauthenticatedReceipt);
    }

    let body = serde_json::to_value(receipt)
        .map_err(|error| ReceiptValidationError::Unserializable(error.to_string()))?;
    if let Some((field, value, term)) = prohibited_names_in(&body).into_iter().next() {
        return Err(ReceiptValidationError::ProhibitedDomainName { field, value, term });
    }

    Ok(())
}

/// Generated whole-application readiness evidence document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApplicationReadinessEvidence {
    /// Schema version of this evidence document.
    pub schema_version: u32,
    /// Schema version of the external consumer receipt this gate admits.
    pub receipt_schema_version: u32,
    /// Generic frontend capability dialects derived from the live registries.
    pub generic_frontend_dialects: Vec<FrontendDialectRecord>,
    /// Artifact modules and target profiles derived from the backend registry.
    pub artifact_modules: Vec<ArtifactModuleRecord>,
    /// One derived whole-application route per generic frontend dialect.
    pub applications: Vec<ApplicationRouteRecord>,
    /// Conformance certificates read from the recorded evidence corpus.
    pub device_execution_certificates: Vec<DeviceCertificateRecord>,
    /// Benchmark evidence read from the recorded release workload roster.
    pub benchmark_evidence: BenchmarkEvidenceSummary,
    /// Blocker judgements if any invariant failed.
    pub blockers: Vec<String>,
}

/// Derived generic frontend dialect capability record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendDialectRecord {
    /// Dialect identifier.
    pub dialect_id: String,
    /// Dialect name.
    pub name: String,
    /// Dialect version.
    pub version: u32,
    /// Minimum supported version.
    pub min_supported_version: u32,
    /// Semantic tier.
    pub tier: String,
    /// Coarse taxonomy category.
    pub category: String,
    /// Operations declared by this dialect.
    pub operations: Vec<FrontendOpRecord>,
}

/// Derived frontend operation record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontendOpRecord {
    /// Operation identifier.
    pub id: String,
    /// Operation name within the dialect.
    pub name: String,
    /// Dialect version that introduced the operation.
    pub version: u32,
    /// Whether the operation composes over existing IR.
    pub is_composable: bool,
    /// Whether the live operation registry carries a program builder for it.
    pub has_registered_builder: bool,
    /// Declared signature input parameter count.
    pub signature_inputs: usize,
    /// Declared signature output parameter count.
    pub signature_outputs: usize,
}

/// Derived artifact module record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactModuleRecord {
    /// Backend identifier.
    pub backend_id: String,
    /// Validated target identity owned by the backend crate.
    pub target_id: String,
    /// Registered target payload format.
    pub target_format: Option<String>,
    /// Whether the backend is a conformance oracle rather than a production target.
    pub reference_oracle: bool,
    /// Whether the backend registers a pure target compiler facet.
    pub has_target_compiler: bool,
    /// Whether the backend registers a device materializer facet.
    pub has_materializer: bool,
    /// Semantic operations the backend's target compiler claims.
    pub semantic_operation_count: usize,
    /// Artifact schema version.
    pub artifact_schema_version: u16,
    /// Target payload schema version.
    pub payload_schema_version: u16,
}

/// One resource the compiler bound for a derived application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRecord {
    /// Stable graph value name.
    pub name: String,
    /// Resolved logical element count.
    pub element_count: u64,
    /// Canonical packed byte count.
    pub byte_count: u64,
    /// Semantic lifetime class.
    pub lifetime: String,
    /// First barrier stage needing the value.
    pub first_stage: u32,
    /// Last barrier stage needing the value.
    pub last_stage: u32,
}

/// One lowered target record for a derived application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationTargetRecord {
    /// Backend that lowered the artifact.
    pub backend_id: String,
    /// Target payload format the backend emitted.
    pub target_format: String,
    /// Executable entry points in the payload.
    pub entry_point_count: usize,
    /// Lowered modules in the target module bundle.
    pub module_count: usize,
    /// Arm assignments recorded beside those modules.
    pub arm_assignment_count: usize,
    /// Execution topology the lowered bundle states.
    pub bundle_topology: String,
    /// Schedule features the lowered target record carries.
    pub target_features: Vec<String>,
    /// Whether the artifact and this payload formed one admitted envelope.
    pub envelope_admitted: bool,
}

/// One derived whole-application production route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationRouteRecord {
    /// Generic frontend dialect the application was derived from.
    pub application_id: String,
    /// Staged operation identifiers in graph order.
    pub stages: Vec<String>,
    /// Nodes in the derived whole-program graph.
    pub graph_node_count: usize,
    /// Values produced by one node and consumed by another.
    pub graph_internal_edge_count: usize,
    /// Graph closure failure, or `None` when the whole graph analyzed.
    pub graph_closure_error: Option<String>,
    /// Compile request rejection, or `None` when the request validated.
    pub compile_request_error: Option<String>,
    /// Compilation failure, or `None` when an artifact was selected.
    pub compile_error: Option<String>,
    /// Artifact schema version, when an artifact was selected.
    pub artifact_schema_version: Option<u16>,
    /// Whether recompiling the same request selected the same artifact identity.
    pub artifact_identity_stable: bool,
    /// Complete resource roster the compiler bound for the application.
    pub resource_roster: Vec<ResourceRecord>,
    /// Graph values with no record in the resource roster.
    pub unbound_graph_values: Vec<String>,
    /// Entry ABI bindings naming a value the roster does not carry.
    pub unresolved_abi_bindings: Vec<String>,
    /// Execution topology the artifact record states.
    pub selected_topology: Option<String>,
    /// Schedule features the artifact record carries.
    pub artifact_features: Vec<String>,
    /// Lowered target records, one per registered production backend.
    pub targets: Vec<ApplicationTargetRecord>,
    /// Backends whose target compiler refused the artifact, with the reason.
    pub target_errors: Vec<String>,
    /// Certificate files recording every staged operation passing on a device.
    pub device_certified_by: Vec<String>,
    /// Staged operations no device certificate records as passing.
    pub uncertified_stages: Vec<String>,
}

/// Validated conformance certificate record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCertificateRecord {
    /// Certificate filename.
    pub certificate_file: String,
    /// Backend the certificate was recorded for.
    pub backend_id: String,
    /// Certificate shape version.
    pub schema_version: u32,
    /// Whether a device took part in producing the record.
    pub device_measured: bool,
    /// Devices named by the provenance record.
    pub devices: Vec<String>,
    /// Operation-backend pairs the record covers.
    pub total_pairs: usize,
    /// Distinct operations the record covers.
    pub distinct_op_count: usize,
    /// Catalog operations the record required.
    pub catalog_required_op_count: usize,
    /// Catalog operations the record covered.
    pub catalog_covered_op_count: usize,
    /// Catalog operations absent from the record.
    pub missing_catalog_ops: Vec<String>,
    /// Blocker sentences the record carries.
    pub blockers: Vec<String>,
}

/// Benchmark evidence summary read from the release workload roster.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEvidenceSummary {
    /// Read failure, or `None` when the roster parsed.
    pub read_error: Option<String>,
    /// Closed workload families the roster requires.
    pub required_closed_families: usize,
    /// Closed workload families the roster matched.
    pub matched_required_families: usize,
    /// Cases in the release suite.
    pub release_suite_case_count: usize,
    /// Required families whose recorded evidence artifact is absent.
    pub missing_evidence_artifacts: Vec<String>,
    /// Required host-baseline families the roster did not match.
    pub missing_required_baseline_families: Vec<String>,
    /// Blocker sentences the roster carries.
    pub blockers: Vec<String>,
}

/// Errors detected in a derived whole-application production route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplicationRouteError {
    /// The derived graph carries fewer than two nodes.
    IsolatedKernel {
        /// Application identifier.
        application_id: String,
        /// Observed node count.
        node_count: usize,
    },
    /// The derived graph carries no value produced by one node and read by another.
    DisconnectedGraph {
        /// Application identifier.
        application_id: String,
    },
    /// Whole-graph closure analysis rejected the derived graph.
    GraphClosureFailed {
        /// Application identifier.
        application_id: String,
        /// Analysis rejection.
        reason: String,
    },
    /// The compile request or the compilation itself was rejected.
    NotCompiled {
        /// Application identifier.
        application_id: String,
        /// Rejection.
        reason: String,
    },
    /// The resource roster does not cover every graph value or ABI binding.
    IncompleteResourceRoster {
        /// Application identifier.
        application_id: String,
        /// Unbound value or unresolved binding.
        detail: String,
    },
    /// Recompiling the same request selected a different artifact identity.
    UnstableArtifactIdentity {
        /// Application identifier.
        application_id: String,
    },
    /// No registered production backend lowered the artifact.
    NoTargetPayload {
        /// Application identifier.
        application_id: String,
    },
    /// A registered production backend refused to lower the artifact.
    TargetLoweringRefused {
        /// Application identifier.
        application_id: String,
        /// Refusal.
        reason: String,
    },
    /// A lowered payload did not form an admitted artifact envelope.
    EnvelopeRefused {
        /// Application identifier.
        application_id: String,
        /// Backend whose payload was refused.
        backend_id: String,
    },
    /// No device certificate records the staged operations passing.
    ReferenceOnlyEvidence {
        /// Application identifier.
        application_id: String,
        /// Staged operations with no device record.
        stages: Vec<String>,
    },
}

impl ApplicationRouteError {
    /// The corrective action for this defect, in one sentence.
    #[must_use]
    pub const fn fix(&self) -> &'static str {
        match self {
            Self::IsolatedKernel { .. } => {
                "stage at least two connected frontend operations into one whole-program graph"
            }
            Self::DisconnectedGraph { .. } => {
                "connect the staged operations through an invocation-scoped value"
            }
            Self::GraphClosureFailed { .. } => {
                "bind every graph input and output to a value contract the analysis accepts"
            }
            Self::NotCompiled { .. } => {
                "compile the whole-program graph through a validated compile request"
            }
            Self::IncompleteResourceRoster { .. } => {
                "record one resource for every graph value and resolve every entry ABI binding"
            }
            Self::UnstableArtifactIdentity { .. } => {
                "make selection reproducible so one certificate names one artifact"
            }
            Self::NoTargetPayload { .. } => {
                "register a production backend whose target compiler lowers the selected artifact"
            }
            Self::TargetLoweringRefused { .. } => {
                "lower the selected artifact through the registered target compiler without rewriting the plan"
            }
            Self::EnvelopeRefused { .. } => {
                "attach the payload the same artifact selected"
            }
            Self::ReferenceOnlyEvidence { .. } => {
                "record a device conformance certificate covering those operations"
            }
        }
    }
}

impl fmt::Display for ApplicationRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IsolatedKernel {
                application_id,
                node_count,
            } => write!(
                f,
                "application `{application_id}` is a {node_count}-node graph, which is an isolated kernel"
            ),
            Self::DisconnectedGraph { application_id } => write!(
                f,
                "application `{application_id}` has no value produced by one node and read by another"
            ),
            Self::GraphClosureFailed {
                application_id,
                reason,
            } => write!(
                f,
                "application `{application_id}` graph closure failed: {reason}"
            ),
            Self::NotCompiled {
                application_id,
                reason,
            } => write!(
                f,
                "application `{application_id}` produced no artifact: {reason}"
            ),
            Self::IncompleteResourceRoster {
                application_id,
                detail,
            } => write!(
                f,
                "application `{application_id}` resource roster is incomplete: {detail}"
            ),
            Self::UnstableArtifactIdentity { application_id } => write!(
                f,
                "application `{application_id}` selected a different artifact identity on recompilation"
            ),
            Self::NoTargetPayload { application_id } => write!(
                f,
                "application `{application_id}` reached no target payload"
            ),
            Self::TargetLoweringRefused {
                application_id,
                reason,
            } => write!(
                f,
                "application `{application_id}` target lowering was refused: {reason}"
            ),
            Self::EnvelopeRefused {
                application_id,
                backend_id,
            } => write!(
                f,
                "application `{application_id}` payload from `{backend_id}` was refused as an artifact envelope"
            ),
            Self::ReferenceOnlyEvidence {
                application_id,
                stages,
            } => write!(
                f,
                "application `{application_id}` has no device record for {stages:?}, and host-only reference evidence certifies the oracle rather than a device"
            ),
        }
    }
}

impl std::error::Error for ApplicationRouteError {}

/// Errors detected in schedule feature agreement between artifact and target records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleFeatureError {
    /// Schedule feature claimed in artifact record is absent from target record.
    FeatureAbsentFromTarget {
        /// Feature name.
        feature: String,
        /// Target name.
        target: String,
    },
    /// Feature present in target record but absent from artifact record.
    FeatureAbsentFromArtifact {
        /// Feature name.
        feature: String,
        /// Artifact name.
        artifact: String,
    },
}

impl ScheduleFeatureError {
    /// The corrective action for this disagreement, in one sentence.
    #[must_use]
    pub const fn fix(&self) -> &'static str {
        match self {
            Self::FeatureAbsentFromTarget { .. } => {
                "carry the selected schedule feature into the lowered target record"
            }
            Self::FeatureAbsentFromArtifact { .. } => {
                "record the lowered schedule feature in the artifact record"
            }
        }
    }
}

impl fmt::Display for ScheduleFeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureAbsentFromTarget { feature, target } => write!(
                f,
                "schedule feature `{feature}` is claimed in the artifact record and absent from target record `{target}`"
            ),
            Self::FeatureAbsentFromArtifact { feature, artifact } => write!(
                f,
                "schedule feature `{feature}` is carried by the target record and absent from artifact record `{artifact}`"
            ),
        }
    }
}

impl std::error::Error for ScheduleFeatureError {}

/// Validate a derived whole-application production route.
///
/// # Errors
///
/// Returns [`ApplicationRouteError`] when the route is an isolated kernel, is
/// disconnected, is not closed, stops short of an artifact or a target payload,
/// binds an incomplete resource roster, selects an unstable identity, or rests
/// on host-only evidence.
pub fn validate_production_route(
    record: &ApplicationRouteRecord,
) -> Result<(), ApplicationRouteError> {
    let application_id = record.application_id.clone();
    if record.graph_node_count < 2 {
        return Err(ApplicationRouteError::IsolatedKernel {
            application_id,
            node_count: record.graph_node_count,
        });
    }
    if record.graph_internal_edge_count == 0 {
        return Err(ApplicationRouteError::DisconnectedGraph { application_id });
    }
    if let Some(reason) = &record.graph_closure_error {
        return Err(ApplicationRouteError::GraphClosureFailed {
            application_id,
            reason: reason.clone(),
        });
    }
    for reason in [&record.compile_request_error, &record.compile_error] {
        if let Some(reason) = reason {
            return Err(ApplicationRouteError::NotCompiled {
                application_id,
                reason: reason.clone(),
            });
        }
    }
    if record.artifact_schema_version.is_none() {
        return Err(ApplicationRouteError::NotCompiled {
            application_id,
            reason: "no artifact schema version was recorded".to_string(),
        });
    }
    if let Some(detail) = record
        .unbound_graph_values
        .first()
        .or_else(|| record.unresolved_abi_bindings.first())
    {
        return Err(ApplicationRouteError::IncompleteResourceRoster {
            application_id,
            detail: detail.clone(),
        });
    }
    if record.resource_roster.is_empty() {
        return Err(ApplicationRouteError::IncompleteResourceRoster {
            application_id,
            detail: "the roster is empty".to_string(),
        });
    }
    if !record.artifact_identity_stable {
        return Err(ApplicationRouteError::UnstableArtifactIdentity { application_id });
    }
    if let Some(reason) = record.target_errors.first() {
        return Err(ApplicationRouteError::TargetLoweringRefused {
            application_id,
            reason: reason.clone(),
        });
    }
    if record.targets.is_empty() {
        return Err(ApplicationRouteError::NoTargetPayload { application_id });
    }
    if let Some(target) = record
        .targets
        .iter()
        .find(|target| !target.envelope_admitted)
    {
        return Err(ApplicationRouteError::EnvelopeRefused {
            application_id,
            backend_id: target.backend_id.clone(),
        });
    }
    if !record.uncertified_stages.is_empty() {
        return Err(ApplicationRouteError::ReferenceOnlyEvidence {
            application_id,
            stages: record.uncertified_stages.clone(),
        });
    }
    if record.device_certified_by.is_empty() {
        return Err(ApplicationRouteError::ReferenceOnlyEvidence {
            application_id,
            stages: record.stages.clone(),
        });
    }
    Ok(())
}

/// Validate that every claimed schedule feature exists in both the artifact record and the target record.
///
/// # Errors
///
/// Returns [`ScheduleFeatureError`] if a feature is claimed in the artifact and absent from the target,
/// or vice versa.
pub fn validate_schedule_feature_coverage(
    artifact_name: &str,
    target_name: &str,
    artifact_features: &[String],
    target_features: &[String],
) -> Result<(), ScheduleFeatureError> {
    for feat in artifact_features {
        if !target_features.contains(feat) {
            return Err(ScheduleFeatureError::FeatureAbsentFromTarget {
                feature: feat.clone(),
                target: target_name.to_string(),
            });
        }
    }
    for feat in target_features {
        if !artifact_features.contains(feat) {
            return Err(ScheduleFeatureError::FeatureAbsentFromArtifact {
                feature: feat.clone(),
                artifact: artifact_name.to_string(),
            });
        }
    }
    Ok(())
}

/// The tag one execution topology is compared under.
///
/// The match is exhaustive with no catch-all, so a new topology variant stops
/// this gate compiling until somebody states what the tag is and what the
/// lowered record has to carry for it.
#[must_use]
pub fn topology_tag(topology: ExecutionTopology) -> String {
    match topology {
        ExecutionTopology::Sequential => "topology:sequential".to_string(),
        ExecutionTopology::ConcurrentQueue { queues } => {
            format!("topology:concurrent_queue:{queues}")
        }
        ExecutionTopology::ResidentPartition { partitions, mode } => {
            let mode = match mode {
                ResidentPartitionMode::FixedSpatialMask => "fixed_spatial_mask",
                ResidentPartitionMode::BoundedWorkQueue => "bounded_work_queue",
            };
            format!("topology:resident_partition:{partitions}:{mode}")
        }
    }
}

/// Derive generic frontend capabilities from the registered dialects and the live operation registry.
#[must_use]
pub fn derive_frontend_capabilities() -> Vec<FrontendDialectRecord> {
    let registry = vyre_registry_link::operation::live_operation_registry();
    let mut builders: BTreeMap<&'static str, bool> = BTreeMap::new();
    for entry in registry.iter() {
        let lowered = builders.entry(entry.id).or_insert(false);
        *lowered |= entry.build.is_some();
    }

    let mut dialect_records = Vec::new();
    for dialect in DialectRegistry::global().values() {
        let mut operations = Vec::new();
        for op in dialect.operations {
            operations.push(FrontendOpRecord {
                id: op.id.to_string(),
                name: op.name.to_string(),
                version: op.version,
                is_composable: op.is_composable,
                has_registered_builder: builders.get(op.id).copied().unwrap_or(false),
                signature_inputs: op.signature.inputs.len(),
                signature_outputs: op.signature.outputs.len(),
            });
        }
        operations.sort_by(|a, b| a.id.cmp(&b.id));

        dialect_records.push(FrontendDialectRecord {
            dialect_id: dialect.id.to_string(),
            name: dialect.name.to_string(),
            version: dialect.version,
            min_supported_version: dialect.min_supported_version,
            tier: format!("{:?}", dialect.tier),
            category: dialect.category.to_string(),
            operations,
        });
    }
    dialect_records.sort_by(|a, b| a.dialect_id.cmp(&b.dialect_id));
    dialect_records
}

/// Derive artifact module records from the frozen backend registry.
///
/// # Errors
///
/// Returns the registry startup error. A fabricated backend list would let a
/// registry that failed to start read as a tree with three working backends.
pub fn derive_artifact_modules() -> Result<Vec<ArtifactModuleRecord>, String> {
    let registrations =
        vyre_registry_link::backend::live_backend_registry().map_err(|error| error.to_string())?;
    let mut modules: Vec<_> = registrations
        .iter()
        .map(|registration| ArtifactModuleRecord {
            backend_id: registration.id.to_string(),
            target_id: registration.target_id.as_str().to_string(),
            target_format: registration.payload_format.map(str::to_string),
            reference_oracle: registration.reference_oracle,
            has_target_compiler: registration.target_compiler.is_some(),
            has_materializer: registration.materializer.is_some(),
            semantic_operation_count: (registration.semantic_operations)().len(),
            artifact_schema_version: vyre_megakernel::ARTIFACT_SCHEMA_VERSION,
            payload_schema_version: vyre_megakernel::TARGET_PAYLOAD_SCHEMA_VERSION,
        })
        .collect();
    modules.sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
    Ok(modules)
}

/// Recorded conformance certificate, as read from the evidence corpus.
#[derive(serde::Deserialize)]
struct CertificateJson {
    schema_version: u32,
    backend_id: String,
    #[serde(default)]
    total_pairs: usize,
    #[serde(default)]
    distinct_op_count: usize,
    #[serde(default)]
    catalog_required_op_count: usize,
    #[serde(default)]
    catalog_covered_op_count: usize,
    #[serde(default)]
    missing_catalog_ops: Vec<String>,
    #[serde(default)]
    blockers: Vec<String>,
    #[serde(default)]
    pairs: Vec<CertificatePairJson>,
    provenance: Option<CertificateProvenanceJson>,
}

#[derive(serde::Deserialize)]
struct CertificatePairJson {
    op_id: String,
    #[serde(default)]
    passed: bool,
}

#[derive(serde::Deserialize)]
struct CertificateProvenanceJson {
    measurement: Option<CertificateMeasurementJson>,
}

#[derive(serde::Deserialize)]
struct CertificateMeasurementJson {
    #[serde(default)]
    state: String,
    #[serde(default)]
    devices: Vec<CertificateDeviceJson>,
}

#[derive(serde::Deserialize)]
struct CertificateDeviceJson {
    #[serde(default)]
    name: String,
}

/// One conformance certificate and the operations it records passing.
pub struct ReadCertificate {
    /// The record written into the readiness evidence.
    pub record: DeviceCertificateRecord,
    /// Operations the record states passed.
    pub passed_ops: BTreeSet<String>,
    /// Operations the record states failed.
    pub failed_ops: BTreeSet<String>,
    /// Read failure, or `None` when the certificate parsed.
    pub read_error: Option<String>,
}

/// Read every conformance certificate in the recorded evidence corpus.
///
/// The roster comes from the directory, so a certificate added or removed is
/// seen rather than matched against a list compiled in here.
#[must_use]
pub fn read_certificate_corpus(root: &Path) -> Vec<ReadCertificate> {
    let directory = root.join(CERTIFICATE_DIR);
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&directory) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with("-conformance.json") {
                names.push(name);
            }
        }
    }
    names.sort();

    let mut certificates = Vec::new();
    for name in names {
        let path = directory.join(&name);
        let read = std::fs::read_to_string(&path)
            .map_err(|error| error.to_string())
            .and_then(|text| {
                serde_json::from_str::<CertificateJson>(&text).map_err(|error| error.to_string())
            });
        match read {
            Ok(value) => {
                let measurement = value
                    .provenance
                    .as_ref()
                    .and_then(|provenance| provenance.measurement.as_ref());
                let passed_ops = value
                    .pairs
                    .iter()
                    .filter(|pair| pair.passed)
                    .map(|pair| pair.op_id.clone())
                    .collect();
                let failed_ops = value
                    .pairs
                    .iter()
                    .filter(|pair| !pair.passed)
                    .map(|pair| pair.op_id.clone())
                    .collect();
                certificates.push(ReadCertificate {
                    record: DeviceCertificateRecord {
                        certificate_file: name,
                        backend_id: value.backend_id,
                        schema_version: value.schema_version,
                        device_measured: measurement
                            .is_some_and(|measurement| measurement.state == "device"),
                        devices: measurement
                            .map(|measurement| {
                                measurement
                                    .devices
                                    .iter()
                                    .map(|device| device.name.clone())
                                    .collect()
                            })
                            .unwrap_or_default(),
                        total_pairs: value.total_pairs,
                        distinct_op_count: value.distinct_op_count,
                        catalog_required_op_count: value.catalog_required_op_count,
                        catalog_covered_op_count: value.catalog_covered_op_count,
                        missing_catalog_ops: value.missing_catalog_ops,
                        blockers: value.blockers,
                    },
                    passed_ops,
                    failed_ops,
                    read_error: None,
                });
            }
            Err(error) => certificates.push(ReadCertificate {
                record: DeviceCertificateRecord {
                    certificate_file: name,
                    backend_id: String::new(),
                    schema_version: 0,
                    device_measured: false,
                    devices: Vec::new(),
                    total_pairs: 0,
                    distinct_op_count: 0,
                    catalog_required_op_count: 0,
                    catalog_covered_op_count: 0,
                    missing_catalog_ops: Vec::new(),
                    blockers: Vec::new(),
                },
                passed_ops: BTreeSet::new(),
                failed_ops: BTreeSet::new(),
                read_error: Some(error),
            }),
        }
    }
    certificates
}

#[derive(serde::Deserialize)]
struct WorkloadMatrixJson {
    #[serde(default)]
    required_closed_families: usize,
    #[serde(default)]
    matched_required_families: usize,
    #[serde(default)]
    release_suite_case_count: usize,
    #[serde(default)]
    missing_required_cpu_sota_100x_families: Vec<String>,
    #[serde(default)]
    families: Vec<WorkloadFamilyJson>,
    #[serde(default)]
    blockers: Vec<String>,
}

#[derive(serde::Deserialize)]
struct WorkloadFamilyJson {
    id: String,
    #[serde(default)]
    required: bool,
    evidence_artifact: Option<String>,
}

/// Read the recorded release workload roster and the artifacts it names.
#[must_use]
pub fn derive_benchmark_evidence(root: &Path) -> BenchmarkEvidenceSummary {
    let path = root.join(WORKLOAD_MATRIX_PATH);
    let read = std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|text| {
            serde_json::from_str::<WorkloadMatrixJson>(&text).map_err(|error| error.to_string())
        });
    let value = match read {
        Ok(value) => value,
        Err(error) => {
            return BenchmarkEvidenceSummary {
                read_error: Some(error),
                required_closed_families: 0,
                matched_required_families: 0,
                release_suite_case_count: 0,
                missing_evidence_artifacts: Vec::new(),
                missing_required_baseline_families: Vec::new(),
                blockers: Vec::new(),
            }
        }
    };

    let mut missing_evidence_artifacts = Vec::new();
    for family in &value.families {
        if !family.required {
            continue;
        }
        match &family.evidence_artifact {
            None => missing_evidence_artifacts.push(family.id.clone()),
            Some(artifact) => {
                if !root.join(artifact).exists() {
                    missing_evidence_artifacts.push(format!("{}: {artifact}", family.id));
                }
            }
        }
    }
    missing_evidence_artifacts.sort();

    BenchmarkEvidenceSummary {
        read_error: None,
        required_closed_families: value.required_closed_families,
        matched_required_families: value.matched_required_families,
        release_suite_case_count: value.release_suite_case_count,
        missing_evidence_artifacts,
        missing_required_baseline_families: value.missing_required_cpu_sota_100x_families,
        blockers: value.blockers,
    }
}

/// One derived whole-program application graph and the operations it stages.
struct DerivedApplication {
    application_id: String,
    stages: Vec<String>,
    graph: ProgramGraph,
    build_error: Option<String>,
}

/// Stage every operation of one dialect into one connected whole-program graph.
///
/// Each stage runs the Program the live registry builds for its operation, with
/// its buffers renamed per stage so a fused rename has one name per value. The
/// first stage reads external values and every later stage consumes the value
/// the stage before it produced, which is what makes the graph connected rather
/// than a set of isolated kernels.
fn derive_application(dialect_id: &str, operations: &[&str]) -> DerivedApplication {
    let registry = vyre_registry_link::operation::live_operation_registry();
    let mut graph = ProgramGraph::new();
    let mut carried: Option<GraphValueId> = None;
    let mut build_error = None;

    for (index, op_id) in operations.iter().enumerate() {
        let Some(entry) = registry.iter().find(|entry| entry.id == *op_id) else {
            build_error = Some(format!("`{op_id}` is absent from the operation registry"));
            break;
        };
        let Some(build) = entry.build else {
            build_error = Some(format!("`{op_id}` registers no program builder"));
            break;
        };
        let built: Program = build();
        let mut program = built.clone();
        for buffer in built.buffers() {
            match rename_buffer(
                &program,
                buffer.name(),
                &format!("s{index}_{}", buffer.name()),
            ) {
                Ok(renamed) => program = renamed,
                Err(error) => {
                    build_error = Some(format!(
                        "`{op_id}` buffer `{}` could not be staged: {error}",
                        buffer.name()
                    ));
                    break;
                }
            }
        }
        if build_error.is_some() {
            break;
        }

        let last = index + 1 == operations.len();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut upstream = carried.take();
        for buffer in program.buffers() {
            let count = u64::from(buffer.count());
            if buffer.is_output() || buffer.is_pipeline_live_out() {
                outputs.push(GraphOutput {
                    buffer: buffer.name().to_string(),
                    name: buffer.name().to_string(),
                    contract: ValueContract::dense_1d(
                        buffer.element(),
                        count,
                        buffer.access(),
                        if last {
                            ValueLifetime::Output
                        } else {
                            ValueLifetime::Invocation
                        },
                    ),
                    retained_successor_of: None,
                });
                continue;
            }
            let contract = ValueContract::dense_1d(
                buffer.element(),
                count,
                buffer.access(),
                ValueLifetime::Invocation,
            );
            let value = match upstream.take() {
                Some(value) => value,
                None => match graph.add_external_value(buffer.name(), contract.clone()) {
                    Ok(value) => value,
                    Err(error) => {
                        build_error = Some(format!(
                            "`{op_id}` input `{}` could not be bound: {error}",
                            buffer.name()
                        ));
                        break;
                    }
                },
            };
            inputs.push(GraphInput {
                buffer: buffer.name().to_string(),
                value,
                contract,
            });
        }
        if build_error.is_some() {
            break;
        }

        match graph.add_node(format!("s{index}"), program, inputs, outputs) {
            Ok((_, produced)) => carried = produced.first().copied(),
            Err(error) => {
                build_error = Some(format!("`{op_id}` could not be staged: {error}"));
                break;
            }
        }
    }

    DerivedApplication {
        application_id: dialect_id.to_string(),
        stages: operations.iter().map(|id| (*id).to_string()).collect(),
        graph,
        build_error,
    }
}

/// The compile request every derived application is compiled through.
fn application_request(graph: ProgramGraph) -> CompileRequest {
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0x11; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(64, 100_000, 8, 0, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 8_000_000),
    )
}

/// The schedule features one artifact record states.
fn artifact_features(artifact: &Artifact) -> Vec<String> {
    let plan = artifact.selected_plan();
    let mut features = BTreeSet::new();
    features.insert(topology_tag(plan.topology));
    features.insert(format!("groups:{}", artifact.fusion().len()));
    if artifact
        .fusion()
        .iter()
        .any(|group| group.members.len() > 1)
    {
        features.insert("fused".to_string());
    }
    for phase in &plan.schedule.phases {
        if phase.vector_width > 1 {
            features.insert(format!("chunked:{}", phase.vector_width));
        }
    }
    features.into_iter().collect()
}

/// The schedule features one lowered target record carries.
fn target_features(bundle: &TargetModuleBundle) -> Vec<String> {
    let mut features = BTreeSet::new();
    features.insert(topology_tag(bundle.topology));
    features.insert(format!("groups:{}", bundle.modules.len()));
    if bundle.modules.iter().any(|module| module.nodes.len() > 1) {
        features.insert("fused".to_string());
    }
    for module in &bundle.modules {
        if let Some(chunk) = module.numeric.chunk {
            features.insert(format!("chunked:{chunk}"));
        }
    }
    features.into_iter().collect()
}

/// Compile and lower one derived application into its route record.
fn route_of(
    application: &DerivedApplication,
    certificates: &[ReadCertificate],
) -> ApplicationRouteRecord {
    let graph = &application.graph;
    let internal_edges = graph
        .values()
        .iter()
        .filter(|value| value.producer.is_some() && !value.consumers.is_empty())
        .count();
    let mut record = ApplicationRouteRecord {
        application_id: application.application_id.clone(),
        stages: application.stages.clone(),
        graph_node_count: graph.nodes().len(),
        graph_internal_edge_count: internal_edges,
        graph_closure_error: application.build_error.clone(),
        compile_request_error: None,
        compile_error: None,
        artifact_schema_version: None,
        artifact_identity_stable: false,
        resource_roster: Vec::new(),
        unbound_graph_values: Vec::new(),
        unresolved_abi_bindings: Vec::new(),
        selected_topology: None,
        artifact_features: Vec::new(),
        targets: Vec::new(),
        target_errors: Vec::new(),
        device_certified_by: Vec::new(),
        uncertified_stages: Vec::new(),
    };

    for certificate in certificates {
        if !certificate.record.device_measured {
            continue;
        }
        if application
            .stages
            .iter()
            .all(|stage| certificate.passed_ops.contains(stage))
        {
            record
                .device_certified_by
                .push(certificate.record.certificate_file.clone());
        }
    }
    if record.device_certified_by.is_empty() {
        let certified: BTreeSet<&String> = certificates
            .iter()
            .filter(|certificate| certificate.record.device_measured)
            .flat_map(|certificate| certificate.passed_ops.iter())
            .collect();
        record.uncertified_stages = application
            .stages
            .iter()
            .filter(|stage| !certified.contains(stage))
            .cloned()
            .collect();
    }

    if record.graph_closure_error.is_some() {
        return record;
    }
    if let Err(error) = graph.analyze() {
        record.graph_closure_error = Some(error.to_string());
        return record;
    }

    let request = match application_request(graph.clone()).validate() {
        Ok(request) => request,
        Err(error) => {
            record.compile_request_error = Some(error.to_string());
            return record;
        }
    };
    let artifact = match compile(&request) {
        Ok(artifact) => artifact,
        Err(error) => {
            record.compile_error = Some(error.to_string());
            return record;
        }
    };

    record.artifact_schema_version = Some(artifact.schema_version());
    record.selected_topology = Some(topology_tag(artifact.selected_plan().topology));
    record.artifact_features = artifact_features(&artifact);
    record.artifact_identity_stable = match application_request(graph.clone()).validate() {
        Ok(replay) => compile(&replay).is_ok_and(|replay| replay.digest() == artifact.digest()),
        Err(_) => false,
    };

    let roster: BTreeMap<_, _> = artifact
        .resources()
        .iter()
        .map(|resource| (resource.value, resource))
        .collect();
    record.resource_roster = artifact
        .resources()
        .iter()
        .map(|resource| ResourceRecord {
            name: resource.name.clone(),
            element_count: resource.element_count,
            byte_count: resource.byte_count,
            lifetime: format!("{:?}", resource.lifetime),
            first_stage: resource.first_stage,
            last_stage: resource.last_stage,
        })
        .collect();
    let bound: BTreeSet<&str> = artifact
        .resources()
        .iter()
        .map(|resource| resource.name.as_str())
        .collect();
    record.unbound_graph_values = graph
        .values()
        .iter()
        .filter(|value| !bound.contains(value.name.as_str()))
        .map(|value| value.name.clone())
        .collect();
    for entry in &artifact.abi().entries {
        for binding in entry.input_bindings.iter().chain(&entry.output_bindings) {
            if !roster.contains_key(&binding.value) {
                record.unresolved_abi_bindings.push(format!(
                    "node {} buffer `{}` names value {} with no resource record",
                    entry.node.0, binding.buffer, binding.value.0
                ));
            }
        }
    }

    let registrations = match vyre_registry_link::backend::live_backend_registry() {
        Ok(registrations) => registrations,
        Err(error) => {
            record.target_errors.push(error.to_string());
            return record;
        }
    };
    for registration in registrations {
        if registration.reference_oracle {
            continue;
        }
        let compiler = match registration.target_compiler() {
            Ok(compiler) => compiler,
            Err(error) => {
                record
                    .target_errors
                    .push(format!("`{}`: {error}", registration.id));
                continue;
            }
        };
        let payload = match compiler.compile(&artifact) {
            Ok(payload) => payload,
            Err(error) => {
                record
                    .target_errors
                    .push(format!("`{}`: {error}", registration.id));
                continue;
            }
        };
        let bundle = match TargetModuleBundle::from_bytes(payload.bytes()) {
            Ok(bundle) => bundle,
            Err(error) => {
                record
                    .target_errors
                    .push(format!("`{}`: {error}", registration.id));
                continue;
            }
        };
        record.targets.push(ApplicationTargetRecord {
            backend_id: registration.id.to_string(),
            target_format: payload.format().identity().to_string(),
            entry_point_count: payload.entries().len(),
            module_count: bundle.modules.len(),
            arm_assignment_count: bundle.arms.len(),
            bundle_topology: topology_tag(bundle.topology),
            target_features: target_features(&bundle),
            envelope_admitted: attach_target(artifact.clone(), compiler.as_ref()).is_ok(),
        });
    }
    record
        .targets
        .sort_by(|a, b| a.backend_id.cmp(&b.backend_id));
    record.target_errors.sort();

    record
}

/// Entry point for the `application-runnable` gate.
pub struct ApplicationRunnable;

impl GateBehavior for ApplicationRunnable {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut inspection = Inspection::new();

        let frontend_dialects = derive_frontend_capabilities();
        if frontend_dialects.is_empty() {
            inspection.find(Finding::new(
                "no generic frontend dialect is registered, so no application can be derived",
                "register a declarative dialect through DialectRegistry",
            ));
        }
        for dialect in &frontend_dialects {
            if dialect.operations.is_empty() {
                inspection.find(Finding::new(
                    format!(
                        "frontend dialect `{}` declares no operation",
                        dialect.dialect_id
                    ),
                    "declare the dialect's operations or retire the dialect",
                ));
            }
            for op in &dialect.operations {
                if !op.has_registered_builder {
                    inspection.find(Finding::new(
                        format!(
                            "frontend operation `{}` registers no program builder, so it has no executable route",
                            op.id
                        ),
                        "submit a LoweringProvider that builds the operation's neutral Program",
                    ));
                }
                if op.signature_outputs == 0 {
                    inspection.find(Finding::new(
                        format!(
                            "frontend operation `{}` declares no signature output, so nothing it computes can be bound",
                            op.id
                        ),
                        "declare the operation's typed output parameters in its signature",
                    ));
                }
            }
        }

        let artifact_modules = match derive_artifact_modules() {
            Ok(modules) => modules,
            Err(error) => {
                inspection.find(Finding::new(
                    format!("the backend registry did not start: {error}"),
                    "resolve the conflicting backend registration",
                ));
                Vec::new()
            }
        };
        if artifact_modules.is_empty() {
            inspection.find(Finding::new(
                "no backend registers an artifact module, so no artifact can be lowered",
                "link a driver crate that submits a BackendRegistration",
            ));
        }
        for module in &artifact_modules {
            if module.reference_oracle {
                continue;
            }
            if module.target_format.is_none() {
                inspection.find(Finding::new(
                    format!(
                        "production backend `{}` registers no target payload format",
                        module.backend_id
                    ),
                    "declare the owner-local payload format in BackendRegistration",
                ));
            }
            if !module.has_target_compiler {
                inspection.find(Finding::new(
                    format!(
                        "production backend `{}` registers no target compiler, so it emits no artifact module",
                        module.backend_id
                    ),
                    "register the backend's target compiler factory",
                ));
            }
            if !module.has_materializer {
                inspection.find(Finding::new(
                    format!(
                        "production backend `{}` registers no materializer, so no device can execute its payload",
                        module.backend_id
                    ),
                    "register the backend's artifact materializer factory",
                ));
            }
        }

        let certificates = read_certificate_corpus(&ctx.root);
        if certificates.is_empty() {
            inspection.find(Finding::new(
                format!("no conformance certificate is recorded under `{CERTIFICATE_DIR}`"),
                "run the conformance suite and commit the recorded certificates",
            ));
        }
        let registered_production: BTreeSet<&str> = artifact_modules
            .iter()
            .filter(|module| !module.reference_oracle)
            .map(|module| module.backend_id.as_str())
            .collect();
        for certificate in &certificates {
            let file = format!("{CERTIFICATE_DIR}/{}", certificate.record.certificate_file);
            if let Some(error) = &certificate.read_error {
                inspection.blocked(
                    &file,
                    format!("conformance certificate could not be read: {error}"),
                    "regenerate the certificate through the conformance gate",
                );
                continue;
            }
            let record = &certificate.record;
            if record.schema_version < MIN_CERTIFICATE_SCHEMA_VERSION {
                inspection.blocked(
                    &file,
                    format!(
                        "certificate records shape {} below the supported floor {MIN_CERTIFICATE_SCHEMA_VERSION}",
                        record.schema_version
                    ),
                    "regenerate the certificate under the current shape",
                );
            }
            if record.total_pairs == 0 || record.distinct_op_count == 0 {
                inspection.blocked(
                    &file,
                    "certificate records no operation-backend pair".to_string(),
                    "regenerate the certificate from a run that dispatched operations",
                );
            }
            if !record.missing_catalog_ops.is_empty() {
                inspection.blocked(
                    &file,
                    format!(
                        "certificate omits {} catalog operation(s), starting with `{}`",
                        record.missing_catalog_ops.len(),
                        record.missing_catalog_ops[0]
                    ),
                    "dispatch the whole registered catalog before recording the certificate",
                );
            }
            if record.catalog_covered_op_count != record.catalog_required_op_count {
                inspection.blocked(
                    &file,
                    format!(
                        "certificate covers {} of {} required catalog operations",
                        record.catalog_covered_op_count, record.catalog_required_op_count
                    ),
                    "dispatch every required catalog operation before recording the certificate",
                );
            }
            for blocker in &record.blockers {
                // A certificate's own blockers are the conformance gate's
                // subject, and one about an operation no derived application
                // stages does not make a route unrunnable. The route-level
                // judgement below reads the pairs instead.
                inspection.notes.push(format!(
                    "{file} carries a conformance blocker: {}",
                    blocker.lines().next().unwrap_or(blocker)
                ));
            }
            if record.device_measured {
                if record.devices.is_empty() {
                    inspection.blocked(
                        &file,
                        "certificate states a device measurement and names no device".to_string(),
                        "record the devices the run dispatched on",
                    );
                }
                if !registered_production.contains(record.backend_id.as_str()) {
                    inspection.blocked(
                        &file,
                        format!(
                            "certificate names backend `{}`, which this build registers no production backend for",
                            record.backend_id
                        ),
                        "link the driver crate that registers that backend, or retire the certificate",
                    );
                }
            }
        }
        if !certificates.is_empty()
            && !certificates
                .iter()
                .any(|certificate| certificate.record.device_measured)
        {
            inspection.find(Finding::new(
                "every recorded conformance certificate is host-only, so nothing certifies device execution",
                "record a conformance certificate from a run that dispatched on a device",
            ));
        }

        let benchmark_evidence = derive_benchmark_evidence(&ctx.root);
        if let Some(error) = &benchmark_evidence.read_error {
            inspection.blocked(
                WORKLOAD_MATRIX_PATH,
                format!("release workload roster could not be read: {error}"),
                "regenerate the roster through `release-workload-matrix --write`",
            );
        } else {
            if benchmark_evidence.required_closed_families == 0 {
                inspection.blocked(
                    WORKLOAD_MATRIX_PATH,
                    "release workload roster requires no closed family".to_string(),
                    "declare the workload families the release must close",
                );
            }
            if benchmark_evidence.matched_required_families
                != benchmark_evidence.required_closed_families
            {
                inspection.blocked(
                    WORKLOAD_MATRIX_PATH,
                    format!(
                        "release workload roster closes {} of {} required families",
                        benchmark_evidence.matched_required_families,
                        benchmark_evidence.required_closed_families
                    ),
                    "measure the unmatched families and regenerate the roster",
                );
            }
            for family in &benchmark_evidence.missing_evidence_artifacts {
                inspection.blocked(
                    WORKLOAD_MATRIX_PATH,
                    format!(
                        "required workload family names no recorded evidence artifact: {family}"
                    ),
                    "record the family's benchmark evidence artifact",
                );
            }
            for family in &benchmark_evidence.missing_required_baseline_families {
                inspection.blocked(
                    WORKLOAD_MATRIX_PATH,
                    format!("required host-baseline family `{family}` is unmatched"),
                    "measure the family against its host baseline",
                );
            }
            for blocker in &benchmark_evidence.blockers {
                inspection.blocked(
                    WORKLOAD_MATRIX_PATH,
                    format!("release workload roster carries a blocker: {blocker}"),
                    "resolve the blocker and regenerate the roster",
                );
            }
        }

        let mut applications = Vec::new();
        for dialect in &frontend_dialects {
            let operations: Vec<&str> =
                dialect.operations.iter().map(|op| op.id.as_str()).collect();
            let derived = derive_application(&dialect.dialect_id, &operations);
            applications.push(route_of(&derived, &certificates));
        }
        for record in &applications {
            for certificate in &certificates {
                if !certificate.record.device_measured {
                    continue;
                }
                for stage in &record.stages {
                    if certificate.failed_ops.contains(stage) {
                        inspection.blocked(
                            &format!(
                                "{CERTIFICATE_DIR}/{}",
                                certificate.record.certificate_file
                            ),
                            format!(
                                "the device record states `{stage}` failed, and application `{}` stages it",
                                record.application_id
                            ),
                            "correct the operation on that backend and record the certificate again",
                        );
                    }
                }
            }
            if let Err(error) = validate_production_route(record) {
                inspection.find(Finding::new(error.to_string(), error.fix()));
            }
            for target in &record.targets {
                if let Err(error) = validate_schedule_feature_coverage(
                    &record.application_id,
                    &target.backend_id,
                    &record.artifact_features,
                    &target.target_features,
                ) {
                    inspection.find(Finding::new(error.to_string(), error.fix()));
                }
                if target.arm_assignment_count != target.module_count {
                    inspection.find(Finding::new(
                        format!(
                            "application `{}` lowered {} module(s) through `{}` with {} arm assignment(s)",
                            record.application_id,
                            target.module_count,
                            target.backend_id,
                            target.arm_assignment_count
                        ),
                        "assign one executable arm to every lowered module",
                    ));
                }
            }
        }

        let blockers = inspection
            .findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect();

        let evidence = ApplicationReadinessEvidence {
            schema_version: READINESS_SCHEMA_VERSION,
            receipt_schema_version: APPLICATION_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            generic_frontend_dialects: frontend_dialects,
            artifact_modules,
            applications,
            device_execution_certificates: certificates
                .into_iter()
                .map(|certificate| certificate.record)
                .collect(),
            benchmark_evidence,
            blockers,
        };
        let application_count = evidence.applications.len();
        let target_count: usize = evidence
            .applications
            .iter()
            .map(|record| record.targets.len())
            .sum();

        inspection.notes.push(format!(
            "{application_count} derived application(s) lowered through {target_count} registered production target(s); {} certificate(s) read",
            evidence.device_execution_certificates.len()
        ));
        inspection.generates_host_evidence(READINESS_EVIDENCE_PATH, &evidence);

        let mut report = xtask::artifact_gate::settle_inspection(ctx, ctx.gate_name()?, inspection);
        report.cover_complete("application-runnable contract paths", 9);
        Ok(report)
    }
}

#[cfg(test)]
#[path = "application_runnable_tests.rs"]
mod tests;
