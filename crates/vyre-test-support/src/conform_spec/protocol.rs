//! Canonical case, worker protocol, receipt, and replay protocol schemas.
//!
//! This module defines the stable data structures and verification invariants
//! used by the conformance coordinator, disposable workers, and certificate
//! issuance pipeline.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Explicit resource budget for one conformance worker process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerBudget {
    /// Wall-clock timeout in milliseconds.
    pub wall_timeout_ms: u64,
    /// CPU execution timeout in milliseconds.
    pub cpu_timeout_ms: u64,
    /// Host memory ceiling in bytes.
    pub max_memory_bytes: u64,
    /// GPU memory ceiling in bytes.
    pub max_gpu_memory_bytes: u64,
    /// Maximum allowed artifact size in bytes.
    pub max_artifact_bytes: u64,
    /// Maximum allowed total output size in bytes.
    pub max_output_bytes: u64,
}

impl Default for WorkerBudget {
    fn default() -> Self {
        Self::default_budget()
    }
}

impl WorkerBudget {
    /// Standard production conformance budget.
    #[must_use]
    pub const fn default_budget() -> Self {
        Self {
            wall_timeout_ms: 120_000,
            cpu_timeout_ms: 120_000,
            max_memory_bytes: 1024 * 1024 * 1024,
            max_gpu_memory_bytes: 1024 * 1024 * 1024,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_output_bytes: 64 * 1024 * 1024,
        }
    }

    /// Fast budget for unit / regression testing.
    #[must_use]
    pub const fn fast_test_budget(timeout_ms: u64) -> Self {
        Self {
            wall_timeout_ms: timeout_ms,
            cpu_timeout_ms: timeout_ms,
            max_memory_bytes: 256 * 1024 * 1024,
            max_gpu_memory_bytes: 256 * 1024 * 1024,
            max_artifact_bytes: 16 * 1024 * 1024,
            max_output_bytes: 16 * 1024 * 1024,
        }
    }

    /// Override the wall-clock timeout in milliseconds.
    #[must_use]
    pub const fn with_wall_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.wall_timeout_ms = timeout_ms;
        self
    }

    /// Override the host memory ceiling in bytes.
    #[must_use]
    pub const fn with_max_memory_bytes(mut self, max_bytes: u64) -> Self {
        self.max_memory_bytes = max_bytes;
        self
    }

    /// Override the max artifact size in bytes.
    #[must_use]
    pub const fn with_max_artifact_bytes(mut self, max_bytes: u64) -> Self {
        self.max_artifact_bytes = max_bytes;
        self
    }

    /// Override the max output size in bytes.
    #[must_use]
    pub const fn with_max_output_bytes(mut self, max_bytes: u64) -> Self {
        self.max_output_bytes = max_bytes;
        self
    }
}

/// Device lease held by one worker process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceLease {
    /// Unique lease identifier.
    pub lease_id: String,
    /// Backend identifier (e.g., "cuda", "wgpu", "spirv").
    pub backend_id: String,
    /// Device ordinal index.
    pub device_ordinal: u32,
    /// Human-readable adapter or device name.
    pub adapter_name: String,
    /// Ephemeral lease token.
    pub lease_token: String,
}

impl DeviceLease {
    /// Construct a new device lease.
    pub fn new(
        lease_id: impl Into<String>,
        backend_id: impl Into<String>,
        device_ordinal: u32,
        adapter_name: impl Into<String>,
        lease_token: impl Into<String>,
    ) -> Self {
        Self {
            lease_id: lease_id.into(),
            backend_id: backend_id.into(),
            device_ordinal,
            adapter_name: adapter_name.into(),
            lease_token: lease_token.into(),
        }
    }
}

/// Execution mode for a disposable worker process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    /// Pure reference interpreter execution.
    Reference,
    /// Production backend execution on device.
    Production,
}

impl fmt::Display for WorkerMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference => f.write_str("reference"),
            Self::Production => f.write_str("production"),
        }
    }
}

/// Numerical agreement policy between reference and backend outputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "policy", content = "parameters", rename_all = "snake_case")]
pub enum NumericalPolicy {
    /// Byte-exact equality across all outputs.
    Exact,
    /// F32 lane comparison within an inclusive ULP bound.
    Float32Ulps {
        /// Maximum allowed ULP distance.
        max_ulps: u32,
    },
}

impl NumericalPolicy {
    /// Check whether `found` outputs satisfy this policy relative to `baseline`.
    ///
    /// # Errors
    ///
    /// Returns [`NumericalMismatch`] if output counts, lengths, bytes, or lane distances
    /// violate the declared policy.
    pub fn check_agreement(
        &self,
        baseline: &[Vec<u8>],
        found: &[Vec<u8>],
    ) -> Result<(), NumericalMismatch> {
        if baseline.len() != found.len() {
            return Err(NumericalMismatch::BufferCount {
                expected: baseline.len(),
                found: found.len(),
            });
        }
        for (buffer_idx, (base_buf, found_buf)) in baseline.iter().zip(found.iter()).enumerate() {
            if base_buf.len() != found_buf.len() {
                return Err(NumericalMismatch::BufferLength {
                    buffer_idx,
                    expected: base_buf.len(),
                    found: found_buf.len(),
                });
            }
            match self {
                Self::Exact => {
                    if let Some(byte_idx) = base_buf
                        .iter()
                        .zip(found_buf.iter())
                        .position(|(a, b)| a != b)
                    {
                        return Err(NumericalMismatch::ExactByteMismatch {
                            buffer_idx,
                            byte_idx,
                            expected: base_buf[byte_idx],
                            found: found_buf[byte_idx],
                        });
                    }
                }
                Self::Float32Ulps { max_ulps } => {
                    if base_buf.len() % 4 != 0 {
                        return Err(NumericalMismatch::LaneAlignment {
                            buffer_idx,
                            bytes: base_buf.len(),
                        });
                    }
                    for (lane_idx, (base_chunk, found_chunk)) in base_buf
                        .chunks_exact(4)
                        .zip(found_buf.chunks_exact(4))
                        .enumerate()
                    {
                        let base_bits = u32::from_le_bytes([
                            base_chunk[0],
                            base_chunk[1],
                            base_chunk[2],
                            base_chunk[3],
                        ]);
                        let found_bits = u32::from_le_bytes([
                            found_chunk[0],
                            found_chunk[1],
                            found_chunk[2],
                            found_chunk[3],
                        ]);
                        let distance = ulp_distance_f32(base_bits, found_bits);
                        if distance > *max_ulps {
                            return Err(NumericalMismatch::LaneMismatch {
                                buffer_idx,
                                lane_idx,
                                expected: base_bits,
                                found: found_bits,
                                distance,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Calculate ULP distance between two IEEE-754 binary32 values.
///
/// NaN, infinity, or sign differences return `u32::MAX`.
#[must_use]
pub fn ulp_distance_f32(left: u32, right: u32) -> u32 {
    if left == right {
        return 0;
    }
    let (a, b) = (f32::from_bits(left), f32::from_bits(right));
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u32::MAX;
    }
    let ordered = |bits: u32| bits & 0x7fff_ffff;
    ordered(left).abs_diff(ordered(right))
}

/// Numerical output mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericalMismatch {
    /// Number of output buffers differs.
    BufferCount {
        /// Expected output count.
        expected: usize,
        /// Observed output count.
        found: usize,
    },
    /// Output buffer size in bytes differs.
    BufferLength {
        /// Output buffer index.
        buffer_idx: usize,
        /// Expected byte length.
        expected: usize,
        /// Observed byte length.
        found: usize,
    },
    /// Byte mismatch under exact policy.
    ExactByteMismatch {
        /// Output buffer index.
        buffer_idx: usize,
        /// Byte offset within buffer.
        byte_idx: usize,
        /// Expected byte.
        expected: u8,
        /// Observed byte.
        found: u8,
    },
    /// F32 lane ULP distance exceeded allowed bound.
    LaneMismatch {
        /// Output buffer index.
        buffer_idx: usize,
        /// F32 lane index.
        lane_idx: usize,
        /// Expected IEEE bit pattern.
        expected: u32,
        /// Observed IEEE bit pattern.
        found: u32,
        /// Computed ULP distance.
        distance: u32,
    },
    /// Buffer byte length is not a multiple of 4 under F32 ULP policy.
    LaneAlignment {
        /// Output buffer index.
        buffer_idx: usize,
        /// Observed buffer byte length.
        bytes: usize,
    },
}

impl fmt::Display for NumericalMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferCount { expected, found } => {
                write!(
                    f,
                    "output buffer count mismatch: expected {expected}, found {found}"
                )
            }
            Self::BufferLength {
                buffer_idx,
                expected,
                found,
            } => {
                write!(
                    f,
                    "output buffer {buffer_idx} length mismatch: expected {expected} bytes, found {found} bytes"
                )
            }
            Self::ExactByteMismatch {
                buffer_idx,
                byte_idx,
                expected,
                found,
            } => {
                write!(
                    f,
                    "output buffer {buffer_idx} byte {byte_idx} mismatch: expected {expected:#04x}, found {found:#04x}"
                )
            }
            Self::LaneMismatch {
                buffer_idx,
                lane_idx,
                expected,
                found,
                distance,
            } => {
                write!(
                    f,
                    "output buffer {buffer_idx} lane {lane_idx} mismatch: expected {expected:#010x}, found {found:#010x}, ULP distance {distance}"
                )
            }
            Self::LaneAlignment { buffer_idx, bytes } => {
                write!(
                    f,
                    "output buffer {buffer_idx} byte length {bytes} is not a multiple of 4 (F32 lane alignment)"
                )
            }
        }
    }
}

impl std::error::Error for NumericalMismatch {}

/// Content-addressed case payload for execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CasePayload {
    /// Content-addressed case ID (blake3 hex digest of program wire, inputs, policy, and schedule).
    pub case_id: String,
    /// Stable operation identifier.
    pub op_id: String,
    /// Canonical program wire bytes.
    pub program_wire: Vec<u8>,
    /// Logical input buffers in declaration order.
    pub inputs: Vec<Vec<u8>>,
    /// Declared numerical policy.
    pub numerical_policy: NumericalPolicy,
    /// Selected schedule family name, if any.
    pub schedule_family: Option<String>,
}

impl CasePayload {
    /// Construct a new content-addressed case payload.
    #[must_use]
    pub fn new(
        op_id: impl Into<String>,
        program_wire: Vec<u8>,
        inputs: Vec<Vec<u8>>,
        numerical_policy: NumericalPolicy,
        schedule_family: Option<String>,
    ) -> Self {
        let op_id = op_id.into();
        let case_id = Self::compute_case_id(
            &op_id,
            &program_wire,
            &inputs,
            &numerical_policy,
            schedule_family.as_deref(),
        );
        Self {
            case_id,
            op_id,
            program_wire,
            inputs,
            numerical_policy,
            schedule_family,
        }
    }

    /// Compute the deterministic content-addressed case ID.
    #[must_use]
    pub fn compute_case_id(
        op_id: &str,
        program_wire: &[u8],
        inputs: &[Vec<u8>],
        numerical_policy: &NumericalPolicy,
        schedule_family: Option<&str>,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre.conform.case.v1");
        hasher.update(&(op_id.len() as u64).to_le_bytes());
        hasher.update(op_id.as_bytes());
        hasher.update(&(program_wire.len() as u64).to_le_bytes());
        hasher.update(program_wire);
        hasher.update(&(inputs.len() as u64).to_le_bytes());
        for input in inputs {
            hasher.update(&(input.len() as u64).to_le_bytes());
            hasher.update(input);
        }
        let policy_bytes = serde_json::to_vec(numerical_policy).unwrap_or_default();
        hasher.update(&(policy_bytes.len() as u64).to_le_bytes());
        hasher.update(&policy_bytes);
        if let Some(family) = schedule_family {
            hasher.update(&(family.len() as u64).to_le_bytes());
            hasher.update(family.as_bytes());
        } else {
            hasher.update(&0u64.to_le_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
}

/// Request sent by the coordinator to a disposable worker process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Execution mode (Reference vs Production).
    pub mode: WorkerMode,
    /// Content-addressed case payload.
    pub case: CasePayload,
    /// Device lease when running in production mode.
    pub device_lease: Option<DeviceLease>,
    /// Explicit resource budget.
    pub budget: WorkerBudget,
    /// Expected compiler binary digest.
    pub compiler_binary_blake3: String,
    /// Expected runner binary digest.
    pub runner_binary_blake3: String,
    /// Expected target facts digest.
    pub expected_target_facts_blake3: Option<String>,
    /// Expected environment digest.
    pub environment_blake3: String,
}

impl WorkerRequest {
    /// Construct a reference worker request.
    #[must_use]
    pub fn new_reference(
        request_id: impl Into<String>,
        case: CasePayload,
        budget: WorkerBudget,
        compiler_binary_blake3: impl Into<String>,
        runner_binary_blake3: impl Into<String>,
        environment_blake3: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            mode: WorkerMode::Reference,
            case,
            device_lease: None,
            budget,
            compiler_binary_blake3: compiler_binary_blake3.into(),
            runner_binary_blake3: runner_binary_blake3.into(),
            expected_target_facts_blake3: None,
            environment_blake3: environment_blake3.into(),
        }
    }

    /// Construct a production worker request.
    #[must_use]
    pub fn new_production(
        request_id: impl Into<String>,
        case: CasePayload,
        device_lease: DeviceLease,
        budget: WorkerBudget,
        compiler_binary_blake3: impl Into<String>,
        runner_binary_blake3: impl Into<String>,
        expected_target_facts_blake3: Option<String>,
        environment_blake3: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            mode: WorkerMode::Production,
            case,
            device_lease: Some(device_lease),
            budget,
            compiler_binary_blake3: compiler_binary_blake3.into(),
            runner_binary_blake3: runner_binary_blake3.into(),
            expected_target_facts_blake3,
            environment_blake3: environment_blake3.into(),
        }
    }
}

/// Outcome status reported by a worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum WorkerStatus {
    /// Step succeeded completely.
    Success,
    /// Step exceeded its wall or CPU budget.
    Timeout {
        /// Elapsed time in milliseconds.
        elapsed_ms: u64,
        /// Allocated budget in milliseconds.
        budget_ms: u64,
    },
    /// Worker process caught an unhandled panic.
    Panicked {
        /// Panic message.
        message: String,
    },
    /// Driver crash, SIGSEGV, SIGABRT, or communication loss.
    DriverLost {
        /// Diagnostic description.
        message: String,
    },
    /// Worker leaked memory, threads, or resources.
    Leak {
        /// Estimated bytes leaked.
        bytes_leaked: u64,
        /// Detail.
        message: String,
    },
    /// Protocol violation (bad serialization, forged auth tag, corrupted output).
    ProtocolViolation {
        /// Protocol violation detail.
        message: String,
    },
    /// Compiler or semantic execution error.
    ExecutionError {
        /// Error message.
        message: String,
    },
    /// Worker was quarantined.
    Quarantined {
        /// Quarantine reason.
        reason: String,
    },
}

impl fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("success"),
            Self::Timeout {
                elapsed_ms,
                budget_ms,
            } => {
                write!(f, "timed out after {elapsed_ms}ms (budget {budget_ms}ms)")
            }
            Self::Panicked { message } => write!(f, "worker panicked: {message}"),
            Self::DriverLost { message } => write!(f, "driver lost: {message}"),
            Self::Leak {
                bytes_leaked,
                message,
            } => {
                write!(f, "resource leak ({bytes_leaked} bytes): {message}")
            }
            Self::ProtocolViolation { message } => {
                write!(f, "protocol violation: {message}")
            }
            Self::ExecutionError { message } => write!(f, "execution error: {message}"),
            Self::Quarantined { reason } => write!(f, "quarantined: {reason}"),
        }
    }
}

/// Authenticated execution receipt emitted by a worker process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerReceipt {
    /// Receipt schema version.
    pub receipt_version: u32,
    /// Request identifier.
    pub request_id: String,
    /// Case identifier.
    pub case_id: String,
    /// Mode executed.
    pub mode: WorkerMode,
    /// Outcome status.
    pub status: WorkerStatus,
    /// Raw outputs in declaration order.
    pub outputs: Vec<Vec<u8>>,
    /// Blake3 digest of raw outputs stream.
    pub outputs_blake3: String,
    /// Blake3 digest of admitted artifact (for production).
    pub artifact_blake3: Option<String>,
    /// Blake3 digest of target payload (for production).
    pub payload_blake3: Option<String>,
    /// Target facts blake3 digest.
    pub target_facts_blake3: Option<String>,
    /// Compiler binary digest.
    pub compiler_binary_blake3: String,
    /// Runner binary digest.
    pub runner_binary_blake3: String,
    /// Environment digest.
    pub environment_blake3: String,
    /// Device lease identifier, if leased.
    pub device_lease_id: Option<String>,
    /// Elapsed wall time in milliseconds.
    pub elapsed_ms: u64,
    /// Peak host memory observed in bytes.
    pub peak_memory_bytes: u64,
    /// Worker report authentication tag.
    pub auth_tag: String,
}

impl WorkerReceipt {
    /// Receipt schema version.
    pub const SCHEMA_VERSION: u32 =
        vyre_spec::schema_registry::SchemaId::ProofReceipt.version_u32();
    /// Compute the authentication tag over the canonical receipt fields.
    #[must_use]
    pub fn compute_auth_tag(&self, secret: &[u8]) -> String {
        let key = derive_auth_key(secret);
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(b"vyre.conform.worker.receipt.v1");
        hasher.update(&self.receipt_version.to_le_bytes());
        hasher.update(&(self.request_id.len() as u64).to_le_bytes());
        hasher.update(self.request_id.as_bytes());
        hasher.update(&(self.case_id.len() as u64).to_le_bytes());
        hasher.update(self.case_id.as_bytes());
        let mode_byte = match self.mode {
            WorkerMode::Reference => 0u8,
            WorkerMode::Production => 1u8,
        };
        hasher.update(&[mode_byte]);
        let status_bytes = serde_json::to_vec(&self.status).unwrap_or_default();
        hasher.update(&(status_bytes.len() as u64).to_le_bytes());
        hasher.update(&status_bytes);
        hasher.update(&(self.outputs_blake3.len() as u64).to_le_bytes());
        hasher.update(self.outputs_blake3.as_bytes());
        if let Some(a) = &self.artifact_blake3 {
            hasher.update(&(a.len() as u64).to_le_bytes());
            hasher.update(a.as_bytes());
        } else {
            hasher.update(&0u64.to_le_bytes());
        }
        if let Some(p) = &self.payload_blake3 {
            hasher.update(&(p.len() as u64).to_le_bytes());
            hasher.update(p.as_bytes());
        } else {
            hasher.update(&0u64.to_le_bytes());
        }
        if let Some(tf) = &self.target_facts_blake3 {
            hasher.update(&(tf.len() as u64).to_le_bytes());
            hasher.update(tf.as_bytes());
        } else {
            hasher.update(&0u64.to_le_bytes());
        }
        hasher.update(&(self.compiler_binary_blake3.len() as u64).to_le_bytes());
        hasher.update(self.compiler_binary_blake3.as_bytes());
        hasher.update(&(self.runner_binary_blake3.len() as u64).to_le_bytes());
        hasher.update(self.runner_binary_blake3.as_bytes());
        hasher.update(&(self.environment_blake3.len() as u64).to_le_bytes());
        hasher.update(self.environment_blake3.as_bytes());
        if let Some(lease) = &self.device_lease_id {
            hasher.update(&(lease.len() as u64).to_le_bytes());
            hasher.update(lease.as_bytes());
        } else {
            hasher.update(&0u64.to_le_bytes());
        }
        hasher.update(&self.elapsed_ms.to_le_bytes());
        hasher.update(&self.peak_memory_bytes.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Authenticate this receipt against the shared secret.
    #[must_use]
    pub fn verify_auth_tag(&self, secret: &[u8]) -> bool {
        self.auth_tag == self.compute_auth_tag(secret)
    }

    /// Whether this receipt represents a successful execution.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self.status, WorkerStatus::Success)
    }
}

/// Compute a 32-byte keyed hash key from arbitrary secret bytes.
#[must_use]
pub fn derive_auth_key(secret: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre.conform.worker.auth_key.v1");
    hasher.update(secret);
    *hasher.finalize().as_bytes()
}

/// Hash a list of output byte buffers to a canonical hex digest.
#[must_use]
pub fn hash_outputs(outputs: &[Vec<u8>]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre.conform.outputs.v1");
    hasher.update(&(outputs.len() as u64).to_le_bytes());
    for buf in outputs {
        hasher.update(&(buf.len() as u64).to_le_bytes());
        hasher.update(buf);
    }
    hasher.finalize().to_hex().to_string()
}

/// Rejection reason when verifying worker receipts for certificate issuance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificateRejection {
    /// Reference and production executed different case IDs.
    MismatchedCaseIdentity {
        /// Reference case ID.
        ref_case: String,
        /// Production case ID.
        prod_case: String,
    },
    /// Binary digest does not match expected binary.
    MismatchedBinary {
        /// Expected binary digest.
        expected: String,
        /// Found binary digest.
        found: String,
    },
    /// Target facts digest does not match expected facts.
    MismatchedTargetFacts {
        /// Expected target facts digest.
        expected: String,
        /// Found target facts digest.
        found: String,
    },
    /// Environment digest does not match expected environment.
    MismatchedEnvironment {
        /// Expected environment digest.
        expected: String,
        /// Found environment digest.
        found: String,
    },
    /// Worker report authentication tag is invalid or tampered.
    InvalidReceiptAuth {
        /// Mode whose receipt was rejected.
        mode: String,
    },
    /// A worker failed to complete successfully.
    WorkerFailed {
        /// Mode whose worker failed.
        mode: String,
        /// Status reported.
        status: WorkerStatus,
    },
    /// Numerical policy was violated between reference and production outputs.
    NumericalPolicyViolation(NumericalMismatch),
    /// Invalid receipt schema version.
    UnsupportedReceiptVersion {
        /// Observed version.
        found: u32,
        /// Supported version.
        supported: u32,
    },
}

impl fmt::Display for CertificateRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedCaseIdentity {
                ref_case,
                prod_case,
            } => {
                write!(
                    f,
                    "case identity mismatch: reference case `{ref_case}` does not match production case `{prod_case}`"
                )
            }
            Self::MismatchedBinary { expected, found } => {
                write!(
                    f,
                    "binary mismatch: expected binary digest `{expected}`, found `{found}`"
                )
            }
            Self::MismatchedTargetFacts { expected, found } => {
                write!(
                    f,
                    "target facts mismatch: expected `{expected}`, found `{found}`"
                )
            }
            Self::MismatchedEnvironment { expected, found } => {
                write!(
                    f,
                    "environment mismatch: expected `{expected}`, found `{found}`"
                )
            }
            Self::InvalidReceiptAuth { mode } => {
                write!(f, "invalid or unauthenticated receipt from {mode} worker")
            }
            Self::WorkerFailed { mode, status } => {
                write!(f, "{mode} worker failed: {status}")
            }
            Self::NumericalPolicyViolation(mismatch) => {
                write!(f, "numerical policy violation: {mismatch}")
            }
            Self::UnsupportedReceiptVersion { found, supported } => {
                write!(
                    f,
                    "unsupported receipt version `{found}`; supported version is `{supported}`"
                )
            }
        }
    }
}

impl std::error::Error for CertificateRejection {}

/// Verify both reference and production receipts for certificate issuance.
///
/// # Errors
///
/// Returns [`CertificateRejection`] naming the exact mismatched receipt, binary, target facts,
/// environment, auth tag, or numerical policy defect.
pub fn verify_receipts_for_certificate(
    ref_receipt: &WorkerReceipt,
    prod_receipt: &WorkerReceipt,
    expected_binary: &str,
    expected_target_facts: Option<&str>,
    expected_environment: &str,
    policy: &NumericalPolicy,
    secret: &[u8],
) -> Result<(), CertificateRejection> {
    if ref_receipt.receipt_version != WorkerReceipt::SCHEMA_VERSION {
        return Err(CertificateRejection::UnsupportedReceiptVersion {
            found: ref_receipt.receipt_version,
            supported: WorkerReceipt::SCHEMA_VERSION,
        });
    }
    if prod_receipt.receipt_version != WorkerReceipt::SCHEMA_VERSION {
        return Err(CertificateRejection::UnsupportedReceiptVersion {
            found: prod_receipt.receipt_version,
            supported: WorkerReceipt::SCHEMA_VERSION,
        });
    }
    if !ref_receipt.verify_auth_tag(secret) {
        return Err(CertificateRejection::InvalidReceiptAuth {
            mode: "reference".to_string(),
        });
    }
    if !prod_receipt.verify_auth_tag(secret) {
        return Err(CertificateRejection::InvalidReceiptAuth {
            mode: "production".to_string(),
        });
    }
    if !ref_receipt.is_success() {
        return Err(CertificateRejection::WorkerFailed {
            mode: "reference".to_string(),
            status: ref_receipt.status.clone(),
        });
    }
    if !prod_receipt.is_success() {
        return Err(CertificateRejection::WorkerFailed {
            mode: "production".to_string(),
            status: prod_receipt.status.clone(),
        });
    }
    if ref_receipt.case_id != prod_receipt.case_id {
        return Err(CertificateRejection::MismatchedCaseIdentity {
            ref_case: ref_receipt.case_id.clone(),
            prod_case: prod_receipt.case_id.clone(),
        });
    }
    if prod_receipt.compiler_binary_blake3 != expected_binary {
        return Err(CertificateRejection::MismatchedBinary {
            expected: expected_binary.to_string(),
            found: prod_receipt.compiler_binary_blake3.clone(),
        });
    }
    if let Some(expected_tf) = expected_target_facts {
        let found_tf = prod_receipt.target_facts_blake3.as_deref().unwrap_or("");
        if found_tf != expected_tf {
            return Err(CertificateRejection::MismatchedTargetFacts {
                expected: expected_tf.to_string(),
                found: found_tf.to_string(),
            });
        }
    }
    if prod_receipt.environment_blake3 != expected_environment {
        return Err(CertificateRejection::MismatchedEnvironment {
            expected: expected_environment.to_string(),
            found: prod_receipt.environment_blake3.clone(),
        });
    }
    policy
        .check_agreement(&ref_receipt.outputs, &prod_receipt.outputs)
        .map_err(CertificateRejection::NumericalPolicyViolation)?;
    Ok(())
}
