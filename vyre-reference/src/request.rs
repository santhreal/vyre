//! Typed, versioned reference interpreter execution request and contract.
//!
//! One submission carries the logical program, the exact resource ABI, the
//! workload envelope, the numerical contract, the schedule policy, and a
//! mandatory work, memory, and recursion budget. Nothing about an execution is
//! implicit, and nothing about it is optional.

use vyre_foundation::ir::{BufferDecl, Program};
use vyre_spec::NumericSemantics;

use crate::error::ReferenceError;
use crate::oob::OobReport;
use crate::value::Value;

/// Stable schema version for [`ReferenceRequest`].
pub const REFERENCE_REQUEST_SCHEMA_VERSION: u32 = 1;

/// Stable reference oracle version reported in certificates.
pub const REFERENCE_ORACLE_VERSION: &str = "0.8.0-ref.row86";

/// Mandatory work, memory, and recursion budget for reference evaluation.
///
/// An absent budget does not compile: evaluation must be explicitly bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ReferenceBudget {
    /// Work ceiling in interpreter steps.
    pub work_ceiling: u64,
    /// Maximum allocated memory across all buffers in bytes.
    pub max_memory_bytes: usize,
    /// Maximum call and block frame recursion depth.
    pub max_recursion_depth: usize,
}

impl ReferenceBudget {
    /// Standard step ceiling for ordinary unit/integration test workloads.
    pub const DEFAULT_WORK_CEILING: u64 = crate::step_budget::MAX_REFERENCE_STEPS;
    /// Standard memory ceiling for workgroup/storage allocations (64 MiB).
    pub const DEFAULT_MAX_MEMORY_BYTES: usize = 64 * 1024 * 1024;
    /// Standard frame depth ceiling.
    pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 256;

    /// Construct an explicit reference execution budget.
    #[must_use]
    pub const fn new(
        work_ceiling: u64,
        max_memory_bytes: usize,
        max_recursion_depth: usize,
    ) -> Self {
        Self {
            work_ceiling,
            max_memory_bytes,
            max_recursion_depth,
        }
    }

    /// Build a standard budget suitable for regular reference evaluation.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            work_ceiling: Self::DEFAULT_WORK_CEILING,
            max_memory_bytes: Self::DEFAULT_MAX_MEMORY_BYTES,
            max_recursion_depth: Self::DEFAULT_MAX_RECURSION_DEPTH,
        }
    }

    /// Construct a tight budget for adversarial termination checks.
    #[must_use]
    pub const fn bounded(steps: u64) -> Self {
        Self {
            work_ceiling: steps,
            max_memory_bytes: Self::DEFAULT_MAX_MEMORY_BYTES,
            max_recursion_depth: Self::DEFAULT_MAX_RECURSION_DEPTH,
        }
    }
}

/// Workload envelope defining workgroup extents and dispatch grid floors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkloadEnvelope {
    /// Workgroup size declared by the program `[sx, sy, sz]`.
    pub workgroup_size: [u32; 3],
    /// Explicit workgroup grid `[gx, gy, gz]` when specified.
    pub workgroup_grid: Option<[u32; 3]>,
    /// Minimum dispatch elements floor when specified.
    pub min_dispatch_elements: Option<u32>,
}

impl WorkloadEnvelope {
    /// Construct a workload envelope for the declared workgroup size.
    #[must_use]
    pub const fn for_workgroup_size(size: [u32; 3]) -> Self {
        Self {
            workgroup_size: size,
            workgroup_grid: None,
            min_dispatch_elements: None,
        }
    }

    /// Construct a workload envelope from a program.
    #[must_use]
    pub fn for_program(program: &Program) -> Self {
        Self::for_workgroup_size(program.workgroup_size())
    }

    /// Set an explicit workgroup grid.
    #[must_use]
    pub const fn with_grid(mut self, grid: [u32; 3]) -> Self {
        self.workgroup_grid = Some(grid);
        self
    }

    /// Set a dispatch elements floor.
    #[must_use]
    pub const fn with_min_dispatch_elements(mut self, min: u32) -> Self {
        self.min_dispatch_elements = Some(min);
        self
    }
}

/// Exact resource ABI: declared buffers and explicit inputs in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactResourceAbi {
    /// Declared buffers from the program specification.
    pub declared_buffers: Vec<BufferDecl>,
    /// Supplied input values in declaration order.
    pub inputs: Vec<Value>,
}

impl ExactResourceAbi {
    /// Construct an exact resource ABI.
    #[must_use]
    pub fn new(declared_buffers: Vec<BufferDecl>, inputs: Vec<Value>) -> Self {
        Self {
            declared_buffers,
            inputs,
        }
    }

    /// Extract exact resource ABI from a program and input slice.
    #[must_use]
    pub fn for_program(program: &Program, inputs: &[Value]) -> Self {
        Self {
            declared_buffers: program.buffers().to_vec(),
            inputs: inputs.to_vec(),
        }
    }
}

/// Deterministic schedule-exploration policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DeterministicSchedulePolicy {
    /// Standard forward step order.
    Forward,
    /// Reversed workgroup and lane order to detect race conditions.
    LaneReversed,
    /// Rotated lane order by `by` positions.
    LaneRotated(u32),
    /// Bounded schedule interleaving.
    BoundedInterleaving,
}

/// Execution strictness mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExecutionStrictness {
    /// Strict mode: any out-of-bounds, type mismatch, missing value, overflow, poison,
    /// or incomplete dispatch returns a structured error.
    Strict,
    /// Diagnostic permissive mode: records what a program absorbed, such as an
    /// out-of-bounds tally. It carries no output value and no certificate.
    DiagnosticPermissive,
}

/// One typed, versioned reference execution request.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRequest {
    /// Schema version for wire / ABI serialization.
    pub version: u32,
    /// The logical program to evaluate.
    pub program: Program,
    /// Exact resource ABI.
    pub resource_abi: ExactResourceAbi,
    /// Workload envelope.
    pub workload_envelope: WorkloadEnvelope,
    /// Numerical contract.
    pub numerical_contract: NumericSemantics,
    /// Deterministic schedule exploration policy.
    pub schedule_policy: DeterministicSchedulePolicy,
    /// Mandatory work and memory budget.
    pub budget: ReferenceBudget,
    /// Strictness mode.
    pub strictness: ExecutionStrictness,
}

impl ReferenceRequest {
    /// Construct a new strict reference request with the given mandatory budget.
    #[must_use]
    pub fn new(program: Program, inputs: Vec<Value>, budget: ReferenceBudget) -> Self {
        let envelope = WorkloadEnvelope::for_program(&program);
        let resource_abi = ExactResourceAbi::for_program(&program, &inputs);
        let numerical_contract = vyre_spec::numeric_semantics_for(&vyre_spec::DataType::F32);
        Self {
            version: REFERENCE_REQUEST_SCHEMA_VERSION,
            program,
            resource_abi,
            workload_envelope: envelope,
            numerical_contract,
            schedule_policy: DeterministicSchedulePolicy::Forward,
            budget,
            strictness: ExecutionStrictness::Strict,
        }
    }

    /// Set an explicit workload envelope.
    #[must_use]
    pub fn with_workload_envelope(mut self, envelope: WorkloadEnvelope) -> Self {
        self.workload_envelope = envelope;
        self
    }

    /// Set an explicit schedule exploration policy.
    #[must_use]
    pub fn with_schedule_policy(mut self, policy: DeterministicSchedulePolicy) -> Self {
        self.schedule_policy = policy;
        self
    }

    /// Set an explicit numerical contract.
    #[must_use]
    pub fn with_numerical_contract(mut self, contract: NumericSemantics) -> Self {
        self.numerical_contract = contract;
        self
    }

    /// Set strictness mode.
    #[must_use]
    pub fn with_strictness(mut self, strictness: ExecutionStrictness) -> Self {
        self.strictness = strictness;
        self
    }

    /// Execute this request strictly and return a verified result with certificate.
    ///
    /// # Errors
    /// Returns a structured [`ReferenceError`] on missing values, type mismatches, poison,
    /// overflow, out-of-bounds access, incomplete dispatch semantics, nontermination, or budget exhaustion.
    pub fn execute(&self) -> Result<StrictExecutionResult, ReferenceError> {
        if self.strictness != ExecutionStrictness::Strict {
            return Err(ReferenceError::type_mismatch(
                "execute() called on a non-strict request; use execute_permissive() for diagnostic permissive requests",
            ));
        }
        let (outputs, steps) = crate::execution::run_with_request(self)?;
        let fingerprint = self
            .program
            .fingerprint()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let certificate = ReferenceCertificate {
            schema_version: REFERENCE_REQUEST_SCHEMA_VERSION,
            program_fingerprint: fingerprint,
            oracle_version: REFERENCE_ORACLE_VERSION.to_string(),
            steps_executed: steps,
            schedule_policy: self.schedule_policy,
        };
        Ok(StrictExecutionResult {
            outputs,
            steps_executed: steps,
            certificate,
        })
    }

    /// Execute in diagnostic permissive mode.
    ///
    /// Diagnostic permissive mode records what the run absorbed. The report it
    /// returns carries no output value and no certificate.
    ///
    /// # Errors
    /// Returns [`ReferenceError`] on unrecoverable host faults.
    pub fn execute_permissive(&self) -> Result<DiagnosticPermissiveReport, ReferenceError> {
        let (outputs, steps, oob) = crate::execution::run_permissive_with_request(self)?;
        let mut anomalies = Vec::new();
        if oob.total() > 0 {
            anomalies.push(format!(
                "out-of-bounds accesses absorbed: loads={}, stores={}, atomics={}",
                oob.oob_loads, oob.oob_stores, oob.oob_atomics
            ));
        }
        Ok(DiagnosticPermissiveReport {
            output_digest: output_digest(&outputs),
            oob_report: oob,
            steps_executed: steps,
            recorded_anomalies: anomalies,
        })
    }
}

/// Verification certificate proving a valid, strict reference oracle execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceCertificate {
    /// Certificate schema version.
    pub schema_version: u32,
    /// Fingerprint of the verified program.
    pub program_fingerprint: String,
    /// Stable oracle version that performed the execution.
    pub oracle_version: String,
    /// Steps executed within the work ceiling.
    pub steps_executed: u64,
    /// Deterministic schedule policy verified.
    pub schedule_policy: DeterministicSchedulePolicy,
}

/// Output of a successful strict reference evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct StrictExecutionResult {
    /// Exact reference outputs.
    pub outputs: Vec<Value>,
    /// Steps executed.
    pub steps_executed: u64,
    /// Verification certificate.
    pub certificate: ReferenceCertificate,
}

/// Digest of the bytes a permissive run produced.
///
/// A digest is the whole record of what permissive mode observed. It is enough
/// to tell two permissive runs apart and not enough to grade a device against,
/// because the bytes it summarizes were computed while out-of-bounds accesses
/// were being absorbed rather than refused.
fn output_digest(outputs: &[Value]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(outputs.len() as u64).to_le_bytes());
    for output in outputs {
        let bytes = output.to_bytes();
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    hasher.finalize().to_hex().to_string()
}

/// Diagnostic report for permissive evaluation.
///
/// Permissive mode cannot issue an expected output or a certificate. That is a
/// property of this type rather than of a check inside it: there is no output
/// value and no certificate anywhere in the report, so no caller can extract
/// one, mistake one for a graded result, or reach one by ignoring a `Result`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticPermissiveReport {
    /// Digest of the bytes the run produced under absorption.
    pub output_digest: String,
    /// Tally of out-of-bounds accesses absorbed.
    pub oob_report: OobReport,
    /// Steps executed.
    pub steps_executed: u64,
    /// Recorded anomaly diagnostics.
    pub recorded_anomalies: Vec<String>,
}
