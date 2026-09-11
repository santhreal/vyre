//! Typed, versioned reference interpreter execution request and contract.
//!
//! One submission carries the logical program, the exact resource ABI, the
//! workload envelope, the numerical contract, the schedule policy, and a
//! mandatory work, memory, and recursion budget. Nothing about an execution is
//! implicit, and nothing about it is optional: [`ReferenceRequest`] is the only
//! type the evaluator accepts and its methods are the only entry points into
//! it.
//!
//! Strictness is the method rather than a field. [`ReferenceRequest::execute`]
//! and [`ReferenceRequest::outputs`] grade a device; they refuse a fault
//! instead of absorbing it. [`ReferenceRequest::execute_permissive`] records
//! what a run absorbed and returns a [`DiagnosticPermissiveReport`], which has
//! no output value and no certificate anywhere in it. A caller therefore
//! cannot extract an expected output from a permissive run by ignoring a
//! `Result` or by reading the wrong field: the type carries none.

use vyre_foundation::ir::{BufferDecl, Program};
use vyre_spec::{numeric_semantics_for, DataType, NumericSemantics};

use crate::error::ReferenceError;
use crate::interleaving::RaceExplorationReport;
use crate::oob::OobReport;
use crate::value::Value;

/// Stable schema version for [`ReferenceRequest`].
pub const REFERENCE_REQUEST_SCHEMA_VERSION: u32 = 2;

/// Stable reference oracle version reported in certificates.
pub const REFERENCE_ORACLE_VERSION: &str = "0.9.0-ref";

/// Mandatory work, memory, and recursion budget for reference evaluation.
///
/// An absent budget does not compile: [`ReferenceRequest::new`] takes one by
/// value and every evaluation arms it, so no path reaches the evaluator
/// unbounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ReferenceBudget {
    /// Work ceiling in interpreter steps.
    pub work_ceiling: u64,
    /// Whether the ceiling rises to the work the program's own constant
    /// extents declare.
    ///
    /// A program whose trip counts are all constants states its work in
    /// advance, and refusing it against a ceiling sized for a smaller corpus
    /// says the oracle cannot evaluate a program whose work it can count. When
    /// this is `false` the ceiling is a hard cap, which is what an adversarial
    /// termination check needs.
    pub admit_declared_work: bool,
    /// Maximum buffer bytes one evaluation may allocate.
    pub max_memory_bytes: usize,
    /// Maximum block, loop, and call frame depth one lane may reach.
    pub max_recursion_depth: usize,
}

impl ReferenceBudget {
    /// Standard step ceiling for ordinary reference workloads.
    pub const DEFAULT_WORK_CEILING: u64 = crate::step_budget::MAX_REFERENCE_STEPS;
    /// Standard allocation ceiling for one evaluation (1 GiB).
    ///
    /// The oracle materializes every declared buffer of a whole dispatch on
    /// the host. The bound exists so a program that asks for more memory than
    /// the host has ends with a structured refusal rather than an allocation
    /// failure that takes the process with it; it is not a workload size.
    pub const DEFAULT_MAX_MEMORY_BYTES: usize = 1024 * 1024 * 1024;
    /// Standard frame depth ceiling.
    ///
    /// Frame depth follows the program's static block nesting, not its trip
    /// counts, so this stands far above any nesting a compiler emits and still
    /// ends a self-referential region before the host stack does.
    pub const DEFAULT_MAX_RECURSION_DEPTH: usize = 1024;

    /// Construct an explicit reference execution budget whose ceiling rises to
    /// the work the program declares.
    #[must_use]
    pub const fn new(
        work_ceiling: u64,
        max_memory_bytes: usize,
        max_recursion_depth: usize,
    ) -> Self {
        Self {
            work_ceiling,
            admit_declared_work: true,
            max_memory_bytes,
            max_recursion_depth,
        }
    }

    /// Build the standard budget for regular reference evaluation.
    #[must_use]
    pub const fn standard() -> Self {
        Self::new(
            Self::DEFAULT_WORK_CEILING,
            Self::DEFAULT_MAX_MEMORY_BYTES,
            Self::DEFAULT_MAX_RECURSION_DEPTH,
        )
    }

    /// Construct a hard step cap for adversarial termination checks.
    ///
    /// The cap does not rise to declared work, so a program that declares a
    /// trip count larger than `steps` is refused rather than admitted.
    #[must_use]
    pub const fn bounded(steps: u64) -> Self {
        Self {
            work_ceiling: steps,
            admit_declared_work: false,
            max_memory_bytes: Self::DEFAULT_MAX_MEMORY_BYTES,
            max_recursion_depth: Self::DEFAULT_MAX_RECURSION_DEPTH,
        }
    }

    /// The standard budget with an explicit work ceiling that still rises to
    /// declared work.
    #[must_use]
    pub const fn with_work_ceiling(steps: u64) -> Self {
        Self::new(
            steps,
            Self::DEFAULT_MAX_MEMORY_BYTES,
            Self::DEFAULT_MAX_RECURSION_DEPTH,
        )
    }

    /// The standard budget with an explicit allocation ceiling.
    #[must_use]
    pub const fn with_max_memory_bytes(bytes: usize) -> Self {
        Self::new(
            Self::DEFAULT_WORK_CEILING,
            bytes,
            Self::DEFAULT_MAX_RECURSION_DEPTH,
        )
    }

    /// The standard budget with an explicit frame depth ceiling.
    #[must_use]
    pub const fn with_max_recursion_depth(depth: usize) -> Self {
        Self::new(
            Self::DEFAULT_WORK_CEILING,
            Self::DEFAULT_MAX_MEMORY_BYTES,
            depth,
        )
    }
}

/// Workload envelope defining workgroup extents and dispatch grid floors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

/// Exact resource ABI: the buffers the program declares and the values the
/// caller supplies for them, both borrowed for the life of the request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactResourceAbi<'a> {
    /// Declared buffers from the program specification.
    pub declared_buffers: &'a [BufferDecl],
    /// Supplied input values in declaration order.
    pub inputs: &'a [Value],
}

impl<'a> ExactResourceAbi<'a> {
    /// Construct an exact resource ABI.
    #[must_use]
    pub const fn new(declared_buffers: &'a [BufferDecl], inputs: &'a [Value]) -> Self {
        Self {
            declared_buffers,
            inputs,
        }
    }

    /// Extract the exact resource ABI a program and input slice describe.
    #[must_use]
    pub fn for_program(program: &'a Program, inputs: &'a [Value]) -> Self {
        Self::new(program.buffers(), inputs)
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

/// One typed, versioned reference execution request.
///
/// The request borrows the program and the inputs, so submitting one copies
/// neither. The evaluator has no other door: every public evaluation entry
/// point on this crate is a method here.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRequest<'a> {
    /// Schema version for wire / ABI serialization.
    pub version: u32,
    /// The logical program to evaluate.
    pub program: &'a Program,
    /// Exact resource ABI.
    pub resource_abi: ExactResourceAbi<'a>,
    /// Workload envelope.
    pub workload_envelope: WorkloadEnvelope,
    /// Numerical contract the caller grades against.
    pub numerical_contract: NumericSemantics,
    /// Deterministic schedule exploration policy.
    pub schedule_policy: DeterministicSchedulePolicy,
    /// Mandatory work, memory, and recursion budget.
    pub budget: ReferenceBudget,
}

impl<'a> ReferenceRequest<'a> {
    /// Construct a reference request under an explicit mandatory budget.
    #[must_use]
    pub fn new(program: &'a Program, inputs: &'a [Value], budget: ReferenceBudget) -> Self {
        Self {
            version: REFERENCE_REQUEST_SCHEMA_VERSION,
            program,
            resource_abi: ExactResourceAbi::for_program(program, inputs),
            workload_envelope: WorkloadEnvelope::for_program(program),
            numerical_contract: numeric_semantics_for(&DataType::F32),
            schedule_policy: DeterministicSchedulePolicy::Forward,
            budget,
        }
    }

    /// Construct a reference request under [`ReferenceBudget::standard`].
    #[must_use]
    pub fn standard(program: &'a Program, inputs: &'a [Value]) -> Self {
        Self::new(program, inputs, ReferenceBudget::standard())
    }

    /// Set an explicit workload envelope.
    #[must_use]
    pub const fn with_workload_envelope(mut self, envelope: WorkloadEnvelope) -> Self {
        self.workload_envelope = envelope;
        self
    }

    /// Set an explicit workgroup grid `[x, y, z]`.
    ///
    /// Buffer-shape inference distributes a dispatch only across workgroup axes
    /// whose size is greater than one, so a program that fans a `[256, 1, 1]`
    /// workgroup across `grid.y` would otherwise collapse to `grid.y == 1` and
    /// cover only the first slice. A caller that knows the real dispatch grid
    /// states it here.
    #[must_use]
    pub const fn with_grid(mut self, grid: [u32; 3]) -> Self {
        self.workload_envelope.workgroup_grid = Some(grid);
        self
    }

    /// Set a dispatch element floor.
    ///
    /// Buffer-shape inference cannot see the per-invocation count of a program
    /// whose scan length is a runtime value: a haystack packed four bytes to a
    /// `u32` infers a quarter of the invocations the dispatch runs, and the
    /// high positions are never visited. The floor states the real count; the
    /// interpreter still runs at least the inferred grid, so `0` changes
    /// nothing.
    #[must_use]
    pub const fn with_min_dispatch_elements(mut self, min: u32) -> Self {
        self.workload_envelope.min_dispatch_elements = Some(min);
        self
    }

    /// Set an explicit schedule exploration policy.
    #[must_use]
    pub const fn with_schedule_policy(mut self, policy: DeterministicSchedulePolicy) -> Self {
        self.schedule_policy = policy;
        self
    }

    /// Set an explicit numerical contract.
    #[must_use]
    pub fn with_numerical_contract(mut self, contract: NumericSemantics) -> Self {
        self.numerical_contract = contract;
        self
    }

    /// Set an explicit budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: ReferenceBudget) -> Self {
        self.budget = budget;
        self
    }

    /// The numerical contract the oracle applies for `datatype`, refusing a
    /// request whose stated contract disagrees with it.
    ///
    /// The oracle's arithmetic comes from `vyre_spec`, so a request that states
    /// a different contract is grading a device against semantics the oracle
    /// never applied. Naming the disagreement is the only answer that is not a
    /// silently wrong verdict.
    fn verify_numerical_contract(&self) -> Result<(), ReferenceError> {
        let authoritative = numeric_semantics_for(&self.numerical_contract.datatype);
        if authoritative == self.numerical_contract {
            return Ok(());
        }
        Err(ReferenceError::type_mismatch(format!(
            "the request states a numerical contract for {:?} that disagrees with the versioned \
             semantics the oracle applies (schema {}). Fix: submit \
             `vyre_spec::numeric_semantics_for` for the datatype, or grade against the semantics \
             the oracle implements.",
            self.numerical_contract.datatype,
            vyre_spec::NUMERIC_SEMANTICS_SCHEMA_VERSION
        )))
    }

    /// Execute this request strictly and return the outputs with a certificate.
    ///
    /// # Errors
    /// Returns a structured [`ReferenceError`] on missing values, type
    /// mismatches, poison, overflow, out-of-bounds access, incomplete dispatch
    /// semantics, nontermination, or budget exhaustion.
    pub fn execute(&self) -> Result<StrictExecutionResult, ReferenceError> {
        self.verify_numerical_contract()?;
        let (outputs, steps) = crate::execution::run_with_request(self)?;
        let fingerprint =
            self.program
                .fingerprint()
                .iter()
                .fold(String::with_capacity(64), |mut hex, byte| {
                    use std::fmt::Write;
                    let _ = write!(hex, "{byte:02x}");
                    hex
                });
        Ok(StrictExecutionResult {
            outputs,
            steps_executed: steps,
            certificate: ReferenceCertificate {
                schema_version: REFERENCE_REQUEST_SCHEMA_VERSION,
                program_fingerprint: fingerprint,
                oracle_version: REFERENCE_ORACLE_VERSION.to_string(),
                steps_executed: steps,
                schedule_policy: self.schedule_policy,
            },
        })
    }

    /// Execute this request strictly and return only the output values.
    ///
    /// The projection a caller comparing bytes against a device reads. It runs
    /// exactly [`execute`](Self::execute) and drops the certificate.
    ///
    /// # Errors
    /// Same as [`execute`](Self::execute).
    pub fn outputs(&self) -> Result<Vec<Value>, ReferenceError> {
        self.execute().map(|result| result.outputs)
    }

    /// Execute this request strictly and return the outputs with the step count
    /// the run charged.
    ///
    /// # Errors
    /// Same as [`execute`](Self::execute).
    pub fn outputs_and_steps(&self) -> Result<(Vec<Value>, u64), ReferenceError> {
        self.execute()
            .map(|result| (result.outputs, result.steps_executed))
    }

    /// Execute in diagnostic permissive mode.
    ///
    /// Permissive mode absorbs an out-of-bounds access deterministically and
    /// records the tally. The report it returns carries no output value and no
    /// certificate, so nothing a device can be graded against leaves this
    /// method.
    ///
    /// # Errors
    /// Returns [`ReferenceError`] for every fault class permissive mode does
    /// not absorb, which is every class except out-of-bounds access.
    pub fn execute_permissive(&self) -> Result<DiagnosticPermissiveReport, ReferenceError> {
        self.verify_numerical_contract()?;
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

    /// Explore a bounded set of deterministic step orders and report every
    /// race the dispatch carries.
    ///
    /// A single-threaded evaluation resolves a cross-lane conflict the same way
    /// every run, so the output looks stable while the device it grades leaves
    /// the winner driver-defined. This terminal runs the same dispatch under
    /// [`declared_race_exploration_orders`](Self::declared_race_exploration_orders)
    /// step orders with shadow memory recording every buffer access, and
    /// reports two hazard classes: an unsynchronized conflict inside one order,
    /// and a byte difference between two orders.
    ///
    /// A race is reported, not refused, so one call names every hazard in the
    /// dispatch rather than the first one. The whole exploration is charged
    /// against one [`ReferenceBudget`], so it cannot run unbounded.
    ///
    /// # Errors
    /// Returns [`ReferenceError`] for every fault class strict evaluation
    /// refuses: missing values, type mismatches, poison, overflow,
    /// out-of-bounds access, incomplete dispatch semantics, nontermination, and
    /// budget exhaustion.
    pub fn explore_races(&self) -> Result<RaceExplorationReport, ReferenceError> {
        self.verify_numerical_contract()?;
        crate::execution::explore_races_with_request(self)
    }

    /// Number of step orders [`explore_races`](Self::explore_races) will run
    /// for this request.
    ///
    /// Derived from the workgroup extent the program declares, so a caller
    /// states the exploration's exact width before running it and never
    /// exceeds [`MAX_RACE_EXPLORATION_ORDERS`](crate::MAX_RACE_EXPLORATION_ORDERS).
    #[must_use]
    pub fn declared_race_exploration_orders(&self) -> usize {
        crate::execution::race_exploration_orders(self.program).len()
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
