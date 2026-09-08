//! Typed, versioned reference execution requests and budgets.
//!
//! Replaces legacy multi-variant evaluation entry points with a single,
//! versioned [`ReferenceRequest`] requiring an explicit [`ReferenceBudget`].

use vyre_foundation::ir::Program;
use vyre_spec::NumericSemantics;

use crate::error::ReferenceError;
use crate::oob::OobReport;
use crate::value::Value;

/// Mandatory resource and termination bounds for a reference evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceBudget {
    /// Maximum interpreter steps admitted before refusal.
    pub max_steps: u64,
    /// Maximum workgroup and storage memory in bytes admitted before refusal.
    pub max_memory_bytes: u64,
    /// Maximum recursion depth admitted before refusal.
    pub max_recursion_depth: u32,
}

impl ReferenceBudget {
    /// Measured standard step ceiling for reference evaluation.
    pub const STANDARD_MAX_STEPS: u64 = crate::execution::step_budget::MAX_REFERENCE_STEPS;
    /// Standard 64 MiB memory bound for reference evaluation.
    pub const STANDARD_MAX_MEMORY_BYTES: u64 = 64 * 1024 * 1024;
    /// Standard recursion depth limit.
    pub const STANDARD_MAX_RECURSION_DEPTH: u32 = 256;

    /// Construct an explicit reference budget with all required limits.
    #[must_use]
    pub const fn new(max_steps: u64, max_memory_bytes: u64, max_recursion_depth: u32) -> Self {
        Self {
            max_steps,
            max_memory_bytes,
            max_recursion_depth,
        }
    }

    /// Construct a standard reference budget using measured defaults.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_steps: Self::STANDARD_MAX_STEPS,
            max_memory_bytes: Self::STANDARD_MAX_MEMORY_BYTES,
            max_recursion_depth: Self::STANDARD_MAX_RECURSION_DEPTH,
        }
    }

    /// Update the maximum step ceiling.
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Update the maximum memory bound in bytes.
    #[must_use]
    pub const fn with_max_memory_bytes(mut self, max_memory_bytes: u64) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }

    /// Update the maximum recursion depth limit.
    #[must_use]
    pub const fn with_max_recursion_depth(mut self, max_recursion_depth: u32) -> Self {
        self.max_recursion_depth = max_recursion_depth;
        self
    }
}

/// Workload dispatch envelope specifying grid and dispatch bounds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkloadEnvelope {
    /// Minimum element count floor for the dispatch grid.
    pub min_dispatch_elements: u32,
    /// Explicit 3-D workgroup grid `[x, y, z]` when overriding inferred dimensions.
    pub grid: Option<[u32; 3]>,
}

impl WorkloadEnvelope {
    /// Construct an empty workload envelope with default inferred bounds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_dispatch_elements: 0,
            grid: None,
        }
    }

    /// Set minimum dispatch element floor.
    #[must_use]
    pub const fn with_min_dispatch_elements(mut self, min: u32) -> Self {
        self.min_dispatch_elements = min;
        self
    }

    /// Set explicit 3-D workgroup grid.
    #[must_use]
    pub const fn with_grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = Some(grid);
        self
    }
}

/// Deterministic schedule-exploration policy for testing concurrency hazards.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScheduleExplorationPolicy {
    /// Standard forward step order.
    #[default]
    Forward,
    /// Reversed invocation/workgroup step order.
    LaneReversed,
    /// Left-rotated invocation step order by a constant lane offset.
    LaneRotated(u32),
}

/// Typed, versioned request for reference interpretation of an IR program.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceRequest<'a> {
    /// Schema version of this request.
    pub version: u32,
    /// Logical IR program to interpret.
    pub program: &'a Program,
    /// Declaration-order input values conforming to the interpreter ABI.
    pub inputs: &'a [Value],
    /// Workload envelope defining grid and dispatch bounds.
    pub envelope: WorkloadEnvelope,
    /// Optional numerical semantics contract.
    pub numerical: Option<NumericSemantics>,
    /// Deterministic schedule-exploration policy.
    pub schedule: ScheduleExplorationPolicy,
    /// Mandatory work, memory, and recursion budget.
    pub budget: ReferenceBudget,
}

impl<'a> ReferenceRequest<'a> {
    /// Current supported schema version for [`ReferenceRequest`].
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Construct a reference request with the mandatory budget.
    #[must_use]
    pub const fn new(program: &'a Program, inputs: &'a [Value], budget: ReferenceBudget) -> Self {
        Self {
            version: Self::CURRENT_SCHEMA_VERSION,
            program,
            inputs,
            envelope: WorkloadEnvelope {
                min_dispatch_elements: 0,
                grid: None,
            },
            numerical: None,
            schedule: ScheduleExplorationPolicy::Forward,
            budget,
        }
    }

    /// Construct a reference request builder.
    #[must_use]
    pub const fn builder(program: &'a Program, inputs: &'a [Value]) -> ReferenceRequestBuilder<'a> {
        ReferenceRequestBuilder::new(program, inputs)
    }

    /// Execute this reference request through the canonical reference evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError`] on validation failure, out-of-bounds access,
    /// or when exceeding the requested budget.
    pub fn execute(&self) -> Result<ReferenceResponse, ReferenceError> {
        crate::execution::reference_eval(self)
    }

    /// Set the workload envelope.
    #[must_use]
    pub const fn with_envelope(mut self, envelope: WorkloadEnvelope) -> Self {
        self.envelope = envelope;
        self
    }

    /// Set the numerical contract.
    #[must_use]
    pub fn with_numerical(mut self, numerical: Option<NumericSemantics>) -> Self {
        self.numerical = numerical;
        self
    }

    /// Set the schedule exploration policy.
    #[must_use]
    pub const fn with_schedule(mut self, schedule: ScheduleExplorationPolicy) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set the execution budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: ReferenceBudget) -> Self {
        self.budget = budget;
        self
    }
}

/// Errors returned during reference request construction and validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReferenceRequestError {
    /// The request lacked a mandatory [`ReferenceBudget`].
    MissingBudget,
    /// Unsupported request schema version.
    UnsupportedVersion(u32),
}

impl std::fmt::Display for ReferenceRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBudget => write!(
                f,
                "ReferenceRequest lacks mandatory ReferenceBudget. Fix: provide ReferenceBudget via .budget() before calling .build()."
            ),
            Self::UnsupportedVersion(v) => write!(
                f,
                "unsupported ReferenceRequest schema version {v}. Fix: use ReferenceRequest::CURRENT_SCHEMA_VERSION."
            ),
        }
    }
}

impl std::error::Error for ReferenceRequestError {}

/// Builder for constructing a typed [`ReferenceRequest`].
#[derive(Clone, Debug)]
pub struct ReferenceRequestBuilder<'a> {
    version: u32,
    program: &'a Program,
    inputs: &'a [Value],
    envelope: WorkloadEnvelope,
    numerical: Option<NumericSemantics>,
    schedule: ScheduleExplorationPolicy,
    budget: Option<ReferenceBudget>,
}

impl<'a> ReferenceRequestBuilder<'a> {
    /// Create a new builder for the given program and inputs.
    #[must_use]
    pub const fn new(program: &'a Program, inputs: &'a [Value]) -> Self {
        Self {
            version: ReferenceRequest::CURRENT_SCHEMA_VERSION,
            program,
            inputs,
            envelope: WorkloadEnvelope {
                min_dispatch_elements: 0,
                grid: None,
            },
            numerical: None,
            schedule: ScheduleExplorationPolicy::Forward,
            budget: None,
        }
    }

    /// Specify schema version.
    #[must_use]
    pub const fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Specify workload envelope.
    #[must_use]
    pub const fn envelope(mut self, envelope: WorkloadEnvelope) -> Self {
        self.envelope = envelope;
        self
    }

    /// Specify numerical semantics contract.
    #[must_use]
    pub fn numerical(mut self, numerical: Option<NumericSemantics>) -> Self {
        self.numerical = numerical;
        self
    }

    /// Specify schedule exploration policy.
    #[must_use]
    pub const fn schedule(mut self, schedule: ScheduleExplorationPolicy) -> Self {
        self.schedule = schedule;
        self
    }

    /// Specify the mandatory reference budget.
    #[must_use]
    pub const fn budget(mut self, budget: ReferenceBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Build the reference request.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceRequestError::MissingBudget`] if no budget was supplied.
    pub fn build(self) -> Result<ReferenceRequest<'a>, ReferenceRequestError> {
        let budget = self.budget.ok_or(ReferenceRequestError::MissingBudget)?;
        if self.version != ReferenceRequest::CURRENT_SCHEMA_VERSION {
            return Err(ReferenceRequestError::UnsupportedVersion(self.version));
        }
        Ok(ReferenceRequest {
            version: self.version,
            program: self.program,
            inputs: self.inputs,
            envelope: self.envelope,
            numerical: self.numerical,
            schedule: self.schedule,
            budget,
        })
    }
}

/// Structured response from reference interpretation.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceResponse {
    /// Return values produced by the evaluated program.
    pub outputs: Vec<Value>,
    /// Total interpreter steps charged during evaluation.
    pub steps_charged: u64,
    /// Tally of out-of-bounds accesses recorded during evaluation.
    pub oob_report: OobReport,
}

impl ReferenceResponse {
    /// Consume the response and return its output values.
    #[must_use]
    pub fn into_outputs(self) -> Vec<Value> {
        self.outputs
    }

    /// Borrow output values as a slice.
    #[must_use]
    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    /// Get total steps executed during evaluation.
    #[must_use]
    pub const fn steps_executed(&self) -> u64 {
        self.steps_charged
    }

    /// Get out-of-bounds report recorded during evaluation.
    #[must_use]
    pub const fn oob_report(&self) -> &OobReport {
        &self.oob_report
    }
}
impl std::ops::Deref for ReferenceResponse {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        &self.outputs
    }
}

impl IntoIterator for ReferenceResponse {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.outputs.into_iter()
    }
}

impl From<ReferenceResponse> for Vec<Value> {
    fn from(response: ReferenceResponse) -> Self {
        response.outputs
    }
}
