//! Conformance oracle session: evaluates a Program directly on the pure-Rust
//! reference interpreter without intermediate backend registration, precedence,
//! or materializers.

use std::sync::Arc;

use thiserror::Error;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

/// Errors returned by [`OracleSession`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OracleError {
    /// Missing or extra input buffers supplied to the oracle.
    #[error(
        "oracle received {received} input buffer(s) for {expected} reference input(s). Fix: pass one buffer per reference input in Program::buffers order without trailing buffers."
    )]
    InputBufferCountMismatch {
        /// Number of reference inputs expected by the program.
        expected: usize,
        /// Number of input buffers received.
        received: usize,
    },

    /// Strict IEEE transcendental expansion failed.
    #[error(
        "oracle cannot lower float mode `{mode}`: {0}. Fix: give the operation an exact f32 expansion in vyre_foundation::fp_expansion.",
        mode = FloatLoweringMode::StrictIeee.cache_label()
    )]
    FloatExpansion(String),

    /// Pure-Rust reference evaluation failed.
    #[error("oracle reference evaluation failed: {0}")]
    Evaluation(String),
}

/// Independent conformance oracle session executing over `vyre-reference`.
///
/// Unlike production device drivers, `OracleSession` does not implement
/// `VyreBackend`, does not appear in backend discovery, capability negotiation,
/// or autoroute, and evaluates programs directly using the pure-Rust reference
/// specification.
#[derive(Debug, Clone)]
pub struct OracleSession {
    program: Arc<Program>,
    float_lowering: FloatLoweringMode,
}

impl OracleSession {
    /// Create a new oracle session for one Program.
    #[must_use]
    pub fn new(program: Program) -> Self {
        Self {
            program: Arc::new(program),
            float_lowering: FloatLoweringMode::Contracted,
        }
    }

    /// Create an oracle session from an existing shared Program reference.
    #[must_use]
    pub fn from_arc(program: Arc<Program>) -> Self {
        Self {
            program,
            float_lowering: FloatLoweringMode::Contracted,
        }
    }

    /// Set the floating-point lowering mode for evaluation.
    #[must_use]
    pub fn with_float_lowering(mut self, mode: FloatLoweringMode) -> Self {
        self.float_lowering = mode;
        self
    }

    /// The program this oracle session evaluates.
    #[must_use]
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Execute the program on caller-supplied input buffers and return declaration-ordered outputs.
    pub fn execute(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, OracleError> {
        for i in 0..inputs.len() {
            for j in (i + 1)..inputs.len() {
                let a = inputs[i];
                let b = inputs[j];
                if !a.is_empty() && !b.is_empty() {
                    let a_start = a.as_ptr() as usize;
                    let a_end = a_start + a.len();
                    let b_start = b.as_ptr() as usize;
                    let b_end = b_start + b.len();
                    if a_start < b_end && b_start < a_end {
                        return Err(OracleError::Evaluation(format!(
                            "aliased input buffers detected between input {i} and input {j}. Fix: provide distinct, non-overlapping input buffers."
                        )));
                    }
                }
            }
        }
        let expanded = if self.float_lowering.blocks_contraction() {
            vyre_foundation::fp_expansion::expand_strict_transcendentals(&self.program)
                .map_err(|error| OracleError::FloatExpansion(error.to_string()))?
        } else {
            None
        };
        let program = expanded.as_ref().unwrap_or(&self.program);
        let values = reference_values(program, inputs)?;
        let request = vyre_reference::ReferenceRequest::new(
            program.clone(),
            values,
            vyre_reference::ReferenceBudget::standard(),
        );
        let result = request
            .execute()
            .map_err(|error| OracleError::Evaluation(error.to_string()))?;

        Ok(result.outputs.into_iter().map(|v| v.to_bytes()).collect())
    }
}

fn reference_values(program: &Program, inputs: &[&[u8]]) -> Result<Vec<Value>, OracleError> {
    // `vyre_reference::reference_input_values` is the interpreter's own input
    // ABI. This walk selected buffers as
    // `access() != Workgroup && !is_backend_allocated_output()`, which admits a
    // `Shared` buffer, a `Persistent` buffer, and a non-read-write
    // `pipeline_live_out` that no backend stages from the host, so the oracle
    // asked a device for one value more than the device consumes.
    vyre_reference::reference_input_values(program, inputs).map_err(|mismatch| {
        OracleError::InputBufferCountMismatch {
            expected: mismatch.expected,
            received: mismatch.received,
        }
    })
}
