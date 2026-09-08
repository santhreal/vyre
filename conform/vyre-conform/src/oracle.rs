//! Conformance oracle session: evaluates a Program directly on the pure-Rust
//! reference interpreter without intermediate backend registration, precedence,
//! or materializers.

use std::sync::Arc;

use thiserror::Error;
use vyre_foundation::fp_parity::FloatLoweringMode;
use vyre_foundation::ir::{BufferAccess, Program};
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
    #[error("oracle cannot lower strict IEEE float mode: {0}")]
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
        let expanded = if self.float_lowering.blocks_contraction() {
            vyre_foundation::fp_expansion::expand_strict_transcendentals(&self.program)
                .map_err(|error| OracleError::FloatExpansion(error.to_string()))?
        } else {
            None
        };
        let program = expanded.as_ref().unwrap_or(&self.program);
        let values = reference_values(program, inputs)?;
        let outputs = vyre_reference::reference_eval(program, &values)
            .map_err(|error| OracleError::Evaluation(error.to_string()))?;

        Ok(outputs.into_iter().map(|v| v.to_bytes()).collect())
    }
}

fn reference_values(program: &Program, inputs: &[&[u8]]) -> Result<Vec<Value>, OracleError> {
    let mut next_input = 0usize;
    let mut values = Vec::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        if buffer.is_backend_allocated_output() {
            continue;
        }
        let input = inputs.get(next_input).ok_or_else(|| {
            OracleError::InputBufferCountMismatch {
                expected: program
                    .buffers()
                    .iter()
                    .filter(|b| {
                        b.access() != BufferAccess::Workgroup && !b.is_backend_allocated_output()
                    })
                    .count(),
                received: inputs.len(),
            }
        })?;
        next_input += 1;
        values.push(Value::from(*input));
    }
    if next_input != inputs.len() {
        return Err(OracleError::InputBufferCountMismatch {
            expected: next_input,
            received: inputs.len(),
        });
    }
    Ok(values)
}
