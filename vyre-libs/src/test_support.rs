//! Shared parity-suite helpers: fixed-point oracles and the reference dispatcher.
//!
//! Compiled in two contexts, matching `scan::test_fixtures`:
//!
//! 1. `#[cfg(test)]` - always available to in-tree tests.
//! 2. `feature = "test-fixtures"` - exported to crates whose own parity
//!    suites dispatch programs built from this crate.
//!
//! ONE PLACE: every `_via` end-to-end parity test in the workspace uses these
//! rather than re-deriving the fixed-point contract or the buffer-bridging
//! rule. Seven per-file copies of the 16.16 multiply drifted before this was
//! consolidated, and six of them silently kept an unsigned form after the
//! kernel was corrected to signed.

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

pub use vyre_primitives::wire::pack_u32_slice as u32_bytes;

/// Concatenate `programs` into one program with a shared workgroup size.
///
/// Buffers and entry nodes are appended in argument order, which is the
/// order a multi-stage parity suite dispatches them in.
#[must_use]
pub fn wrap_program_sequence(programs: &[&Program], workgroup_size: [u32; 3]) -> Program {
    let buffer_count = programs.iter().map(|program| program.buffers().len()).sum();
    let entry_count = programs.iter().map(|program| program.entry().len()).sum();
    let mut buffers = Vec::with_capacity(buffer_count);
    let mut entry = Vec::with_capacity(entry_count);

    for program in programs {
        buffers.extend_from_slice(program.buffers());
        entry.extend_from_slice(program.entry());
    }

    Program::wrapped(buffers, workgroup_size, entry)
}

/// Signed 16.16 fixed-point multiply: bits `[16..48]` of the signed 64-bit
/// product, identical to the IR's `crate::math::fixed::fixed_mul_16_16_expr`.
///
/// Operands are two's-complement `i32` carried in a `u32`. A weighted-Jacobi
/// residual, a sheaf coupling, and a gradient are all routinely negative, so
/// the multiply must be signed; the unsigned form silently corrupts negative
/// operands. For non-negative operands it is bit-identical to the unsigned
/// form.
#[must_use]
pub fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// Multiply a square 16.16 matrix by a 16.16 vector with wrapping accumulation.
#[must_use]
pub fn fixed_matvec(matrix: &[u32], vector: &[u32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|row| {
            let mut acc = 0u32;
            for column in 0..n {
                acc = acc.wrapping_add(fixed_mul(matrix[row * n + column], vector[column]));
            }
            acc
        })
        .collect()
}

/// Advance the deterministic xorshift32 generator used by parity sweeps.
pub fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn signed_fixed_with_mask(state: &mut u32, magnitude_mask: u32) -> u32 {
    let magnitude = (xorshift32(state) & magnitude_mask) as i32;
    if xorshift32(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Generate a signed 16.16 sample in approximately `[-8, 8)`.
pub fn signed_fixed_19(state: &mut u32) -> u32 {
    signed_fixed_with_mask(state, 0x0007_FFFF)
}

/// Generate a signed 16.16 sample in approximately `[-2, 2)`.
pub fn signed_fixed_17(state: &mut u32) -> u32 {
    signed_fixed_with_mask(state, 0x0001_FFFF)
}

/// Signed integer division by a known-positive divisor, truncating toward
/// zero, identical to the IR's `crate::math::fixed::fixed_sdiv_by_positive_expr`.
///
/// Mirrors the fixed weighted-Jacobi `delta` divide, whose numerator is
/// negative whenever the residual is negative.
#[must_use]
pub fn fixed_sdiv_by_positive(numerator: u32, denominator: u32) -> u32 {
    ((numerator as i32) / (denominator as i32)) as u32
}

/// The one `ProgramDispatcher` that actually executes a vyre `Program`.
///
/// Backing the dispatch boundary with `vyre_reference::reference_eval` lets
/// every `*_via` production entry point be tested end to end against its `_cpu`
/// oracle without a GPU backend. Hand-written per-op CPU oracles recognize only
/// the programs they were written for; this one runs any program.
///
/// The bridge models the real backend's dispatch-input contract exactly, so a
/// `_via` consumer that passes here also runs correctly on hardware. The
/// canonical mapping is `vyre-driver`'s `role_for_buffer` /
/// [`vyre_foundation::ir::BufferDecl::is_backend_allocated_output`]: a buffer is
/// backend-allocated (the backend creates it, no dispatch input) only when it is
/// `is_output` / `WriteOnly` / `pipeline_live_out && ReadWrite`. Every other
/// non-workgroup buffer - `ReadOnly` (role `Input`), plain `ReadWrite` (role
/// `InputOutput`, whose zero or initial contents the caller supplies), and
/// `Uniform` - consumes one dispatch input, in buffer order. The real backend
/// validates this strictly (`inputs.len() == input_indices.len()`), so a
/// consumer must pass one `Vec<u8>` per input-consuming buffer in buffer order,
/// zero-filled for plain-`ReadWrite` outputs. `reference_eval` has the same
/// requirement, so this bridge forwards each input-consuming buffer's dispatch
/// bytes straight through. No zero-synthesis, which would silently diverge from
/// the backend when a plain-`ReadWrite` buffer precedes a `ReadOnly` input. The
/// returned values are the writable buffers in binding order, matching the
/// dispatch contract.
pub struct ReferenceEvalDispatcher;

impl ProgramDispatcher for ReferenceEvalDispatcher {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut values: Vec<vyre_reference::value::Value> = Vec::new();
        let mut next_input = inputs.iter();
        for buffer in program.buffers() {
            if buffer.access() == BufferAccess::Workgroup {
                continue;
            }
            // Backend-allocated outputs (is_output / WriteOnly / pipeline_live_out&&RW) are created
            // by the backend and consume NO dispatch input, mirroring `role_for_buffer`.
            if buffer.is_backend_allocated_output() {
                continue;
            }
            // Every remaining buffer is input-consuming per `role_for_buffer`: ReadOnly (Input),
            // plain ReadWrite (InputOutput, whose zero/initial contents come from the caller),
            // Uniform. Take the next dispatch input in buffer order, as the real backend does.
            let bytes = next_input.next().ok_or_else(|| {
                DispatchError::BadInputs(format!(
                    "ReferenceEvalDispatcher: program declares more input-consuming buffers than the \
                     {} dispatch inputs provided (at buffer `{}`). The backend requires one input per \
                     {{ReadOnly, plain-ReadWrite, Uniform}} buffer in buffer order; pass a zero-filled \
                     slot for each plain-ReadWrite output.",
                    inputs.len(),
                    buffer.name()
                ))
            })?;
            values.push(vyre_reference::value::Value::from(bytes.clone()));
        }
        // Faithful to the backend's strict count validation: reject leftover inputs so an
        // over-feeding consumer is caught here, not silently on hardware.
        if next_input.next().is_some() {
            return Err(DispatchError::BadInputs(format!(
                "ReferenceEvalDispatcher: {} dispatch inputs provided but the program has fewer \
                 input-consuming buffers. The backend requires exactly one input per {{ReadOnly, \
                 plain-ReadWrite, Uniform}} buffer; do not pass slots for backend-allocated outputs.",
                inputs.len()
            )));
        }
        let outputs = vyre_reference::reference_eval(program, &values).map_err(|err| {
            DispatchError::BackendError(format!(
                "ReferenceEvalDispatcher: reference_eval failed. {err}"
            ))
        })?;
        Ok(outputs
            .iter()
            .map(vyre_reference::value::Value::to_bytes)
            .collect())
    }
}

/// A dispatcher whose being called is the test failure.
///
/// Reaching the backend at all is what a reject-before-dispatch, short-circuit, or cache-hit
/// contract forbids, so the assertion has to live in `dispatch` rather than after the call. The
/// message names the contract that was supposed to stop first.
#[cfg(test)]
pub(crate) struct NeverDispatches(pub(crate) &'static str);

#[cfg(test)]
impl ProgramDispatcher for NeverDispatches {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        panic!("{}", self.0);
    }
}

/// A dispatcher that returns fixed output buffers and states its dispatch-side
/// contract as data.
///
/// Fourteen structs across the graph-dispatch, encoding, analysis and solver
/// suites carried this one shape. Each restated the same three checks by hand:
/// an expected grid override, an expected dispatch-input count with a `Fix:`
/// message, and one input buffer recorded as `u32`s for a later assertion. The
/// copies had already drifted - one recorded through a `RefCell` while its
/// siblings used a `Mutex`, so the same double was `Sync` in some suites and
/// not in others, and two spelled the same input-count rejection with different
/// wording.
///
/// `contract` names what the dispatcher stands in for, so a failure reads as
/// the contract that was violated rather than as an anonymous double.
#[cfg(test)]
pub(crate) struct StaticOutputs {
    contract: &'static str,
    outputs: Vec<Vec<u8>>,
    expect_inputs: &'static [usize],
    expect_grid: Option<[u32; 3]>,
    expect_input_bytes: Option<(usize, usize)>,
    record_input: Option<usize>,
    recorded: std::sync::Mutex<Vec<Vec<u32>>>,
}

#[cfg(test)]
impl StaticOutputs {
    /// Returns `outputs` from every dispatch, checking nothing.
    pub(crate) fn new(contract: &'static str, outputs: Vec<Vec<u8>>) -> Self {
        Self {
            contract,
            outputs,
            expect_inputs: &[],
            expect_grid: None,
            expect_input_bytes: None,
            record_input: None,
            recorded: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Rejects a dispatch whose input count is not one of `counts`.
    ///
    /// More than one count is a real contract: a builder that grew an optional
    /// buffer accepts both the shape with it and the shape without.
    pub(crate) fn expecting_inputs(mut self, counts: &'static [usize]) -> Self {
        self.expect_inputs = counts;
        self
    }

    /// Asserts the planner passed `grid` as the grid override.
    pub(crate) fn expecting_grid(mut self, grid: [u32; 3]) -> Self {
        self.expect_grid = Some(grid);
        self
    }

    /// Rejects a dispatch whose input at `index` is not `bytes` long.
    pub(crate) fn expecting_input_bytes(mut self, index: usize, bytes: usize) -> Self {
        self.expect_input_bytes = Some((index, bytes));
        self
    }

    /// Records the input at `index`, decoded as little-endian `u32`s, once per
    /// dispatch.
    pub(crate) fn recording_input(mut self, index: usize) -> Self {
        self.record_input = Some(index);
        self
    }

    /// The recorded inputs in dispatch order.
    pub(crate) fn recorded(&self) -> Vec<Vec<u32>> {
        self.recorded
            .lock()
            .expect("Fix: static-output dispatcher recorder mutex should not be poisoned")
            .clone()
    }
}

#[cfg(test)]
impl ProgramDispatcher for StaticOutputs {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        if let Some(grid) = self.expect_grid {
            assert_eq!(
                grid_override,
                Some(grid),
                "{} must dispatch at {grid:?}",
                self.contract
            );
        }
        if !self.expect_inputs.is_empty() && !self.expect_inputs.contains(&inputs.len()) {
            let expected = self
                .expect_inputs
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" or ");
            return Err(DispatchError::BadInputs(format!(
                "Fix: {} expected {expected} dispatch inputs, got {}.",
                self.contract,
                inputs.len()
            )));
        }
        if let Some((index, bytes)) = self.expect_input_bytes {
            if inputs[index].len() != bytes {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: {} expected input {index} to be {bytes} bytes, got {}.",
                    self.contract,
                    inputs[index].len()
                )));
            }
        }
        if let Some(index) = self.record_input {
            self.recorded
                .lock()
                .expect("Fix: static-output dispatcher recorder mutex should not be poisoned")
                .push(crate::dispatch_buffers::read_u32s(&inputs[index]));
        }
        Ok(self.outputs.clone())
    }
}
