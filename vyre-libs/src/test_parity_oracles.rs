//! Dispatcher doubles and program sequencing for this crate's own unit tests.
//!
//! The 16.16 oracles that used to live here are arithmetic with no dependency
//! on this crate, and their only callers are integration suites. They are owned
//! by `vyre_test_support::fixed_point`, a dev-dependency, so a library consumer
//! no longer sees test tooling in this crate's published surface.

use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Concatenate `programs` into one program with a shared workgroup size.
///
/// Buffers and entry nodes are appended in argument order, which is the
/// order a multi-stage parity suite dispatches them in.
#[must_use]
pub(crate) fn wrap_program_sequence(programs: &[&Program], workgroup_size: [u32; 3]) -> Program {
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

/// A dispatcher whose being called is the test failure.
///
/// Reaching the backend at all is what a reject-before-dispatch, short-circuit, or cache-hit
/// contract forbids, so the assertion has to live in `dispatch` rather than after the call. The
/// message names the contract that was supposed to stop first.
pub(crate) struct NeverDispatches(pub(crate) &'static str);

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
pub(crate) struct StaticOutputs {
    contract: &'static str,
    outputs: Vec<Vec<u8>>,
    expect_inputs: &'static [usize],
    expect_grid: Option<[u32; 3]>,
    expect_input_bytes: Option<(usize, usize)>,
    record_input: Option<usize>,
    recorded: std::sync::Mutex<Vec<Vec<u32>>>,
}

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

/// A dispatcher that returns sequential output buffers across multiple dispatches.
#[allow(dead_code)]
pub(crate) struct SequentialOutputs {
    contract: &'static str,
    steps: std::sync::Mutex<Vec<Vec<Vec<u8>>>>,
}

#[allow(dead_code)]
impl SequentialOutputs {
    pub(crate) fn new(contract: &'static str, steps: Vec<Vec<Vec<u8>>>) -> Self {
        Self {
            contract,
            steps: std::sync::Mutex::new(steps),
        }
    }
}

impl ProgramDispatcher for SequentialOutputs {
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut guard = self
            .steps
            .lock()
            .expect("Fix: sequential-output dispatcher mutex should not be poisoned");
        if guard.is_empty() {
            return Err(DispatchError::BackendError(format!(
                "{}: sequential dispatcher ran out of expected steps",
                self.contract
            )));
        }
        Ok(guard.remove(0))
    }
}
