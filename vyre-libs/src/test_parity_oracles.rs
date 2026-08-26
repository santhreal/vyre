//! Dispatcher doubles and program sequencing for this crate's own unit tests.
//!
//! The 16.16 oracles that used to live here are arithmetic with no dependency
//! on this crate, and their only callers are integration suites. They are owned
//! by `vyre_test_support::fixed_point`, a dev-dependency, so a library consumer
//! no longer sees test tooling in this crate's published surface.

use std::collections::BTreeMap;

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionOutput, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};

pub(crate) fn policy() -> SemanticExecutionPolicy {
    SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([3; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}

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

impl SemanticExecutor for NeverDispatches {
    fn execute(
        &self,
        _request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        panic!("{}", self.0);
    }
}

/// A semantic executor that returns fixed output buffers.
///
/// `contract` states the logical input contract represented by the double.
pub(crate) struct StaticOutputs {
    contract: &'static str,
    outputs: Vec<Vec<u8>>,
    expect_inputs: &'static [usize],
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

impl SemanticExecutor for StaticOutputs {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = canonical_inputs(request)?;
        if !self.expect_inputs.is_empty() && !self.expect_inputs.contains(&inputs.len()) {
            let expected = self
                .expect_inputs
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" or ");
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: {} expected {expected} semantic inputs, got {}.",
                self.contract,
                inputs.len()
            )));
        }
        if let Some((index, bytes)) = self.expect_input_bytes {
            if inputs[index].len() != bytes {
                return Err(SemanticExecutionError::InvalidRequest(format!(
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
        semantic_output(request, self.outputs.clone())
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

impl SemanticExecutor for SequentialOutputs {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let mut guard = self
            .steps
            .lock()
            .expect("Fix: sequential-output dispatcher mutex should not be poisoned");
        if guard.is_empty() {
            return Err(SemanticExecutionError::Backend(format!(
                "{}: sequential executor ran out of expected steps",
                self.contract
            )));
        }
        semantic_output(request, guard.remove(0))
    }
}

pub(crate) fn canonical_inputs(
    request: &SemanticExecutionRequest<'_>,
) -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
    let graph = request.logical().graph();
    let node = graph.nodes().first().ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "Fix: semantic test executor requires one graph node.".to_string(),
        )
    })?;
    node.inputs
        .iter()
        .map(|port| {
            request
                .inputs()
                .get(&port.value)
                .map(|bytes| bytes.to_vec())
                .ok_or_else(|| {
                    SemanticExecutionError::InvalidRequest(format!(
                        "Fix: semantic test executor missing input graph value {}.",
                        port.value.0
                    ))
                })
        })
        .collect()
}

/// Map a positional output list onto the graph values every node writes.
///
/// A backend returns one buffer per writable graph value, so a test double owes
/// the same set. The list is positional in Program buffer declaration order and
/// repeats for each node, which is what a per-iteration oracle produced before
/// a loop became one multi-node graph. Any writable value the list does not
/// reach is empty, so a wrapper that reads it fails on decode instead of
/// reading plausible bytes. Supplying more buffers than a node writes is
/// rejected, which is what an executor returning an undeclared output looks
/// like here.
///
/// A program that declares read-write working storage writes buffers a wrapper
/// never reads, and their declaration order is not the wrapper's. Use
/// [`semantic_output_named`] there.
pub(crate) fn semantic_output(
    request: &SemanticExecutionRequest<'_>,
    ordered: Vec<Vec<u8>>,
) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
    let graph = request.logical().graph();
    if graph.nodes().is_empty() {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: semantic test executor requires one graph node.".to_string(),
        ));
    }
    let mut outputs = BTreeMap::new();
    for node in graph.nodes() {
        let written = vyre_megakernel::writable_graph_values(node);
        if ordered.len() > written.len() {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: semantic test executor returned {} output buffers for {} written graph values.",
                ordered.len(),
                written.len()
            )));
        }
        let mut supplied = ordered.iter();
        for value in written {
            outputs.insert(value, supplied.next().cloned().unwrap_or_default());
        }
    }
    Ok(SemanticExecutionOutput {
        artifact: Digest([1; 32]),
        payload: Digest([2; 32]),
        outputs,
    })
}

/// Map named Program buffers onto the graph values every node writes.
///
/// Every writable value the caller does not name is empty, which is what a
/// backend leaves in read-write working storage a wrapper never reads. A name
/// the program does not write is rejected: it is a stale test, not a backend
/// that returned too much.
pub(crate) fn semantic_output_named(
    request: &SemanticExecutionRequest<'_>,
    named: Vec<(&str, Vec<u8>)>,
) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
    let graph = request.logical().graph();
    if graph.nodes().is_empty() {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: semantic test executor requires one graph node.".to_string(),
        ));
    }
    let mut outputs = BTreeMap::new();
    let mut matched = vec![false; named.len()];
    for node in graph.nodes() {
        for (value, buffer) in vyre_megakernel::writable_graph_value_buffers(node) {
            let supplied = named
                .iter()
                .zip(matched.iter_mut())
                .find(|((name, _), _)| *name == buffer.as_str());
            match supplied {
                Some(((_, bytes), seen)) => {
                    *seen = true;
                    outputs.insert(value, bytes.clone());
                }
                None => {
                    outputs.insert(value, Vec::new());
                }
            }
        }
    }
    if let Some((name, _)) = named
        .iter()
        .zip(matched)
        .find_map(|(entry, seen)| (!seen).then_some(entry))
    {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {name} is not a written Program buffer in this graph."
        )));
    }
    Ok(SemanticExecutionOutput {
        artifact: Digest([1; 32]),
        payload: Digest([2; 32]),
        outputs,
    })
}
