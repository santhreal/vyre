//! Dispatcher doubles, request decoders, and program sequencing for semantic parity tests.
//!
//! A parity oracle runs against an executor double to verify graph construction,
//! dispatch invariants, or output decoding without invoking a physical device backend.

use std::collections::BTreeMap;

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Digest, SearchBudget, SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionPolicy,
    SemanticExecutionRequest, SemanticExecutor,
};

/// Project `ordered` after padding it to the node's declared output count.
///
/// A parity oracle computes only the outputs the arm it took produces, and the
/// request declares how many the node has. Fifteen solver oracles each padded
/// and projected in their own copy of the same epilogue, so a change to how a
/// short result is padded reached one of them and not the rest.
pub fn semantic_output_padded(
    request: &SemanticExecutionRequest<'_>,
    mut ordered: Vec<Vec<u8>>,
) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
    let output_count = request.logical().graph().nodes()[0].outputs.len();
    if ordered.len() < output_count {
        ordered.resize(output_count, Vec::new());
    }
    semantic_output(request, ordered)
}

/// The operation id of the single region generator `program` dispatches.
///
/// Every parity oracle in this crate dispatches on it. Five copies of the walk
/// disagreed about what a program carrying no region is: two panicked, one
/// reported `None` and fell through to another operation's arm, and one returned
/// an invalid-request error. A program with no region generator is a request no
/// oracle can serve, so it is one error here.
pub fn region_operation_id(program: &Program) -> Result<&str, SemanticExecutionError> {
    program
        .entry()
        .iter()
        .find_map(|node| match node {
            vyre_foundation::ir::Node::Region { generator, .. } => Some(generator.as_str()),
            _ => None,
        })
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(
                "Fix: dispatch a primitive program whose entry carries a region generator."
                    .to_string(),
            )
        })
}

/// The default execution policy used for semantic requests in parity suites.
pub fn policy() -> SemanticExecutionPolicy {
    crate::semantic_requests::unknown_policy(
        Digest([3; 32]),
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}

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

/// A dispatcher whose being called is the test failure.
///
/// Reaching the backend at all is what a reject-before-dispatch, short-circuit, or cache-hit
/// contract forbids, so the assertion has to live in `dispatch` rather than after the call. The
/// message names the contract that was supposed to stop first.
pub struct NeverDispatches(pub &'static str);

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
pub struct StaticOutputs {
    contract: &'static str,
    outputs: Vec<Vec<u8>>,
    expect_inputs: &'static [usize],
    expect_input_bytes: Option<(usize, usize)>,
    record_input: Option<usize>,
    recorded: std::sync::Mutex<Vec<Vec<u32>>>,
}

impl StaticOutputs {
    /// Returns `outputs` from every dispatch, checking nothing.
    pub fn new(contract: &'static str, outputs: Vec<Vec<u8>>) -> Self {
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
    pub fn expecting_inputs(mut self, counts: &'static [usize]) -> Self {
        self.expect_inputs = counts;
        self
    }

    /// Rejects a dispatch whose input at `index` is not `bytes` long.
    pub fn expecting_input_bytes(mut self, index: usize, bytes: usize) -> Self {
        self.expect_input_bytes = Some((index, bytes));
        self
    }

    /// Records the input at `index`, decoded as little-endian `u32`s, once per
    /// dispatch.
    pub fn recording_input(mut self, index: usize) -> Self {
        self.record_input = Some(index);
        self
    }

    /// The recorded inputs in dispatch order.
    pub fn recorded(&self) -> Vec<Vec<u32>> {
        self.recorded
            .lock()
            .expect("Fix: static-output dispatcher recorder mutex should not be poisoned")
            .clone()
    }
}

fn read_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            u32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("Fix: four-byte chunk is a u32 word; state a buffer that holds the word."),
            )
        })
        .collect()
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
                .push(read_u32s(&inputs[index]));
        }
        semantic_output(request, self.outputs.clone())
    }
}

/// A dispatcher that returns sequential output buffers across multiple dispatches.
///
/// The one consumer is the motif dispatch suite, which the `graph` and
/// `graph-dispatch` features carry, so this is declared on the same pair. A
/// blanket `dead_code` allowance is what stood here before, and it hid the
/// unselected configurations from the only lint that reports them.
pub struct SequentialOutputs {
    contract: &'static str,
    steps: std::sync::Mutex<Vec<Vec<Vec<u8>>>>,
}

impl SequentialOutputs {
    /// Create a new sequential output dispatcher with the expected output steps.
    pub fn new(contract: &'static str, steps: Vec<Vec<Vec<u8>>>) -> Self {
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

/// Collect canonical input buffers in the order declared on the graph's first node.
pub fn canonical_inputs(
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

/// Map a positional output list onto the graph values a completion returns.
///
/// A backend returns one buffer per returned graph value, so a test double owes
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
pub fn semantic_output(
    request: &SemanticExecutionRequest<'_>,
    ordered: Vec<Vec<u8>>,
) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
    let graph = request.logical().graph();
    if graph.nodes().is_empty() {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: semantic test executor requires one graph node.".to_string(),
        ));
    }
    let returned = vyre_megakernel::returned_graph_values(graph);
    let mut outputs: BTreeMap<_, Vec<u8>> =
        returned.iter().map(|value| (*value, Vec::new())).collect();
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
            let bytes = supplied.next().cloned().unwrap_or_default();
            if let Some(slot) = outputs.get_mut(&value) {
                *slot = bytes;
            }
        }
    }
    Ok(SemanticExecutionOutput {
        artifact: Digest([1; 32]),
        payload: Digest([2; 32]),
        outputs,
    })
}

/// Map named Program buffers onto the graph values a completion returns.
///
/// Every writable value the caller does not name is empty, which is what a
/// backend leaves in read-write working storage a wrapper never reads. A name
/// the program does not write is rejected: it is a stale test, not a backend
/// that returned too much.
pub fn semantic_output_named(
    request: &SemanticExecutionRequest<'_>,
    named: Vec<(&str, Vec<u8>)>,
) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
    let graph = request.logical().graph();
    if graph.nodes().is_empty() {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: semantic test executor requires one graph node.".to_string(),
        ));
    }
    let returned = vyre_megakernel::returned_graph_values(graph);
    let mut outputs: BTreeMap<_, Vec<u8>> =
        returned.iter().map(|value| (*value, Vec::new())).collect();
    let mut matched = vec![false; named.len()];
    for node in graph.nodes() {
        for (value, buffer) in vyre_megakernel::writable_graph_value_buffers(node) {
            let supplied = named
                .iter()
                .zip(matched.iter_mut())
                .find(|((name, _), _)| *name == buffer.as_str());
            if let Some(((_, bytes), seen)) = supplied {
                *seen = true;
                if let Some(slot) = outputs.get_mut(&value) {
                    *slot = bytes.clone();
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

/// Run a program through the reference interpreter and hand back raw buffers.
///
/// Every module test that wanted a reference answer used to pack its own
/// buffers, call `reference_eval`, and decode the result, so the same eight
/// lines existed once per module under a local `run`. Two of those copies had
/// already drifted into passing a differently sized output buffer than the
/// program declared. This is the one place the call is made.
///
/// `buffers` is the complete argument list in declaration order, outputs
/// included: a zeroed vector of the right length is what the interpreter
/// writes into, and a zero-length one is a real argument, not an omission.
pub fn eval_bytes(
    label: &str,
    program: &Program,
    buffers: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    try_eval_bytes(program, buffers).unwrap_or_else(|error| {
        panic!("Fix: {label} program must execute in the reference interpreter: {error:?}")
    })
}

/// Run a program that is expected to trap, and hand back the refusal.
///
/// A trap contract asserts on the error, so it cannot go through
/// [`eval_bytes`], which panics. Both share this body so the interpreter is
/// still reached from one place.
pub fn try_eval_bytes(
    program: &Program,
    buffers: Vec<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, vyre_reference::ReferenceError> {
    let values = vyre_reference::reference_inputs(program, buffers);
    Ok(vyre_reference::reference_eval(program, &values)?
        .iter()
        .map(|value| value.to_bytes())
        .collect())
}
