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
                chunk.try_into().expect(
                    "Fix: four-byte chunk is a u32 word; state a buffer that holds the word.",
                ),
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
pub fn eval_bytes(label: &str, program: &Program, buffers: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
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
    Ok(vyre_reference::ReferenceRequest::standard(program, &values)
        .outputs()?
        .iter()
        .map(|value| value.to_bytes())
        .collect())
}

/// Every registration checked against u32 bytes is a hash, linalg, pattern-dfa
/// or representation op.
pub fn u32_bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

/// Every registration checked against f32 bytes is a conv, fft, weighted-sum,
/// strassen or fused-activation op.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

/// Decode an entire slice of little-endian bytes into f32 words.
pub fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    vyre_primitives::wire::decode_f32_le_bytes_all(bytes)
}

/// Decode the first f32 value from little-endian bytes.
pub fn decode_f32_one(bytes: &[u8]) -> f32 {
    match try_decode_f32_one(bytes) {
        Ok(value) => value,
        Err(_) => f32::NAN,
    }
}

/// Try to decode the first f32 value from little-endian bytes.
pub fn try_decode_f32_one(bytes: &[u8]) -> Result<f32, String> {
    vyre_primitives::wire::read_f32_le_word(bytes, 0, "f32 scalar fixture output")
}

/// Decode the first u32 value from little-endian bytes.
pub fn decode_u32_one(bytes: &[u8]) -> u32 {
    match try_decode_u32_one(bytes) {
        Ok(value) => value,
        Err(_) => u32::MAX,
    }
}

/// Try to decode the first u32 value from little-endian bytes.
pub fn try_decode_u32_one(bytes: &[u8]) -> Result<u32, String> {
    vyre_primitives::wire::read_u32_le_word(bytes, 0, "u32 scalar fixture output")
}

/// Decode an entire slice of little-endian bytes into u32 words.
pub fn bytes_to_u32(slice: &[u8]) -> Vec<u32> {
    vyre_primitives::wire::decode_u32_le_bytes_all(slice)
}

/// The `(pattern_id, origin, end)` records a pattern program emitted, sorted.
///
/// `outputs[0]` holds the single-word match count and `outputs[1]` the record
/// buffer, three words per record. Every suite comparing two pattern programs
/// decodes that pair, and each wrote the decode out: a copy that dropped the
/// sort compares emission order instead of the match set, which is the one
/// thing a coalesced emit is allowed to change.
#[must_use]
pub fn sorted_match_triples(outputs: &[Vec<u8>]) -> Vec<(u32, u32, u32)> {
    let count = bytes_to_u32(&outputs[0])[0] as usize;
    let words = bytes_to_u32(&outputs[1]);
    let mut decoded: Vec<(u32, u32, u32)> = words[..count.saturating_mul(3)]
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();
    decoded.sort_unstable();
    decoded
}

/// Run a program and hand back its buffers, refusing an access that left a
/// declared buffer.
///
/// The strict oracle answers an out-of-bounds access with a structured
/// refusal, so a program that only "works" because the interpreter absorbed
/// the access never reaches an output here.
///
/// # Panics
/// Panics when the program does not evaluate, including when it accesses a
/// buffer out of bounds.
pub fn eval_bytes_in_bounds(label: &str, program: &Program, buffers: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let values = vyre_reference::reference_inputs(program, buffers);
    vyre_reference::ReferenceRequest::standard(program, &values)
        .outputs()
        .unwrap_or_else(|error| {
            panic!(
                "Fix: {label} program must execute in the reference interpreter without leaving \
                 a declared buffer: {error:?}"
            )
        })
        .iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Run a program with the interpreter's lanes in declaration order, or
/// reversed.
pub fn eval_bytes_lane_order(
    label: &str,
    program: &Program,
    buffers: Vec<Vec<u8>>,
    reversed: bool,
) -> Vec<Vec<u8>> {
    let values = vyre_reference::reference_inputs(program, buffers);
    let results = if reversed {
        vyre_reference::ReferenceRequest::standard(program, &values)
            .with_schedule_policy(vyre_reference::DeterministicSchedulePolicy::LaneReversed)
            .outputs()
    } else {
        vyre_reference::ReferenceRequest::standard(program, &values).outputs()
    };
    results
        .unwrap_or_else(|error| {
            panic!("Fix: {label} program must execute in the reference interpreter: {error:?}")
        })
        .iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Run a program whose arguments and one output are all f32.
pub fn eval_f32(label: &str, program: &Program, inputs: &[&[f32]], output_len: usize) -> Vec<f32> {
    let mut buffers: Vec<Vec<u8>> = inputs.iter().map(|input| f32_bytes(input)).collect();
    buffers.push(vec![0u8; output_len * 4]);
    decode_f32(&eval_bytes(label, program, buffers)[0])
}

/// Run a program whose arguments and one output are all u32.
pub fn eval_u32(label: &str, program: &Program, inputs: &[&[u32]], output_len: usize) -> Vec<u32> {
    let mut buffers: Vec<Vec<u8>> = inputs.iter().map(|input| u32_bytes(input)).collect();
    buffers.push(vec![0u8; output_len * 4]);
    bytes_to_u32(&eval_bytes(label, program, buffers)[0])
}

/// Run a one-in one-out f32 program through the reference interpreter.
pub fn eval_f32_unary(label: &str, input: &[f32], program: &Program) -> Vec<f32> {
    eval_f32(label, program, &[input], input.len())
}

/// Compare a tiled f32 program against its scalar reference, lane by lane.
pub fn assert_tiled_matches_reference(
    label: &str,
    input: &[f32],
    tolerance: f32,
    tiled: &Program,
    reference: &Program,
) {
    let actual = eval_f32_unary(label, input, tiled);
    let expected = eval_f32_unary(label, input, reference);
    assert_eq!(
        actual.len(),
        expected.len(),
        "Fix: {label} must write the same lane count as its reference."
    );
    for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= tolerance,
            "{label} mismatch at lane {idx}: tiled={lhs:?} reference={rhs:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        let original = vec![1, 2, 3, 0xFFFFFFFF, 0x12345678];
        let bytes = u32_bytes(&original);
        let back = bytes_to_u32(&bytes);
        assert_eq!(original, back);
    }

    #[test]
    fn test_empty_input() {
        let original: Vec<u32> = vec![];
        let bytes = u32_bytes(&original);
        assert!(bytes.is_empty());
        let back = bytes_to_u32(&bytes);
        assert!(back.is_empty());
    }

    #[test]
    fn test_f32_bit_exact_pack() {
        let bytes = f32_bytes(&[1.0, -0.0, f32::INFINITY, f32::NAN]);
        let unpacked =
            vyre_primitives::wire::unpack_f32_slice(&bytes, 4, "test_f32_bit_exact_pack")
                .expect("Fix: f32 test fixture pack must round-trip.");
        assert_eq!(unpacked[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(unpacked[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(unpacked[2].to_bits(), f32::INFINITY.to_bits());
        assert!(unpacked[3].is_nan());
    }
}
