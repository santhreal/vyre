//! Dead-code elimination through semantic execution of the reachability kernel.

use vyre_foundation::ir::Program;
use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::persistent_bfs::validate_persistent_bfs_converged_flag;
use vyre_libs::graph::program_graph::ProgramGraphShape;

use vyre_libs::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};

use super::dce_program::build_dce_bfs_program;
use super::encode::{apply_live_mask, encode_program, EncodeError, EncodedProgram, ROOT_GRAPH_ID};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

#[derive(Debug, Default)]
struct DceKernelScratch {
    inputs: Vec<Vec<u8>>,
    seed: Vec<u32>,
    frontier: Vec<u32>,
    changed: Vec<u32>,
    converged: Vec<u32>,
}

/// Errors surfaced by semantic DCE execution.
#[derive(Debug)]
pub enum DceError {
    /// Encoder did not accept the input shape.
    Encode(EncodeError),
    /// Semantic execution or canonical output decoding failed.
    Semantic(SemanticExecutionError),
}

impl std::fmt::Display for DceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_dce encode error: {err:?}"),
            Self::Semantic(err) => write!(f, "gpu_dce semantic execution error: {err}"),
        }
    }
}

impl std::error::Error for DceError {}

/// Run DCE through one semantic execution of the reachability kernel.
pub fn gpu_dce(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<Program, DceError> {
    let encoded = encode_program(&program).map_err(DceError::Encode)?;
    let mut scratch = DceKernelScratch::default();
    let mut live = Vec::with_capacity(encoded.node_count as usize);
    compute_live_mask_with_scratch_into(&encoded, executor, policy, &mut scratch, &mut live)
        .map_err(DceError::Semantic)?;
    Ok(apply_live_mask(&program, &encoded, &live))
}

fn compute_live_mask_with_scratch_into(
    encoded: &EncodedProgram,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    scratch: &mut DceKernelScratch,
    live: &mut Vec<bool>,
) -> Result<(), SemanticExecutionError> {
    let n = encoded.node_count;
    if n == 0 {
        live.clear();
        return Ok(());
    }

    // Build the DCE analysis Program for this exact graph shape. Buffer
    // names + binding indices match the persistent BFS layout, including the
    // converged word.
    let shape = ProgramGraphShape::new(encoded.node_count, encoded.edge_count);
    let program = build_dce_bfs_program(shape, n.max(1));

    let words = bitset_words(n) as usize;
    scratch.seed.clear();
    scratch.seed.resize(words.max(1), 0);
    let root = ROOT_GRAPH_ID as usize;
    scratch.seed[root / 32] |= 1u32 << (root % 32);

    ensure_input_slots(&mut scratch.inputs, 9);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], &encoded.nodes);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &encoded.edge_offsets);
    write_padded_one_u32_bytes(&mut scratch.inputs[2], &encoded.edge_targets);
    write_padded_one_u32_bytes(&mut scratch.inputs[3], &encoded.edge_kind_mask);
    write_u32_slice_le_bytes(&mut scratch.inputs[4], &encoded.node_tags);
    write_u32_slice_le_bytes(&mut scratch.inputs[5], &scratch.seed);
    write_zero_bytes(
        &mut scratch.inputs[6],
        words.max(1) * std::mem::size_of::<u32>(),
    );
    write_zero_bytes(&mut scratch.inputs[7], std::mem::size_of::<u32>());
    write_zero_bytes(&mut scratch.inputs[8], std::mem::size_of::<u32>());

    let mut outputs =
        super::execute_retained_program(executor, policy, "dce", program, &scratch.inputs)?;
    if outputs.len() != 3 {
        return Err(SemanticExecutionError::Backend(format!(
            "dce semantic execution expected three canonical outputs, got {}",
            outputs.len()
        )));
    }
    let converged_bytes = outputs.remove(2);
    let changed_bytes = outputs.remove(1);
    let frontier_bytes = outputs.remove(0);
    decode_u32_output_exact(
        &frontier_bytes,
        words,
        "gpu_dce frontier_out",
        &mut scratch.frontier,
    )
    .map_err(|error| SemanticExecutionError::Backend(format!("dce frontier output: {error}")))?;
    decode_u32_output_exact(&changed_bytes, 1, "gpu_dce changed", &mut scratch.changed)
        .map_err(|error| SemanticExecutionError::Backend(format!("dce changed output: {error}")))?;
    decode_u32_output_exact(
        &converged_bytes,
        1,
        "gpu_dce converged",
        &mut scratch.converged,
    )
    .map_err(|error| SemanticExecutionError::Backend(format!("dce converged output: {error}")))?;
    let converged = scratch.converged.first().copied().unwrap_or_default();
    validate_persistent_bfs_converged_flag(converged)
        .map_err(|reason| SemanticExecutionError::Backend(format!("gpu_dce {reason}")))?;
    if converged != 1 {
        return Err(SemanticExecutionError::Backend(format!(
            "gpu_dce liveness closure did not converge within its {} iteration budget for a {}-node graph",
            n.max(1),
            n
        )));
    }

    live.clear();
    live.resize(n as usize, false);
    for graph_id in 0..(n as usize) {
        let word = scratch.frontier.get(graph_id / 32).copied().unwrap_or(0);
        if word & (1u32 << (graph_id % 32)) != 0 {
            live[graph_id] = true;
        }
    }
    Ok(())
}

fn write_padded_one_u32_bytes(out: &mut Vec<u8>, buf: &[u32]) {
    if buf.is_empty() {
        write_zero_bytes(out, std::mem::size_of::<u32>());
    } else {
        write_u32_slice_le_bytes(out, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vyre_foundation::ir::{Expr, GraphValueId, Node, Program};
    use vyre_libs::bitset::bitset_words;
    use vyre_libs::dispatch_buffers::u32_slice_to_le_bytes;
    use vyre_megakernel::{
        Digest, SemanticExecutionOutput, SemanticExecutionRequest, SemanticExecutor,
    };

    use super::super::arena_kernel::semantic_test_policy;

    fn wrapped_program(entry: Vec<Node>) -> Program {
        Program::wrapped(Vec::new(), [1, 1, 1], entry)
    }

    struct MalformedExecutor {
        outputs: Vec<Vec<u8>>,
        extra_output: bool,
        omit_output: bool,
    }

    impl MalformedExecutor {
        fn new(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                extra_output: false,
                omit_output: false,
            }
        }

        fn with_extra_output(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                extra_output: true,
                omit_output: false,
            }
        }

        fn with_omitted_output(outputs: Vec<Vec<u8>>) -> Self {
            Self {
                outputs,
                extra_output: false,
                omit_output: true,
            }
        }
    }

    impl SemanticExecutor for MalformedExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            let node = &request.logical().graph().nodes()[0];
            let written = vyre_megakernel::writable_graph_values(node);
            let mut outputs = BTreeMap::new();
            for (idx, (val, bytes)) in written.into_iter().zip(self.outputs.iter()).enumerate() {
                if self.omit_output && idx == 2 {
                    // Omit converged output
                    continue;
                }
                outputs.insert(val, bytes.clone());
            }
            if self.extra_output {
                outputs.insert(GraphValueId(u32::MAX), vec![0u8; 4]);
            }
            Ok(SemanticExecutionOutput {
                artifact: Digest([0; 32]),
                payload: Digest([1; 32]),
                outputs,
            })
        }
    }
    #[test]
    fn dce_rejects_extra_dispatch_outputs() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let executor = MalformedExecutor::with_extra_output(vec![
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[1]),
        ]);
        let err = gpu_dce(program, &executor, &semantic_test_policy())
            .expect_err("extra outputs must be rejected");
        assert!(
            matches!(
                &err,
                DceError::Semantic(SemanticExecutionError::Backend(msg))
                    if msg.contains("undeclared output values")
                        || msg.contains("expected three canonical outputs")
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn dce_rejects_a_dispatch_missing_the_converged_output() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let encoded = encode_program(&program).expect("Fix: encoder accepts store");
        let words = bitset_words(encoded.node_count) as usize;
        let executor = MalformedExecutor::with_omitted_output(vec![
            u32_slice_to_le_bytes(&vec![1; words]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[1]),
        ]);
        let err = gpu_dce(program, &executor, &semantic_test_policy())
            .expect_err("missing converged must be rejected");
        assert!(
            matches!(
                &err,
                DceError::Semantic(SemanticExecutionError::Backend(msg))
                    if msg.contains("omitted canonical output value")
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn dce_fails_closed_when_the_liveness_closure_did_not_converge() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let encoded = encode_program(&program).expect("Fix: encoder accepts store");
        let words = bitset_words(encoded.node_count) as usize;
        let executor = MalformedExecutor::new(vec![
            u32_slice_to_le_bytes(&vec![1; words]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[0]),
        ]);
        let err = gpu_dce(program, &executor, &semantic_test_policy())
            .expect_err("non-converged closure must fail");
        let DceError::Semantic(SemanticExecutionError::Backend(message)) = err else {
            panic!("expected a backend error naming the truncated closure, got {err:?}");
        };
        assert!(
            message.contains("did not converge within its"),
            "the error must say why a partial closure is unusable, got: {message}"
        );
    }

    #[test]
    fn dce_rejects_a_converged_word_that_is_neither_zero_nor_one() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let encoded = encode_program(&program).expect("Fix: encoder accepts store");
        let words = bitset_words(encoded.node_count) as usize;
        let executor = MalformedExecutor::new(vec![
            u32_slice_to_le_bytes(&vec![1; words]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[7]),
        ]);
        let err = gpu_dce(program, &executor, &semantic_test_policy())
            .expect_err("a non-boolean converged is rejected");
        assert!(
            matches!(
                &err,
                DceError::Semantic(SemanticExecutionError::Backend(msg))
                    if msg.contains("converged flag readback must be 0 or 1, got 7")
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn dce_rejects_trailing_changed_bytes() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let encoded = encode_program(&program).expect("Fix: encoder accepts store");
        let words = bitset_words(encoded.node_count) as usize;
        let executor = MalformedExecutor::new(vec![
            u32_slice_to_le_bytes(&vec![1; words]),
            vec![0, 0, 0, 0, 1],
            u32_slice_to_le_bytes(&[1]),
        ]);
        let err = gpu_dce(program, &executor, &semantic_test_policy())
            .expect_err("trailing changed bytes rejected");
        assert!(
            matches!(
                &err,
                DceError::Semantic(SemanticExecutionError::Backend(msg))
                    if msg.contains("dce changed output") && msg.contains("expected 4 output bytes")
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn live_mask_with_scratch_reuses_dispatch_decode_and_output_storage() {
        let program = wrapped_program(vec![Node::store("buf", Expr::u32(0), Expr::u32(1))]);
        let encoded = encode_program(&program).expect("Fix: encoder accepts store");
        let words = bitset_words(encoded.node_count) as usize;
        let executor = MalformedExecutor::new(vec![
            u32_slice_to_le_bytes(&vec![u32::MAX; words]),
            vec![0, 0, 0, 0],
            u32_slice_to_le_bytes(&[1]),
        ]);
        let mut scratch = DceKernelScratch::default();
        let mut live = Vec::with_capacity(encoded.node_count as usize);

        compute_live_mask_with_scratch_into(
            &encoded,
            &executor,
            &semantic_test_policy(),
            &mut scratch,
            &mut live,
        )
        .expect("Fix: dispatch succeeds");

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let seed_capacity = scratch.seed.capacity();
        let frontier_capacity = scratch.frontier.capacity();
        let changed_capacity = scratch.changed.capacity();
        let converged_capacity = scratch.converged.capacity();
        let live_capacity = live.capacity();

        compute_live_mask_with_scratch_into(
            &encoded,
            &executor,
            &semantic_test_policy(),
            &mut scratch,
            &mut live,
        )
        .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(scratch.seed.capacity(), seed_capacity);
        assert_eq!(scratch.frontier.capacity(), frontier_capacity);
        assert_eq!(scratch.changed.capacity(), changed_capacity);
        assert_eq!(scratch.converged.capacity(), converged_capacity);
        assert_eq!(live.capacity(), live_capacity);
        assert!(live.iter().all(|&is_live| is_live));
    }
}
