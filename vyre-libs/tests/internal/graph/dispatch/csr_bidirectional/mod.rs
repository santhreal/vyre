use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::test_parity_oracles::{policy, NeverDispatches, StaticOutputs};
use vyre_megakernel::SemanticExecutionError;

const BIDIRECTIONAL_CONTRACT: &str = "bidirectional step semantic execution";

fn linear_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![u32::MAX; 3])
}

#[test]
fn via_step_decodes_exact_output_into_reused_buffer() {
    let executor = StaticOutputs::new(
        BIDIRECTIONAL_CONTRACT,
        vec![u32_slice_to_le_bytes(&[0b0110])],
    )
    .expecting_inputs(&[7]);
    let (offsets, targets, kinds) = linear_graph();
    let mut out = Vec::with_capacity(4);
    let pointer = out.as_ptr();

    bidirectional_step_via_into(
        &executor,
        &policy(),
        4,
        &offsets,
        &targets,
        &kinds,
        &[0b0001],
        u32::MAX,
        &mut out,
    )
    .expect("Fix: a valid semantic bidirectional step must execute");

    assert_eq!(out, vec![0b0110]);
    assert_eq!(out.as_ptr(), pointer);
}

#[test]
fn via_step_with_scratch_reuses_storage_and_refreshes_graph_content() {
    let executor = StaticOutputs::new(
        BIDIRECTIONAL_CONTRACT,
        vec![u32_slice_to_le_bytes(&[0b0110])],
    )
    .expecting_inputs(&[7])
    .recording_input(2);
    let (offsets, targets, kinds) = linear_graph();
    let mut scratch = BidirectionalGpuScratch::default();
    let mut out = Vec::new();

    bidirectional_step_via_with_scratch_into(
        &executor,
        &policy(),
        4,
        &offsets,
        &targets,
        &kinds,
        &[0b0001],
        u32::MAX,
        &mut scratch,
        &mut out,
    )
    .expect("Fix: first semantic bidirectional step must execute");
    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    assert_eq!(scratch.program_builds(), 1);

    let changed_targets = [2, 3, 0];
    bidirectional_step_via_with_scratch_into(
        &executor,
        &policy(),
        4,
        &offsets,
        &changed_targets,
        &kinds,
        &[0b0010],
        u32::MAX,
        &mut scratch,
        &mut out,
    )
    .expect("Fix: same-shape graph content must refresh semantic inputs");

    assert_eq!(scratch.program_builds(), 1);
    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(executor.recorded(), vec![targets, changed_targets.to_vec()]);
}

#[test]
fn malformed_graph_is_rejected_before_semantic_execution() {
    let executor = NeverDispatches("Fix: mismatched bidirectional edge arrays must not execute");
    let error = bidirectional_step_via(
        &executor,
        &policy(),
        4,
        &[0, 1, 2, 3, 3],
        &[1, 2, 3],
        &[u32::MAX; 2],
        &[0b0001],
        u32::MAX,
    )
    .expect_err("mismatched edge arrays must be rejected");

    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));
}

#[test]
fn empty_graph_validates_without_semantic_execution() {
    let executor = NeverDispatches("Fix: empty bidirectional graph must not execute");
    let output = bidirectional_step_via(&executor, &policy(), 0, &[0], &[], &[], &[], u32::MAX)
        .expect("Fix: an empty graph is a valid no-work query");

    assert!(output.is_empty());
}

#[test]
fn malformed_semantic_outputs_are_rejected() {
    let (offsets, targets, kinds) = linear_graph();
    let extra = StaticOutputs::new(
        BIDIRECTIONAL_CONTRACT,
        vec![u32_slice_to_le_bytes(&[0]), u32_slice_to_le_bytes(&[0])],
    );
    let extra_error = bidirectional_step_via(
        &extra,
        &policy(),
        4,
        &offsets,
        &targets,
        &kinds,
        &[1],
        u32::MAX,
    )
    .expect_err("an extra graph output must be rejected");
    assert!(matches!(extra_error, SemanticExecutionError::Backend(_)));

    let trailing = StaticOutputs::new(BIDIRECTIONAL_CONTRACT, vec![vec![0, 0, 0, 0, 1]]);
    let trailing_error = bidirectional_step_via(
        &trailing,
        &policy(),
        4,
        &offsets,
        &targets,
        &kinds,
        &[1],
        u32::MAX,
    )
    .expect_err("trailing output bytes must be rejected");
    assert!(matches!(trailing_error, SemanticExecutionError::Backend(_)));
}
