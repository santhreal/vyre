use super::*;

#[test]
fn validate_zero_nodes_zero_edges_placeholder() {
    let shape = ProgramGraphShape::new(0, 0);
    // edge_count == 0 → read_only_buffers() emits count=1 placeholders
    let result = validate_program_graph(shape, &[], &[0], &[0], &[0], &[]);
    assert!(result.is_ok(), "0-node/0-edge placeholder must validate");
}

#[test]
fn validate_nonzero_nodes_zero_edges_placeholder() {
    let shape = ProgramGraphShape::new(3, 0);
    // 3 nodes, 0 edges: edge_targets & edge_kind_mask must still be length 1
    let result = validate_program_graph(
        shape,
        &[0, 0, 0],    // nodes
        &[0, 0, 0, 0], // edge_offsets (3+1 entries, all zero)
        &[0],          // edge_targets placeholder
        &[0],          // edge_kind_mask placeholder
        &[0, 0, 0],    // node_tags
    );
    assert!(
        result.is_ok(),
        ">0 nodes with 0 edges placeholder must validate"
    );
}

#[test]
fn csr_cpu_ref_zero_nodes_zero_edges() {
    let out = csr_cpu_ref(0, &[0], &[], &[], &[], 0xFFFF_FFFF);
    assert!(out.is_empty());
}

#[test]
fn csr_cpu_ref_nonzero_nodes_zero_edges() {
    let out = csr_cpu_ref(
        3,
        &[0, 0, 0, 0], // no outgoing edges for any node
        &[0],          // placeholder target
        &[0],          // placeholder mask
        &[0b0001],     // frontier on node 0
        0xFFFF_FFFF,
    );
    assert_eq!(out, vec![0], "zero edges → empty output frontier");
}

// ---------------------------------------------------------------------------
// Malformed CSR lengths
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_short_edge_offsets() {
    let shape = ProgramGraphShape::new(3, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 1, 2], // need 4 entries, got 3
        &[1, 2],
        &[1, 1],
        &[0, 0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeOffsetsLen { got: 3, .. }
    ));
}

#[test]
fn validate_rejects_long_edge_offsets() {
    let shape = ProgramGraphShape::new(2, 1);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[0, 1, 1, 1], // need 3 entries, got 4
        &[0],
        &[0],
        &[0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeOffsetsLen { got: 4, .. }
    ));
}

#[test]
fn validate_rejects_short_edge_targets() {
    let shape = ProgramGraphShape::new(2, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[0, 1, 2],
        &[1], // need 2 entries, got 1
        &[1, 1],
        &[0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeTargetsLen { got: 1, .. }
    ));
}

#[test]
fn validate_rejects_long_edge_targets() {
    let shape = ProgramGraphShape::new(2, 1);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[0, 1, 1],
        &[0, 0], // need 1 entry (max(1,1)=1), but got 2
        &[0],
        &[0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeTargetsLen { got: 2, .. }
    ));
}

#[test]
fn validate_rejects_short_edge_kind_mask() {
    let shape = ProgramGraphShape::new(2, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[0, 1, 2],
        &[1, 2],
        &[1], // need 2 entries, got 1
        &[0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeKindMaskLen { got: 1, .. }
    ));
}

#[test]
fn validate_rejects_short_nodes() {
    let shape = ProgramGraphShape::new(3, 0);
    let err = validate_program_graph(
        shape,
        &[0, 0], // need 3 entries, got 2
        &[0, 0, 0, 0],
        &[0],
        &[0],
        &[0, 0, 0],
    )
    .unwrap_err();
    assert!(matches!(err, GraphValidationError::NodesLen { got: 2, .. }));
}

#[test]
fn validate_rejects_short_node_tags() {
    let shape = ProgramGraphShape::new(3, 0);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 0, 0, 0],
        &[0],
        &[0],
        &[0, 0], // need 3 entries, got 2
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::NodeTagsLen { got: 2, .. }
    ));
}

#[test]
#[should_panic(expected = "node_count + 1 CSR offsets")]
fn csr_cpu_ref_rejects_short_edge_offsets() {
    let _ = csr_cpu_ref(3, &[0, 1], &[1, 2], &[1, 1], &[0b0001], 0xFFFF_FFFF);
}

#[test]
#[should_panic(expected = "complete CSR edge buffers")]
fn csr_cpu_ref_rejects_short_edge_targets_vs_offsets() {
    // edge_offsets says 2 edges, but edge_targets only has 1
    let _ = csr_cpu_ref(2, &[0, 1, 2], &[0], &[0, 0], &[0b0001], 0xFFFF_FFFF);
}

#[test]
#[should_panic(expected = "complete CSR edge buffers")]
fn csr_cpu_ref_rejects_short_edge_kind_mask_vs_offsets() {
    // edge_offsets says 2 edges, but edge_kind_mask only has 1
    let _ = csr_cpu_ref(2, &[0, 1, 2], &[0, 0], &[0], &[0b0001], 0xFFFF_FFFF);
}

// ---------------------------------------------------------------------------
// Non-monotonic offsets
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_first_offset_nonzero() {
    let shape = ProgramGraphShape::new(2, 1);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[1, 1, 1], // first offset must be 0
        &[0],
        &[0],
        &[0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(err, GraphValidationError::NonMonotonicOffsets { index: 0 }),
        "first offset nonzero must be rejected at index 0"
    );
}

#[test]
fn validate_rejects_strictly_decreasing_offsets() {
    let shape = ProgramGraphShape::new(3, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 3, 1, 2], // 3 → 1 is a decrease
        &[1, 2],
        &[1, 1],
        &[0, 0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(err, GraphValidationError::NonMonotonicOffsets { index: 1 }),
        "decrease at index 1 must be caught"
    );
}

#[test]
fn validate_rejects_equal_then_decrease_offsets() {
    let shape = ProgramGraphShape::new(4, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0, 0],
        &[0, 2, 2, 1, 2], // equal (ok), then decrease 2→1
        &[1, 2],
        &[1, 1],
        &[0, 0, 0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(err, GraphValidationError::NonMonotonicOffsets { index: 2 }),
        "decrease at index 2 must be caught"
    );
}

#[test]
#[should_panic(expected = "non-monotonic CSR offsets")]
fn csr_cpu_ref_rejects_non_monotonic_edge_start_gt_end() {
    let _ = csr_cpu_ref(
        2,
        &[0, 2, 1],
        &[1, 0],
        &[1, 1],
        &[0b0001], // only node 0 is in frontier
        0xFFFF_FFFF,
    );
}

// ---------------------------------------------------------------------------
// OOB targets
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_target_equal_to_node_count() {
    let shape = ProgramGraphShape::new(2, 1);
    let err = validate_program_graph(
        shape,
        &[0, 0],
        &[0, 1, 1],
        &[2], // == node_count
        &[0],
        &[0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            GraphValidationError::EdgeOutOfRange {
                target: 2,
                node_count: 2,
                ..
            }
        ),
        "target == node_count must be OOB"
    );
}

#[test]
fn validate_rejects_target_u32_max() {
    let shape = ProgramGraphShape::new(2, 1);
    let err =
        validate_program_graph(shape, &[0, 0], &[0, 1, 1], &[u32::MAX], &[0], &[0, 0]).unwrap_err();
    assert!(
        matches!(
            err,
            GraphValidationError::EdgeOutOfRange {
                target: u32::MAX,
                ..
            }
        ),
        "u32::MAX target must be OOB"
    );
}

#[test]
fn csr_cpu_ref_oob_target_equal_to_node_count_when_fits_in_word() {
    let out = csr_cpu_ref(
        2,
        &[0, 2, 2],
        &[1, 2], // target 2 == node_count, but fits in word 0
        &[1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(out, vec![0b0010], "cpu_ref must drop dst == node_count");
}

#[test]
fn csr_cpu_ref_oob_target_dropped_when_dst_word_exceeds_out_len() {
    // For node_count=2, dst=32 maps to word 1, which is >= out.len()=1, so dropped.
    let out = csr_cpu_ref(2, &[0, 2, 2], &[1, 32], &[1, 1], &[0b0001], 0xFFFF_FFFF);
    assert_eq!(out, vec![0b0010], "dst=32 dropped because word 1 is OOB");
}

#[test]
fn csr_cpu_ref_oob_target_u32_max_silently_dropped() {
    let out = csr_cpu_ref(
        2,
        &[0, 2, 2],
        &[1, u32::MAX],
        &[1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(
        out,
        vec![0b0010],
        "u32::MAX destination must be silently ignored"
    );
}

#[test]
fn csr_cpu_ref_oob_target_with_multiword_bitset() {
    // 40 nodes across 2 words; frontier on node 0; edge to node 39 (valid) and 40 (OOB)
    let mut offsets = vec![2u32; 41];
    offsets[0] = 0;
    offsets[1] = 2;
    let mut frontier = zero_frontier(40);
    frontier[0] = 1; // node 0 set
    let out = csr_cpu_ref(
        40,
        &offsets,
        &[39, 40], // 39 valid, 40 == node_count must be dropped
        &[1, 1],
        &frontier,
        0xFFFF_FFFF,
    );
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 0);
    assert_eq!(
        out[1],
        1u32 << 7,
        "node 39 is in second word, bit 7; node 40 must be dropped"
    );
}

// ---------------------------------------------------------------------------
// Edge mask filtering
// ---------------------------------------------------------------------------

#[test]
fn csr_cpu_ref_edge_mask_zero_allow_mask_blocks_all() {
    let out = csr_cpu_ref(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[1, 1, 1, 1],
        &[0b0001],
        0, // allow_mask == 0
    );
    assert_eq!(out, vec![0], "zero allow_mask must block every edge");
}

#[test]
fn csr_cpu_ref_edge_mask_zero_kind_mask_blocks_all() {
    let out = csr_cpu_ref(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[0, 0, 0, 0], // every edge has mask 0
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(out, vec![0], "zero edge_kind_mask must block every edge");
}
