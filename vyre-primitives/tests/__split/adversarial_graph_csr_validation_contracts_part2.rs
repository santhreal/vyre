use super::*;

#[test]
fn csr_cpu_ref_edge_mask_partial_filter() {
    // Graph: 0→1 (mask 0b01), 0→2 (mask 0b10), 1→3 (mask 0b01), 2→3 (mask 0b01)
    let out = csr_cpu_ref(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[0b01, 0b10, 0b01, 0b01],
        &[0b0001], // frontier = {0}
        0b01,      // only allow 0b01 edges
    );
    assert_eq!(out, vec![0b0010], "only node 1 reached via allowed edge");
}

#[test]
fn csr_cpu_ref_edge_mask_no_overlap() {
    // Every edge has mask 0b1000, allow_mask is 0b0100 → no overlap
    let out = csr_cpu_ref(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[0b1000, 0b1000],
        &[0b0001],
        0b0100,
    );
    assert_eq!(out, vec![0], "no overlapping bits → empty frontier");
}

#[test]
fn csr_cpu_ref_edge_mask_multi_source_mixed() {
    // Graph: 0→1 (mask 0b01), 1→2 (mask 0b10)
    // Frontier {0,1}, allow 0b01 → only 0→1 contributes
    let out = csr_cpu_ref(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[0b01, 0b10],
        &[0b0011], // nodes 0 and 1
        0b01,
    );
    assert_eq!(out, vec![0b0010], "only node 1 reached");
}

#[test]
fn validate_rejects_wrong_edge_kind_mask_len_for_zero_edges() {
    // shape.edge_count == 0 → expected len == 1 (placeholder)
    let shape = ProgramGraphShape::new(1, 0);
    let err = validate_program_graph(
        shape,
        &[0],
        &[0, 0],
        &[0],
        &[], // empty instead of placeholder 1
        &[0],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        GraphValidationError::EdgeKindMaskLen { got: 0, .. }
    ));
}

// ---------------------------------------------------------------------------
// High node counts / multi-word bitsets
// ---------------------------------------------------------------------------

#[test]
fn csr_cpu_ref_33_nodes_two_words() {
    // 33 nodes: frontier on node 32 (second word, bit 0)
    // Node 32 has one edge to node 0
    let mut offsets = vec![0u32; 34];
    for i in 0..34 {
        offsets[i] = if i <= 32 { 0 } else { 1 };
    }
    let mut frontier = zero_frontier(33);
    frontier[1] = 1; // node 32

    let out = csr_cpu_ref(
        33,
        &offsets,
        &[0], // 32 → 0
        &[1],
        &frontier,
        0xFFFF_FFFF,
    );
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 1, "node 0 set");
    assert_eq!(out[1], 0, "node 32 not in output");
}

#[test]
fn csr_cpu_ref_64_nodes_exactly_two_words() {
    // 64 nodes: frontier on node 63 (word 1, bit 31)
    // Node 63 → node 0
    let mut offsets = vec![0u32; 65];
    for i in 0..65 {
        offsets[i] = if i <= 63 { 0 } else { 1 };
    }
    let mut frontier = zero_frontier(64);
    frontier[1] = 1u32 << 31; // node 63

    let out = csr_cpu_ref(64, &offsets, &[0], &[1], &frontier, 0xFFFF_FFFF);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 1);
    assert_eq!(out[1], 0);
}

#[test]
fn csr_cpu_ref_65_nodes_three_words() {
    // 65 nodes: frontier on node 64 (word 2, bit 0)
    // Node 64 → node 64 (self-loop)
    let mut offsets = vec![0u32; 66];
    for i in 0..66 {
        offsets[i] = if i <= 64 { 0 } else { 1 };
    }
    let mut frontier = zero_frontier(65);
    frontier[2] = 1; // node 64

    let out = csr_cpu_ref(65, &offsets, &[64], &[1], &frontier, 0xFFFF_FFFF);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0], 0);
    assert_eq!(out[1], 0);
    assert_eq!(out[2], 1, "node 64 self-loop preserved");
}

#[test]
fn csr_cpu_ref_all_nodes_self_loop_preserves_actual_nodes() {
    // 100 nodes, each node has a self-loop. Frontier = all bits set (including garbage).
    // cpu_ref only processes src in 0..node_count, so only real nodes propagate.
    let n = 100u32;
    let words = bitset_words(n);
    let mut offsets = vec![0u32; n as usize + 1];
    for i in 0..=n {
        offsets[i as usize] = i;
    }
    let targets: Vec<u32> = (0..n).collect();
    let masks = vec![1u32; n as usize];
    let frontier = vec![0xFFFF_FFFF; words];

    let out = csr_cpu_ref(n, &offsets, &targets, &masks, &frontier, 0xFFFF_FFFF);
    assert_eq!(out.len(), words);
    // Real nodes 0..99 are preserved via self-loops. Garbage bits 100..127 are NOT
    // preserved because there is no src >= 100 to iterate them.
    assert_eq!(out[0], 0xFFFF_FFFF);
    assert_eq!(out[1], 0xFFFF_FFFF);
    assert_eq!(out[2], 0xFFFF_FFFF);
    assert_eq!(
        out[3], 0x0000_000F,
        "only nodes 96..99 preserved in last word"
    );
}

#[test]
fn validate_high_node_count_zero_edges() {
    // Cannot allocate u32::MAX nodes, but we can test a large count with 0 edges
    // to ensure the validation logic doesn't overflow on large counts.
    let shape = ProgramGraphShape::new(1_000_000, 0);
    // We won't allocate 1M arrays here; instead test shape invariants directly.
    let bufs = shape.read_only_buffers();
    assert_eq!(bufs[0].count(), 1_000_000);
    assert_eq!(bufs[1].count(), 1_000_001);
    assert_eq!(bufs[2].count(), 1); // placeholder
    assert_eq!(bufs[3].count(), 1); // placeholder
    assert_eq!(bufs[4].count(), 1_000_000);
}

// ---------------------------------------------------------------------------
// edge_offsets last count mismatch
// ---------------------------------------------------------------------------

#[test]
fn validate_passes_when_offsets_last_matches_edge_count() {
    let shape = ProgramGraphShape::new(3, 3);
    let result = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 1, 2, 3], // last == 3 == edge_count
        &[1, 2, 0],
        &[1, 1, 1],
        &[0, 0, 0],
    );
    assert!(result.is_ok(), "offsets.last() == edge_count must validate");
}

#[test]
fn validate_rejects_offsets_last_less_than_edge_count() {
    let shape = ProgramGraphShape::new(3, 3);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 1, 2, 2], // last == 2, but edge_count == 3
        &[1, 2, 0],    // len == 3, matches edge_count
        &[1, 1, 1],
        &[0, 0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            GraphValidationError::EdgeCountMismatch {
                expected: 3,
                got: 2
            }
        ),
        "offsets[last] < edge_count must be rejected"
    );
}

#[test]
fn validate_rejects_offsets_last_greater_than_edge_count() {
    let shape = ProgramGraphShape::new(3, 2);
    let err = validate_program_graph(
        shape,
        &[0, 0, 0],
        &[0, 1, 2, 5], // last == 5, but edge_count == 2
        &[1, 2],       // len == 2, matches edge_count
        &[1, 1],
        &[0, 0, 0],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            GraphValidationError::EdgeCountMismatch {
                expected: 2,
                got: 5
            }
        ),
        "offsets[last] > edge_count must be rejected"
    );
}

#[test]
fn csr_cpu_ref_uses_offsets_last_as_authoritative_edge_count() {
    // cpu_ref derives edge_count from edge_offsets.last(), not from a shape parameter.
    // offsets = [0,1,2,2] means node 0 has 1 edge (0→1), node 1 has 1 edge (1→2).
    // Frontier = {0}, so only edge 0 is processed.
    let out = csr_cpu_ref(
        3,
        &[0, 1, 2, 2], // offsets say 2 edges total
        &[1, 2],
        &[1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(
        out,
        vec![0b0010],
        "node 0 reaches node 1 via its single edge"
    );
}

#[test]
fn csr_cpu_ref_offsets_last_less_than_provided_targets_ignores_extras() {
    // offsets say 1 edge, but we provide 3 targets.
    // cpu_ref only iterates up to offsets.last() == 1.
    let out = csr_cpu_ref(
        3,
        &[0, 1, 1, 1], // last == 1
        &[1, 2, 0],    // 3 targets provided
        &[1, 1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
    assert_eq!(
        out,
        vec![0b0010],
        "only first edge (to node 1) is considered; extras ignored"
    );
}

#[test]
#[should_panic(expected = "complete CSR edge buffers")]
fn csr_cpu_ref_offsets_last_greater_than_provided_targets_fails_loudly() {
    // offsets say 5 edges, but we provide only 2 targets.
    let _ = csr_cpu_ref(
        3,
        &[0, 5, 5, 5], // last == 5
        &[1, 2],       // only 2 targets
        &[1, 1, 1, 1, 1],
        &[0b0001],
        0xFFFF_FFFF,
    );
}

// ---------------------------------------------------------------------------
// Property-style invariants
// ---------------------------------------------------------------------------

#[test]
fn csr_cpu_ref_empty_frontier_invariant() {
    // For any graph, empty frontier → empty output.
    for n in [1, 2, 5, 32, 33, 64, 65] {
        let offsets = vec![0u32; n + 1];
        let frontier = zero_frontier(n as u32);
        let out = csr_cpu_ref(
            n as u32,
            &offsets,
            &[0], // placeholder
            &[0], // placeholder
            &frontier,
            0xFFFF_FFFF,
        );
        assert_eq!(
            out, frontier,
            "empty frontier must produce empty output for n={n}"
        );
    }
}

#[test]
fn csr_cpu_ref_garbage_frontier_bits_not_propagated_beyond_node_count() {
    // 35 nodes → 2 words. Input frontier has garbage bits 35..63 set in word 1.
    // cpu_ref starts output at zero and only ORs from edges, so garbage bits
    // do not appear in output.
    let n = 35u32;
    let mut offsets = vec![0u32; n as usize + 1];
    for i in 0..=n {
        offsets[i as usize] = i;
    }
    let targets: Vec<u32> = (0..n).collect();
    let masks = vec![1u32; n as usize];
    let frontier = vec![0xFFFF_FFFF; 2];

    let out = csr_cpu_ref(n, &offsets, &targets, &masks, &frontier, 0xFFFF_FFFF);
    // Self-loops preserve real nodes 0..34. Bits 35..63 are not set.
    assert_eq!(out[0], 0xFFFF_FFFF);
    assert_eq!(
        out[1], 0x0000_0007,
        "only nodes 32,33,34 preserved; bits 35..63 zero"
    );
}

#[test]
fn csr_cpu_ref_frontier_word_oob_is_safely_skipped() {
    // frontier_in has fewer words than needed, but cpu_ref checks word_idx < len.
    let out = csr_cpu_ref(
        40,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        &[0],
        &[0],
        &[0], // only 1 word for 40 nodes (need 2)
        0xFFFF_FFFF,
    );
    assert_eq!(out, vec![0, 0], "short frontier must be safely handled");
}

#[test]
fn csr_cpu_ref_monotonic_in_allow_mask() {
    // Larger allow_mask cannot block more edges than a smaller one.
    let offsets = &[0, 2, 2, 2];
    let targets = &[1, 2];
    let masks = &[0b01, 0b10];
    let frontier = &[0b0001];

    let out_narrow = csr_cpu_ref(3, offsets, targets, masks, frontier, 0b01);
    let out_wide = csr_cpu_ref(3, offsets, targets, masks, frontier, 0b11);

    // out_wide must be a superset of out_narrow (bitwise)
    assert_eq!(out_narrow, vec![0b0010]);
    assert_eq!(out_wide, vec![0b0110]);
    assert!(
        (out_wide[0] & out_narrow[0]) == out_narrow[0],
        "wider allow_mask must be superset of narrower"
    );
}

#[test]
fn program_graph_shape_new_roundtrip() {
    let s = ProgramGraphShape::new(42, 99);
    assert_eq!(s.node_count, 42);
    assert_eq!(s.edge_count, 99);
}

#[test]
fn program_graph_shape_read_only_buffers_nonzero_edge() {
    let s = ProgramGraphShape::new(5, 3);
    let bufs = s.read_only_buffers();
    assert_eq!(bufs[2].count(), 3);
    assert_eq!(bufs[3].count(), 3);
}

#[test]
fn program_graph_shape_read_only_buffers_zero_edge_placeholder() {
    let s = ProgramGraphShape::new(5, 0);
    let bufs = s.read_only_buffers();
    assert_eq!(bufs[2].count(), 1);
    assert_eq!(bufs[3].count(), 1);
}
