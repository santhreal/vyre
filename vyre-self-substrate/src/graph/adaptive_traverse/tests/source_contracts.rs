#[test]
fn matches_primitive_directly_by_wiring_release_programs() {
    let upload_source = include_str!("../upload.rs");
    let resident_source = include_str!("../resident_steps.rs");
    let release_path = format!("{upload_source}\n{resident_source}");

    // Dense-mode Programs are wired straight from vyre-primitives here.
    for primitive_call in [
        "primitive_adaptive_sparse_dense_step(",
        "primitive_adaptive_four_russians_dense_step(",
        "primitive_four_russians_dense_lut_from_adj_rows(",
    ] {
        assert!(
            release_path.contains(primitive_call),
            "adaptive traversal release path must call primitive output wiring {primitive_call}"
        );
    }

    // Sparse-queue Programs are wired through the one resident CSR queue owner,
    // which is itself pinned byte-for-byte against the vyre-primitives builders
    // by graph::csr_frontier_queue_programs::tests. Naming a primitive builder
    // directly here would be a second copy of that wiring.
    for owner_call in [
        "resident_csr_queue_len_init_program(",
        "resident_csr_queue_atomic_word_scan_program(",
        "resident_csr_queue_clear_frontier_out_program(",
        "resident_csr_queue_word_counts_program(",
        "resident_csr_queue_block_offsets_program(",
        "resident_csr_queue_word_prefix_queue_program(",
        "resident_csr_queue_traverse_program(",
        "resident_csr_queue_split_low_program(",
    ] {
        assert!(
            release_path.contains(owner_call),
            "adaptive traversal release path must build its sparse-queue Programs through the \
             shared resident CSR queue owner {owner_call}"
        );
    }
}

#[test]
fn release_resident_paths_do_not_call_cpu_or_local_saturating_helpers() {
    let upload_source = include_str!("../upload.rs");
    let resident_source = include_str!("../resident_steps.rs");
    let release_path = format!("{upload_source}\n{resident_source}");

    assert!(!release_path.contains("reference_adaptive_sparse_dense_step("));
    assert!(!release_path.contains("cpu_sparse_dense_step("));
    assert!(!release_path.contains("saturating_mul"));
    assert!(!release_path.contains(concat!("checked_mul", "(std::mem::size_of::<u32>())")));
    assert!(release_path.contains("u32_word_bytes("));
    assert!(!release_path.contains(".div_ceil(256)"));
    assert!(release_path.contains("plan_adaptive_resident_frontier_step"));
    assert!(release_path.contains("plan_adaptive_resident_sparse_queue_step"));
    assert!(release_path.contains("plan_adaptive_resident_auto_step"));
}
