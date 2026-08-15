use super::*;

/// Assert the GPU reproduces the CPU oracle from the token stream through the
/// classifier.
///
/// Binds this crate's [`GpuArm`] to the shared parity sequence in
/// `tests/support/c_frontend/parity_matrix`. The sequence, the programs and the
/// comparisons have one owner there because the CPU-reference arm in
/// `vyre-libs/tests` runs the same three stages; only the dispatch differs.
///
/// This stops at the classifier. `assert_family_parity` continues into the
/// property-graph lowerer, which is why the three C-AST families that iterate a
/// `CASES` table call that instead.
pub(crate) fn assert_full_pipeline_parity(fix: &Fixture, label: &str) {
    crate::c_frontend::parity_matrix::assert_arm_parity_through_classify(&GpuArm, fix, label);
}

pub(crate) fn assert_gpu_pg_parity(fix: &Fixture, typed_vast: &[u8], label: &str) {
    let node_count = node_count_from_vast(typed_vast);
    let cpu_pg = reference_ast_to_pg_nodes(typed_vast);
    let gpu_pg = run_gpu_pg_lower_with_count(typed_vast, node_count);
    assert_eq!(
        gpu_pg,
        cpu_pg,
        "{label}: GPU PG lower output diverged from reference for {} bytes",
        fix.source.len()
    );
}

pub(crate) fn run_gpu_vast_builder_from_parts(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> Vec<u8> {
    arm_raw_vast(&GpuArm, tok_types, tok_starts, tok_lens)
}

pub(crate) fn run_gpu_classifier(annotated_vast: &[u8]) -> Vec<u8> {
    arm_typed_vast(&GpuArm, annotated_vast)
}

pub(crate) fn run_gpu_classifier_with_count(annotated_vast: &[u8], num_nodes: u32) -> Vec<u8> {
    let outputs = dispatch_gpu_program(
        "GPU VAST classifier",
        program::classify(num_nodes),
        vec![annotated_vast.to_vec()],
    );
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}

pub(crate) fn run_gpu_expr_shape(raw_vast: &[u8], typed_vast: &[u8]) -> Vec<u8> {
    crate::c_frontend::parity_matrix::arm_expr_shape(&GpuArm, raw_vast, typed_vast)
}

/// Assert the GPU reproduces the CPU oracle's expression-shape and
/// property-graph rows for every case of an expression-operator family.
///
/// The run lives in `tests/support/c_frontend/parity_matrix`; only the case
/// table differs per family.
pub(crate) fn assert_expression_shape_parity(cases: &[ExpressionCase]) {
    crate::c_frontend::parity_matrix::assert_expression_shape_family_parity(&GpuArm, cases);
}

pub(crate) fn run_gpu_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    arm_pg_nodes(&GpuArm, typed_vast)
}

pub(crate) fn run_gpu_pg_lower_with_count(typed_vast: &[u8], num_nodes: u32) -> Vec<u8> {
    let outputs = dispatch_gpu_program(
        "GPU AST-to-PG lower",
        program::lower_pg(num_nodes),
        vec![typed_vast.to_vec()],
    );
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}

pub(crate) fn run_gpu_semantic_pg_lower(typed_vast: &[u8]) -> (Vec<u8>, Vec<u8>) {
    run_gpu_semantic_pg_lower_with_count(typed_vast, node_count_from_vast(typed_vast))
}

pub(crate) fn run_gpu_semantic_pg_lower_with_count(
    typed_vast: &[u8],
    num_nodes: u32,
) -> (Vec<u8>, Vec<u8>) {
    let program = c_lower_ast_to_pg_semantic_graph(
        "vast_nodes",
        Expr::u32(num_nodes),
        "out_pg_nodes",
        "out_pg_edges",
    );
    let outputs = dispatch_gpu_program(
        "GPU semantic AST-to-PG lower",
        program,
        vec![typed_vast.to_vec()],
    );
    assert_eq!(outputs.len(), 2);
    (outputs[0].clone(), outputs[1].clone())
}

pub(crate) fn run_gpu_c_sema_scope_from_parts(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
    haystack: &[u8],
) -> Vec<u8> {
    let program = c_sema_scope(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(haystack.len() as u32),
        Expr::u32(tok_types.len() as u32),
        "out_scope_tree",
    );
    let tok_type_bytes = bytes(tok_types);
    let tok_start_bytes = bytes(tok_starts);
    let tok_len_bytes = bytes(tok_lens);
    let haystack_bytes = haystack_words(haystack);
    let outputs = dispatch_gpu_program(
        "GPU C semantic scope",
        program,
        vec![
            tok_type_bytes,
            tok_start_bytes,
            tok_len_bytes,
            haystack_bytes,
        ],
    );
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}
