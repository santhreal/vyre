// The declaration-container parity arm's differential: CPU reference build,
// CPU reference classify, GPU classify, compared. Backend acquisition and the
// stage dispatch are owned by `c_ast_gpu_parity_support`; the row accessors are
// owned by `tests/support/c_frontend/rows.rs`.

use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
};

#[allow(unused_imports)]
pub(crate) use crate::c_ast_gpu_parity_support::{
    run_gpu_classifier_with_count as run_gpu_classifier,
    run_gpu_vast_builder_from_parts as run_gpu_vast_builder,
};
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::rows::{
    bytes, node_count_from_vast, row_indices as typed_indices, starts_for_lens, word_at,
    VAST_STRIDE_U32,
};

pub(crate) fn cpu_gpu_classified(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> Vec<u8> {
    let raw = reference_c11_build_vast_nodes(tok_types, tok_starts, tok_lens);
    let expected = reference_c11_classify_vast_node_kinds(&raw);
    let gpu = run_gpu_classifier(&raw, node_count_from_vast(&raw));
    assert_eq!(
        gpu, expected,
        "GPU classifier must match CPU for declaration container fixture"
    );
    expected
}
