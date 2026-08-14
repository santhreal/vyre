// Backend dispatch for the declaration-container parity arm. The row
// accessors it builds on are owned by `tests/support/c_frontend/rows.rs`.

use std::sync::OnceLock;

use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::parse::vast::{
    c11_build_vast_nodes, c11_classify_vast_node_kinds, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};

#[allow(unused_imports)]
pub(crate) use crate::c_frontend::rows::{
    bytes, node_count_from_vast, row_indices as typed_indices, starts_for_lens, word_at,
    VAST_STRIDE_U32,
};

pub(crate) fn gpu_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "WgpuBackend::acquire failed on a machine that must have a GPU. \
             Per project GPU rule, this is a configuration bug, not a graceful skip.",
        )
    })
}

pub(crate) fn run_gpu_vast_builder(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> Vec<u8> {
    let program = c11_build_vast_nodes(
        "tok_types",
        "tok_starts",
        "tok_lens",
        Expr::u32(tok_types.len() as u32),
        "out_vast_nodes",
        "out_count",
    );
    let tok_type_bytes = bytes(tok_types);
    let tok_start_bytes = bytes(tok_starts);
    let tok_len_bytes = bytes(tok_lens);
    let inputs: Vec<&[u8]> = vec![&tok_type_bytes, &tok_start_bytes, &tok_len_bytes];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU VAST builder dispatch must succeed");
    assert_eq!(outputs.len(), 2);
    outputs[0].clone()
}

pub(crate) fn run_gpu_classifier(raw_vast: &[u8], num_nodes: u32) -> Vec<u8> {
    let program =
        c11_classify_vast_node_kinds("vast_nodes", Expr::u32(num_nodes), "typed_vast_nodes");
    let inputs: Vec<&[u8]> = vec![raw_vast];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU classifier dispatch must succeed");
    assert_eq!(outputs.len(), 1);
    outputs[0].clone()
}

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
