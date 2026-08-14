// Backend dispatch for the expression-shape parity arm. The CPU reference
// pipeline it compares against is owned by
// `tests/support/c_frontend/expression_pipeline.rs`.

use std::sync::OnceLock;

use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lower::c_lower_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::c11_build_expression_shape_nodes;

#[allow(unused_imports)]
pub(crate) use crate::c_frontend::expression_pipeline::{
    assert_kind, assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_row,
    run_pipeline, run_reference_pg_lower, PipelineRows,
};
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::rows::{
    node_count_from_vast, row_indices_by_stride as row_indices, starts_for_lens, word_at,
    PG_STRIDE_U32, SENTINEL, VAST_STRIDE_U32,
};

pub(crate) fn gpu_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "WgpuBackend::acquire failed on a machine that must have a GPU. \
             This is a configuration bug, not a graceful skip.",
        )
    })
}

pub(crate) fn run_gpu_expr_shape(raw_vast: &[u8], typed_vast: &[u8]) -> Vec<u8> {
    let program = c11_build_expression_shape_nodes(
        "raw_vast_nodes",
        "typed_vast_nodes",
        Expr::u32(node_count_from_vast(raw_vast)),
        "expr_shape_nodes",
    );
    let inputs: Vec<&[u8]> = vec![raw_vast, typed_vast];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU expression-shape dispatch must succeed");
    assert_eq!(outputs.len(), 1, "expected one expression-shape output");
    outputs[0].clone()
}

pub(crate) fn run_gpu_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    let program = c_lower_ast_to_pg_nodes(
        "vast_nodes",
        Expr::u32(node_count_from_vast(typed_vast)),
        "pg_nodes",
    );
    let inputs: Vec<&[u8]> = vec![typed_vast];
    let outputs = gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU PG lowerer dispatch must succeed");
    assert_eq!(outputs.len(), 1, "expected one PG output");
    outputs[0].clone()
}
