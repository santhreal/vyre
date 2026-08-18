//! Linear-algebra sub-dialect: dot product, matmul, tiled matmul,
//! Strassen base case.
mod dot;
mod matmul;
mod matmul_strassen;
pub(crate) mod matmul_tiled;

pub use dot::{dot, Dot};
pub use matmul::{matmul, matmul_bias, Matmul, MatmulBias};
pub use matmul_strassen::{matmul_strassen_2x2, matmul_strassen_one_level};

// Keep the tiled builders on the linear-algebra sub-dialect surface.
pub use matmul_tiled::{matmul_bias_tiled, matmul_tiled, MatmulBiasTiled, MatmulTiled};
#[cfg(feature = "nn-linear-4bit")]
pub(crate) use matmul_tiled::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, MatrixShape,
};

use crate::builder::gemm::ContractionComposer;
use crate::builder::BuildOptions;
use crate::plumbing::operand::tensor_ref::TensorRef;

#[must_use]
pub(crate) fn matmul_2d_dims(a: &TensorRef, b: &TensorRef) -> (u32, u32, u32) {
    let m = if a.shape.len() == 2 { a.shape[0] } else { 0 };
    let k = if a.shape.len() == 2 { a.shape[1] } else { 0 };
    let n = if b.shape.len() == 2 { b.shape[1] } else { 0 };
    (m, k, n)
}

#[must_use]
pub(crate) fn apply_contraction_options(
    mut composer: ContractionComposer,
    options: &BuildOptions,
) -> ContractionComposer {
    if let Some(workgroup) = options.workgroup_size {
        composer = composer.with_workgroup_size(workgroup);
    }
    if let Some(generator) = options.region_generator {
        composer = composer.with_region_generator(generator);
    }
    if let Some(tenant_id) = options.tenant_id {
        composer = composer.with_tenant_id(tenant_id);
    }
    composer
}

#[must_use]
pub(crate) fn matmul_bias_2x2_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        crate::fixture_bytes::u32_bytes(&[1, 2, 3, 4]),
        crate::fixture_bytes::u32_bytes(&[5, 6, 7, 8]),
        crate::fixture_bytes::u32_bytes(&[10, 20]),
    ]]
}

#[must_use]
pub(crate) fn matmul_bias_2x2_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![
        0x1d, 0x00, 0x00, 0x00, // 29
        0x2a, 0x00, 0x00, 0x00, // 42
        0x35, 0x00, 0x00, 0x00, // 53
        0x46, 0x00, 0x00, 0x00, // 70
    ]]]
}
