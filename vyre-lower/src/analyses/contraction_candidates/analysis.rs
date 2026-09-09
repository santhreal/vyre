//! Contraction candidate detection and strategy planning.
//!
//! Logical contraction IR lowers into measured scalar, SIMD/SIMT tiled, and
//! target matrix-instruction candidates without domain ownership. This analysis
//! inspects the lowered descriptor and surfaces the candidate execution strategies
//! for each contraction site.

use super::plan::{ContractionCandidate, ContractionPlan, ContractionStrategy};
use crate::descriptor::{
    KernelBody, KernelDescriptor, KernelOpKind, MatrixMmaElement, MatrixMmaLayout, MatrixTileShape,
};
use vyre_foundation::ir::DataType;

/// Analyze a kernel descriptor and surface contraction candidate execution strategies.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> ContractionPlan {
    let (has_mma, has_fma, has_reduction) = scan_body(&desc.body);

    let mut candidates = Vec::new();

    // If the kernel contains contraction operations (MMA, FMA chain, or reduction loop)
    // or has matrix/vector bindings:
    let is_contraction = has_mma || has_fma || has_reduction || has_matrix_bindings(desc);

    if is_contraction {
        // 1. Scalar baseline candidate (always eligible, reference element order)
        candidates.push(ContractionCandidate {
            contraction_id: format!("{}_scalar", desc.id),
            strategy: ContractionStrategy::Scalar,
            estimated_speedup_factor: 1.0,
            supported_dtypes: vec![
                DataType::F32,
                DataType::F16,
                DataType::BF16,
                DataType::U32,
                DataType::I32,
            ],
        });

        // 2. SIMT workgroup-shared memory cooperative tiled candidate
        candidates.push(ContractionCandidate {
            contraction_id: format!("{}_simt_tiled_16x16", desc.id),
            strategy: ContractionStrategy::SimtTiled {
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
                workgroup_size: [16, 16, 1],
            },
            estimated_speedup_factor: 4.0,
            supported_dtypes: vec![
                DataType::F32,
                DataType::F16,
                DataType::BF16,
                DataType::U32,
                DataType::I32,
            ],
        });

        // 3. Target matrix-multiply-accumulate (MMA / tensor-core) candidate
        candidates.push(ContractionCandidate {
            contraction_id: format!("{}_mma_m16n8k16", desc.id),
            strategy: ContractionStrategy::MatrixInstruction {
                tile: MatrixTileShape {
                    m: 16,
                    n: 8,
                    k: 16,
                },
                left_layout: MatrixMmaLayout::RowMajor,
                right_layout: MatrixMmaLayout::ColMajor,
                left_element: MatrixMmaElement::F16,
                right_element: MatrixMmaElement::F16,
                acc_element: MatrixMmaElement::F32,
            },
            estimated_speedup_factor: 10.0,
            supported_dtypes: vec![DataType::F16, DataType::BF16],
        });
    }

    ContractionPlan {
        kernel_id: desc.id.clone(),
        candidates,
    }
}

fn scan_body(body: &KernelBody) -> (bool, bool, bool) {
    let mut has_mma = false;
    let mut has_fma = false;
    let mut has_reduction = false;

    for op in &body.ops {
        match &op.kind {
            KernelOpKind::MatrixMma(_) => has_mma = true,
            KernelOpKind::Fma => has_fma = true,
            KernelOpKind::StructuredForLoop { .. } => has_reduction = true,
            _ => {}
        }
    }

    for child in &body.child_bodies {
        let (c_mma, c_fma, c_red) = scan_body(child);
        has_mma |= c_mma;
        has_fma |= c_fma;
        has_reduction |= c_red;
    }

    (has_mma, has_fma, has_reduction)
}

fn has_matrix_bindings(desc: &KernelDescriptor) -> bool {
    desc.bindings
        .slots
        .iter()
        .any(|slot| slot.element_count.is_some_and(|c| c >= 4))
}
