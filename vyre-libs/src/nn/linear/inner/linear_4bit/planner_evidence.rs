//! Cost evidence comparing the fused grouped INT4 kernel against a
//! dequantize-then-matmul plan.

use vyre_foundation::ir::DataType;

use super::grouped_layout::AFFINE_GROUPED_LANES_PER_OUTPUT;
use super::{
    QuantizedLinear4BitPlannerEvidence, QuantizedLinear4BitSpec,
    LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE,
};
use crate::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, MatrixShape,
};

/// Build planner evidence for
/// [`linear_4bit_affine_grouped_typed`](super::linear_4bit_affine_grouped_typed).
///
/// # Errors
/// Returns `Err` when quantized metadata or dimensions are invalid.
pub fn linear_4bit_affine_grouped_planner_evidence(
    spec: &QuantizedLinear4BitSpec,
) -> Result<QuantizedLinear4BitPlannerEvidence, String> {
    let group_size = spec.affine_group_size()?;
    quantized_linear_4bit_planner_evidence(spec.in_dim, spec.out_dim, group_size)
}

fn quantized_linear_4bit_planner_evidence(
    in_dim: u32,
    out_dim: u32,
    group_size: u32,
) -> Result<QuantizedLinear4BitPlannerEvidence, String> {
    if in_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires in_dim > 0.".to_string(),
        );
    }
    if out_dim == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires out_dim > 0.".to_string(),
        );
    }
    if group_size == 0 {
        return Err(
            "Fix: linear_4bit_affine_grouped planner evidence requires group_size > 0.".to_string(),
        );
    }
    if in_dim % 8 != 0 {
        return Err(format!(
            "Fix: linear_4bit_affine_grouped planner evidence in_dim={in_dim} is not divisible by 8."
        ));
    }

    let packed_words = (in_dim / 8).checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped planner evidence packed weights overflow u32.".to_string()
    })?;
    let group_count = in_dim.div_ceil(group_size);
    let sidecar_values = group_count.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_4bit_affine_grouped planner evidence sidecars overflow u32.".to_string()
    })?;
    let matmul_shape = MatrixShape {
        m: out_dim,
        k: in_dim,
        n: 1,
    };
    let matmul_tile = AFFINE_GROUPED_LANES_PER_OUTPUT;
    let matmul_plan = plan_matmul_kernel(
        &DataType::F32,
        matmul_shape,
        matmul_tile,
        1,
        F32MatmulMode::StrictF32,
        MatmulKernelCapabilities::current_codegen(),
    );
    let dequantized_weight_bytes = u64::from(in_dim)
        .saturating_mul(u64::from(out_dim))
        .saturating_mul(core::mem::size_of::<f32>() as u64);
    let packed_weight_bytes = u64::from(packed_words) * core::mem::size_of::<u32>() as u64;
    let sidecar_bytes = u64::from(sidecar_values)
        .saturating_mul((core::mem::size_of::<f32>() + core::mem::size_of::<u32>()) as u64);
    let output_bytes = u64::from(out_dim) * core::mem::size_of::<f32>() as u64;

    Ok(QuantizedLinear4BitPlannerEvidence {
        in_dim,
        out_dim,
        group_size,
        group_count,
        packed_weight_bytes,
        dequantized_weight_bytes,
        sidecar_bytes,
        bias_bytes: output_bytes,
        output_bytes,
        dequant_bytes_elided: dequantized_weight_bytes,
        matmul_m: matmul_shape.m,
        matmul_k: matmul_shape.k,
        matmul_n: matmul_shape.n,
        matmul_tile,
        matmul_selected_path: matmul_path_label(matmul_plan.selected_path),
        matmul_candidate_path: matmul_plan.candidate_path.map(matmul_path_label),
        matmul_fallback_reason: matmul_fallback_label(&matmul_plan),
        tensor_core_eligible: matmul_plan.selected_path != MatmulKernelPath::Cooperative,
        output_drift_abs_tolerance: LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE,
    })
}

fn matmul_path_label(path: MatmulKernelPath) -> &'static str {
    match path {
        MatmulKernelPath::Cooperative => "cooperative",
        MatmulKernelPath::TensorCoreF16M16N8K16 => "tensor_core_f16_m16n8k16",
        MatmulKernelPath::TensorCoreBf16M16N8K16 => "tensor_core_bf16_m16n8k16",
        MatmulKernelPath::TensorCoreTf32M16N8K4 => "tensor_core_tf32_m16n8k4",
    }
}

fn matmul_fallback_label(plan: &MatmulKernelPlan) -> Option<&'static str> {
    match plan.fallback_reason {
        Some(MatmulFallbackReason::StrictF32Requested) => Some("strict_f32_requested"),
        Some(MatmulFallbackReason::UnsupportedDtype) => Some("unsupported_dtype"),
        Some(MatmulFallbackReason::TileSizeMismatch { .. }) => Some("tile_size_mismatch"),
        Some(MatmulFallbackReason::RaggedTileUnsupported) => Some("ragged_tile_unsupported"),
        Some(MatmulFallbackReason::SplitKUnsupported) => Some("split_k_unsupported"),
        Some(MatmulFallbackReason::TensorCoreDtypeUnsupported) => {
            Some("tensor_core_dtype_unsupported")
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::grouped_layout::AFFINE_GROUPED_LANES_PER_OUTPUT;
    use super::super::{
        QuantizedLinear4BitSpec, LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE,
    };
    use super::linear_4bit_affine_grouped_planner_evidence;

    #[test]
    fn typed_affine_grouped_planner_evidence_records_matmul_and_dequant_contract() {
        let spec = QuantizedLinear4BitSpec::affine_grouped(256, 4096, 64);
        let evidence = linear_4bit_affine_grouped_planner_evidence(&spec)
            .expect("Fix: release grouped INT4 evidence must build");

        assert_eq!(evidence.in_dim, 256);
        assert_eq!(evidence.out_dim, 4096);
        assert_eq!(evidence.group_size, 64);
        assert_eq!(evidence.group_count, 4);
        assert_eq!(evidence.packed_weight_bytes, 524_288);
        assert_eq!(evidence.dequantized_weight_bytes, 4_194_304);
        assert_eq!(
            evidence.dequant_bytes_elided,
            evidence.dequantized_weight_bytes
        );
        assert_eq!(evidence.sidecar_bytes, 131_072);
        assert_eq!(evidence.bias_bytes, 16_384);
        assert_eq!(evidence.output_bytes, 16_384);
        assert_eq!(evidence.matmul_m, 4096);
        assert_eq!(evidence.matmul_k, 256);
        assert_eq!(evidence.matmul_n, 1);
        assert_eq!(evidence.matmul_tile, AFFINE_GROUPED_LANES_PER_OUTPUT);
        assert_eq!(evidence.matmul_selected_path, "cooperative");
        assert_eq!(evidence.matmul_candidate_path, None);
        assert_eq!(
            evidence.matmul_fallback_reason,
            Some("strict_f32_requested")
        );
        assert!(!evidence.tensor_core_eligible);
        assert_eq!(
            evidence.output_drift_abs_tolerance,
            LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE
        );
    }
}
