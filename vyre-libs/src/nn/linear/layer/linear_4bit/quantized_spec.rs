//! Validation of first-class quantized weight metadata and the spec-driven
//! program builders.

use vyre_foundation::ir::{DataType, Program};
use vyre_spec::{QuantizationScale, QuantizationZeroPoint};

use super::affine_grouped::{linear_4bit_affine_grouped, linear_4bit_affine_grouped_batched};
use super::QuantizedLinear4BitSpec;

impl QuantizedLinear4BitSpec {
    /// Build a grouped affine INT4 metadata spec.
    #[must_use]
    pub fn affine_grouped(in_dim: u32, out_dim: u32, group_size: u32) -> Self {
        Self {
            in_dim,
            out_dim,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I4),
                scale: QuantizationScale::PerGroup { group_size },
                zero_point: QuantizationZeroPoint::PerGroup { group_size },
            },
        }
    }

    pub(super) fn affine_group_size(&self) -> Result<u32, String> {
        match &self.weight_type {
            DataType::Quantized {
                storage,
                scale: QuantizationScale::PerGroup { group_size },
                zero_point:
                    QuantizationZeroPoint::PerGroup {
                        group_size: zp_group_size,
                    },
            } => {
                if storage.as_ref() != &DataType::I4 {
                    return Err(format!(
                        "Fix: grouped INT4 linear requires DataType::Quantized storage I4, got {storage}."
                    ));
                }
                if group_size != zp_group_size {
                    return Err(format!(
                        "Fix: grouped INT4 linear requires scale and zero-point group sizes to match, got scale={group_size}, zero_point={zp_group_size}."
                    ));
                }
                if *group_size == 0 {
                    return Err(
                        "Fix: grouped INT4 linear requires quantized group_size > 0.".to_string()
                    );
                }
                Ok(*group_size)
            }
            other => Err(format!(
                "Fix: grouped INT4 linear requires DataType::Quantized<I4; PerGroup scale; PerGroup zero-point>, got {other}."
            )),
        }
    }
}

/// Build [`linear_4bit_affine_grouped`] from first-class quantized metadata.
///
/// # Errors
/// Returns `Err` when the spec is not `Quantized<I4; PerGroup; PerGroup>`,
/// when scale/zero-point group sizes differ, or when dimensions are invalid.
pub fn linear_4bit_affine_grouped_typed(
    spec: &QuantizedLinear4BitSpec,
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
) -> Result<Program, String> {
    let group_size = spec.affine_group_size()?;
    linear_4bit_affine_grouped(
        x,
        w_packed,
        scale,
        zero_point,
        b,
        out,
        spec.in_dim,
        spec.out_dim,
        group_size,
    )
}

/// Build [`linear_4bit_affine_grouped_batched`] from first-class quantized
/// metadata.
///
/// # Errors
/// Returns `Err` when quantized metadata, dimensions, or `batch_size` are
/// invalid.
pub fn linear_4bit_affine_grouped_batched_typed(
    spec: &QuantizedLinear4BitSpec,
    batch_size: u32,
    x: &str,
    w_packed: &str,
    scale: &str,
    zero_point: &str,
    b: &str,
    out: &str,
) -> Result<Program, String> {
    let group_size = spec.affine_group_size()?;
    linear_4bit_affine_grouped_batched(
        x,
        w_packed,
        scale,
        zero_point,
        b,
        out,
        spec.in_dim,
        spec.out_dim,
        group_size,
        batch_size,
    )
}

#[cfg(test)]
mod tests {
    use vyre_foundation::ir::DataType;
    use vyre_spec::{QuantizationScale, QuantizationZeroPoint};

    use super::super::QuantizedLinear4BitSpec;
    use super::linear_4bit_affine_grouped_typed;

    #[test]
    fn typed_affine_grouped_builder_uses_quantized_metadata() {
        let spec = QuantizedLinear4BitSpec::affine_grouped(32, 7, 8);
        let program = linear_4bit_affine_grouped_typed(&spec, "x", "w", "scale", "zp", "b", "out")
            .expect("Fix: valid typed grouped INT4 spec must build");

        assert_eq!(program.buffers()[1].name(), "w");
        assert_eq!(program.buffers()[1].element(), DataType::U32);
        assert_eq!(program.buffers()[1].count(), 28);
        assert!(matches!(
            spec.weight_type,
            DataType::Quantized {
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 8 },
                ..
            }
        ));
    }

    #[test]
    fn typed_affine_grouped_builder_rejects_mismatched_quantized_metadata() {
        let bad_storage = QuantizedLinear4BitSpec {
            in_dim: 32,
            out_dim: 4,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I8),
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 8 },
            },
        };
        let error =
            linear_4bit_affine_grouped_typed(&bad_storage, "x", "w", "scale", "zp", "b", "out")
                .unwrap_err();
        assert!(
            error.contains("storage I4"),
            "Fix: storage mismatch should be explicit: {error}"
        );

        let bad_sidecar = QuantizedLinear4BitSpec {
            in_dim: 32,
            out_dim: 4,
            weight_type: DataType::Quantized {
                storage: Box::new(DataType::I4),
                scale: QuantizationScale::PerGroup { group_size: 8 },
                zero_point: QuantizationZeroPoint::PerGroup { group_size: 16 },
            },
        };
        let error =
            linear_4bit_affine_grouped_typed(&bad_sidecar, "x", "w", "scale", "zp", "b", "out")
                .unwrap_err();
        assert!(
            error.contains("group sizes to match"),
            "Fix: sidecar mismatch should be explicit: {error}"
        );
    }

    #[test]
    fn generated_typed_affine_grouped_specs_build_or_reject_by_metadata_contract() {
        let mut accepted = 0usize;
        let mut rejected = 0usize;
        for in_dim in [8u32, 10, 16, 18, 24, 32, 64, 128] {
            for out_dim in [1u32, 2, 3, 7, 16, 31] {
                for group_size in [1u32, 2, 4, 8, 16, 32] {
                    let spec = QuantizedLinear4BitSpec::affine_grouped(in_dim, out_dim, group_size);
                    let result = linear_4bit_affine_grouped_typed(
                        &spec, "x", "w", "scale", "zp", "b", "out",
                    );
                    if in_dim % 8 == 0 {
                        let program = result.expect("Fix: generated valid typed spec must build");
                        let output = &program.buffers()[5];
                        assert!(
                            output.count() >= out_dim,
                            "Fix: grouped INT4 output storage must cover the logical outputs after launch padding."
                        );
                        assert_eq!(
                            output.output_byte_range(),
                            Some(0..(out_dim as usize * core::mem::size_of::<f32>())),
                            "Fix: grouped INT4 output byte range must trim padded launch storage to the logical tensor."
                        );
                        accepted += 1;
                    } else {
                        let error = result.expect_err(
                            "Fix: generated indivisible typed spec must reject before dispatch",
                        );
                        assert!(error.contains("divisible by 8"));
                        rejected += 1;
                    }
                }
            }
        }

        assert!(
            accepted + rejected >= 216,
            "Fix: generated typed quantized specs should cover hundreds of layouts"
        );
    }
}
