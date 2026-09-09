//! Tensor-core path selection for tiled matmul.

use vyre_foundation::ir::DataType;

use super::shape::MatrixShape;

/// Selected kernel lowering path for tiled matrix multiplication.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MatmulKernelPath {
    /// Cooperative workgroup tiled execution.
    Cooperative,
    /// F16 tensor core execution.
    TensorCoreF16M16N8K16,
    /// BF16 tensor core execution.
    TensorCoreBf16M16N8K16,
    /// TF32 tensor core execution.
    TensorCoreTf32M16N8K4,
}

/// Floating-point precision mode for F32 matrix multiplication.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum F32MatmulMode {
    /// Strict F32 arithmetic without contraction.
    StrictF32,
    /// TF32 tensor core arithmetic.
    Tf32TensorCore,
}

/// Tensor core tile geometry.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TensorCoreTileShape {
    /// 16x8x16 tile.
    M16N8K16,
    /// 16x8x4 tile.
    M16N8K4,
}

/// Reason why a tensor core path was not selected.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MatmulFallbackReason {
    /// Strict F32 arithmetic was requested.
    StrictF32Requested,
    /// Data type is unsupported for tensor cores.
    UnsupportedDtype,
    /// Tile size mismatch.
    TileSizeMismatch {
        /// Required k-tile.
        required_k_tile: u32,
        /// Found tile.
        found_tile: u32,
    },
    /// Ragged tiles are unsupported on this target.
    RaggedTileUnsupported,
    /// Split-k is unsupported on this target.
    SplitKUnsupported,
    /// Tensor core data type is unsupported.
    TensorCoreDtypeUnsupported,
}

/// Hardware capabilities for matrix multiplication.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatmulKernelCapabilities {
    /// Whether F16 tensor cores are supported.
    pub f16_tensor_cores: bool,
    /// Whether BF16 tensor cores are supported.
    pub bf16_tensor_cores: bool,
    /// Whether TF32 tensor cores are supported.
    pub tf32_tensor_cores: bool,
    /// Whether split-K is supported.
    pub split_k: bool,
    /// Whether ragged tensor tiles are supported.
    pub ragged_tensor_tiles: bool,
}
impl MatmulKernelCapabilities {
    /// Default capabilities for current target code generation.
    pub const fn current_codegen() -> Self {
        Self {
            f16_tensor_cores: true,
            bf16_tensor_cores: false,
            tf32_tensor_cores: false,
            split_k: false,
            ragged_tensor_tiles: false,
        }
    }

    #[cfg(test)]
    const fn all_tensor_core_modes() -> Self {
        Self {
            f16_tensor_cores: true,
            bf16_tensor_cores: true,
            tf32_tensor_cores: true,
            split_k: true,
            ragged_tensor_tiles: true,
        }
    }
}

/// Selected kernel plan for matrix multiplication.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatmulKernelPlan {
    /// Selected lowering path.
    pub selected_path: MatmulKernelPath,
    /// Candidate tensor core path.
    pub candidate_path: Option<MatmulKernelPath>,
    /// Candidate tile shape.
    pub tile_shape: Option<TensorCoreTileShape>,
    /// Number of split-k slices.
    pub split_k_slices: u32,
    /// Whether ragged tiles are needed.
    pub ragged_tiles: bool,
    /// Reason for falling back to cooperative path.
    pub fallback_reason: Option<MatmulFallbackReason>,
}

pub(crate) fn select_matmul_kernel(
    dtype: &DataType,
    shape: MatrixShape,
    tile: u32,
) -> MatmulKernelPath {
    plan_matmul_kernel(
        dtype,
        shape,
        tile,
        1,
        F32MatmulMode::StrictF32,
        MatmulKernelCapabilities::current_codegen(),
    )
    .selected_path
}

/// Plan the kernel execution path for matrix multiplication given data type, shape, and capabilities.
pub fn plan_matmul_kernel(
    dtype: &DataType,
    shape: MatrixShape,
    tile: u32,
    split_k_slices: u32,
    f32_mode: F32MatmulMode,
    capabilities: MatmulKernelCapabilities,
) -> MatmulKernelPlan {
    let split_k_slices = split_k_slices.max(1);
    let Some((candidate_path, tile_shape, required_k_tile, dtype_supported)) =
        tensor_core_candidate(dtype, f32_mode, capabilities)
    else {
        return cooperative_plan(
            None,
            None,
            split_k_slices,
            false,
            if *dtype == DataType::F32 && f32_mode == F32MatmulMode::StrictF32 {
                MatmulFallbackReason::StrictF32Requested
            } else {
                MatmulFallbackReason::UnsupportedDtype
            },
        );
    };

    if !dtype_supported {
        return cooperative_plan(
            Some(candidate_path),
            Some(tile_shape),
            split_k_slices,
            false,
            MatmulFallbackReason::TensorCoreDtypeUnsupported,
        );
    }

    if tile != required_k_tile {
        return cooperative_plan(
            Some(candidate_path),
            Some(tile_shape),
            split_k_slices,
            false,
            MatmulFallbackReason::TileSizeMismatch {
                required_k_tile,
                found_tile: tile,
            },
        );
    }

    let ragged_tiles = shape.m % 16 != 0 || shape.n % 8 != 0 || shape.k % required_k_tile != 0;
    if ragged_tiles && !capabilities.ragged_tensor_tiles {
        return cooperative_plan(
            Some(candidate_path),
            Some(tile_shape),
            split_k_slices,
            true,
            MatmulFallbackReason::RaggedTileUnsupported,
        );
    }

    if split_k_slices > 1 && !capabilities.split_k {
        return cooperative_plan(
            Some(candidate_path),
            Some(tile_shape),
            split_k_slices,
            ragged_tiles,
            MatmulFallbackReason::SplitKUnsupported,
        );
    }

    MatmulKernelPlan {
        selected_path: candidate_path,
        candidate_path: Some(candidate_path),
        tile_shape: Some(tile_shape),
        split_k_slices,
        ragged_tiles,
        fallback_reason: None,
    }
}

fn tensor_core_candidate(
    dtype: &DataType,
    f32_mode: F32MatmulMode,
    capabilities: MatmulKernelCapabilities,
) -> Option<(MatmulKernelPath, TensorCoreTileShape, u32, bool)> {
    match dtype {
        DataType::F16 => Some((
            MatmulKernelPath::TensorCoreF16M16N8K16,
            TensorCoreTileShape::M16N8K16,
            16,
            capabilities.f16_tensor_cores,
        )),
        DataType::BF16 => Some((
            MatmulKernelPath::TensorCoreBf16M16N8K16,
            TensorCoreTileShape::M16N8K16,
            16,
            capabilities.bf16_tensor_cores,
        )),
        DataType::F32 if f32_mode == F32MatmulMode::Tf32TensorCore => Some((
            MatmulKernelPath::TensorCoreTf32M16N8K4,
            TensorCoreTileShape::M16N8K4,
            4,
            capabilities.tf32_tensor_cores,
        )),
        _ => None,
    }
}

fn cooperative_plan(
    candidate_path: Option<MatmulKernelPath>,
    tile_shape: Option<TensorCoreTileShape>,
    split_k_slices: u32,
    ragged_tiles: bool,
    fallback_reason: MatmulFallbackReason,
) -> MatmulKernelPlan {
    MatmulKernelPlan {
        selected_path: MatmulKernelPath::Cooperative,
        candidate_path,
        tile_shape,
        split_k_slices,
        ragged_tiles,
        fallback_reason: Some(fallback_reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_current_codegen_policy_only_accepts_full_f16_m16n8k16_tiles() {
        let mut cases = 0usize;
        for m in [1, 15, 16, 31, 32] {
            for k in [1, 15, 16, 17, 32] {
                for n in [1, 7, 8, 9, 16] {
                    for tile in [1, 8, 16, 32] {
                        let path =
                            select_matmul_kernel(&DataType::F16, MatrixShape { m, k, n }, tile);
                        let expected_mma = tile == 16 && m % 16 == 0 && n % 8 == 0 && k % 16 == 0;
                        assert_eq!(
                            path == MatmulKernelPath::TensorCoreF16M16N8K16,
                            expected_mma,
                            "m={m} k={k} n={n} tile={tile}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 500);
    }

    #[test]
    fn planner_selects_exact_dtype_tile_paths_with_capabilities() {
        let shape = MatrixShape {
            m: 32,
            k: 16,
            n: 16,
        };
        let caps = MatmulKernelCapabilities::all_tensor_core_modes();

        let f16 = plan_matmul_kernel(&DataType::F16, shape, 16, 1, F32MatmulMode::StrictF32, caps);
        assert_eq!(f16.selected_path, MatmulKernelPath::TensorCoreF16M16N8K16);
        assert_eq!(f16.tile_shape, Some(TensorCoreTileShape::M16N8K16));
        assert_eq!(f16.fallback_reason, None);

        let bf16 = plan_matmul_kernel(
            &DataType::BF16,
            shape,
            16,
            1,
            F32MatmulMode::StrictF32,
            caps,
        );
        assert_eq!(bf16.selected_path, MatmulKernelPath::TensorCoreBf16M16N8K16);
        assert_eq!(bf16.tile_shape, Some(TensorCoreTileShape::M16N8K16));
        assert_eq!(bf16.fallback_reason, None);

        let tf32 = plan_matmul_kernel(
            &DataType::F32,
            MatrixShape {
                m: 32,
                k: 16,
                n: 16,
            },
            4,
            1,
            F32MatmulMode::Tf32TensorCore,
            caps,
        );
        assert_eq!(tf32.selected_path, MatmulKernelPath::TensorCoreTf32M16N8K4);
        assert_eq!(tf32.tile_shape, Some(TensorCoreTileShape::M16N8K4));
        assert_eq!(tf32.fallback_reason, None);

        let strict_f32 =
            plan_matmul_kernel(&DataType::F32, shape, 16, 1, F32MatmulMode::StrictF32, caps);
        assert_eq!(strict_f32.selected_path, MatmulKernelPath::Cooperative);
        assert_eq!(
            strict_f32.fallback_reason,
            Some(MatmulFallbackReason::StrictF32Requested)
        );
    }

    #[test]
    fn planner_records_split_k_and_ragged_fallbacks_exactly() {
        let shape = MatrixShape {
            m: 32,
            k: 16,
            n: 16,
        };
        let current = MatmulKernelCapabilities::current_codegen();

        let unsupported_split_k = plan_matmul_kernel(
            &DataType::F16,
            shape,
            16,
            4,
            F32MatmulMode::StrictF32,
            current,
        );
        assert_eq!(
            unsupported_split_k.selected_path,
            MatmulKernelPath::Cooperative
        );
        assert_eq!(
            unsupported_split_k.candidate_path,
            Some(MatmulKernelPath::TensorCoreF16M16N8K16)
        );
        assert_eq!(unsupported_split_k.split_k_slices, 4);
        assert_eq!(
            unsupported_split_k.fallback_reason,
            Some(MatmulFallbackReason::SplitKUnsupported)
        );

        let split_k = plan_matmul_kernel(
            &DataType::F16,
            shape,
            16,
            4,
            F32MatmulMode::StrictF32,
            MatmulKernelCapabilities::all_tensor_core_modes(),
        );
        assert_eq!(
            split_k.selected_path,
            MatmulKernelPath::TensorCoreF16M16N8K16
        );
        assert_eq!(split_k.split_k_slices, 4);
        assert_eq!(split_k.fallback_reason, None);

        let ragged = plan_matmul_kernel(
            &DataType::F16,
            MatrixShape { m: 33, k: 17, n: 9 },
            16,
            1,
            F32MatmulMode::StrictF32,
            current,
        );
        assert_eq!(ragged.selected_path, MatmulKernelPath::Cooperative);
        assert!(ragged.ragged_tiles);
        assert_eq!(
            ragged.fallback_reason,
            Some(MatmulFallbackReason::RaggedTileUnsupported)
        );

        let ragged_supported = plan_matmul_kernel(
            &DataType::F16,
            MatrixShape { m: 33, k: 17, n: 9 },
            16,
            1,
            F32MatmulMode::StrictF32,
            MatmulKernelCapabilities::all_tensor_core_modes(),
        );
        assert_eq!(
            ragged_supported.selected_path,
            MatmulKernelPath::TensorCoreF16M16N8K16
        );
        assert!(ragged_supported.ragged_tiles);
        assert_eq!(ragged_supported.fallback_reason, None);
    }

    #[test]
    fn planner_records_tile_and_dtype_fallbacks_exactly() {
        let caps = MatmulKernelCapabilities::all_tensor_core_modes();
        let shape = MatrixShape {
            m: 32,
            k: 16,
            n: 16,
        };

        let wrong_tile =
            plan_matmul_kernel(&DataType::F16, shape, 8, 1, F32MatmulMode::StrictF32, caps);
        assert_eq!(wrong_tile.selected_path, MatmulKernelPath::Cooperative);
        assert_eq!(
            wrong_tile.fallback_reason,
            Some(MatmulFallbackReason::TileSizeMismatch {
                required_k_tile: 16,
                found_tile: 8,
            })
        );

        let unsupported_dtype =
            plan_matmul_kernel(&DataType::I32, shape, 16, 1, F32MatmulMode::StrictF32, caps);
        assert_eq!(
            unsupported_dtype.selected_path,
            MatmulKernelPath::Cooperative
        );
        assert_eq!(
            unsupported_dtype.fallback_reason,
            Some(MatmulFallbackReason::UnsupportedDtype)
        );

        let unsupported_bf16_codegen = plan_matmul_kernel(
            &DataType::BF16,
            shape,
            16,
            1,
            F32MatmulMode::StrictF32,
            MatmulKernelCapabilities::current_codegen(),
        );
        assert_eq!(
            unsupported_bf16_codegen.candidate_path,
            Some(MatmulKernelPath::TensorCoreBf16M16N8K16)
        );
        assert_eq!(
            unsupported_bf16_codegen.fallback_reason,
            Some(MatmulFallbackReason::TensorCoreDtypeUnsupported)
        );
    }
}
