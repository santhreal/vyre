//! Canonical matrix multiplication and contraction IR composer.
//!
//! Unifies dense 2D GEMM, 3D batched GEMM, row-batched linear projections,
//! semiring matrix multiplications, fixed-point contractions, cooperative
//! tiled GEMM, and fused epilogues (bias, activation, scaling).

use std::sync::Arc;
use vyre_foundation::ir::{DataType, Expr, Program};
use vyre_spec::Semiring;

#[path = "gemm_algebra.rs"]
mod gemm_algebra;
pub use gemm_algebra::*;

#[path = "gemm_programs.rs"]
mod gemm_programs;
use gemm_programs::*;

use crate::builder::{check_tensors, BuildOptions};
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};
/// Fused post-accumulation transformation applied to each output element.
#[derive(Clone)]
pub enum ContractionEpilogue {
    /// Store accumulated result directly.
    None,
    /// Add bias vector: `acc + bias[col]`.
    Bias {
        /// Bias buffer name.
        buffer: String,
        /// Number of bias elements.
        count: u32,
        /// Element data type.
        dtype: DataType,
    },
    /// Fused elementwise activation: `activation(acc)`.
    Activation {
        /// Optional bias buffer name.
        bias: Option<String>,
        /// Activation transformation function over [`Expr`].
        activation: Arc<dyn Fn(Expr) -> Expr + Send + Sync>,
    },
    /// Linear scaling epilogue for quantized matmul.
    QuantizedScale {
        /// Row scale buffer name.
        row_scales: String,
        /// Batch scale buffer name.
        batch_scales: String,
    },
}

impl core::fmt::Debug for ContractionEpilogue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Bias {
                buffer,
                count,
                dtype,
            } => f
                .debug_struct("Bias")
                .field("buffer", buffer)
                .field("count", count)
                .field("dtype", dtype)
                .finish(),
            Self::Activation { bias, .. } => {
                f.debug_struct("Activation").field("bias", bias).finish()
            }
            Self::QuantizedScale {
                row_scales,
                batch_scales,
            } => f
                .debug_struct("QuantizedScale")
                .field("row_scales", row_scales)
                .field("batch_scales", batch_scales)
                .finish(),
        }
    }
}

/// Contraction execution geometry and tiling strategy.
#[derive(Clone, Debug)]
pub enum ContractionTiling {
    /// 1D linear invocation grid. Each invocation computes one output element.
    Linear {
        /// Workgroup size configuration.
        workgroup_size: [u32; 3],
    },
    /// 2D cooperative shared-memory tiling with optional MMA tensor core acceleration.
    CooperativeShared {
        /// Tile dimension size.
        tile: u32,
        /// Shared memory buffer name for LHS tiles.
        a_tile_name: String,
        /// Shared memory buffer name for RHS tiles.
        b_tile_name: String,
    },
    /// 1D block-tiled loop over the reduction dimension (reference / oracle structure).
    Block1D {
        /// Tile dimension size.
        tile: u32,
    },
}

/// Geometry of contraction tensors.
#[derive(Clone, Debug)]
pub enum ContractionGeometry {
    /// 2D GEMM: `a: [m, k]`, `b: [k, n]`, `out: [m, n]`.
    Matmul2D {
        /// Row dimension of matrix A and output matrix.
        m: u32,
        /// Shared contraction dimension between A and B.
        k: u32,
        /// Column dimension of matrix B and output matrix.
        n: u32,
    },
    /// 3D Batched GEMM: `a: [batch, m, k]`, `b: [batch, k, n]`, `out: [batch, m, n]`.
    BatchedMatmul3D {
        /// Batch dimension count.
        batch: u32,
        /// Row dimension of matrix A.
        m: u32,
        /// Shared contraction dimension.
        k: u32,
        /// Column dimension of matrix B.
        n: u32,
    },
    /// Row-batched affine projection: `x: [rows, in_dim]`, `w: [in_dim, out_dim]` (or `[out_dim, in_dim]` if `weight_out_in`).
    BatchedRows {
        /// Number of input and output rows.
        rows: u32,
        /// Input projection dimension.
        in_dim: u32,
        /// Output projection dimension.
        out_dim: u32,
        /// True if weights are transposed `[out_dim, in_dim]`.
        weight_out_in: bool,
    },
    /// Matrix-Vector product: `matrix: [n, n]`, `vector: [n]`, `out: [n]`.
    Matvec {
        /// Matrix and vector linear dimension.
        n: u32,
        /// Total cells in the matrix buffer.
        matrix_cells: u32,
    },
    /// 2x2 Strassen 7-multiplication closed form.
    Strassen2x2,
    /// 1-level recursive Strassen 7-multiplication block formula.
    StrassenOneLevel {
        /// Matrix dimension (must be even).
        n: u32,
    },
}

/// Canonical composer for matrix multiplication, projections, and tensor contractions.
#[derive(Clone, Debug)]
pub struct ContractionComposer {
    /// Canonical operation identifier.
    pub op_id: &'static str,
    /// Region generator identifier override.
    pub generator: Option<&'static str>,
    /// Left-hand input tensor descriptor.
    pub a: TensorRef,
    /// Right-hand input tensor descriptor.
    pub b: TensorRef,
    /// Output tensor descriptor.
    pub out: TensorRef,
    /// Optional bias tensor descriptor.
    pub bias: Option<TensorRef>,
    /// Tensor element data type.
    pub dtype: DataType,
    /// Accumulator data type.
    pub acc_dtype: DataType,
    /// Contraction algebraic structure (standard, semiring, fixed-point, custom).
    pub semiring: ContractionSemiring,
    /// Tiling and execution strategy.
    pub tiling: ContractionTiling,
    /// Post-accumulation fused transformation.
    pub epilogue: ContractionEpilogue,
    /// Shape geometry and dimensionality contract.
    pub geometry: ContractionGeometry,
    /// Category-A build options (workgroup override, tenant id).
    pub options: BuildOptions,
}

impl ContractionComposer {
    fn base_linear(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        dtype: DataType,
        acc_dtype: DataType,
        geometry: ContractionGeometry,
    ) -> Self {
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias: None,
            dtype,
            acc_dtype,
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry,
            options: BuildOptions::default(),
        }
    }

    /// Create a standard 2D GEMM composer.
    #[must_use]
    pub fn matmul_2d(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        m: u32,
        k: u32,
        n: u32,
    ) -> Self {
        let dtype = a.dtype.clone();
        Self::base_linear(
            op_id,
            a,
            b,
            out,
            dtype.clone(),
            dtype,
            ContractionGeometry::Matmul2D { m, k, n },
        )
    }

    /// Create a 2D GEMM composer with fused bias.
    #[must_use]
    pub fn matmul_bias_2d(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        bias: TensorRef,
        out: TensorRef,
        m: u32,
        k: u32,
        n: u32,
    ) -> Self {
        let dtype = a.dtype.clone();
        let bias_name = bias.name_str().to_string();
        let mut composer = Self::base_linear(
            op_id,
            a,
            b,
            out,
            dtype.clone(),
            dtype.clone(),
            ContractionGeometry::Matmul2D { m, k, n },
        );
        composer.bias = Some(bias);
        composer.epilogue = ContractionEpilogue::Bias {
            buffer: bias_name,
            count: n,
            dtype,
        };
        composer
    }

    /// Create a cooperative tiled 2D GEMM composer.
    #[must_use]
    pub fn tiled_2d(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        bias: Option<TensorRef>,
        out: TensorRef,
        m: u32,
        k: u32,
        n: u32,
        tile: u32,
    ) -> Self {
        let dtype = a.dtype.clone();
        let epilogue = bias
            .as_ref()
            .map(|b| ContractionEpilogue::Bias {
                buffer: b.name_str().to_string(),
                count: n,
                dtype: dtype.clone(),
            })
            .unwrap_or(ContractionEpilogue::None);
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias,
            dtype: dtype.clone(),
            acc_dtype: dtype,
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::CooperativeShared {
                tile,
                a_tile_name: "matmul_a_tile".to_string(),
                b_tile_name: "matmul_b_tile".to_string(),
            },
            epilogue,
            geometry: ContractionGeometry::Matmul2D { m, k, n },
            options: BuildOptions::default(),
        }
    }

    /// Create a semiring GEMM composer.
    #[must_use]
    pub fn semiring_2d(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        m: u32,
        k: u32,
        n: u32,
        semiring: Semiring,
    ) -> Self {
        let mut composer = Self::base_linear(
            op_id,
            a,
            b,
            out,
            DataType::U32,
            DataType::U32,
            ContractionGeometry::Matmul2D { m, k, n },
        );
        composer.semiring = ContractionSemiring::Closed(semiring);
        composer
    }

    /// Create a 3D batched GEMM composer.
    #[must_use]
    pub fn batched_matmul_3d(
        op_id: &'static str,
        a: TensorRef,
        b: TensorRef,
        out: TensorRef,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) -> Self {
        let dtype = a.dtype.clone();
        Self::base_linear(
            op_id,
            a,
            b,
            out,
            dtype.clone(),
            dtype,
            ContractionGeometry::BatchedMatmul3D { batch, m, k, n },
        )
    }

    /// Create a row-batched affine projection composer.
    #[must_use]
    pub fn batched_rows(
        op_id: &'static str,
        x: TensorRef,
        w: TensorRef,
        bias: Option<TensorRef>,
        out: TensorRef,
        rows: u32,
        in_dim: u32,
        out_dim: u32,
        dtype: DataType,
        weight_out_in: bool,
    ) -> Self {
        let epilogue = bias
            .as_ref()
            .map(|b| ContractionEpilogue::Bias {
                buffer: b.name_str().to_string(),
                count: out_dim,
                dtype: dtype.clone(),
            })
            .unwrap_or(ContractionEpilogue::None);
        let mut composer = Self::base_linear(
            op_id,
            x,
            w,
            out,
            dtype,
            DataType::F32,
            ContractionGeometry::BatchedRows {
                rows,
                in_dim,
                out_dim,
                weight_out_in,
            },
        );
        composer.tiling = ContractionTiling::Linear {
            workgroup_size: [64, 1, 1],
        };
        composer.bias = bias;
        composer.epilogue = epilogue;
        composer
    }

    /// Create a fixed-point u32 matrix-vector contraction composer.
    #[must_use]
    pub fn fixed_u32_matvec(
        op_id: &'static str,
        matrix: TensorRef,
        vector: TensorRef,
        out: TensorRef,
        n: u32,
        matrix_cells: u32,
    ) -> Self {
        let mut composer = Self::base_linear(
            op_id,
            matrix,
            vector,
            out,
            DataType::U32,
            DataType::U32,
            ContractionGeometry::Matvec { n, matrix_cells },
        );
        composer.semiring = ContractionSemiring::Fixed16_16;
        composer
    }

    /// Create a custom u32 matrix contraction composer.
    #[must_use]
    pub fn custom_u32_2d<C, A>(
        op_id: &'static str,
        lhs: TensorRef,
        rhs: TensorRef,
        out: TensorRef,
        m: u32,
        k: u32,
        n: u32,
        identity: u32,
        combine: C,
        accumulate: A,
    ) -> Self
    where
        C: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
        A: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
    {
        let mut composer = Self::base_linear(
            op_id,
            lhs,
            rhs,
            out,
            DataType::U32,
            DataType::U32,
            ContractionGeometry::Matmul2D { m, k, n },
        );
        composer.semiring = ContractionSemiring::Custom {
            identity,
            combine: Arc::new(combine),
            accumulate: Arc::new(accumulate),
        };
        composer
    }
    /// Set workgroup size override.
    #[must_use]
    pub fn with_workgroup_size(mut self, size: [u32; 3]) -> Self {
        self.options = self.options.with_workgroup_size(size);
        self
    }

    /// Set region generator override.
    #[must_use]
    pub fn with_region_generator(mut self, name: &'static str) -> Self {
        self.generator = Some(name);
        self.options = self.options.with_region_generator(name);
        self
    }

    /// Set tenant id.
    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: u32) -> Self {
        self.options = self.options.with_tenant_id(tenant_id);
        self
    }

    /// Set semiring algebra.
    #[must_use]
    pub fn with_semiring(mut self, semiring: ContractionSemiring) -> Self {
        self.semiring = semiring;
        self
    }

    /// Set fused epilogue.
    #[must_use]
    pub fn with_epilogue(mut self, epilogue: ContractionEpilogue) -> Self {
        self.epilogue = epilogue;
        self
    }

    /// Set accumulation data type.
    #[must_use]
    pub fn with_acc_dtype(mut self, acc_dtype: DataType) -> Self {
        self.acc_dtype = acc_dtype;
        self
    }

    /// Set tiling strategy.
    #[must_use]
    pub fn with_tiling(mut self, tiling: ContractionTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Validate tensors and assemble the contraction Program.
    ///
    /// # Errors
    /// Returns [`TensorRefError`] on shape mismatch, dtype mismatch, or element overflow.
    pub fn build(self) -> Result<Program, TensorRefError> {
        let generator = self
            .generator
            .unwrap_or(self.options.region_generator.unwrap_or(self.op_id));

        match &self.geometry {
            ContractionGeometry::Matmul2D { m, k, n } => {
                let m = *m;
                let k = *k;
                let n = *n;

                // Validate tensor shapes and types.
                if let Some(bias) = self.bias.as_ref() {
                    check_tensors(
                        self.op_id,
                        &[
                            (&self.a, self.dtype.clone()),
                            (&self.b, self.dtype.clone()),
                            (bias, bias.dtype.clone()),
                            (&self.out, self.dtype.clone()),
                        ],
                    )?;
                } else {
                    check_tensors(
                        self.op_id,
                        &[
                            (&self.a, self.dtype.clone()),
                            (&self.b, self.dtype.clone()),
                            (&self.out, self.dtype.clone()),
                        ],
                    )?;
                }

                let shape_name = if self.bias.is_some() {
                    "a/b/bias/out"
                } else {
                    "a/b/out"
                };

                let bias_valid = self.bias.as_ref().is_none_or(|b| b.shape.len() == 1);
                if self.a.shape.len() != 2
                    || self.b.shape.len() != 2
                    || !bias_valid
                    || self.out.shape.len() != 2
                {
                    return Err(TensorRefError::ShapeMismatch {
                        name: shape_name.into(),
                        found: vec![],
                        expected: vec![0, 0],
                        op: self.op_id,
                    });
                }
                if m == 0 || k == 0 || n == 0 {
                    return Err(TensorRefError::ShapeMismatch {
                        name: shape_name.into(),
                        found: vec![m, k, n],
                        expected: vec![1, 1, 1],
                        op: self.op_id,
                    });
                }
                if self.b.shape[0] != k {
                    return Err(TensorRefError::ShapeMismatch {
                        name: self.b.name_str().to_string(),
                        found: self.b.shape.to_vec(),
                        expected: vec![k, n],
                        op: self.op_id,
                    });
                }
                if let Some(bias) = self.bias.as_ref() {
                    if bias.shape[0] != n {
                        return Err(TensorRefError::ShapeMismatch {
                            name: bias.name_str().to_string(),
                            found: bias.shape.to_vec(),
                            expected: vec![n],
                            op: self.op_id,
                        });
                    }
                }
                if self.out.shape.as_ref() != [m, n] {
                    return Err(TensorRefError::ShapeMismatch {
                        name: self.out.name_str().to_string(),
                        found: self.out.shape.to_vec(),
                        expected: vec![m, n],
                        op: self.op_id,
                    });
                }

                match &self.tiling {
                    ContractionTiling::Linear { workgroup_size } => {
                        let wg = self.options.workgroup_size.unwrap_or(*workgroup_size);
                        let linear_wg = [
                            wg[0]
                                .max(1)
                                .saturating_mul(wg[1].max(1))
                                .saturating_mul(wg[2].max(1)),
                            1,
                            1,
                        ];
                        build_matmul_2d_linear(
                            self.op_id,
                            generator,
                            self.a.name_str(),
                            self.b.name_str(),
                            self.bias.as_ref().map(TensorRef::name_str),
                            self.out.name_str(),
                            m,
                            k,
                            n,
                            &self.dtype,
                            &self.semiring,
                            &self.epilogue,
                            linear_wg,
                        )
                    }
                    ContractionTiling::CooperativeShared {
                        tile,
                        a_tile_name,
                        b_tile_name,
                    } => {
                        let tile = *tile;
                        if tile == 0 {
                            return Err(TensorRefError::ShapeMismatch {
                                name: "tile".into(),
                                found: vec![0],
                                expected: vec![1],
                                op: self.op_id,
                            });
                        }
                        #[cfg(feature = "math-linalg")]
                        {
                            crate::math::linalg::matmul_tiled::program::build_matmul_tiled_program(
                                crate::math::linalg::matmul_tiled::program::MatmulTiledProgramSpec {
                                    op_id: self.op_id,
                                    a: self.a.name_str(),
                                    b: self.b.name_str(),
                                    bias: self.bias.as_ref().map(TensorRef::name_str),
                                    out: self.out.name_str(),
                                    m,
                                    k,
                                    n,
                                    tile,
                                    workgroup: self
                                        .options
                                        .workgroup_size
                                        .unwrap_or([16, 16, 1]),
                                    generator,
                                    dtype: self.dtype,
                                    a_tile_name,
                                    b_tile_name,
                                    mma_capabilities: crate::math::linalg::matmul_tiled::mma_fragment::MmaCapabilityRecord::all_descriptor_mma_shapes(),
                                },
                            )
                        }
                        #[cfg(not(feature = "math-linalg"))]
                        {
                            let _ = (generator, a_tile_name, b_tile_name);
                            Err(TensorRefError::ShapeMismatch {
                                name: "tiled".into(),
                                found: vec![tile],
                                expected: vec![],
                                op: self.op_id,
                            })
                        }
                    }
                    ContractionTiling::Block1D { tile } => build_block_1d_contraction(
                        self.op_id,
                        generator,
                        self.a.name_str(),
                        self.b.name_str(),
                        self.bias.as_ref().map(TensorRef::name_str),
                        self.out.name_str(),
                        m,
                        k,
                        n,
                        *tile,
                        &self.dtype,
                    ),
                }
            }
            ContractionGeometry::BatchedMatmul3D { batch, m, k, n } => {
                let wg = self.options.workgroup_size.unwrap_or([256, 1, 1]);
                build_batched_3d_contraction(
                    self.op_id,
                    generator,
                    self.a.name_str(),
                    self.b.name_str(),
                    self.out.name_str(),
                    *batch,
                    *m,
                    *k,
                    *n,
                    &self.dtype,
                    wg,
                )
            }
            ContractionGeometry::BatchedRows {
                rows,
                in_dim,
                out_dim,
                weight_out_in,
            } => {
                let wg = self.options.workgroup_size.unwrap_or([64, 1, 1]);
                build_batched_rows_contraction(
                    self.op_id,
                    generator,
                    self.a.name_str(),
                    self.b.name_str(),
                    self.bias.as_ref().map(TensorRef::name_str),
                    self.out.name_str(),
                    *rows,
                    *in_dim,
                    *out_dim,
                    &self.dtype,
                    &self.acc_dtype,
                    *weight_out_in,
                    wg,
                )
            }
            ContractionGeometry::Matvec { n, matrix_cells } => {
                let wg = self.options.workgroup_size.unwrap_or([256, 1, 1]);
                build_matvec_contraction(
                    self.op_id,
                    generator,
                    self.a.name_str(),
                    self.b.name_str(),
                    self.out.name_str(),
                    *n,
                    *matrix_cells,
                    &self.dtype,
                    &self.semiring,
                    wg,
                )
            }
            ContractionGeometry::Strassen2x2 => build_strassen_2x2(
                self.op_id,
                generator,
                self.a.name_str(),
                self.b.name_str(),
                self.out.name_str(),
            ),
            ContractionGeometry::StrassenOneLevel { n } => build_strassen_one_level(
                self.op_id,
                generator,
                self.a.name_str(),
                self.b.name_str(),
                self.out.name_str(),
                *n,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::BufferAccess;
    #[test]
    fn test_contraction_composer_2d_matmul_u32() {
        let a = TensorRef::u32_2d("a", 2, 3);
        let b = TensorRef::u32_2d("b", 3, 2);
        let out = TensorRef::u32_2d("out", 2, 2);
        let program = ContractionComposer::matmul_2d("test_op", a, b, out, 2, 3, 2)
            .build()
            .expect("matmul_2d should build");
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_2d_matmul_bias_u32() {
        let a = TensorRef::u32_2d("a", 2, 3);
        let b = TensorRef::u32_2d("b", 3, 2);
        let bias = TensorRef::u32_1d("bias", 2);
        let out = TensorRef::u32_2d("out", 2, 2);
        let program = ContractionComposer::matmul_bias_2d("test_op", a, b, bias, out, 2, 3, 2)
            .build()
            .expect("matmul_bias_2d should build");
        assert_eq!(program.buffers().len(), 4);
    }

    #[cfg(feature = "math-linalg")]
    #[test]
    fn test_contraction_composer_tiled_2d() {
        let a = TensorRef::u32_2d("a", 16, 16);
        let b = TensorRef::u32_2d("b", 16, 16);
        let out = TensorRef::u32_2d("out", 16, 16);
        let program = ContractionComposer::tiled_2d("test_tiled", a, b, None, out, 16, 16, 16, 16)
            .build()
            .expect("tiled_2d should build");
        assert!(program
            .buffers()
            .iter()
            .any(|b| matches!(b.access, BufferAccess::Workgroup)));
    }

    #[test]
    fn test_contraction_composer_semirings() {
        for semiring in [
            Semiring::Real,
            Semiring::MinPlus,
            Semiring::MaxPlus,
            Semiring::BoolOr,
            Semiring::BoolAnd,
            Semiring::MaxTimes,
            Semiring::Lineage,
            Semiring::Gf2,
        ] {
            let a = TensorRef::u32_2d("a", 2, 2);
            let b = TensorRef::u32_2d("b", 2, 2);
            let out = TensorRef::u32_2d("out", 2, 2);
            let program =
                ContractionComposer::semiring_2d("test_semiring", a, b, out, 2, 2, 2, semiring)
                    .build()
                    .expect("semiring_2d should build");
            assert_eq!(program.buffers().len(), 3);
        }
    }

    #[test]
    fn test_contraction_composer_batched_3d() {
        let a = TensorRef::new("a", DataType::F32, vec![2, 3, 4]);
        let b = TensorRef::new("b", DataType::F32, vec![2, 4, 5]);
        let out = TensorRef::new("out", DataType::F32, vec![2, 3, 5]);
        let program = ContractionComposer::batched_matmul_3d("test_batch", a, b, out, 2, 3, 4, 5)
            .build()
            .expect("batched_matmul_3d should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_batched_rows() {
        let x = TensorRef::new("x", DataType::F32, vec![4, 8]);
        let w = TensorRef::new("w", DataType::F32, vec![8, 16]);
        let bias = TensorRef::new("b", DataType::F32, vec![16]);
        let out = TensorRef::new("out", DataType::F32, vec![4, 16]);
        let program = ContractionComposer::batched_rows(
            "test_rows",
            x,
            w,
            Some(bias),
            out,
            4,
            8,
            16,
            DataType::F32,
            false,
        )
        .build()
        .expect("batched_rows should build");
        assert_eq!(program.buffers().len(), 4);
    }

    #[test]
    fn test_contraction_composer_fixed_matvec() {
        let m = TensorRef::u32_2d("matrix", 4, 4);
        let v = TensorRef::u32_1d("vector", 4);
        let out = TensorRef::u32_1d("out", 4);
        let program = ContractionComposer::fixed_u32_matvec("test_matvec", m, v, out, 4, 16)
            .build()
            .expect("fixed_u32_matvec should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_strassen_2x2() {
        let a = TensorRef::f32_2d("a", 2, 2);
        let b = TensorRef::f32_2d("b", 2, 2);
        let c = TensorRef::f32_2d("c", 2, 2);
        let mut composer = ContractionComposer::matmul_2d("test_strassen", a, b, c, 2, 2, 2);
        composer.geometry = ContractionGeometry::Strassen2x2;
        let program = composer.build().expect("strassen 2x2 should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_strassen_one_level() {
        let a = TensorRef::f32_2d("a", 4, 4);
        let b = TensorRef::f32_2d("b", 4, 4);
        let c = TensorRef::f32_2d("c", 4, 4);
        let mut composer = ContractionComposer::matmul_2d("test_strassen_4", a, b, c, 4, 4, 4);
        composer.geometry = ContractionGeometry::StrassenOneLevel { n: 4 };
        let program = composer.build().expect("strassen one level should build");
        assert_eq!(program.buffers().len(), 3);
    }

    #[test]
    fn test_contraction_composer_rejects_zero_dims() {
        let a = TensorRef::u32_2d("a", 0, 4);
        let b = TensorRef::u32_2d("b", 4, 4);
        let out = TensorRef::u32_2d("out", 0, 4);
        let err = ContractionComposer::matmul_2d("test_err", a, b, out, 0, 4, 4)
            .build()
            .expect_err("zero dim must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_contraction_composer_rejects_shared_dim_mismatch() {
        let a = TensorRef::u32_2d("a", 4, 3);
        let b = TensorRef::u32_2d("b", 5, 4);
        let out = TensorRef::u32_2d("out", 4, 4);
        let err = ContractionComposer::matmul_2d("test_err", a, b, out, 4, 3, 4)
            .build()
            .expect_err("shared dim mismatch must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }

    #[cfg(feature = "math-linalg")]
    #[test]
    fn test_contraction_composer_rejects_tile_zero() {
        let a = TensorRef::u32_2d("a", 4, 4);
        let b = TensorRef::u32_2d("b", 4, 4);
        let out = TensorRef::u32_2d("out", 4, 4);
        let err = ContractionComposer::tiled_2d("test_tiled_zero", a, b, None, out, 4, 4, 4, 0)
            .build()
            .expect_err("tile=0 must fail");
        assert!(matches!(err, TensorRefError::ShapeMismatch { .. }));
    }
}
