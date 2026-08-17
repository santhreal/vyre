//! Canonical matrix multiplication and contraction IR composer.
//!
//! Unifies dense 2D GEMM, 3D batched GEMM, row-batched linear projections,
//! semiring matrix multiplications, fixed-point contractions, cooperative
//! tiled GEMM, and fused epilogues (bias, activation, scaling).

use std::sync::Arc;
use vyre_foundation::composition::{wrap_anonymous_region, wrap_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::Semiring;

use crate::builder::{check_tensors, BuildOptions};
use crate::plumbing::operand::tensor_ref::{TensorRef, TensorRefError};

/// Algebraic structure for the contraction inner product: `acc = ⊕ (lhs ⊗ rhs)`.
#[derive(Clone)]
pub enum ContractionSemiring {
    /// Standard arithmetic: `⊗ = *`, `⊕ = +`, identity = 0.
    Standard,
    /// Canonical semirings from [`Semiring`]:
    /// `Real`, `MinPlus`, `MaxPlus`, `BoolOr`, `BoolAnd`, `MaxTimes`, `Lineage`, `Gf2`.
    Closed(Semiring),
    /// Unsigned 16.16 fixed-point arithmetic (`fixed_mul_16_16`, `+`, identity = 0).
    Fixed16_16,
    /// Custom combine and accumulate expressions over `DataType::U32`.
    Custom {
        /// Additive identity value for initializing accumulator.
        identity: u32,
        /// Scalar combine operation.
        combine: Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>,
        /// Scalar accumulate operation.
        accumulate: Arc<dyn Fn(Expr, Expr) -> Expr + Send + Sync>,
    },
}

impl core::fmt::Debug for ContractionSemiring {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::Closed(s) => write!(f, "Closed({s:?})"),
            Self::Fixed16_16 => write!(f, "Fixed16_16"),
            Self::Custom { identity, .. } => {
                f.debug_struct("Custom").field("identity", identity).finish()
            }
        }
    }
}

impl ContractionSemiring {
    /// Additive identity used to initialize accumulators.
    #[must_use]
    pub fn identity_expr(&self, dtype: &DataType) -> Expr {
        match self {
            Self::Standard => match dtype {
                DataType::F32 => Expr::f32(0.0),
                DataType::F64 => Expr::f64(0.0),
                _ => Expr::u32(0),
            },
            Self::Closed(s) => match dtype {
                DataType::F32 => match s {
                    Semiring::MinPlus | Semiring::BoolAnd => Expr::f32(f32::INFINITY),
                    _ => Expr::f32(0.0),
                },
                _ => Expr::u32(s.identity()),
            },
            Self::Fixed16_16 => Expr::u32(0),
            Self::Custom { identity, .. } => Expr::u32(*identity),
        }
    }

    /// Scalar combine operation: `lhs ⊗ rhs`.
    #[must_use]
    pub fn combine_expr(&self, a: Expr, b: Expr) -> Expr {
        match self {
            Self::Standard => Expr::mul(a, b),
            Self::Closed(s) => semiring_combine_expr(*s, a, b),
            Self::Fixed16_16 => fixed_mul_16_16_signed_expr(a, b),
            Self::Custom { combine, .. } => combine(a, b),
        }
    }

    /// Accumulator update: `acc ⊕ value`.
    #[must_use]
    pub fn accumulate_expr(&self, acc: Expr, val: Expr) -> Expr {
        match self {
            Self::Standard => Expr::add(acc, val),
            Self::Closed(s) => semiring_accumulate_expr(*s, acc, val),
            Self::Fixed16_16 => Expr::add(acc, val),
            Self::Custom { accumulate, .. } => accumulate(acc, val),
        }
    }
}

/// Combine expression for canonical semirings.
#[must_use]
pub fn semiring_combine_expr(semiring: Semiring, a: Expr, b: Expr) -> Expr {
    match semiring {
        Semiring::Real | Semiring::MaxTimes => Expr::mul(a, b),
        Semiring::MinPlus => {
            let max_const = Expr::u32(u32::MAX);
            let either_inf = Expr::or(
                Expr::eq(a.clone(), max_const.clone()),
                Expr::eq(b.clone(), max_const.clone()),
            );
            Expr::select(either_inf, max_const, Expr::add(a, b))
        }
        Semiring::MaxPlus => Expr::add(a, b),
        Semiring::BoolOr | Semiring::Gf2 => Expr::bitand(a, b),
        Semiring::BoolAnd => Expr::bitor(a, b),
        Semiring::Lineage => {
            let either_zero = Expr::or(
                Expr::eq(a.clone(), Expr::u32(0)),
                Expr::eq(b.clone(), Expr::u32(0)),
            );
            Expr::select(either_zero, Expr::u32(0), Expr::bitor(a, b))
        }
    }
}

/// Accumulate expression for canonical semirings.
#[must_use]
pub fn semiring_accumulate_expr(semiring: Semiring, acc: Expr, val: Expr) -> Expr {
    match semiring {
        Semiring::Real | Semiring::MaxPlus => Expr::add(acc, val),
        Semiring::MinPlus => Expr::min(acc, val),
        Semiring::MaxTimes => Expr::max(acc, val),
        Semiring::BoolOr | Semiring::Lineage => Expr::bitor(acc, val),
        Semiring::BoolAnd => Expr::bitand(acc, val),
        Semiring::Gf2 => Expr::bitxor(acc, val),
    }
}

/// Signed 16.16 fixed-point multiplication over [`Expr`].
#[must_use]
pub fn fixed_mul_16_16_signed_expr(left: Expr, right: Expr) -> Expr {
    let low = Expr::mul(left.clone(), right.clone());
    let unsigned_high = Expr::mulhi(left.clone(), right.clone());
    let left_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(left.clone(), Expr::u32(31)));
    let right_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(right.clone(), Expr::u32(31)));
    let correction_left = Expr::bitand(left_sign_mask, right);
    let correction_right = Expr::bitand(right_sign_mask, left);
    let signed_high = Expr::sub(Expr::sub(unsigned_high, correction_left), correction_right);
    Expr::bitor(
        Expr::shr(low, Expr::u32(16)),
        Expr::shl(signed_high, Expr::u32(16)),
    )
}

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
            Self::Bias { buffer, count, dtype } => f
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
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias: None,
            dtype: dtype.clone(),
            acc_dtype: dtype,
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry: ContractionGeometry::Matmul2D { m, k, n },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias: Some(bias),
            dtype: dtype.clone(),
            acc_dtype: dtype.clone(),
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::Bias {
                buffer: bias_name,
                count: n,
                dtype,
            },
            geometry: ContractionGeometry::Matmul2D { m, k, n },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias: None,
            dtype: DataType::U32,
            acc_dtype: DataType::U32,
            semiring: ContractionSemiring::Closed(semiring),
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry: ContractionGeometry::Matmul2D { m, k, n },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a,
            b,
            out,
            bias: None,
            dtype: dtype.clone(),
            acc_dtype: dtype,
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry: ContractionGeometry::BatchedMatmul3D { batch, m, k, n },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a: x,
            b: w,
            out,
            bias,
            dtype,
            acc_dtype: DataType::F32,
            semiring: ContractionSemiring::Standard,
            tiling: ContractionTiling::Linear {
                workgroup_size: [64, 1, 1],
            },
            epilogue,
            geometry: ContractionGeometry::BatchedRows {
                rows,
                in_dim,
                out_dim,
                weight_out_in,
            },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a: matrix,
            b: vector,
            out,
            bias: None,
            dtype: DataType::U32,
            acc_dtype: DataType::U32,
            semiring: ContractionSemiring::Fixed16_16,
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry: ContractionGeometry::Matvec { n, matrix_cells },
            options: BuildOptions::default(),
        }
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
        Self {
            op_id,
            generator: None,
            a: lhs,
            b: rhs,
            out,
            bias: None,
            dtype: DataType::U32,
            acc_dtype: DataType::U32,
            semiring: ContractionSemiring::Custom {
                identity,
                combine: Arc::new(combine),
                accumulate: Arc::new(accumulate),
            },
            tiling: ContractionTiling::Linear {
                workgroup_size: [256, 1, 1],
            },
            epilogue: ContractionEpilogue::None,
            geometry: ContractionGeometry::Matmul2D { m, k, n },
            options: BuildOptions::default(),
        }
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
        let generator = self.generator.unwrap_or(
            self.options
                .region_generator
                .unwrap_or(self.op_id),
        );

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
                        let wg = self
                            .options
                            .workgroup_size
                            .unwrap_or(*workgroup_size);
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
                    ContractionTiling::Block1D { tile } => {
                        build_block_1d_contraction(
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
                        )
                    }
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
            ContractionGeometry::Strassen2x2 => {
                build_strassen_2x2(
                    self.op_id,
                    generator,
                    self.a.name_str(),
                    self.b.name_str(),
                    self.out.name_str(),
                )
            }
            ContractionGeometry::StrassenOneLevel { n } => {
                build_strassen_one_level(
                    self.op_id,
                    generator,
                    self.a.name_str(),
                    self.b.name_str(),
                    self.out.name_str(),
                    *n,
                )
            }
        }
    }
}

/// Assemble 2D GEMM with 1D linear dispatch.
fn build_matmul_2d_linear(
    op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    bias: Option<&str>,
    out: &str,
    m: u32,
    k: u32,
    n: u32,
    dtype: &DataType,
    semiring: &ContractionSemiring,
    epilogue: &ContractionEpilogue,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let a_count = m
        .checked_mul(k)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![m, k],
        })?;
    let b_count = k
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![k, n],
        })?;
    let out_count = m
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![m, n],
        })?;

    let idx = Expr::InvocationId { axis: 0 };
    let row_expr = Expr::div(idx.clone(), Expr::u32(n));
    let col_expr = Expr::rem(idx.clone(), Expr::u32(n));

    let a_load = Expr::load(
        a,
        Expr::add(Expr::mul(Expr::var("row"), Expr::u32(k)), Expr::var("kk")),
    );
    let b_load = Expr::load(
        b,
        Expr::add(Expr::mul(Expr::var("kk"), Expr::u32(n)), Expr::var("col")),
    );
    let combined = semiring.combine_expr(a_load, b_load);
    let folded = semiring.accumulate_expr(Expr::var("acc"), combined);

    let (initial_acc, store_value) = match epilogue {
        ContractionEpilogue::None => (semiring.identity_expr(dtype), Expr::var("acc")),
        ContractionEpilogue::Bias { buffer, .. } => (
            semiring.identity_expr(dtype),
            Expr::add(Expr::var("acc"), Expr::load(buffer, Expr::var("col"))),
        ),
        ContractionEpilogue::Activation {
            bias: Some(bias_buf),
            activation,
        } => (
            Expr::load(bias_buf, Expr::var("col")),
            activation(Expr::var("acc")),
        ),
        ContractionEpilogue::Activation {
            bias: None,
            activation,
        } => (semiring.identity_expr(dtype), activation(Expr::var("acc"))),
        ContractionEpilogue::QuantizedScale {
            row_scales,
            batch_scales,
        } => (
            semiring.identity_expr(dtype),
            Expr::mul(
                Expr::mul(Expr::var("acc"), Expr::load(row_scales, Expr::var("row"))),
                Expr::load(batch_scales, Expr::var("batch")),
            ),
        ),
    };

    let body = vec![Node::if_then(
        Expr::lt(idx.clone(), Expr::buf_len(out)),
        vec![
            Node::let_bind("row", row_expr),
            Node::let_bind("col", col_expr),
            Node::let_bind("acc", initial_acc),
            Node::loop_for(
                "kk",
                Expr::u32(0),
                Expr::u32(k),
                vec![Node::assign("acc", folded)],
            ),
            Node::Store {
                buffer: out.into(),
                index: idx,
                value: store_value,
            },
        ],
    )];

    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
    ];
    let out_slot = if let Some(bias_name) = bias {
        buffers.push(
            BufferDecl::storage(bias_name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, out_slot, dtype.clone()).with_count(out_count));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 3D batched GEMM: `out[b, i, j] = sum_k a[b, i, k] * b[b, k, j]`.
fn build_batched_3d_contraction(
    _op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    out: &str,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    dtype: &DataType,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let a_batch_stride = m
        .checked_mul(k)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![batch, m, k],
        })?;
    let b_batch_stride = k
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![batch, k, n],
        })?;
    let out_batch_stride = m
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![batch, m, n],
        })?;
    let a_count = batch
        .checked_mul(a_batch_stride)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![batch, m, k],
        })?;
    let b_count = batch
        .checked_mul(b_batch_stride)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![batch, k, n],
        })?;
    let out_count = batch
        .checked_mul(out_batch_stride)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![batch, m, n],
        })?;

    let idx = Expr::var("idx");
    let batch_idx = Expr::var("batch_idx");
    let row = Expr::var("row");
    let col = Expr::var("col");
    let local_idx = Expr::var("local_idx");

    let body = vec![
        Node::let_bind("idx", Expr::InvocationId { axis: 0 }),
        Node::let_bind(
            "batch_idx",
            Expr::div(idx.clone(), Expr::u32(out_batch_stride)),
        ),
        Node::let_bind(
            "local_idx",
            Expr::rem(idx.clone(), Expr::u32(out_batch_stride)),
        ),
        Node::let_bind("row", Expr::div(local_idx.clone(), Expr::u32(n))),
        Node::let_bind("col", Expr::rem(local_idx.clone(), Expr::u32(n))),
        Node::if_then(
            Expr::lt(idx.clone(), Expr::buf_len(out)),
            vec![
                Node::let_bind("acc", Expr::f32(0.0)),
                Node::loop_for(
                    "kk",
                    Expr::u32(0),
                    Expr::u32(k),
                    vec![Node::assign(
                        "acc",
                        Expr::add(
                            Expr::var("acc"),
                            Expr::mul(
                                Expr::load(
                                    a,
                                    Expr::add(
                                        Expr::mul(batch_idx.clone(), Expr::u32(a_batch_stride)),
                                        Expr::add(
                                            Expr::mul(row.clone(), Expr::u32(k)),
                                            Expr::var("kk"),
                                        ),
                                    ),
                                ),
                                Expr::load(
                                    b,
                                    Expr::add(
                                        Expr::mul(batch_idx.clone(), Expr::u32(b_batch_stride)),
                                        Expr::add(
                                            Expr::mul(Expr::var("kk"), Expr::u32(n)),
                                            col.clone(),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: idx,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
        BufferDecl::output(out, 2, dtype.clone()).with_count(out_count),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble row-batched affine projection with F32 accumulation.
fn build_batched_rows_contraction(
    _op_id: &'static str,
    generator: &'static str,
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    dtype: &DataType,
    acc_dtype: &DataType,
    weight_out_in: bool,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let input_count = rows
        .checked_mul(in_dim)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: x.to_string(),
            shape: vec![rows, in_dim],
        })?;
    let output_count = rows
        .checked_mul(out_dim)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![rows, out_dim],
        })?;
    let weight_count = in_dim
        .checked_mul(out_dim)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: w.to_string(),
            shape: vec![in_dim, out_dim],
        })?;

    let index = Expr::var("index");
    let row = Expr::div(index.clone(), Expr::u32(out_dim));
    let column = Expr::rem(index.clone(), Expr::u32(out_dim));
    let accumulator = bias.map_or_else(
        || Expr::f32(0.0),
        |name| Expr::cast(acc_dtype.clone(), Expr::load(name, Expr::var("column"))),
    );
    let weight_index = if weight_out_in {
        Expr::add(
            Expr::mul(Expr::var("column"), Expr::u32(in_dim)),
            Expr::var("inner"),
        )
    } else {
        Expr::add(
            Expr::mul(Expr::var("inner"), Expr::u32(out_dim)),
            Expr::var("column"),
        )
    };

    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(output_count)),
            vec![
                Node::let_bind("row", row),
                Node::let_bind("column", column),
                Node::let_bind("accumulator", accumulator),
                Node::loop_for(
                    "inner",
                    Expr::u32(0),
                    Expr::u32(in_dim),
                    vec![Node::assign(
                        "accumulator",
                        Expr::add(
                            Expr::var("accumulator"),
                            Expr::mul(
                                Expr::cast(
                                    acc_dtype.clone(),
                                    Expr::load(
                                        x,
                                        Expr::add(
                                            Expr::mul(Expr::var("row"), Expr::u32(in_dim)),
                                            Expr::var("inner"),
                                        ),
                                    ),
                                ),
                                Expr::cast(acc_dtype.clone(), Expr::load(w, weight_index)),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index,
                    value: Expr::cast(dtype.clone(), Expr::var("accumulator")),
                },
            ],
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(input_count),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let output_slot = if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(output_count));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 1D block-tiled reference contraction loop.
fn build_block_1d_contraction(
    _op_id: &'static str,
    generator: &'static str,
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    _m: u32,
    in_dim: u32,
    out_dim: u32,
    tile: u32,
    dtype: &DataType,
) -> Result<Program, TensorRefError> {
    let weight_count = in_dim
        .checked_mul(out_dim)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: w.to_string(),
            shape: vec![in_dim, out_dim],
        })?;
    let tile_count = in_dim.div_ceil(tile);
    let lane = Expr::var("lane");
    let kk = Expr::var("kk");

    let initial_acc = bias.map_or_else(
        || Expr::u32(0),
        |b| Expr::load(b, lane.clone()),
    );

    let body = vec![
        Node::let_bind("lane", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(lane.clone(), Expr::u32(out_dim)),
            vec![
                Node::let_bind("acc", initial_acc),
                Node::loop_for(
                    "tile_idx",
                    Expr::u32(0),
                    Expr::u32(tile_count),
                    vec![
                        Node::let_bind(
                            "tile_base",
                            Expr::mul(Expr::var("tile_idx"), Expr::u32(tile)),
                        ),
                        Node::loop_for(
                            "tile_k",
                            Expr::u32(0),
                            Expr::u32(tile),
                            vec![
                                Node::let_bind(
                                    "kk",
                                    Expr::add(Expr::var("tile_base"), Expr::var("tile_k")),
                                ),
                                Node::if_then(
                                    Expr::lt(kk.clone(), Expr::u32(in_dim)),
                                    vec![Node::assign(
                                        "acc",
                                        Expr::add(
                                            Expr::var("acc"),
                                            Expr::mul(
                                                Expr::load(x, kk.clone()),
                                                Expr::load(
                                                    w,
                                                    Expr::add(
                                                        Expr::mul(kk.clone(), Expr::u32(out_dim)),
                                                        lane.clone(),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )],
                                ),
                            ],
                        ),
                    ],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: lane,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(in_dim),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let output_slot = if let Some(b) = bias {
        buffers.push(
            BufferDecl::storage(b, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(out_dim));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [256, 1, 1], vec![region]))
}

/// Assemble Matrix-Vector contraction.
fn build_matvec_contraction(
    _op_id: &'static str,
    generator: &'static str,
    matrix: &str,
    vector: &str,
    out: &str,
    n: u32,
    matrix_cells: u32,
    dtype: &DataType,
    semiring: &ContractionSemiring,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let row = Expr::InvocationId { axis: 0 };
    let row_base = Expr::mul(row.clone(), Expr::u32(n));

    let combined = semiring.combine_expr(
        Expr::load(matrix, Expr::add(row_base, Expr::var("j"))),
        Expr::load(vector, Expr::var("j")),
    );
    let folded = semiring.accumulate_expr(Expr::var("acc"), combined);

    let body = vec![Node::if_then(
        Expr::lt(row.clone(), Expr::u32(n)),
        vec![
            Node::let_bind("acc", semiring.identity_expr(dtype)),
            Node::let_bind("row_base", Expr::mul(row.clone(), Expr::u32(n))),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::assign("acc", folded)],
            ),
            Node::store(out, row, Expr::var("acc")),
        ],
    )];

    let buffers = vec![
        BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, dtype.clone())
            .with_count(matrix_cells),
        BufferDecl::storage(vector, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
        BufferDecl::storage(out, 2, BufferAccess::ReadWrite, dtype.clone()).with_count(n),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 2x2 Strassen 7-multiplication closed-form Program.
fn build_strassen_2x2(
    _op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    c: &str,
) -> Result<Program, TensorRefError> {
    let body = vec![
        Node::let_bind("a00", Expr::load(a, Expr::u32(0))),
        Node::let_bind("a01", Expr::load(a, Expr::u32(1))),
        Node::let_bind("a10", Expr::load(a, Expr::u32(2))),
        Node::let_bind("a11", Expr::load(a, Expr::u32(3))),
        Node::let_bind("b00", Expr::load(b, Expr::u32(0))),
        Node::let_bind("b01", Expr::load(b, Expr::u32(1))),
        Node::let_bind("b10", Expr::load(b, Expr::u32(2))),
        Node::let_bind("b11", Expr::load(b, Expr::u32(3))),
        Node::let_bind(
            "m1",
            Expr::mul(
                Expr::add(Expr::var("a00"), Expr::var("a11")),
                Expr::add(Expr::var("b00"), Expr::var("b11")),
            ),
        ),
        Node::let_bind(
            "m2",
            Expr::mul(
                Expr::add(Expr::var("a10"), Expr::var("a11")),
                Expr::var("b00"),
            ),
        ),
        Node::let_bind(
            "m3",
            Expr::mul(
                Expr::var("a00"),
                Expr::sub(Expr::var("b01"), Expr::var("b11")),
            ),
        ),
        Node::let_bind(
            "m4",
            Expr::mul(
                Expr::var("a11"),
                Expr::sub(Expr::var("b10"), Expr::var("b00")),
            ),
        ),
        Node::let_bind(
            "m5",
            Expr::mul(
                Expr::add(Expr::var("a00"), Expr::var("a01")),
                Expr::var("b11"),
            ),
        ),
        Node::let_bind(
            "m6",
            Expr::mul(
                Expr::sub(Expr::var("a10"), Expr::var("a00")),
                Expr::add(Expr::var("b00"), Expr::var("b01")),
            ),
        ),
        Node::let_bind(
            "m7",
            Expr::mul(
                Expr::sub(Expr::var("a01"), Expr::var("a11")),
                Expr::add(Expr::var("b10"), Expr::var("b11")),
            ),
        ),
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(0),
            value: Expr::add(
                Expr::sub(Expr::add(Expr::var("m1"), Expr::var("m4")), Expr::var("m5")),
                Expr::var("m7"),
            ),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(1),
            value: Expr::add(Expr::var("m3"), Expr::var("m5")),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(2),
            value: Expr::add(Expr::var("m2"), Expr::var("m4")),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(3),
            value: Expr::add(
                Expr::add(Expr::sub(Expr::var("m1"), Expr::var("m2")), Expr::var("m3")),
                Expr::var("m6"),
            ),
        },
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::output(c, 2, DataType::F32).with_count(4),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [1, 1, 1], vec![region]))
}

/// Assemble 1-level recursive Strassen 7-multiplication block Program.
fn build_strassen_one_level(
    _op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    c: &str,
    n: u32,
) -> Result<Program, TensorRefError> {
    let half = n / 2;
    let total = n
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: c.to_string(),
            shape: vec![n, n],
        })?;

    let body = vec![
        Node::let_bind("flat", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("flat"), Expr::u32(total)),
            vec![
                Node::let_bind("row", Expr::div(Expr::var("flat"), Expr::u32(n))),
                Node::let_bind("col", Expr::rem(Expr::var("flat"), Expr::u32(n))),
                Node::let_bind("q_row", Expr::div(Expr::var("row"), Expr::u32(half))),
                Node::let_bind("q_col", Expr::div(Expr::var("col"), Expr::u32(half))),
                Node::let_bind("sr", Expr::rem(Expr::var("row"), Expr::u32(half))),
                Node::let_bind("sc", Expr::rem(Expr::var("col"), Expr::u32(half))),
                Node::let_bind("c_val", Expr::f32(0.0)),
                Node::let_bind("m1", Expr::f32(0.0)),
                Node::let_bind("m2", Expr::f32(0.0)),
                Node::let_bind("m3", Expr::f32(0.0)),
                Node::let_bind("m4", Expr::f32(0.0)),
                Node::let_bind("m5", Expr::f32(0.0)),
                Node::let_bind("m6", Expr::f32(0.0)),
                Node::let_bind("m7", Expr::f32(0.0)),
                Node::loop_for(
                    "k",
                    Expr::u32(0),
                    Expr::u32(half),
                    vec![
                        Node::let_bind(
                            "a11",
                            Expr::load(
                                a,
                                Expr::add(Expr::mul(Expr::var("sr"), Expr::u32(n)), Expr::var("k")),
                            ),
                        ),
                        Node::let_bind(
                            "a12",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(Expr::var("sr"), Expr::u32(n)),
                                    Expr::add(Expr::u32(half), Expr::var("k")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "a21",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("sr"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::var("k"),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "a22",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("sr"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::add(Expr::u32(half), Expr::var("k")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b11",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(Expr::var("k"), Expr::u32(n)),
                                    Expr::var("sc"),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b12",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(Expr::var("k"), Expr::u32(n)),
                                    Expr::add(Expr::u32(half), Expr::var("sc")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b21",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("k"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::var("sc"),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b22",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("k"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::add(Expr::u32(half), Expr::var("sc")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m1",
                            Expr::add(
                                Expr::var("m1"),
                                Expr::mul(
                                    Expr::add(Expr::var("a11"), Expr::var("a22")),
                                    Expr::add(Expr::var("b11"), Expr::var("b22")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m2",
                            Expr::add(
                                Expr::var("m2"),
                                Expr::mul(
                                    Expr::add(Expr::var("a21"), Expr::var("a22")),
                                    Expr::var("b11"),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m3",
                            Expr::add(
                                Expr::var("m3"),
                                Expr::mul(
                                    Expr::var("a11"),
                                    Expr::sub(Expr::var("b12"), Expr::var("b22")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m4",
                            Expr::add(
                                Expr::var("m4"),
                                Expr::mul(
                                    Expr::var("a22"),
                                    Expr::sub(Expr::var("b21"), Expr::var("b11")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m5",
                            Expr::add(
                                Expr::var("m5"),
                                Expr::mul(
                                    Expr::add(Expr::var("a11"), Expr::var("a12")),
                                    Expr::var("b22"),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m6",
                            Expr::add(
                                Expr::var("m6"),
                                Expr::mul(
                                    Expr::sub(Expr::var("a21"), Expr::var("a11")),
                                    Expr::add(Expr::var("b11"), Expr::var("b12")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m7",
                            Expr::add(
                                Expr::var("m7"),
                                Expr::mul(
                                    Expr::sub(Expr::var("a12"), Expr::var("a22")),
                                    Expr::add(Expr::var("b21"), Expr::var("b22")),
                                ),
                            ),
                        ),
                    ],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(0)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(0)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(
                            Expr::sub(
                                Expr::add(Expr::var("m1"), Expr::var("m4")),
                                Expr::var("m5"),
                            ),
                            Expr::var("m7"),
                        ),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(0)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(1)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(Expr::var("m3"), Expr::var("m5")),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(1)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(0)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(Expr::var("m2"), Expr::var("m4")),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(1)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(1)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(
                            Expr::add(
                                Expr::sub(Expr::var("m1"), Expr::var("m2")),
                                Expr::var("m3"),
                            ),
                            Expr::var("m6"),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: c.into(),
                    index: Expr::var("flat"),
                    value: Expr::var("c_val"),
                },
            ],
        ),
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::F32).with_count(total),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::F32).with_count(total),
        BufferDecl::output(c, 2, DataType::F32).with_count(total),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [64, 1, 1], vec![region]))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(program.buffers().iter().any(|b| matches!(b.access, BufferAccess::Workgroup)));
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
            let program = ContractionComposer::semiring_2d("test_semiring", a, b, out, 2, 2, 2, semiring)
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
