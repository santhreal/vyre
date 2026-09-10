//! Canonical matrix multiplication and contraction IR composer.
//!
//! Unifies dense 2D GEMM, 3D batched GEMM, row-batched linear projections,
//! semiring matrix multiplications, fixed-point contractions, cooperative
//! tiled GEMM, and fused epilogues (bias, activation, scaling).

use std::sync::Arc;
use vyre_foundation::ir::{DataType, Expr, Program};
use vyre_megakernel::DeviceFacts;
use vyre_spec::Semiring;

#[path = "contraction_buffers.rs"]
mod contraction_buffers;

#[path = "gemm_algebra.rs"]
mod gemm_algebra;
pub use gemm_algebra::*;

#[cfg(test)]
#[path = "gemm_contracts.rs"]
mod gemm_contracts;

#[path = "gemm_programs.rs"]
mod gemm_programs;
use gemm_programs::*;
#[path = "strassen_programs.rs"]
mod strassen_programs;
use strassen_programs::*;
#[path = "tiled_gemm_programs.rs"]
mod tiled_gemm_programs;
use tiled_gemm_programs::*;

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
    /// Register-tiled invocation grid. Each invocation accumulates a
    /// `rows x columns` tile of outputs across the whole contraction
    /// dimension, so one staged left value serves `columns` accumulators and
    /// one staged right value serves `rows` of them.
    RegisterTiled {
        /// Output rows one invocation accumulates.
        rows: u32,
        /// Output columns one invocation accumulates.
        columns: u32,
        /// Workgroup size configuration.
        workgroup_size: [u32; 3],
    },
    /// 1D block-tiled loop over the reduction dimension (reference / oracle structure).
    Block1D {
        /// Tile dimension size.
        tile: u32,
    },
}

/// Output tile one invocation of a contraction accumulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractionOutputTile {
    /// Output rows one invocation accumulates.
    pub rows: u32,
    /// Output columns one invocation accumulates.
    pub columns: u32,
}

impl ContractionOutputTile {
    /// Live scalars the tiled body holds besides its accumulators and its
    /// staged operands: the flat tile index, the two tile coordinates, the
    /// two tile origins, and the contraction induction variable.
    const BODY_LIVE_SCALARS: u32 = 6;

    /// Registers one invocation holds for a `rows x columns` output tile.
    ///
    /// The tiled body keeps `rows * columns` accumulators live across the
    /// contraction loop, and stages one left value per tile row and one right
    /// value per tile column inside each iteration.
    #[must_use]
    pub const fn register_footprint(rows: u32, columns: u32) -> u32 {
        rows.saturating_mul(columns)
            .saturating_add(rows)
            .saturating_add(columns)
            .saturating_add(Self::BODY_LIVE_SCALARS)
    }

    /// Largest output tile the declared extents and the stated device budgets
    /// admit for one invocation.
    ///
    /// The tile grows squarely while its register footprint fits the stated
    /// per-invocation budget, then extends along whichever declared extent
    /// still has room. Facts that state no register budget admit no tile, and
    /// so does a geometry whose output is one element wide in both extents:
    /// the caller keeps the untiled candidate in both cases rather than
    /// receiving a tile derived from an assumed budget.
    #[must_use]
    pub fn derive(rows: u32, columns: u32, facts: &DeviceFacts) -> Option<Self> {
        let budget = facts.registers_per_invocation();
        if rows == 0 || columns == 0 || budget <= Self::BODY_LIVE_SCALARS {
            return None;
        }

        let mut tile = Self {
            rows: 1,
            columns: 1,
        };
        while tile.rows < rows
            && tile.columns < columns
            && Self::register_footprint(tile.rows + 1, tile.columns + 1) <= budget
        {
            tile.rows += 1;
            tile.columns += 1;
        }
        while tile.columns < columns
            && Self::register_footprint(tile.rows, tile.columns + 1) <= budget
        {
            tile.columns += 1;
        }
        while tile.rows < rows && Self::register_footprint(tile.rows + 1, tile.columns) <= budget {
            tile.rows += 1;
        }

        (tile.rows > 1 || tile.columns > 1).then_some(tile)
    }
}

/// Workgroup the stated device admits for a tiled contraction launch.
///
/// The invocation count is the largest whole number of subgroups the stated
/// per-workgroup invocation limit holds. A device that states no subgroup size
/// contributes only its invocation limit, and one that states neither reports
/// a bound at all.
#[must_use]
pub fn tiled_workgroup_bound(facts: &DeviceFacts) -> Option<u32> {
    let limit = facts.max_invocations_per_workgroup();
    if limit == 0 {
        return None;
    }
    let subgroup = facts.subgroup_size();
    let invocations = if subgroup == 0 {
        limit
    } else {
        (limit / subgroup) * subgroup
    };
    (invocations != 0).then_some(invocations)
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
    /// Facts of the device the contraction will run on. Absent facts state no
    /// budget, which admits no physical tile.
    pub device: DeviceFacts,
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
            device: DeviceFacts::unknown(),
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
            device: DeviceFacts::unknown(),
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

    /// State the facts of the device the contraction will run on and derive
    /// the register tile those facts admit.
    ///
    /// A row-batched contraction accumulates the largest output tile the
    /// declared extents and the stated register budget admit. The launch keeps
    /// the declared workgroup. The invocation limit the device states is a
    /// fact, reported by `tiled_workgroup_bound`, and ranking a launch shape
    /// against it is one cost model's decision rather than this builder's:
    /// widening a launch here would also move geometry the artifact froze.
    #[must_use]
    pub fn with_device_facts(mut self, facts: DeviceFacts) -> Self {
        self.device = facts;
        if let ContractionGeometry::BatchedRows {
            rows,
            out_dim,
            in_dim: _,
            weight_out_in: _,
        } = self.geometry
        {
            if let Some(tile) = ContractionOutputTile::derive(rows, out_dim, &facts) {
                let declared = match self.tiling {
                    ContractionTiling::Linear { workgroup_size }
                    | ContractionTiling::RegisterTiled { workgroup_size, .. } => workgroup_size,
                    ContractionTiling::CooperativeShared { tile, .. } => [tile, tile, 1],
                    ContractionTiling::Block1D { tile } => [tile, 1, 1],
                };
                self.tiling = ContractionTiling::RegisterTiled {
                    rows: tile.rows,
                    columns: tile.columns,
                    workgroup_size: declared,
                };
            }
        }
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

                        let wg = self.options.workgroup_size.unwrap_or([tile, tile, 1]);
                        build_matmul_2d_cooperative(
                            generator,
                            self.a.name_str(),
                            self.b.name_str(),
                            self.bias.as_ref().map(TensorRef::name_str),
                            self.out.name_str(),
                            m,
                            k,
                            n,
                            tile,
                            a_tile_name,
                            b_tile_name,
                            &self.dtype,
                            &self.semiring,
                            &self.epilogue,
                            wg,
                        )
                    }
                    ContractionTiling::Block1D { tile } => build_block_1d_contraction(
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
                    ContractionTiling::RegisterTiled { .. } => {
                        Err(TensorRefError::UnsupportedTiling {
                            tiling: "RegisterTiled",
                            op: self.op_id,
                        })
                    }
                }
            }
            ContractionGeometry::BatchedMatmul3D { batch, m, k, n } => {
                let wg = self.options.workgroup_size.unwrap_or([256, 1, 1]);
                build_batched_3d_contraction(
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
                match &self.tiling {
                    ContractionTiling::RegisterTiled {
                        rows: tile_rows,
                        columns,
                        workgroup_size,
                    } => {
                        let wg = self.options.workgroup_size.unwrap_or(*workgroup_size);
                        build_batched_rows_register_tiled(
                            generator,
                            self.a.name_str(),
                            self.b.name_str(),
                            self.bias.as_ref().map(TensorRef::name_str),
                            self.out.name_str(),
                            *rows,
                            *in_dim,
                            *out_dim,
                            *tile_rows,
                            *columns,
                            &self.dtype,
                            &self.acc_dtype,
                            *weight_out_in,
                            wg,
                        )
                    }
                    ContractionTiling::Linear { workgroup_size } => {
                        let wg = self.options.workgroup_size.unwrap_or(*workgroup_size);
                        build_batched_rows_contraction(
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
                    ContractionTiling::CooperativeShared { .. } => {
                        Err(TensorRefError::UnsupportedTiling {
                            tiling: "CooperativeShared",
                            op: self.op_id,
                        })
                    }
                    ContractionTiling::Block1D { .. } => Err(TensorRefError::UnsupportedTiling {
                        tiling: "Block1D",
                        op: self.op_id,
                    }),
                }
            }
            ContractionGeometry::Matvec { n, matrix_cells } => {
                let wg = self.options.workgroup_size.unwrap_or([256, 1, 1]);
                build_matvec_contraction(
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
                generator,
                self.a.name_str(),
                self.b.name_str(),
                self.out.name_str(),
            ),
            ContractionGeometry::StrassenOneLevel { n } => build_strassen_one_level(
                generator,
                self.a.name_str(),
                self.b.name_str(),
                self.out.name_str(),
                *n,
            ),
        }
    }
}
