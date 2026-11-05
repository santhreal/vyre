//! Output type for the contraction candidate selection analysis.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;

use crate::analyses::candidate_plan::CandidatePlan;
use crate::descriptor::{MatrixMmaElement, MatrixMmaLayout, MatrixTileShape};

/// Strategy for lowering and executing a contraction site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContractionStrategy {
    /// Scalar baseline: one output per invocation, sequential reduction loop.
    Scalar,
    /// SIMT cooperative 2D workgroup memory tiling.
    SimtTiled {
        /// M tile extent.
        tile_m: u32,
        /// N tile extent.
        tile_n: u32,
        /// K tile extent.
        tile_k: u32,
        /// Workgroup size configuration.
        workgroup_size: [u32; 3],
    },
    /// Target matrix-multiply-accumulate (MMA / tensor-core) fragment instruction.
    MatrixInstruction {
        /// Matrix tile shape (M, N, K).
        tile: MatrixTileShape,
        /// Left fragment matrix layout.
        left_layout: MatrixMmaLayout,
        /// Right fragment matrix layout.
        right_layout: MatrixMmaLayout,
        /// Left element type.
        left_element: MatrixMmaElement,
        /// Right element type.
        right_element: MatrixMmaElement,
        /// Accumulator element type.
        acc_element: MatrixMmaElement,
        /// Where the tile extents and fragment element types came from.
        source: MatrixInstructionSource,
    },
}

/// Where a matrix-instruction candidate's extents came from.
///
/// A candidate whose extents no descriptor stated would claim a fragment
/// packing the program never expressed, so the only source is the descriptor
/// itself. The variant is recorded rather than assumed so a candidate cannot
/// be read as target-derived when it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixInstructionSource {
    /// Extents and fragment element types read from a `MatrixMma` operation in
    /// the lowered body.
    DeclaredByDescriptor,
}

/// One candidate strategy for a contraction site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractionCandidate {
    /// Identifier or label of the contraction site.
    pub contraction_id: String,
    /// Selected strategy for this candidate.
    pub strategy: ContractionStrategy,
    /// Operand elements read from memory per multiply-accumulate under this
    /// strategy.
    ///
    /// This is counted work, not a measured time: the scalar baseline reads
    /// both operands of every multiply-accumulate and so counts 2.0, and a
    /// strategy that stages a tile counts the reuse that staging buys. A
    /// device outcome is a measurement made above this analysis.
    pub operand_loads_per_fma: f32,
    /// Data types supported by this candidate, taken from the bindings the
    /// kernel reads.
    pub supported_dtypes: Vec<DataType>,
    /// What the extents, element types and load count were derived from.
    pub derivation: String,
}

impl ContractionCandidate {
    /// Operand loads this candidate avoids per multiply-accumulate, relative
    /// to the scalar baseline's two.
    ///
    /// A ratio of counted loads. It bounds what reuse can buy on a
    /// load-bound contraction and states nothing about a device.
    #[must_use]
    pub fn operand_reuse_factor(&self) -> f32 {
        if self.operand_loads_per_fma <= 0.0 {
            return f32::INFINITY;
        }
        SCALAR_OPERAND_LOADS_PER_FMA / self.operand_loads_per_fma
    }
}

/// Operand elements the scalar baseline reads per multiply-accumulate.
pub const SCALAR_OPERAND_LOADS_PER_FMA: f32 = 2.0;

/// Contraction candidate plan for a kernel descriptor.
pub type ContractionPlan = CandidatePlan<ContractionCandidate>;
