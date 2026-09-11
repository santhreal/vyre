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
    },
}

/// One candidate strategy for a contraction site.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractionCandidate {
    /// Identifier or label of the contraction site.
    pub contraction_id: String,
    /// Selected strategy for this candidate.
    pub strategy: ContractionStrategy,
    /// Estimated speedup factor over the un-tiled scalar baseline.
    pub estimated_speedup_factor: f32,
    /// Data types supported by this candidate.
    pub supported_dtypes: Vec<DataType>,
}

/// Contraction candidate plan for a kernel descriptor.
pub type ContractionPlan = CandidatePlan<ContractionCandidate>;
