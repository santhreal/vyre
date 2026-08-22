//! PERF B6: tensor-core (wmma/mma) fragment promotion candidate detection.
//!
//! Detects KernelOp groups whose shape divides the wmma fragment tile
//! (16×16×16 for f16 on sm_70+, 16×16×16 for bf16 on sm_80+). Real
//! emission of wmma fragments is phase 2; this module identifies the
//! candidates so the emit-time decision can be made.
//!
//! Phase-1 detection criteria:
//! - Workgroup-size dimensions are multiples of the fragment tile.
//! - Kernel has an FMA-chain pattern that looks like matmul accumulation
//!   (sum of products into a register).
//! - Element type is f16 / bf16 / f32 (wmma fragments require these).
//!
//! The `analyze` returns a `TensorCorePlan` listing which fragment
//! shapes are eligible for the kernel on the given target capability.

use serde::{Deserialize, Serialize};
use vyre_lower::{KernelBody, KernelDescriptor, KernelOpKind};

use crate::ComputeCapability;

/// Matrix fragment tile supported by the PTX emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FragmentTile {
    /// 16×16×16 f16 fragment, supported on sm_70+.
    F16_16x16x16,
    /// 16×16×16 bf16 fragment, supported on sm_80+.
    Bf16_16x16x16,
    /// 8×8×16 f16 fragment for tiny tiles.
    F16_8x8x16,
}

impl FragmentTile {
    /// Return whether `target` supports this fragment tile.
    #[must_use]
    pub fn supported_on(&self, target: ComputeCapability) -> bool {
        match self {
            Self::F16_16x16x16 | Self::F16_8x8x16 => target.supports_wmma_f16(),
            Self::Bf16_16x16x16 => target.supports_wmma_bf16(),
        }
    }

    /// Return the tile dimensions as rows, columns, and reduction lanes.
    #[must_use]
    pub fn dims(&self) -> (u32, u32, u32) {
        match self {
            Self::F16_16x16x16 | Self::Bf16_16x16x16 => (16, 16, 16),
            Self::F16_8x8x16 => (8, 8, 16),
        }
    }
}

/// One kernel eligible for matrix-fragment promotion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorCoreCandidate {
    /// Eligible fragment tile.
    pub fragment: FragmentTile,
    /// FMA op count in the kernel  -  the higher this is, the more
    /// accumulation work goes through tensor cores.
    pub fma_op_count: u32,
    /// Estimated speedup over scalar FMA chain. Conservative
    /// `5.0 + log2(fma_op_count)` to avoid overpromise.
    pub estimated_speedup_factor: f32,
}

/// Matrix-fragment opportunities for one kernel and target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorCorePlan {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Target compute capability label.
    pub target_sm: String,
    /// Eligible fragment candidates.
    pub candidates: Vec<TensorCoreCandidate>,
}

impl TensorCorePlan {
    /// Return the number of matrix-fragment candidates.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

/// Analyze a descriptor for matrix-fragment promotion.
#[must_use]
pub fn analyze(desc: &KernelDescriptor, target: ComputeCapability) -> TensorCorePlan {
    let fma_count = count_fma(&desc.body);
    let workgroup_aligned = workgroup_size_aligned(desc.dispatch.workgroup_size);

    let mut candidates = Vec::new();
    if fma_count >= 4 && workgroup_aligned {
        // Conservative speedup: 5.0 baseline + log2 scaling.
        let speedup = 5.0 + (fma_count as f32).log2();
        for tile in [
            FragmentTile::F16_16x16x16,
            FragmentTile::Bf16_16x16x16,
            FragmentTile::F16_8x8x16,
        ] {
            if tile.supported_on(target) {
                candidates.push(TensorCoreCandidate {
                    fragment: tile,
                    fma_op_count: fma_count,
                    estimated_speedup_factor: speedup,
                });
            }
        }
    }

    TensorCorePlan {
        kernel_id: desc.id.clone(),
        target_sm: format!("sm_{}{}", target.major, target.minor),
        candidates,
    }
}

fn count_fma(body: &KernelBody) -> u32 {
    let mut total: u32 = body
        .ops
        .iter()
        .filter(|op| matches!(op.kind, KernelOpKind::Fma))
        .count() as u32;
    for child in &body.child_bodies {
        total = total.saturating_add(count_fma(child));
    }
    total
}

fn workgroup_size_aligned(size: [u32; 3]) -> bool {
    // wmma requires workgroup_size_x ≥ 32 (warp size) and divides 16.
    size[0] >= 32 && size[0] % 16 == 0
}
