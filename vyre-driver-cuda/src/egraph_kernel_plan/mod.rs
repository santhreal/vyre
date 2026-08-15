//! CUDA launch-wave planning for resident e-graph device images.
//!
//! Equality-saturation kernels need deterministic row, child-edge, and
//! e-class-group work partitions. This module converts the checked resident
//! image view into bounded launch waves without rebuilding graph metadata or
//! depending on e-graph semantics in the CUDA backend.

mod backend_canonicalization;
mod backend_rewrite;
mod backend_structural;
mod device_image_rows;
mod error;
mod kernel_abi;
mod launch_waves;
mod plan_equivalence;
mod plan_kernel_work;
mod plan_signature;
mod plan_union;
mod signature_pair_ordinals;
mod types_canonicalization;
mod types_launch;
mod types_signature;
mod types_snapshot;
mod types_union;

mod args;
mod ptx;

#[cfg(test)]
mod tests;

pub use error::CudaEGraphKernelPlanError;
pub use kernel_abi::{
    CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_ENTRY, CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS, CUDA_EGRAPH_SIGNATURE_BUCKET_RECORD_WORDS,
    CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_ENTRY, CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_ENTRY,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_PARAM_COUNT,
};
pub use plan_equivalence::{
    collect_cuda_egraph_structural_equivalences, pack_cuda_egraph_signature_bucket_device_image,
    plan_cuda_egraph_structural_equivalence_launch_artifact,
    plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan,
    plan_cuda_egraph_structural_equivalence_output, plan_cuda_egraph_structural_equivalences,
};
pub use plan_kernel_work::plan_cuda_egraph_kernel_work;
pub use plan_signature::{
    plan_cuda_egraph_signature_buckets, plan_cuda_egraph_signature_buckets_from_resident_snapshot,
    plan_cuda_egraph_signature_buckets_from_signature_snapshot,
};
pub use plan_union::{
    pack_cuda_egraph_canonical_rewrite_device_image, plan_cuda_egraph_union_compaction,
};
pub use ptx::{
    cuda_egraph_canonical_rewrite_kernel_ptx, cuda_egraph_signature_refresh_kernel_ptx,
    cuda_egraph_structural_equivalence_kernel_ptx,
};
pub use signature_pair_ordinals::cuda_egraph_signature_pair_rows;
pub use types_canonicalization::{
    CudaEGraphFixedPointReadback, CudaEGraphStructuralCanonicalizationFixedPointReport,
    CudaEGraphStructuralCanonicalizationFixedPointResult,
    CudaEGraphStructuralCanonicalizationRoundResult,
};
pub use types_launch::{
    CudaEGraphKernelLaunchConfig, CudaEGraphKernelPass, CudaEGraphKernelWave,
    CudaEGraphKernelWorkPlan,
};
pub use types_signature::{
    CudaEGraphSignatureBucket, CudaEGraphSignatureBucketDeviceImage, CudaEGraphSignatureBucketPlan,
    CudaEGraphSignaturePairWave, CudaEGraphStructuralEquivalenceKernelPtx,
    CudaEGraphStructuralEquivalenceKernelResult, CudaEGraphStructuralEquivalenceLaunchArtifact,
    CudaEGraphStructuralEquivalenceOutputPlan, CudaEGraphStructuralEquivalencePlan,
};
pub use types_snapshot::{CudaEGraphResidentColumnSnapshot, CudaEGraphResidentSignatureSnapshot};
pub use types_union::{
    CudaEGraphCanonicalRewrite, CudaEGraphCanonicalRewriteDeviceImage,
    CudaEGraphCanonicalRewriteKernelPtx, CudaEGraphCanonicalRewriteKernelResult,
    CudaEGraphSignatureRefreshKernelPtx, CudaEGraphSignatureRefreshKernelResult,
    CudaEGraphUnionCompactionPass, CudaEGraphUnionCompactionPlan, CudaEGraphUnionCompactionWave,
};
