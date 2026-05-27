//! CUDA launch-wave planning for resident e-graph device images.
//!
//! Equality-saturation kernels need deterministic row, child-edge, and
//! e-class-group work partitions. This module converts the checked resident
//! image view into bounded launch waves without rebuilding graph metadata or
//! depending on e-graph semantics in the CUDA backend.

use std::fmt;

use crate::backend::ordering::{sort_unstable_by_key_if_needed, sort_unstable_if_needed};
use crate::backend::staging_reserve::{reserved_typed_vec, CudaStorageReserveFailure};
use crate::backend::{CudaBackend, CudaResidentBuffer};
use crate::egraph_device_image::{
    plan_cuda_egraph_device_upload_from_image_ref, CudaEGraphDeviceKernelView,
};
use crate::egraph_readback::{
    cleanup_egraph_kernel_handles, decode_unique_equivalence_pairs, device_ptr_at,
    download_structural_equivalence_output_ranges, egraph_column_snapshot_readback_bytes,
    egraph_column_snapshot_spans, read_resident_u32_range, read_u64_le,
    upload_structural_equivalence_scratch, upload_u32_words,
};
use crate::numeric::CUDA_NUMERIC;
use crate::CudaResidentEGraphDeviceImage;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vyre_driver::BackendError;
use vyre_driver::LaunchPlan;
use vyre_foundation::optimizer::eqsat_gpu::{Equivalence, GpuEGraphDeviceImage};

const DEFAULT_THREADS_PER_BLOCK: u32 = 256;
const DEFAULT_MAX_BLOCKS_PER_LAUNCH: u32 = 65_535;
const SIGNATURE_BUCKET_RECORD_WORDS: usize = 5;

/// CUDA structural-equivalence kernel entry symbol.
pub const CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_ENTRY: &str = "main";

/// Number of u32 words in one packed signature-bucket record.
pub const CUDA_EGRAPH_SIGNATURE_BUCKET_RECORD_WORDS: usize = SIGNATURE_BUCKET_RECORD_WORDS;

/// Number of parameters in the structural-equivalence PTX kernel ABI.
pub const CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_PARAM_COUNT: usize = 13;

/// CUDA canonical-rewrite kernel entry symbol.
pub const CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_ENTRY: &str = "main";

/// Number of parameters in the canonical-rewrite PTX kernel ABI.
pub const CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_PARAM_COUNT: usize = 7;

/// Number of u32 words in one packed canonical-rewrite record.
pub const CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS: usize = 2;

/// CUDA row-signature refresh kernel entry symbol.
pub const CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_ENTRY: &str = "main";

/// Number of parameters in the row-signature refresh PTX kernel ABI.
pub const CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_PARAM_COUNT: usize = 7;

mod args;
use args::{EGraphCanonicalRewriteKernelArgs, EGraphSignatureRefreshKernelArgs};
mod ptx;
pub use ptx::{
    cuda_egraph_canonical_rewrite_kernel_ptx, cuda_egraph_signature_refresh_kernel_ptx,
    cuda_egraph_structural_equivalence_kernel_ptx,
};

/// E-graph kernel pass represented in CUDA launch planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaEGraphKernelPass {
    /// Per-row canonicalization or op/arity scanning.
    RowScan,
    /// Per-child-edge traversal over the flat child e-class column.
    ChildEdgeScan,
    /// Per-e-class grouped-row processing.
    EclassGroupScan,
    /// Per-candidate structural-signature row-pair comparison.
    StructuralSignaturePairScan,
}

/// Launch-shaping controls for e-graph kernel work planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphKernelLaunchConfig {
    /// CUDA threads per block.
    pub threads_per_block: u32,
    /// Maximum blocks emitted into one launch wave.
    pub max_blocks_per_launch: u32,
}

impl Default for CudaEGraphKernelLaunchConfig {
    fn default() -> Self {
        Self {
            threads_per_block: DEFAULT_THREADS_PER_BLOCK,
            max_blocks_per_launch: DEFAULT_MAX_BLOCKS_PER_LAUNCH,
        }
    }
}

/// One bounded CUDA launch wave for an e-graph kernel pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphKernelWave {
    /// Kernel pass.
    pub pass: CudaEGraphKernelPass,
    /// First logical row/edge/group item handled by this wave.
    pub first_item: u64,
    /// Logical row/edge/group item count handled by this wave.
    pub item_count: u64,
    /// CUDA blocks for this wave.
    pub blocks: u32,
    /// CUDA threads per block for this wave.
    pub threads_per_block: u32,
}

/// Complete launch plan for resident e-graph kernel passes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphKernelWorkPlan {
    /// Checked resident image view used by kernels.
    pub view: CudaEGraphDeviceKernelView,
    /// Bounded launch waves in deterministic pass order.
    pub waves: Vec<CudaEGraphKernelWave>,
    /// Sum of logical items across all waves.
    pub total_items: u64,
    /// Sum of CUDA blocks across all waves.
    pub total_blocks: u64,
}

/// Host snapshot of the CUDA-resident e-graph columns needed to plan another
/// structural canonicalization round after device mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphResidentColumnSnapshot {
    /// One e-class id per row.
    pub row_eclass_ids: Vec<u32>,
    /// One language op id per row.
    pub row_language_op_ids: Vec<u32>,
    /// Child-column offset per row.
    pub row_children_offsets: Vec<u32>,
    /// Child count per row.
    pub row_children_lens: Vec<u32>,
    /// One structural signature per row.
    pub row_signatures: Vec<u32>,
    /// Flat child e-class column.
    pub children: Vec<u32>,
    /// Number of e-class groups in the resident image.
    pub eclass_group_count: usize,
}

impl CudaEGraphResidentColumnSnapshot {
    /// Build a resident-column snapshot from a foundation-packed image using
    /// fallible exact-reserve copies for every large column.
    pub fn try_from_device_image(
        image: &GpuEGraphDeviceImage,
    ) -> Result<Self, CudaEGraphKernelPlanError> {
        Ok(Self {
            row_eclass_ids: copy_u32_snapshot_column(
                image.row_eclass_ids(),
                "resident snapshot row eclass ids",
            )?,
            row_language_op_ids: copy_u32_snapshot_column(
                image.row_language_op_ids(),
                "resident snapshot row language op ids",
            )?,
            row_children_offsets: copy_u32_snapshot_column(
                image.row_children_offsets(),
                "resident snapshot row child offsets",
            )?,
            row_children_lens: copy_u32_snapshot_column(
                image.row_children_lens(),
                "resident snapshot row child lengths",
            )?,
            row_signatures: copy_u32_snapshot_column(
                image.row_signatures(),
                "resident snapshot row signatures",
            )?,
            children: copy_u32_snapshot_column(image.children(), "resident snapshot children")?,
            eclass_group_count: image.layout().eclass_group_count(),
        })
    }

    /// Build a resident-column snapshot from a foundation-packed image before
    /// any CUDA-side mutation has occurred.
    #[must_use]
    pub fn from_device_image(image: &GpuEGraphDeviceImage) -> Self {
        Self {
            row_eclass_ids: image.row_eclass_ids().to_vec(),
            row_language_op_ids: image.row_language_op_ids().to_vec(),
            row_children_offsets: image.row_children_offsets().to_vec(),
            row_children_lens: image.row_children_lens().to_vec(),
            row_signatures: image.row_signatures().to_vec(),
            children: image.children().to_vec(),
            eclass_group_count: image.layout().eclass_group_count(),
        }
    }

    /// Number of rows in the snapshot.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_signatures.len()
    }

    /// Number of child entries in the snapshot.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }
}

/// Lightweight host snapshot of the CUDA-resident row-signature column needed
/// to plan the next structural candidate buckets after device mutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphResidentSignatureSnapshot {
    /// One structural signature per row.
    pub row_signatures: Vec<u32>,
    /// Number of child entries in the resident image.
    pub child_count: usize,
    /// Number of e-class groups in the resident image.
    pub eclass_group_count: usize,
}

impl CudaEGraphResidentSignatureSnapshot {
    /// Build a signature snapshot from a foundation-packed image using a
    /// fallible exact-reserve copy for the row-signature column.
    pub fn try_from_device_image(
        image: &GpuEGraphDeviceImage,
    ) -> Result<Self, CudaEGraphKernelPlanError> {
        Ok(Self {
            row_signatures: copy_u32_snapshot_column(
                image.row_signatures(),
                "resident signature snapshot row signatures",
            )?,
            child_count: image.layout().child_count(),
            eclass_group_count: image.layout().eclass_group_count(),
        })
    }

    /// Build a signature snapshot from a foundation-packed image before any
    /// CUDA-side mutation has occurred.
    #[must_use]
    pub fn from_device_image(image: &GpuEGraphDeviceImage) -> Self {
        Self {
            row_signatures: image.row_signatures().to_vec(),
            child_count: image.layout().child_count(),
            eclass_group_count: image.layout().eclass_group_count(),
        }
    }

    /// Build a signature snapshot from a full resident-column snapshot.
    #[must_use]
    pub fn from_column_snapshot(snapshot: &CudaEGraphResidentColumnSnapshot) -> Self {
        Self {
            row_signatures: snapshot.row_signatures.clone(),
            child_count: snapshot.child_count(),
            eclass_group_count: snapshot.eclass_group_count,
        }
    }

    /// Build a signature snapshot from a full resident-column snapshot using
    /// fallible exact-reserve storage for the copied signature column.
    pub fn try_from_column_snapshot(
        snapshot: &CudaEGraphResidentColumnSnapshot,
    ) -> Result<Self, CudaEGraphKernelPlanError> {
        Ok(Self {
            row_signatures: copy_u32_snapshot_column(
                &snapshot.row_signatures,
                "resident signature snapshot from full row signatures",
            )?,
            child_count: snapshot.child_count(),
            eclass_group_count: snapshot.eclass_group_count,
        })
    }

    /// Number of rows in the snapshot.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_signatures.len()
    }

    /// Number of child entries in the resident image.
    #[must_use]
    pub const fn child_count(&self) -> usize {
        self.child_count
    }
}

fn copy_u32_snapshot_column(
    column: &[u32],
    field: &'static str,
) -> Result<Vec<u32>, CudaEGraphKernelPlanError> {
    let mut out = reserved_typed_vec(column.len(), field)?;
    out.extend_from_slice(column);
    Ok(out)
}

/// Rows that share one structural signature and therefore need exact
/// device-side comparison before any equivalence is emitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignatureBucket {
    /// Shared structural row signature.
    pub signature: u32,
    /// First row index for this bucket inside
    /// [`CudaEGraphSignatureBucketPlan::bucket_rows`].
    pub first_bucket_row: u32,
    /// Number of rows in this signature bucket.
    pub row_count: u32,
    /// Number of unordered row pairs represented by this bucket.
    pub candidate_pair_count: u64,
}

/// Bounded CUDA launch wave for exact comparison of signature-candidate pairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignaturePairWave {
    /// Index into [`CudaEGraphSignatureBucketPlan::buckets`].
    pub bucket_index: u32,
    /// First pair ordinal inside the bucket's triangular pair space.
    pub first_pair: u64,
    /// Number of pair ordinals compared by this wave.
    pub pair_count: u64,
    /// CUDA blocks for this wave.
    pub blocks: u32,
    /// CUDA threads per block for this wave.
    pub threads_per_block: u32,
}

/// Device-work plan for structural duplicate discovery from row signatures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignatureBucketPlan {
    /// Checked resident image view used by kernels.
    pub view: CudaEGraphDeviceKernelView,
    /// Candidate buckets sorted by signature, then row index.
    pub buckets: Vec<CudaEGraphSignatureBucket>,
    /// Row ids for all buckets, concatenated by bucket.
    pub bucket_rows: Vec<u32>,
    /// Bounded pair-comparison launch waves.
    pub pair_waves: Vec<CudaEGraphSignaturePairWave>,
    /// Total exact row comparisons required after signature filtering.
    pub candidate_pair_count: u64,
    /// Sum of CUDA blocks across all pair waves.
    pub total_blocks: u64,
}

/// Exact structural-equivalence output derived from signature buckets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralEquivalencePlan {
    /// Signature bucket plan that bounded exact comparison work.
    pub signature_plan: CudaEGraphSignatureBucketPlan,
    /// Unique e-class merge candidates proven by exact packed-row comparison.
    pub equivalences: Vec<Equivalence>,
    /// Candidate pairs that survived exact row comparison.
    pub exact_pair_count: u64,
    /// Exact pairs already inside the same e-class and therefore redundant.
    pub redundant_pair_count: u64,
    /// Candidate pairs rejected after exact op/arity/child comparison.
    pub rejected_candidate_pair_count: u64,
    /// U32 words required by a compact `(left, right)` equivalence output.
    pub equivalence_output_words: usize,
}

/// Device-packed signature-bucket table consumed by structural-equivalence
/// kernels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignatureBucketDeviceImage {
    /// Fixed-width records: `(signature, first_bucket_row, row_count,
    /// candidate_pair_count_lo, candidate_pair_count_hi)`.
    pub bucket_words: Vec<u32>,
    /// Concatenated row ids referenced by bucket records.
    pub bucket_rows: Vec<u32>,
    /// Number of bucket records.
    pub bucket_count: usize,
    /// Number of u32 words per bucket record.
    pub bucket_record_words: usize,
    /// Total candidate pairs represented by the bucket table.
    pub candidate_pair_count: u64,
}

/// Bounded output buffers needed by a structural-equivalence kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralEquivalenceOutputPlan {
    /// Worst-case emitted equivalences before exact duplicate compaction.
    pub max_equivalences: u64,
    /// U32 words required for `(left, right)` output pairs.
    pub output_pair_words: usize,
    /// Bytes required for `(left, right)` output pairs.
    pub output_pair_bytes: usize,
    /// U32 words required for the atomic output counter.
    pub output_counter_words: usize,
    /// Bytes required for the atomic output counter.
    pub output_counter_bytes: usize,
}

/// Complete resident-kernel launch artifact for structural e-graph
/// equivalence discovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralEquivalenceLaunchArtifact {
    /// Packed signature-bucket metadata uploaded beside the e-graph image.
    pub bucket_image: CudaEGraphSignatureBucketDeviceImage,
    /// Output buffer sizing for the kernel.
    pub output: CudaEGraphStructuralEquivalenceOutputPlan,
    /// Bounded pair-comparison waves to launch.
    pub pair_waves: Vec<CudaEGraphSignaturePairWave>,
}

/// PTX source and ABI metadata for the structural-equivalence kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralEquivalenceKernelPtx {
    /// CUDA SM target used by the PTX preamble.
    pub target_sm: u32,
    /// PTX ISA version emitted in the preamble.
    pub ptx_version: &'static str,
    /// Entry symbol resolved by the CUDA module loader.
    pub entry_name: &'static str,
    /// Number of kernel parameters in the ABI.
    pub parameter_count: usize,
    /// Number of u32 words per signature-bucket record.
    pub bucket_record_words: usize,
    /// Complete PTX source.
    pub source: String,
}

/// Result produced by launching the structural-equivalence CUDA kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralEquivalenceKernelResult {
    /// Raw emitted e-class pair count before host deduplication and excluding
    /// device-side overflow beyond the planned output capacity.
    pub emitted_pair_count: u64,
    /// Unique sorted e-class pairs after host compaction.
    pub unique: Vec<Equivalence>,
    /// Number of pairs reported by the device atomic counter before capping to
    /// the planned output capacity.
    pub device_reported_count: u64,
    /// Whether device output exceeded the planned output-pair capacity.
    pub overflowed_output_capacity: bool,
}

/// CUDA pass used when applying structural e-graph equivalence output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaEGraphUnionCompactionPass {
    /// Merge canonicalized `(left, right)` e-class pairs into a device-side
    /// union-find parent column.
    UnionPairs,
    /// Rewrite non-representative e-classes to their deterministic canonical
    /// representative after path compression.
    CanonicalRewrites,
}

/// One bounded CUDA launch wave for e-graph union/compaction work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphUnionCompactionWave {
    /// Union/compaction pass.
    pub pass: CudaEGraphUnionCompactionPass,
    /// First logical pair or rewrite handled by this wave.
    pub first_item: u64,
    /// Logical pair or rewrite count handled by this wave.
    pub item_count: u64,
    /// CUDA blocks for this wave.
    pub blocks: u32,
    /// CUDA threads per block for this wave.
    pub threads_per_block: u32,
}

/// Deterministic e-class canonicalization rewrite produced after union
/// compaction planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphCanonicalRewrite {
    /// E-class id that should be rewritten.
    pub eclass_id: u32,
    /// Stable representative e-class id.
    pub representative: u32,
}

/// Deterministic apply plan for CUDA structural-equivalence output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphUnionCompactionPlan {
    /// Unique sorted non-self e-class pairs to union.
    pub canonical_pairs: Vec<Equivalence>,
    /// Sorted e-class ids touched by the union batch.
    pub affected_eclasses: Vec<u32>,
    /// Stable non-representative to representative rewrites after union
    /// closure.
    pub canonical_rewrites: Vec<CudaEGraphCanonicalRewrite>,
    /// Bounded union and rewrite launch waves.
    pub waves: Vec<CudaEGraphUnionCompactionWave>,
    /// Number of self-pairs dropped before planning.
    pub ignored_self_pair_count: u64,
    /// Number of duplicate or reversed duplicate pairs removed before
    /// planning.
    pub duplicate_pair_count: u64,
    /// Sum of logical union/rewrite items across all waves.
    pub total_items: u64,
    /// Sum of CUDA blocks across all waves.
    pub total_blocks: u64,
}

/// Device-packed canonical rewrite table consumed by the CUDA e-graph
/// canonicalization kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphCanonicalRewriteDeviceImage {
    /// Fixed-width records: `(eclass_id, representative)`.
    pub rewrite_words: Vec<u32>,
    /// Number of rewrite records.
    pub rewrite_count: usize,
    /// Number of u32 words per rewrite record.
    pub rewrite_record_words: usize,
}

/// PTX source and ABI metadata for the canonical-rewrite kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphCanonicalRewriteKernelPtx {
    /// CUDA SM target used by the PTX preamble.
    pub target_sm: u32,
    /// PTX ISA version emitted in the preamble.
    pub ptx_version: &'static str,
    /// Entry symbol resolved by the CUDA module loader.
    pub entry_name: &'static str,
    /// Number of kernel parameters in the ABI.
    pub parameter_count: usize,
    /// Number of u32 words per canonical-rewrite record.
    pub rewrite_record_words: usize,
    /// Complete PTX source.
    pub source: String,
}

/// Result produced by launching the canonical-rewrite CUDA kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphCanonicalRewriteKernelResult {
    /// Number of rewrite records uploaded for binary-search lookup.
    pub rewrite_count: usize,
    /// Number of row e-class ids covered by the launch.
    pub row_count: usize,
    /// Number of child e-class ids covered by the launch.
    pub child_count: usize,
    /// Number of CUDA launch waves issued.
    pub launch_count: usize,
    /// Sum of logical row/child items covered by launches.
    pub total_items: u64,
}

/// PTX source and ABI metadata for the row-signature refresh kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignatureRefreshKernelPtx {
    /// CUDA SM target used by the PTX preamble.
    pub target_sm: u32,
    /// PTX ISA version emitted in the preamble.
    pub ptx_version: &'static str,
    /// Entry symbol resolved by the CUDA module loader.
    pub entry_name: &'static str,
    /// Number of kernel parameters in the ABI.
    pub parameter_count: usize,
    /// Complete PTX source.
    pub source: String,
}

/// Result produced by launching the row-signature refresh CUDA kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaEGraphSignatureRefreshKernelResult {
    /// Number of snapshot rows covered by the launch.
    pub row_count: usize,
    /// Number of CUDA launch waves issued.
    pub launch_count: usize,
    /// Sum of logical row items covered by launches.
    pub total_rows: u64,
}

/// End-to-end result for one CUDA-resident e-graph structural
/// canonicalization round.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralCanonicalizationRoundResult {
    /// Exact structural duplicate discovery result.
    pub discovery: CudaEGraphStructuralEquivalenceKernelResult,
    /// Deterministic union/rewrite plan derived from discovered pairs.
    pub union_plan: CudaEGraphUnionCompactionPlan,
    /// Device-side canonical rewrite launch result.
    pub rewrite: CudaEGraphCanonicalRewriteKernelResult,
    /// Device-side row-signature refresh launch result after rewrites.
    pub signature_refresh: CudaEGraphSignatureRefreshKernelResult,
}

/// Result for iterative CUDA-resident e-graph canonicalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralCanonicalizationFixedPointResult {
    /// Rounds executed, including the final no-op round when convergence is
    /// proven before `max_rounds`.
    pub rounds: Vec<CudaEGraphStructuralCanonicalizationRoundResult>,
    /// Resident columns after the last executed round.
    pub final_snapshot: CudaEGraphResidentColumnSnapshot,
    /// `true` iff a no-op discovery round proved fixed-point convergence.
    pub converged: bool,
    /// Maximum number of rounds requested by the caller.
    pub max_rounds: usize,
    /// Total unique equivalence pairs discovered across all rounds.
    pub total_discovered_pairs: u64,
    /// Total canonical rewrite records applied across all rounds.
    pub total_rewrites: u64,
}

/// Final host readback policy for iterative CUDA-resident e-graph
/// canonicalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaEGraphFixedPointReadback {
    /// Do not copy any final resident columns back to the host. This is the
    /// hot path when the caller keeps the e-graph resident for later kernels.
    None,
    /// Copy only refreshed row signatures back to the host. This is sufficient
    /// for planning later structural candidate buckets without transferring
    /// row ids, op ids, child offsets, child lengths, or children.
    Signatures,
    /// Copy all planning columns back to the host.
    FullColumns,
}

impl Default for CudaEGraphFixedPointReadback {
    fn default() -> Self {
        Self::FullColumns
    }
}

/// Policy-aware result for iterative CUDA-resident e-graph canonicalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaEGraphStructuralCanonicalizationFixedPointReport {
    /// Rounds executed, including the final no-op round when convergence is
    /// proven before `max_rounds`.
    pub rounds: Vec<CudaEGraphStructuralCanonicalizationRoundResult>,
    /// Full resident columns after the last executed round when requested by
    /// [`CudaEGraphFixedPointReadback::FullColumns`].
    pub final_snapshot: Option<CudaEGraphResidentColumnSnapshot>,
    /// Signature-only resident snapshot after the last executed round when
    /// requested directly or derivable from a full-column readback.
    pub final_signature_snapshot: Option<CudaEGraphResidentSignatureSnapshot>,
    /// Final host readback policy used by this run.
    pub final_readback: CudaEGraphFixedPointReadback,
    /// Bytes a full final resident-column readback would transfer.
    pub final_full_readback_bytes: usize,
    /// Bytes represented by the final row-signature column.
    pub final_signature_snapshot_bytes: usize,
    /// Additional bytes transferred at the final readback boundary by the
    /// selected policy. Signature-only reports reuse the already-current
    /// planning snapshot and therefore do not force another device-to-host
    /// copy.
    pub final_additional_readback_bytes: usize,
    /// Full-column final readback bytes avoided by the selected policy.
    pub avoided_final_readback_bytes: usize,
    /// `true` iff a no-op discovery round proved fixed-point convergence.
    pub converged: bool,
    /// Maximum number of rounds requested by the caller.
    pub max_rounds: usize,
    /// Total unique equivalence pairs discovered across all rounds.
    pub total_discovered_pairs: u64,
    /// Total canonical rewrite records applied across all rounds.
    pub total_rewrites: u64,
}

/// Error returned when e-graph kernel work cannot be planned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaEGraphKernelPlanError {
    /// Threads per block was zero.
    ZeroThreadsPerBlock,
    /// Maximum blocks per launch was zero.
    ZeroMaxBlocksPerLaunch,
    /// Count arithmetic overflowed.
    CountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// PTX generation received an invalid CUDA SM target.
    InvalidPtxTarget {
        /// Invalid `sm_XX` target.
        target_sm: u32,
    },
    /// Packed image metadata did not match the checked CUDA kernel view.
    ImageViewMismatch {
        /// Field that disagreed.
        field: &'static str,
        /// Count from the packed foundation image.
        image: usize,
        /// Count from the CUDA kernel view.
        view: usize,
    },
    /// A packed row child span pointed outside the packed child column.
    ImageColumnOutOfBounds {
        /// Column being decoded.
        column: &'static str,
        /// Row being decoded.
        row: u32,
        /// Start index into the column.
        start: usize,
        /// End index into the column.
        end: usize,
        /// Column length.
        len: usize,
    },
    /// A pair ordinal did not identify a valid row pair in a signature bucket.
    SignaturePairOrdinalOutOfBounds {
        /// Signature bucket being decoded.
        bucket_index: u32,
        /// Pair ordinal inside the bucket's triangular pair space.
        pair_ordinal: u64,
        /// Number of candidate pairs in the bucket.
        candidate_pair_count: u64,
    },
    /// A signature bucket's row range pointed outside the bucket row table.
    SignatureBucketRowsOutOfBounds {
        /// Signature bucket being decoded.
        bucket_index: u32,
        /// First row offset in the bucket row table.
        first_bucket_row: usize,
        /// Bucket row count.
        row_count: usize,
        /// Available row table length.
        bucket_rows_len: usize,
    },
    /// Planner storage reservation failed.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Requested element count.
        requested: usize,
        /// Allocator error text.
        message: String,
    },
}

impl CudaStorageReserveFailure for CudaEGraphKernelPlanError {
    fn storage_reserve_failed(field: &'static str, requested: usize, message: String) -> Self {
        Self::StorageReserveFailed {
            field,
            requested,
            message,
        }
    }
}

impl fmt::Display for CudaEGraphKernelPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroThreadsPerBlock => write!(
                f,
                "CUDA e-graph kernel planner received zero threads per block. Fix: choose a non-zero launch width before planning equality-saturation work."
            ),
            Self::ZeroMaxBlocksPerLaunch => write!(
                f,
                "CUDA e-graph kernel planner received zero max blocks per launch. Fix: choose a non-zero launch partition limit."
            ),
            Self::CountOverflow { field } => write!(
                f,
                "CUDA e-graph kernel planner overflowed while computing {field}. Fix: shard the resident e-graph image before launch planning."
            ),
            Self::InvalidPtxTarget { target_sm } => write!(
                f,
                "CUDA e-graph structural-equivalence PTX generation received invalid sm_{target_sm}. Fix: pass the backend's probed CUDA PTX target."
            ),
            Self::ImageViewMismatch { field, image, view } => write!(
                f,
                "CUDA e-graph kernel planner received mismatched {field}: packed image has {image}, kernel view has {view}. Fix: build the view from the same upload plan/image."
            ),
            Self::ImageColumnOutOfBounds {
                column,
                row,
                start,
                end,
                len,
            } => write!(
                f,
                "CUDA e-graph kernel planner decoded row {row} span {column}[{start}..{end}) but {column} has {len} entries. Fix: rebuild the packed e-graph image from a validated snapshot."
            ),
            Self::SignaturePairOrdinalOutOfBounds {
                bucket_index,
                pair_ordinal,
                candidate_pair_count,
            } => write!(
                f,
                "CUDA e-graph signature bucket {bucket_index} pair ordinal {pair_ordinal} is outside {candidate_pair_count} candidate pairs. Fix: launch only planned pair-wave ranges."
            ),
            Self::SignatureBucketRowsOutOfBounds {
                bucket_index,
                first_bucket_row,
                row_count,
                bucket_rows_len,
            } => write!(
                f,
                "CUDA e-graph signature bucket {bucket_index} row range [{first_bucket_row}..{}) exceeds bucket row table length {bucket_rows_len}. Fix: rebuild the signature bucket plan.",
                first_bucket_row.saturating_add(*row_count)
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "CUDA e-graph kernel planner could not reserve {requested} {field} entries: {message}. Fix: shard the resident e-graph image before launch planning."
            ),
        }
    }
}

impl std::error::Error for CudaEGraphKernelPlanError {}

/// Plan bounded CUDA launch waves for a resident e-graph image.
pub fn plan_cuda_egraph_kernel_work(
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphKernelWorkPlan, CudaEGraphKernelPlanError> {
    if config.threads_per_block == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroThreadsPerBlock);
    }
    if config.max_blocks_per_launch == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch);
    }

    let row_count = usize_to_u64(view.row_count(), "row count")?;
    let child_count = usize_to_u64(view.child_count(), "child count")?;
    let group_count = usize_to_u64(view.eclass_group_count(), "eclass group count")?;
    let row_waves = wave_count_for(row_count, config)?;
    let child_waves = wave_count_for(child_count, config)?;
    let group_waves = wave_count_for(group_count, config)?;
    let wave_count = row_waves
        .checked_add(child_waves)
        .and_then(|count| count.checked_add(group_waves))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "wave count",
        })?;
    let mut waves = reserved_typed_vec(
        usize::try_from(wave_count).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
            field: "wave count usize conversion",
        })?,
        "egraph kernel waves",
    )?;

    let mut total_items = 0_u64;
    let mut total_blocks = 0_u64;
    append_pass_waves(
        &mut waves,
        &mut total_items,
        &mut total_blocks,
        CudaEGraphKernelPass::RowScan,
        row_count,
        config,
    )?;
    append_pass_waves(
        &mut waves,
        &mut total_items,
        &mut total_blocks,
        CudaEGraphKernelPass::ChildEdgeScan,
        child_count,
        config,
    )?;
    append_pass_waves(
        &mut waves,
        &mut total_items,
        &mut total_blocks,
        CudaEGraphKernelPass::EclassGroupScan,
        group_count,
        config,
    )?;

    Ok(CudaEGraphKernelWorkPlan {
        view,
        waves,
        total_items,
        total_blocks,
    })
}

/// Plan structural-signature candidate buckets for GPU-side e-graph
/// equivalence discovery.
///
/// Row signatures are a prefilter only: kernels must still compare
/// language-op ids, child lengths, and child columns before emitting an
/// equivalence. The value of this plan is that the expensive exact comparison
/// runs only on compact candidate buckets instead of every row pair.
pub fn plan_cuda_egraph_signature_buckets(
    image: &GpuEGraphDeviceImage,
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphSignatureBucketPlan, CudaEGraphKernelPlanError> {
    plan_cuda_egraph_signature_buckets_from_column(
        image.row_signatures(),
        image.layout().row_count(),
        image.layout().child_count(),
        image.layout().eclass_group_count(),
        view,
        config,
    )
}

/// Plan structural-signature candidate buckets from a current CUDA-resident
/// column snapshot.
///
/// This is the planning path used after a CUDA kernel mutates resident row or
/// child e-class ids and refreshes the resident signature column. Exact
/// comparison still happens against device memory; this host snapshot only
/// bounds candidate-pair launch work.
pub fn plan_cuda_egraph_signature_buckets_from_resident_snapshot(
    snapshot: &CudaEGraphResidentColumnSnapshot,
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphSignatureBucketPlan, CudaEGraphKernelPlanError> {
    plan_cuda_egraph_signature_buckets_from_column(
        &snapshot.row_signatures,
        snapshot.row_count(),
        snapshot.child_count(),
        snapshot.eclass_group_count,
        view,
        config,
    )
}

/// Plan structural-signature candidate buckets from a lightweight
/// CUDA-resident signature snapshot.
pub fn plan_cuda_egraph_signature_buckets_from_signature_snapshot(
    snapshot: &CudaEGraphResidentSignatureSnapshot,
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphSignatureBucketPlan, CudaEGraphKernelPlanError> {
    plan_cuda_egraph_signature_buckets_from_column(
        &snapshot.row_signatures,
        snapshot.row_count(),
        snapshot.child_count(),
        snapshot.eclass_group_count,
        view,
        config,
    )
}

fn plan_cuda_egraph_signature_buckets_from_column(
    signatures: &[u32],
    row_count: usize,
    child_count: usize,
    eclass_group_count: usize,
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphSignatureBucketPlan, CudaEGraphKernelPlanError> {
    if config.threads_per_block == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroThreadsPerBlock);
    }
    if config.max_blocks_per_launch == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch);
    }
    if row_count != view.row_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "row count",
            image: row_count,
            view: view.row_count(),
        });
    }
    if child_count != view.child_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "child count",
            image: child_count,
            view: view.child_count(),
        });
    }
    if eclass_group_count != view.eclass_group_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "eclass group count",
            image: eclass_group_count,
            view: view.eclass_group_count(),
        });
    }

    let mut sorted_rows = reserved_typed_vec(signatures.len(), "egraph signature sorted rows")?;
    for row in 0..signatures.len() {
        sorted_rows.push(u32::try_from(row).map_err(|_| {
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature row index",
            }
        })?);
    }
    sort_unstable_by_key_if_needed(&mut sorted_rows, |&row| (signatures[row as usize], row));

    let mut buckets = reserved_typed_vec(signatures.len(), "egraph signature buckets")?;
    let mut bucket_rows = reserved_typed_vec(signatures.len(), "egraph signature bucket rows")?;
    let mut candidate_pair_count = 0_u64;

    let mut cursor = 0_usize;
    while cursor < sorted_rows.len() {
        let signature = signatures[sorted_rows[cursor] as usize];
        let start = cursor;
        cursor += 1;
        while cursor < sorted_rows.len() && signatures[sorted_rows[cursor] as usize] == signature {
            cursor += 1;
        }

        let row_count = cursor - start;
        if row_count < 2 {
            continue;
        }
        let first_bucket_row = u32::try_from(bucket_rows.len()).map_err(|_| {
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature bucket row offset",
            }
        })?;
        bucket_rows.extend_from_slice(&sorted_rows[start..cursor]);
        let pair_count = unordered_pair_count(row_count as u64)?;
        candidate_pair_count = candidate_pair_count.checked_add(pair_count).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature candidate pair count",
            },
        )?;
        buckets.push(CudaEGraphSignatureBucket {
            signature,
            first_bucket_row,
            row_count: u32::try_from(row_count).map_err(|_| {
                CudaEGraphKernelPlanError::CountOverflow {
                    field: "signature bucket row count",
                }
            })?,
            candidate_pair_count: pair_count,
        });
    }

    let pair_wave_count = buckets.iter().try_fold(0_u64, |acc, bucket| {
        wave_count_for(bucket.candidate_pair_count, config).and_then(|count| {
            acc.checked_add(count)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "signature pair wave count",
                })
        })
    })?;
    let mut pair_waves = reserved_typed_vec(
        usize::try_from(pair_wave_count).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
            field: "signature pair wave count usize conversion",
        })?,
        "egraph signature pair waves",
    )?;
    let mut total_blocks = 0_u64;
    for (bucket_index, bucket) in buckets.iter().enumerate() {
        append_signature_pair_waves(
            &mut pair_waves,
            &mut total_blocks,
            u32::try_from(bucket_index).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "signature bucket index",
            })?,
            bucket.candidate_pair_count,
            config,
        )?;
    }

    Ok(CudaEGraphSignatureBucketPlan {
        view,
        buckets,
        bucket_rows,
        pair_waves,
        candidate_pair_count,
        total_blocks,
    })
}

/// Build signature buckets and emit exact structural e-class equivalences from
/// the packed columns.
///
/// This is the host-side mirror of the CUDA duplicate-discovery kernel:
/// signatures bound the search space, then exact op/arity/child comparison
/// prevents hash-collision false positives before emitting merge candidates.
pub fn plan_cuda_egraph_structural_equivalences(
    image: &GpuEGraphDeviceImage,
    view: CudaEGraphDeviceKernelView,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphStructuralEquivalencePlan, CudaEGraphKernelPlanError> {
    let signature_plan = plan_cuda_egraph_signature_buckets(image, view, config)?;
    collect_cuda_egraph_structural_equivalences(image, signature_plan)
}

/// Emit exact structural equivalences from an existing signature-bucket plan.
///
/// # Errors
///
/// Returns [`CudaEGraphKernelPlanError`] if the packed image does not match
/// the bucket plan's checked view or if a planned bucket/pair range is invalid.
pub fn collect_cuda_egraph_structural_equivalences(
    image: &GpuEGraphDeviceImage,
    signature_plan: CudaEGraphSignatureBucketPlan,
) -> Result<CudaEGraphStructuralEquivalencePlan, CudaEGraphKernelPlanError> {
    validate_image_view_matches(image, signature_plan.view)?;

    let mut equivalence_keys = reserved_typed_vec(
        signature_plan.buckets.len(),
        "egraph structural equivalences",
    )?;
    let mut exact_pair_count = 0_u64;
    let mut redundant_pair_count = 0_u64;
    let mut rejected_candidate_pair_count = 0_u64;

    for bucket_index in 0..signature_plan.buckets.len() {
        let bucket = &signature_plan.buckets[bucket_index];
        for pair_ordinal in 0..bucket.candidate_pair_count {
            let (left_row, right_row) = cuda_egraph_signature_pair_rows(
                &signature_plan,
                bucket_index as u32,
                pair_ordinal,
            )?;
            if !packed_rows_structurally_equal(image, left_row, right_row)? {
                rejected_candidate_pair_count = rejected_candidate_pair_count
                    .checked_add(1)
                    .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                        field: "rejected structural candidate count",
                    })?;
                continue;
            }
            exact_pair_count = exact_pair_count.checked_add(1).ok_or(
                CudaEGraphKernelPlanError::CountOverflow {
                    field: "exact structural pair count",
                },
            )?;

            let left_eclass = image.row_eclass_ids()[left_row as usize];
            let right_eclass = image.row_eclass_ids()[right_row as usize];
            if left_eclass == right_eclass {
                redundant_pair_count = redundant_pair_count.checked_add(1).ok_or(
                    CudaEGraphKernelPlanError::CountOverflow {
                        field: "redundant structural pair count",
                    },
                )?;
                continue;
            }
            equivalence_keys.push(if left_eclass < right_eclass {
                (left_eclass, right_eclass)
            } else {
                (right_eclass, left_eclass)
            });
        }
    }

    sort_unstable_if_needed(&mut equivalence_keys);
    equivalence_keys.dedup();
    let equivalence_output_words =
        equivalence_keys
            .len()
            .checked_mul(2)
            .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                field: "structural equivalence output words",
            })?;
    let mut equivalences = reserved_typed_vec(
        equivalence_keys.len(),
        "egraph structural equivalence output",
    )?;
    equivalences.extend(
        equivalence_keys
            .into_iter()
            .map(|(left, right)| Equivalence { left, right }),
    );

    Ok(CudaEGraphStructuralEquivalencePlan {
        signature_plan,
        equivalences,
        exact_pair_count,
        redundant_pair_count,
        rejected_candidate_pair_count,
        equivalence_output_words,
    })
}

/// Pack signature-bucket metadata into a fixed-width u32 table for resident
/// CUDA kernels.
///
/// The table is intentionally separate from the foundation e-graph image:
/// foundation owns canonical e-graph columns, while CUDA owns launch-local
/// work partitioning.
pub fn pack_cuda_egraph_signature_bucket_device_image(
    signature_plan: &CudaEGraphSignatureBucketPlan,
) -> Result<CudaEGraphSignatureBucketDeviceImage, CudaEGraphKernelPlanError> {
    let bucket_words = pack_cuda_egraph_signature_bucket_words(signature_plan)?;
    Ok(CudaEGraphSignatureBucketDeviceImage {
        bucket_words,
        bucket_rows: signature_plan.bucket_rows.clone(),
        bucket_count: signature_plan.buckets.len(),
        bucket_record_words: SIGNATURE_BUCKET_RECORD_WORDS,
        candidate_pair_count: signature_plan.candidate_pair_count,
    })
}

/// Pack signature-bucket metadata while consuming the host plan.
///
/// The borrowed packing API is retained for callers that need to inspect the
/// plan after packing. Release execution usually creates the plan only to
/// launch the CUDA kernel, so this consuming variant moves the large
/// `bucket_rows` and `pair_waves` vectors into the launch artifact instead of
/// cloning them.
pub fn plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan(
    signature_plan: CudaEGraphSignatureBucketPlan,
) -> Result<CudaEGraphStructuralEquivalenceLaunchArtifact, CudaEGraphKernelPlanError> {
    let output = plan_cuda_egraph_structural_equivalence_output(&signature_plan)?;
    let bucket_words = pack_cuda_egraph_signature_bucket_words(&signature_plan)?;
    let CudaEGraphSignatureBucketPlan {
        buckets,
        bucket_rows,
        pair_waves,
        candidate_pair_count,
        ..
    } = signature_plan;
    Ok(CudaEGraphStructuralEquivalenceLaunchArtifact {
        bucket_image: CudaEGraphSignatureBucketDeviceImage {
            bucket_words,
            bucket_rows,
            bucket_count: buckets.len(),
            bucket_record_words: SIGNATURE_BUCKET_RECORD_WORDS,
            candidate_pair_count,
        },
        output,
        pair_waves,
    })
}

fn pack_cuda_egraph_signature_bucket_words(
    signature_plan: &CudaEGraphSignatureBucketPlan,
) -> Result<Vec<u32>, CudaEGraphKernelPlanError> {
    let bucket_word_count = signature_plan
        .buckets
        .len()
        .checked_mul(SIGNATURE_BUCKET_RECORD_WORDS)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "signature bucket device word count",
        })?;
    let mut bucket_words =
        reserved_typed_vec(bucket_word_count, "egraph signature bucket device words")?;
    for (bucket_index, bucket) in signature_plan.buckets.iter().enumerate() {
        let start = bucket.first_bucket_row as usize;
        let end = start.checked_add(bucket.row_count as usize).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature bucket device row range end",
            },
        )?;
        if end > signature_plan.bucket_rows.len() {
            return Err(CudaEGraphKernelPlanError::SignatureBucketRowsOutOfBounds {
                bucket_index: u32::try_from(bucket_index).map_err(|_| {
                    CudaEGraphKernelPlanError::CountOverflow {
                        field: "signature bucket device index",
                    }
                })?,
                first_bucket_row: start,
                row_count: bucket.row_count as usize,
                bucket_rows_len: signature_plan.bucket_rows.len(),
            });
        }
        let pair_bytes = bucket.candidate_pair_count.to_le_bytes();
        bucket_words.extend_from_slice(&[
            bucket.signature,
            bucket.first_bucket_row,
            bucket.row_count,
            u32::from_le_bytes([pair_bytes[0], pair_bytes[1], pair_bytes[2], pair_bytes[3]]),
            u32::from_le_bytes([pair_bytes[4], pair_bytes[5], pair_bytes[6], pair_bytes[7]]),
        ]);
    }
    Ok(bucket_words)
}

/// Plan the worst-case structural-equivalence output buffers for a signature
/// bucket plan.
pub fn plan_cuda_egraph_structural_equivalence_output(
    signature_plan: &CudaEGraphSignatureBucketPlan,
) -> Result<CudaEGraphStructuralEquivalenceOutputPlan, CudaEGraphKernelPlanError> {
    let output_pair_words = usize::try_from(signature_plan.candidate_pair_count)
        .map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
            field: "structural equivalence output pair count usize conversion",
        })?
        .checked_mul(2)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "structural equivalence output pair words",
        })?;
    let output_pair_bytes = output_pair_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "structural equivalence output pair bytes",
        })?;
    let output_counter_words = 2_usize;
    let output_counter_bytes = output_counter_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "structural equivalence output counter bytes",
        })?;
    Ok(CudaEGraphStructuralEquivalenceOutputPlan {
        max_equivalences: signature_plan.candidate_pair_count,
        output_pair_words,
        output_pair_bytes,
        output_counter_words,
        output_counter_bytes,
    })
}

/// Build the resident launch artifact consumed by a structural-equivalence
/// CUDA kernel.
pub fn plan_cuda_egraph_structural_equivalence_launch_artifact(
    signature_plan: &CudaEGraphSignatureBucketPlan,
) -> Result<CudaEGraphStructuralEquivalenceLaunchArtifact, CudaEGraphKernelPlanError> {
    Ok(CudaEGraphStructuralEquivalenceLaunchArtifact {
        bucket_image: pack_cuda_egraph_signature_bucket_device_image(signature_plan)?,
        output: plan_cuda_egraph_structural_equivalence_output(signature_plan)?,
        pair_waves: signature_plan.pair_waves.clone(),
    })
}

/// Generate the concrete PTX kernel that compares packed e-graph rows inside
/// one signature bucket wave and emits exact e-class equivalences.
///
/// The kernel expects the packed columns produced by
/// [`GpuEGraphDeviceImage`], the bucket table produced by
/// [`pack_cuda_egraph_signature_bucket_device_image`], and the output buffers
/// sized by [`plan_cuda_egraph_structural_equivalence_output`].
pub fn plan_cuda_egraph_union_compaction(
    equivalences: &[Equivalence],
    config: CudaEGraphKernelLaunchConfig,
) -> Result<CudaEGraphUnionCompactionPlan, CudaEGraphKernelPlanError> {
    if config.threads_per_block == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroThreadsPerBlock);
    }
    if config.max_blocks_per_launch == 0 {
        return Err(CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch);
    }
    if equivalences.is_empty() {
        return Ok(CudaEGraphUnionCompactionPlan {
            canonical_pairs: Vec::new(),
            affected_eclasses: Vec::new(),
            canonical_rewrites: Vec::new(),
            waves: Vec::new(),
            ignored_self_pair_count: 0,
            duplicate_pair_count: 0,
            total_items: 0,
            total_blocks: 0,
        });
    }

    let mut ignored_self_pair_count = 0_u64;
    let mut canonical_pairs =
        reserved_typed_vec(equivalences.len(), "egraph union canonical pairs")?;
    for pair in equivalences {
        if pair.left == pair.right {
            ignored_self_pair_count = ignored_self_pair_count.checked_add(1).ok_or(
                CudaEGraphKernelPlanError::CountOverflow {
                    field: "ignored self pair count",
                },
            )?;
            continue;
        }
        let (left, right) = if pair.left < pair.right {
            (pair.left, pair.right)
        } else {
            (pair.right, pair.left)
        };
        canonical_pairs.push(Equivalence { left, right });
    }
    let pair_count_before_dedup = canonical_pairs.len();
    sort_unstable_by_key_if_needed(&mut canonical_pairs, |pair| (pair.left, pair.right));
    canonical_pairs.dedup();
    let duplicate_pair_count = pair_count_before_dedup
        .checked_sub(canonical_pairs.len())
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "duplicate pair count",
        })? as u64;

    let affected_capacity =
        canonical_pairs
            .len()
            .checked_mul(2)
            .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                field: "affected eclass capacity",
            })?;
    let mut affected_eclasses =
        reserved_typed_vec(affected_capacity, "egraph union affected eclasses")?;
    for pair in &canonical_pairs {
        affected_eclasses.push(pair.left);
        affected_eclasses.push(pair.right);
    }
    sort_unstable_if_needed(&mut affected_eclasses);
    affected_eclasses.dedup();

    let mut parents = reserved_typed_vec(affected_eclasses.len(), "egraph union parents")?;
    for index in 0..affected_eclasses.len() {
        parents.push(index);
    }
    let mut eclass_indices = FxHashMap::<u32, usize>::default();
    eclass_indices
        .try_reserve(affected_eclasses.len())
        .map_err(|error| CudaEGraphKernelPlanError::StorageReserveFailed {
            field: "egraph union eclass index",
            requested: affected_eclasses.len(),
            message: error.to_string(),
        })?;
    for (index, &eclass_id) in affected_eclasses.iter().enumerate() {
        eclass_indices.insert(eclass_id, index);
    }
    for pair in &canonical_pairs {
        let left =
            *eclass_indices
                .get(&pair.left)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "left eclass lookup",
                })?;
        let right =
            *eclass_indices
                .get(&pair.right)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "right eclass lookup",
                })?;
        union_min_parent(&mut parents, left, right);
    }

    let mut canonical_rewrites =
        reserved_typed_vec(affected_eclasses.len(), "egraph canonical rewrites")?;
    for index in 0..affected_eclasses.len() {
        let root = find_union_parent(&mut parents, index);
        let representative = affected_eclasses[root];
        let eclass_id = affected_eclasses[index];
        if representative != eclass_id {
            canonical_rewrites.push(CudaEGraphCanonicalRewrite {
                eclass_id,
                representative,
            });
        }
    }

    let union_items = usize_to_u64(canonical_pairs.len(), "canonical union pair count")?;
    let rewrite_items = usize_to_u64(canonical_rewrites.len(), "canonical rewrite count")?;
    let union_wave_count = wave_count_for(union_items, config)?;
    let rewrite_wave_count = wave_count_for(rewrite_items, config)?;
    let wave_count = union_wave_count.checked_add(rewrite_wave_count).ok_or(
        CudaEGraphKernelPlanError::CountOverflow {
            field: "union compaction wave count",
        },
    )?;
    let mut waves = reserved_typed_vec(
        usize::try_from(wave_count).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
            field: "union compaction wave count usize conversion",
        })?,
        "egraph union compaction waves",
    )?;
    let mut total_items = 0_u64;
    let mut total_blocks = 0_u64;
    append_union_compaction_waves(
        &mut waves,
        &mut total_items,
        &mut total_blocks,
        CudaEGraphUnionCompactionPass::UnionPairs,
        union_items,
        config,
    )?;
    append_union_compaction_waves(
        &mut waves,
        &mut total_items,
        &mut total_blocks,
        CudaEGraphUnionCompactionPass::CanonicalRewrites,
        rewrite_items,
        config,
    )?;

    Ok(CudaEGraphUnionCompactionPlan {
        canonical_pairs,
        affected_eclasses,
        canonical_rewrites,
        waves,
        ignored_self_pair_count,
        duplicate_pair_count,
        total_items,
        total_blocks,
    })
}

/// Pack canonical e-class rewrites into fixed-width device records sorted by
/// source e-class id.
///
/// # Errors
///
/// Returns [`CudaEGraphKernelPlanError`] if packed word-count arithmetic
/// overflows host addressing.
pub fn pack_cuda_egraph_canonical_rewrite_device_image(
    plan: &CudaEGraphUnionCompactionPlan,
) -> Result<CudaEGraphCanonicalRewriteDeviceImage, CudaEGraphKernelPlanError> {
    let word_count = plan
        .canonical_rewrites
        .len()
        .checked_mul(CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "canonical rewrite word count",
        })?;
    let mut rewrite_words = reserved_typed_vec(word_count, "canonical rewrite words")?;
    for rewrite in &plan.canonical_rewrites {
        rewrite_words.push(rewrite.eclass_id);
        rewrite_words.push(rewrite.representative);
    }
    Ok(CudaEGraphCanonicalRewriteDeviceImage {
        rewrite_words,
        rewrite_count: plan.canonical_rewrites.len(),
        rewrite_record_words: CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS,
    })
}

/// Generate PTX for applying canonical e-class rewrites directly to a
/// CUDA-resident packed e-graph image.
///
/// The kernel scans `row_eclass_ids || children`, binary-searches the sorted
/// rewrite table, and overwrites matching ids with their canonical
/// representative. This keeps equality-saturation compaction on the GPU after
/// structural-equivalence discovery.
///
/// # Errors
///
/// Returns [`CudaEGraphKernelPlanError::InvalidPtxTarget`] when `target_sm` is
/// zero.
fn append_union_compaction_waves(
    waves: &mut Vec<CudaEGraphUnionCompactionWave>,
    total_items: &mut u64,
    total_blocks: &mut u64,
    pass: CudaEGraphUnionCompactionPass,
    item_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<(), CudaEGraphKernelPlanError> {
    let mut remaining = item_count;
    let mut first_item = 0_u64;
    let max_items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "union compaction max items per wave",
        })?;
    while remaining > 0 {
        let wave_items = remaining.min(max_items_per_wave);
        let blocks = ceil_div_u64(wave_items, u64::from(config.threads_per_block))?;
        let blocks =
            u32::try_from(blocks).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "blocks per union compaction launch wave",
            })?;
        waves.push(CudaEGraphUnionCompactionWave {
            pass,
            first_item,
            item_count: wave_items,
            blocks,
            threads_per_block: config.threads_per_block,
        });
        *total_items = total_items.checked_add(wave_items).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "union compaction total items",
            },
        )?;
        *total_blocks = total_blocks.checked_add(u64::from(blocks)).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "union compaction total blocks",
            },
        )?;
        first_item =
            first_item
                .checked_add(wave_items)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "union compaction wave first item",
                })?;
        remaining -= wave_items;
    }
    Ok(())
}

fn find_union_parent(parents: &mut [usize], index: usize) -> usize {
    let parent = parents[index];
    if parent == index {
        return index;
    }
    let root = find_union_parent(parents, parent);
    parents[index] = root;
    root
}

fn union_min_parent(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find_union_parent(parents, left);
    let right_root = find_union_parent(parents, right);
    if left_root == right_root {
        return;
    }
    if left_root < right_root {
        parents[right_root] = left_root;
    } else {
        parents[left_root] = right_root;
    }
}

impl CudaBackend {
    /// Generate and warm-load the structural e-graph equivalence kernel through
    /// the CUDA module cache.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if PTX generation fails, the CUDA driver
    /// rejects the PTX module, or the `main` entry symbol cannot be resolved.
    pub fn warm_egraph_structural_equivalence_kernel(
        &self,
    ) -> Result<CudaEGraphStructuralEquivalenceKernelPtx, BackendError> {
        self.warm_egraph_structural_equivalence_kernel_with_key()
            .map(|(kernel, _)| kernel)
    }

    fn warm_egraph_structural_equivalence_kernel_with_key(
        &self,
    ) -> Result<
        (
            CudaEGraphStructuralEquivalenceKernelPtx,
            cudarc::driver::sys::CUfunction,
        ),
        BackendError,
    > {
        let kernel = cuda_egraph_structural_equivalence_kernel_ptx(self.ptx_target_sm()).map_err(
            |error| BackendError::InvalidProgram {
                fix: error.to_string(),
            },
        )?;
        let module_key = self.module_cache_key_for_raw_ptx_artifact(&kernel.source)?;
        let function = self.module_for_ptx_with_key(&kernel.source, module_key)?;
        if function.is_null() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph structural-equivalence kernel loaded but resolved a null `main` function. Inspect generated PTX entry metadata before launch.".to_string(),
            });
        }
        Ok((kernel, function))
    }

    /// Launch the structural e-graph equivalence kernel over a resident packed
    /// e-graph image.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if any scratch allocation, upload, kernel
    /// launch, readback, or cleanup step fails.
    pub fn run_egraph_structural_equivalence_kernel(
        &self,
        image: CudaResidentEGraphDeviceImage,
        artifact: &CudaEGraphStructuralEquivalenceLaunchArtifact,
    ) -> Result<CudaEGraphStructuralEquivalenceKernelResult, BackendError> {
        let view = self.egraph_device_kernel_view(image)?;
        let (_kernel, func) = self.warm_egraph_structural_equivalence_kernel_with_key()?;
        let mut handles = SmallVec::<[CudaResidentBuffer; 4]>::new();
        let result =
            self.run_egraph_structural_equivalence_kernel_inner(view, artifact, func, &mut handles);
        let cleanup = cleanup_egraph_kernel_handles(self, &handles);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) | (Err(_), Err(error)) => Err(error),
        }
    }

    /// Upload a packed foundation e-graph image, discover exact structural
    /// equivalences on CUDA, and free the temporary resident image.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if upload planning, resident upload, bucket
    /// planning, kernel execution, readback, or resident cleanup fails.
    pub fn discover_egraph_structural_equivalences(
        &self,
        image: GpuEGraphDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphStructuralEquivalenceKernelResult, BackendError> {
        let upload_plan =
            plan_cuda_egraph_device_upload_from_image_ref(&image).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        let resident = self.upload_egraph_device_image_borrowed_plan(upload_plan)?;
        let result = (|| {
            let view = self.egraph_device_kernel_view(resident)?;
            let signature_plan =
                plan_cuda_egraph_signature_buckets(&image, view, config).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: error.to_string(),
                    }
                })?;
            let artifact =
                plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan(signature_plan)
                    .map_err(|error| BackendError::InvalidProgram {
                    fix: error.to_string(),
                })?;
            self.run_egraph_structural_equivalence_kernel(resident, &artifact)
        })();
        let cleanup = self.free_resident(resident.handle());
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) | (Err(_), Err(error)) => Err(error),
        }
    }

    /// Generate and warm-load the canonical e-graph rewrite kernel through the
    /// CUDA module cache.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if PTX generation fails, the CUDA driver
    /// rejects the PTX module, or the `main` entry symbol cannot be resolved.
    pub fn warm_egraph_canonical_rewrite_kernel(
        &self,
    ) -> Result<CudaEGraphCanonicalRewriteKernelPtx, BackendError> {
        self.warm_egraph_canonical_rewrite_kernel_with_key()
            .map(|(kernel, _)| kernel)
    }

    fn warm_egraph_canonical_rewrite_kernel_with_key(
        &self,
    ) -> Result<
        (
            CudaEGraphCanonicalRewriteKernelPtx,
            cudarc::driver::sys::CUfunction,
        ),
        BackendError,
    > {
        let kernel =
            cuda_egraph_canonical_rewrite_kernel_ptx(self.ptx_target_sm()).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        let module_key = self.module_cache_key_for_raw_ptx_artifact(&kernel.source)?;
        let function = self.module_for_ptx_with_key(&kernel.source, module_key)?;
        if function.is_null() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph canonical-rewrite kernel loaded but resolved a null `main` function. Inspect generated PTX entry metadata before launch.".to_string(),
            });
        }
        Ok((kernel, function))
    }

    /// Apply canonical e-class rewrites directly to a resident packed e-graph
    /// image on CUDA.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if rewrite metadata is malformed, launch
    /// dimensions are invalid, or CUDA allocation, upload, launch,
    /// synchronization, or cleanup fails.
    pub fn run_egraph_canonical_rewrite_kernel(
        &self,
        image: CudaResidentEGraphDeviceImage,
        rewrites: &CudaEGraphCanonicalRewriteDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphCanonicalRewriteKernelResult, BackendError> {
        if config.threads_per_block == 0 {
            return Err(BackendError::InvalidProgram {
                fix: CudaEGraphKernelPlanError::ZeroThreadsPerBlock.to_string(),
            });
        }
        if config.max_blocks_per_launch == 0 {
            return Err(BackendError::InvalidProgram {
                fix: CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch.to_string(),
            });
        }
        if rewrites.rewrite_record_words != CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite table uses {} words per record, expected {}.",
                    rewrites.rewrite_record_words,
                    CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS
                ),
            });
        }
        let expected_words = rewrites
            .rewrite_count
            .checked_mul(CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph canonical rewrite table word count overflowed host usize addressing.".to_string(),
            })?;
        if expected_words != rewrites.rewrite_words.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite table has {} words for {} records, expected {expected_words}.",
                    rewrites.rewrite_words.len(),
                    rewrites.rewrite_count
                ),
            });
        }

        let view = self.egraph_device_kernel_view(image)?;
        let row_items =
            usize_to_u64(view.row_count(), "canonical rewrite row count").map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        let child_items = usize_to_u64(view.child_count(), "canonical rewrite child count")
            .map_err(|error| BackendError::InvalidProgram {
                fix: error.to_string(),
            })?;
        let total_items = row_items
            .checked_add(child_items)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph canonical rewrite item count overflowed u64; shard the image before launch.".to_string(),
            })?;
        if total_items == 0 || rewrites.rewrite_count == 0 {
            return Ok(CudaEGraphCanonicalRewriteKernelResult {
                rewrite_count: rewrites.rewrite_count,
                row_count: view.row_count(),
                child_count: view.child_count(),
                launch_count: 0,
                total_items: 0,
            });
        }

        let (_kernel, func) = self.warm_egraph_canonical_rewrite_kernel_with_key()?;
        let rewrite_buffer = upload_u32_words(self, &rewrites.rewrite_words)?;
        let result = self.run_egraph_canonical_rewrite_kernel_inner(
            view,
            rewrites.rewrite_count,
            total_items,
            rewrite_buffer,
            func,
            config,
        );
        let cleanup = self.free_resident(rewrite_buffer);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) | (Err(_), Err(error)) => Err(error),
        }
    }

    /// Generate and warm-load the row-signature refresh kernel through the
    /// CUDA module cache.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if PTX generation fails, the CUDA driver
    /// rejects the PTX module, or the `main` entry symbol cannot be resolved.
    pub fn warm_egraph_signature_refresh_kernel(
        &self,
    ) -> Result<CudaEGraphSignatureRefreshKernelPtx, BackendError> {
        self.warm_egraph_signature_refresh_kernel_with_key()
            .map(|(kernel, _)| kernel)
    }

    fn warm_egraph_signature_refresh_kernel_with_key(
        &self,
    ) -> Result<
        (
            CudaEGraphSignatureRefreshKernelPtx,
            cudarc::driver::sys::CUfunction,
        ),
        BackendError,
    > {
        let kernel =
            cuda_egraph_signature_refresh_kernel_ptx(self.ptx_target_sm()).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        let module_key = self.module_cache_key_for_raw_ptx_artifact(&kernel.source)?;
        let function = self.module_for_ptx_with_key(&kernel.source, module_key)?;
        if function.is_null() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph row-signature refresh kernel loaded but resolved a null `main` function. Inspect generated PTX entry metadata before launch.".to_string(),
            });
        }
        Ok((kernel, function))
    }

    /// Refresh resident e-graph row signatures on CUDA after canonical rewrites.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if launch dimensions are invalid, resident
    /// pointer resolution fails, PTX loading fails, kernel launch fails, or
    /// synchronization fails.
    pub fn run_egraph_signature_refresh_kernel(
        &self,
        image: CudaResidentEGraphDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphSignatureRefreshKernelResult, BackendError> {
        if config.threads_per_block == 0 {
            return Err(BackendError::InvalidProgram {
                fix: CudaEGraphKernelPlanError::ZeroThreadsPerBlock.to_string(),
            });
        }
        if config.max_blocks_per_launch == 0 {
            return Err(BackendError::InvalidProgram {
                fix: CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch.to_string(),
            });
        }
        let view = self.egraph_device_kernel_view(image)?;
        let row_count =
            usize_to_u64(view.row_count(), "signature refresh row count").map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        if row_count == 0 {
            return Ok(CudaEGraphSignatureRefreshKernelResult {
                row_count: view.row_count(),
                launch_count: 0,
                total_rows: 0,
            });
        }
        let (_kernel, func) = self.warm_egraph_signature_refresh_kernel_with_key()?;
        self.run_egraph_signature_refresh_kernel_inner(view, row_count, func, config)
    }

    /// Download the current CUDA-resident e-graph planning columns.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if resident download fails or any packed u32
    /// span is malformed.
    pub fn download_egraph_resident_column_snapshot(
        &self,
        image: CudaResidentEGraphDeviceImage,
    ) -> Result<CudaEGraphResidentColumnSnapshot, BackendError> {
        let layout = image.byte_layout();
        let spans = egraph_column_snapshot_spans(layout);
        let ranges = spans.map(|span| (image.handle(), span.offset(), span.byte_len()));
        let mut row_eclass_bytes = Vec::new();
        let mut row_language_op_bytes = Vec::new();
        let mut row_children_offset_bytes = Vec::new();
        let mut row_children_len_bytes = Vec::new();
        let mut row_signature_bytes = Vec::new();
        let mut child_bytes = Vec::new();
        let mut outputs: [&mut Vec<u8>; 6] = [
            &mut row_eclass_bytes,
            &mut row_language_op_bytes,
            &mut row_children_offset_bytes,
            &mut row_children_len_bytes,
            &mut row_signature_bytes,
            &mut child_bytes,
        ];
        self.download_resident_ranges_into(&ranges, &mut outputs)?;
        Ok(CudaEGraphResidentColumnSnapshot {
            row_eclass_ids: read_resident_u32_range(
                &row_eclass_bytes,
                layout.row_count(),
                "row eclass ids",
            )?,
            row_language_op_ids: read_resident_u32_range(
                &row_language_op_bytes,
                layout.row_count(),
                "row language op ids",
            )?,
            row_children_offsets: read_resident_u32_range(
                &row_children_offset_bytes,
                layout.row_count(),
                "row child offsets",
            )?,
            row_children_lens: read_resident_u32_range(
                &row_children_len_bytes,
                layout.row_count(),
                "row child lengths",
            )?,
            row_signatures: read_resident_u32_range(
                &row_signature_bytes,
                layout.row_count(),
                "row signatures",
            )?,
            children: read_resident_u32_range(&child_bytes, layout.child_count(), "children")?,
            eclass_group_count: layout.eclass_group_count(),
        })
    }

    /// Download only the current CUDA-resident row-signature column needed for
    /// planning the next fixed-point round.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if resident download fails or the signature
    /// span is malformed.
    pub fn download_egraph_resident_signature_snapshot(
        &self,
        image: CudaResidentEGraphDeviceImage,
    ) -> Result<CudaEGraphResidentSignatureSnapshot, BackendError> {
        let layout = image.byte_layout();
        let signature_span = layout.row_signatures();
        let bytes = self.download_resident_range(
            image.handle(),
            signature_span.offset(),
            signature_span.byte_len(),
        )?;
        Ok(CudaEGraphResidentSignatureSnapshot {
            row_signatures: read_resident_u32_range(&bytes, layout.row_count(), "row signatures")?,
            child_count: layout.child_count(),
            eclass_group_count: layout.eclass_group_count(),
        })
    }

    /// Discover structural e-class equivalences, derive deterministic
    /// canonical representatives, and mutate the resident e-graph image on
    /// CUDA in one round.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if resident view construction, signature
    /// planning, structural discovery, union planning, rewrite packing, kernel
    /// launch, synchronization, or cleanup fails.
    pub fn run_egraph_structural_canonicalization_round(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        image: &GpuEGraphDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphStructuralCanonicalizationRoundResult, BackendError> {
        let view = self.egraph_device_kernel_view(resident)?;
        let signature_plan =
            plan_cuda_egraph_signature_buckets(image, view, config).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        self.run_egraph_structural_canonicalization_round_from_signature_plan(
            resident,
            view,
            signature_plan,
            config,
        )
    }

    /// Run one CUDA-resident structural canonicalization round using a current
    /// resident-column planning snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if resident view construction, signature
    /// planning, structural discovery, union planning, rewrite packing, kernel
    /// launch, synchronization, or cleanup fails.
    pub fn run_egraph_structural_canonicalization_round_from_snapshot(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        snapshot: &CudaEGraphResidentColumnSnapshot,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphStructuralCanonicalizationRoundResult, BackendError> {
        let view = self.egraph_device_kernel_view(resident)?;
        let signature_plan =
            plan_cuda_egraph_signature_buckets_from_resident_snapshot(snapshot, view, config)
                .map_err(|error| BackendError::InvalidProgram {
                    fix: error.to_string(),
                })?;
        self.run_egraph_structural_canonicalization_round_from_signature_plan(
            resident,
            view,
            signature_plan,
            config,
        )
    }

    /// Run one CUDA-resident structural canonicalization round using a current
    /// resident signature-column planning snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if resident view construction, signature
    /// planning, structural discovery, union planning, rewrite packing, kernel
    /// launch, synchronization, or cleanup fails.
    pub fn run_egraph_structural_canonicalization_round_from_signature_snapshot(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        snapshot: &CudaEGraphResidentSignatureSnapshot,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphStructuralCanonicalizationRoundResult, BackendError> {
        let view = self.egraph_device_kernel_view(resident)?;
        let signature_plan =
            plan_cuda_egraph_signature_buckets_from_signature_snapshot(snapshot, view, config)
                .map_err(|error| BackendError::InvalidProgram {
                    fix: error.to_string(),
                })?;
        self.run_egraph_structural_canonicalization_round_from_signature_plan(
            resident,
            view,
            signature_plan,
            config,
        )
    }

    fn run_egraph_structural_canonicalization_round_from_signature_plan(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        view: CudaEGraphDeviceKernelView,
        signature_plan: CudaEGraphSignatureBucketPlan,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphStructuralCanonicalizationRoundResult, BackendError> {
        let artifact =
            plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan(signature_plan)
                .map_err(|error| BackendError::InvalidProgram {
                    fix: error.to_string(),
                })?;
        let discovery = self.run_egraph_structural_equivalence_kernel(resident, &artifact)?;
        let union_plan =
            plan_cuda_egraph_union_compaction(&discovery.unique, config).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        if union_plan.canonical_rewrites.is_empty() {
            return Ok(CudaEGraphStructuralCanonicalizationRoundResult {
                discovery,
                union_plan,
                rewrite: CudaEGraphCanonicalRewriteKernelResult {
                    rewrite_count: 0,
                    row_count: view.row_count(),
                    child_count: view.child_count(),
                    launch_count: 0,
                    total_items: 0,
                },
                signature_refresh: CudaEGraphSignatureRefreshKernelResult {
                    row_count: view.row_count(),
                    launch_count: 0,
                    total_rows: 0,
                },
            });
        }
        let rewrite_image =
            pack_cuda_egraph_canonical_rewrite_device_image(&union_plan).map_err(|error| {
                BackendError::InvalidProgram {
                    fix: error.to_string(),
                }
            })?;
        let rewrite = self.run_egraph_canonical_rewrite_kernel(resident, &rewrite_image, config)?;
        let signature_refresh = if rewrite.rewrite_count == 0 {
            CudaEGraphSignatureRefreshKernelResult {
                row_count: view.row_count(),
                launch_count: 0,
                total_rows: 0,
            }
        } else {
            self.run_egraph_signature_refresh_kernel(resident, config)?
        };
        Ok(CudaEGraphStructuralCanonicalizationRoundResult {
            discovery,
            union_plan,
            rewrite,
            signature_refresh,
        })
    }

    /// Iterate CUDA-resident structural canonicalization until a no-op
    /// discovery round proves fixed-point convergence or `max_rounds` is
    /// reached.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if any round fails or resident snapshot
    /// readback fails between rounds.
    pub fn run_egraph_structural_canonicalization_fixed_point(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        initial_image: &GpuEGraphDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
        max_rounds: usize,
    ) -> Result<CudaEGraphStructuralCanonicalizationFixedPointResult, BackendError> {
        let report = self.run_egraph_structural_canonicalization_fixed_point_with_readback(
            resident,
            initial_image,
            config,
            max_rounds,
            CudaEGraphFixedPointReadback::FullColumns,
        )?;
        let final_snapshot =
            report
                .final_snapshot
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: "Fix: CUDA e-graph fixed-point full-column readback was requested but no final snapshot was produced."
                        .to_string(),
                })?;
        Ok(CudaEGraphStructuralCanonicalizationFixedPointResult {
            rounds: report.rounds,
            final_snapshot,
            converged: report.converged,
            max_rounds: report.max_rounds,
            total_discovered_pairs: report.total_discovered_pairs,
            total_rewrites: report.total_rewrites,
        })
    }

    /// Iterate CUDA-resident structural canonicalization with explicit control
    /// over the final host readback volume.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if any round fails or the requested resident
    /// snapshot readback fails.
    pub fn run_egraph_structural_canonicalization_fixed_point_with_readback(
        &self,
        resident: CudaResidentEGraphDeviceImage,
        initial_image: &GpuEGraphDeviceImage,
        config: CudaEGraphKernelLaunchConfig,
        max_rounds: usize,
        final_readback: CudaEGraphFixedPointReadback,
    ) -> Result<CudaEGraphStructuralCanonicalizationFixedPointReport, BackendError> {
        let mut rounds = reserved_typed_vec(max_rounds, "egraph fixed point rounds").map_err(
            |error: CudaEGraphKernelPlanError| BackendError::InvalidProgram {
                fix: error.to_string(),
            },
        )?;
        let mut signature_snapshot: Option<CudaEGraphResidentSignatureSnapshot> = None;
        let mut signature_snapshot_current = false;
        let mut total_discovered_pairs = 0_u64;
        let mut total_rewrites = 0_u64;
        let mut converged = false;
        let layout = resident.byte_layout();
        let final_full_readback_bytes = egraph_column_snapshot_readback_bytes(layout)?;
        let final_signature_snapshot_bytes = layout.row_signatures().byte_len();

        for round_index in 0..max_rounds {
            let round = if round_index == 0 && signature_snapshot.is_none() {
                self.run_egraph_structural_canonicalization_round(resident, initial_image, config)?
            } else {
                let snapshot = signature_snapshot.as_ref().ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: "Fix: CUDA e-graph fixed-point planner lost the current signature snapshot before a follow-up round.".to_string(),
                    }
                })?;
                self.run_egraph_structural_canonicalization_round_from_signature_snapshot(
                    resident, snapshot, config,
                )?
            };
            let discovered_pairs = usize_to_u64(
                round.discovery.unique.len(),
                "fixed point discovered pair count",
            )
            .map_err(|error| BackendError::InvalidProgram {
                fix: error.to_string(),
            })?;
            let rewrite_count =
                usize_to_u64(round.rewrite.rewrite_count, "fixed point rewrite count").map_err(
                    |error| BackendError::InvalidProgram {
                        fix: error.to_string(),
                    },
                )?;
            total_discovered_pairs = total_discovered_pairs
                .checked_add(discovered_pairs)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: "Fix: CUDA e-graph fixed-point discovered pair count overflowed u64."
                        .to_string(),
                })?;
            total_rewrites = total_rewrites.checked_add(rewrite_count).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: "Fix: CUDA e-graph fixed-point rewrite count overflowed u64.".to_string(),
                }
            })?;
            rounds.push(round);
            if discovered_pairs == 0 || rewrite_count == 0 {
                converged = true;
                break;
            }
            signature_snapshot_current = false;
            if round_index + 1 == max_rounds {
                break;
            }
            signature_snapshot = Some(self.download_egraph_resident_signature_snapshot(resident)?);
            signature_snapshot_current = true;
        }

        let final_snapshot = match final_readback {
            CudaEGraphFixedPointReadback::FullColumns => {
                Some(self.download_egraph_resident_column_snapshot(resident)?)
            }
            CudaEGraphFixedPointReadback::None | CudaEGraphFixedPointReadback::Signatures => None,
        };
        let mut final_additional_readback_bytes = match final_readback {
            CudaEGraphFixedPointReadback::FullColumns => final_full_readback_bytes,
            CudaEGraphFixedPointReadback::None | CudaEGraphFixedPointReadback::Signatures => 0,
        };
        let final_signature_snapshot = match final_readback {
            CudaEGraphFixedPointReadback::None => None,
            CudaEGraphFixedPointReadback::Signatures => {
                if signature_snapshot_current {
                    signature_snapshot
                } else if total_rewrites == 0 {
                    Some(
                        CudaEGraphResidentSignatureSnapshot::try_from_device_image(initial_image)
                            .map_err(|error| BackendError::InvalidProgram {
                            fix: error.to_string(),
                        })?,
                    )
                } else {
                    final_additional_readback_bytes = final_signature_snapshot_bytes;
                    Some(self.download_egraph_resident_signature_snapshot(resident)?)
                }
            }
            CudaEGraphFixedPointReadback::FullColumns => final_snapshot
                .as_ref()
                .map(CudaEGraphResidentSignatureSnapshot::try_from_column_snapshot)
                .transpose()
                .map_err(|error| BackendError::InvalidProgram {
                    fix: error.to_string(),
                })?,
        };
        let avoided_final_readback_bytes = final_full_readback_bytes
            .checked_sub(final_additional_readback_bytes)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph final readback accounting underflowed.".to_string(),
            })?;

        Ok(CudaEGraphStructuralCanonicalizationFixedPointReport {
            rounds,
            final_snapshot,
            final_signature_snapshot,
            final_readback,
            final_full_readback_bytes,
            final_signature_snapshot_bytes,
            final_additional_readback_bytes,
            avoided_final_readback_bytes,
            converged,
            max_rounds,
            total_discovered_pairs,
            total_rewrites,
        })
    }

    fn run_egraph_structural_equivalence_kernel_inner(
        &self,
        view: CudaEGraphDeviceKernelView,
        artifact: &CudaEGraphStructuralEquivalenceLaunchArtifact,
        func: cudarc::driver::sys::CUfunction,
        handles: &mut SmallVec<[CudaResidentBuffer; 4]>,
    ) -> Result<CudaEGraphStructuralEquivalenceKernelResult, BackendError> {
        let scratch = upload_structural_equivalence_scratch(self, artifact)?;
        handles.push(scratch.handle);
        let scratch_base_ptr = self.resident_device_ptr(scratch.handle)?;
        let bucket_words_ptr = device_ptr_at(
            scratch_base_ptr,
            scratch.bucket_words_offset,
            "bucket words",
        )?;
        let bucket_rows_ptr =
            device_ptr_at(scratch_base_ptr, scratch.bucket_rows_offset, "bucket rows")?;
        let output_pairs_ptr = device_ptr_at(
            scratch_base_ptr,
            scratch.output_pairs_offset,
            "output pairs",
        )?;
        let output_count_ptr = device_ptr_at(
            scratch_base_ptr,
            scratch.output_count_offset,
            "output count",
        )?;
        let stream = crate::stream::CudaStream::non_blocking()?;
        let mut kernel_args = SmallVec::<[*mut std::ffi::c_void; 8]>::new();
        for wave in &artifact.pair_waves {
            let launch = LaunchPlan {
                element_count: u32::try_from(wave.pair_count).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA e-graph structural-equivalence wave pair count {} does not fit u32 launch accounting: {error}. Split the wave before launch.",
                            wave.pair_count
                        ),
                    }
                })?,
                workgroup: [wave.threads_per_block, 1, 1],
                grid: [wave.blocks, 1, 1],
                param_words: Vec::new(),
                max_binding_alignment: std::mem::size_of::<u64>(),
            };
            let mut args = EGraphStructuralKernelArgs {
                row_eclass_ids_ptr: view.row_eclass_ids_ptr(),
                row_language_op_ids_ptr: view.row_language_op_ids_ptr(),
                row_children_offsets_ptr: view.row_children_offsets_ptr(),
                row_children_lens_ptr: view.row_children_lens_ptr(),
                row_signatures_ptr: view.row_signatures_ptr(),
                children_ptr: view.children_ptr(),
                bucket_words_ptr,
                bucket_rows_ptr,
                output_pairs_ptr,
                output_count_ptr,
                bucket_index: wave.bucket_index,
                first_pair: wave.first_pair,
                pair_count: wave.pair_count,
            };
            args.write_kernel_args_into(&mut kernel_args)?;
            self.launch_resolved_function(
                func,
                &mut kernel_args,
                &launch,
                stream.raw(),
                false,
                false,
            )?;
        }
        stream.synchronize()?;

        let (count_bytes, pair_bytes) =
            download_structural_equivalence_output_ranges(self, &scratch)?;
        let count_bytes = count_bytes
            .get(..8)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph fused scratch readback did not contain the 8-byte structural equivalence counter.".to_string(),
            })?;
        let device_reported_count = read_u64_le(count_bytes, "structural equivalence count")?;
        let planned_capacity = artifact.output.max_equivalences;
        let capped_count = device_reported_count.min(planned_capacity);
        let pair_bytes = pair_bytes
            .get(..scratch.output_pairs_bytes)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph fused scratch readback did not contain the planned structural equivalence output-pair range.".to_string(),
            })?;
        let (emitted_pair_count, unique) =
            decode_unique_equivalence_pairs(&pair_bytes, capped_count)?;
        Ok(CudaEGraphStructuralEquivalenceKernelResult {
            emitted_pair_count,
            unique,
            device_reported_count,
            overflowed_output_capacity: device_reported_count > planned_capacity,
        })
    }

    fn run_egraph_canonical_rewrite_kernel_inner(
        &self,
        view: CudaEGraphDeviceKernelView,
        rewrite_count: usize,
        total_items: u64,
        rewrite_buffer: CudaResidentBuffer,
        func: cudarc::driver::sys::CUfunction,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphCanonicalRewriteKernelResult, BackendError> {
        let rewrite_words_ptr = self.resident_device_ptr(rewrite_buffer)?;
        let stream = crate::stream::CudaStream::non_blocking()?;
        let row_count = u32::try_from(view.row_count()).map_err(|error| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite row count {} does not fit u32 kernel ABI: {error}. Shard the image before launch.",
                    view.row_count()
                ),
            }
        })?;
        let child_count = u32::try_from(view.child_count()).map_err(|error| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite child count {} does not fit u32 kernel ABI: {error}. Shard the image before launch.",
                    view.child_count()
                ),
            }
        })?;
        let rewrite_count_u32 = u32::try_from(rewrite_count).map_err(|error| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite count {rewrite_count} does not fit u32 kernel ABI: {error}. Shard the rewrite table before launch."
                ),
            }
        })?;
        let items_per_wave = u64::from(config.threads_per_block)
            .checked_mul(u64::from(config.max_blocks_per_launch))
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph canonical rewrite launch dimensions overflowed u64 item accounting.".to_string(),
            })?;
        let mut first_item = 0_u64;
        let mut launch_count = 0_usize;
        let mut kernel_args = SmallVec::<[*mut std::ffi::c_void; 8]>::new();
        while first_item < total_items {
            let wave_items = (total_items - first_item).min(items_per_wave);
            let blocks =
                ceil_div_u64(wave_items, u64::from(config.threads_per_block)).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: error.to_string(),
                    }
                })?;
            let blocks = u32::try_from(blocks).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph canonical rewrite block count does not fit u32 launch ABI: {error}."
                ),
            })?;
            let launch = LaunchPlan {
                element_count: u32::try_from(wave_items).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA e-graph canonical rewrite wave item count {wave_items} does not fit u32 launch accounting: {error}. Split the wave before launch."
                        ),
                    }
                })?,
                workgroup: [config.threads_per_block, 1, 1],
                grid: [blocks, 1, 1],
                param_words: Vec::new(),
                max_binding_alignment: std::mem::size_of::<u64>(),
            };
            let mut args = EGraphCanonicalRewriteKernelArgs {
                row_eclass_ids_ptr: view.row_eclass_ids_ptr(),
                children_ptr: view.children_ptr(),
                rewrite_words_ptr,
                rewrite_count: rewrite_count_u32,
                row_count,
                child_count,
                first_item,
            };
            args.write_kernel_args_into(&mut kernel_args)?;
            self.launch_resolved_function(
                func,
                &mut kernel_args,
                &launch,
                stream.raw(),
                false,
                false,
            )?;
            first_item =
                first_item
                    .checked_add(wave_items)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: "Fix: CUDA e-graph canonical rewrite launch wave cursor overflowed u64 item accounting.".to_string(),
                    })?;
            launch_count =
                launch_count
                    .checked_add(1)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: "Fix: CUDA e-graph canonical rewrite launch count overflowed usize."
                            .to_string(),
                    })?;
        }
        stream.synchronize()?;
        Ok(CudaEGraphCanonicalRewriteKernelResult {
            rewrite_count,
            row_count: view.row_count(),
            child_count: view.child_count(),
            launch_count,
            total_items,
        })
    }

    fn run_egraph_signature_refresh_kernel_inner(
        &self,
        view: CudaEGraphDeviceKernelView,
        row_count: u64,
        func: cudarc::driver::sys::CUfunction,
        config: CudaEGraphKernelLaunchConfig,
    ) -> Result<CudaEGraphSignatureRefreshKernelResult, BackendError> {
        let stream = crate::stream::CudaStream::non_blocking()?;
        let row_count_u32 = u32::try_from(view.row_count()).map_err(|error| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph signature refresh row count {} does not fit u32 kernel ABI: {error}. Shard the image before launch.",
                    view.row_count()
                ),
            }
        })?;
        let items_per_wave = u64::from(config.threads_per_block)
            .checked_mul(u64::from(config.max_blocks_per_launch))
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA e-graph signature refresh launch dimensions overflowed u64 item accounting.".to_string(),
            })?;
        let mut first_row = 0_u64;
        let mut launch_count = 0_usize;
        let mut kernel_args = SmallVec::<[*mut std::ffi::c_void; 8]>::new();
        while first_row < row_count {
            let wave_rows = (row_count - first_row).min(items_per_wave);
            let blocks =
                ceil_div_u64(wave_rows, u64::from(config.threads_per_block)).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: error.to_string(),
                    }
                })?;
            let blocks = u32::try_from(blocks).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA e-graph signature refresh block count does not fit u32 launch ABI: {error}."
                ),
            })?;
            let launch = LaunchPlan {
                element_count: u32::try_from(wave_rows).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA e-graph signature refresh wave row count {wave_rows} does not fit u32 launch accounting: {error}. Split the wave before launch."
                        ),
                    }
                })?,
                workgroup: [config.threads_per_block, 1, 1],
                grid: [blocks, 1, 1],
                param_words: Vec::new(),
                max_binding_alignment: std::mem::size_of::<u64>(),
            };
            let mut args = EGraphSignatureRefreshKernelArgs {
                row_language_op_ids_ptr: view.row_language_op_ids_ptr(),
                row_children_offsets_ptr: view.row_children_offsets_ptr(),
                row_children_lens_ptr: view.row_children_lens_ptr(),
                row_signatures_ptr: view.row_signatures_ptr(),
                children_ptr: view.children_ptr(),
                row_count: row_count_u32,
                first_row,
            };
            args.write_kernel_args_into(&mut kernel_args)?;
            self.launch_resolved_function(
                func,
                &mut kernel_args,
                &launch,
                stream.raw(),
                false,
                false,
            )?;
            first_row = first_row.checked_add(wave_rows).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: "Fix: CUDA e-graph signature refresh launch wave cursor overflowed u64 row accounting.".to_string(),
                }
            })?;
            launch_count =
                launch_count
                    .checked_add(1)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: "Fix: CUDA e-graph signature refresh launch count overflowed usize."
                            .to_string(),
                    })?;
        }
        stream.synchronize()?;
        Ok(CudaEGraphSignatureRefreshKernelResult {
            row_count: view.row_count(),
            launch_count,
            total_rows: row_count,
        })
    }
}

struct EGraphStructuralKernelArgs {
    row_eclass_ids_ptr: u64,
    row_language_op_ids_ptr: u64,
    row_children_offsets_ptr: u64,
    row_children_lens_ptr: u64,
    row_signatures_ptr: u64,
    children_ptr: u64,
    bucket_words_ptr: u64,
    bucket_rows_ptr: u64,
    output_pairs_ptr: u64,
    output_count_ptr: u64,
    bucket_index: u32,
    first_pair: u64,
    pair_count: u64,
}

/// Decode a signature-bucket pair ordinal to the concrete row ids kernels must
/// compare.
///
/// Pair ordinals enumerate the upper triangle of each bucket in row-major
/// order: `(0, 1), (0, 2), ..., (1, 2), ...`. CUDA kernels can use this same
/// arithmetic to map a thread's pair ordinal to two row ids without materializing
/// all candidate pairs.
///
/// # Errors
///
/// Returns [`CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds`] when
/// `bucket_index` or `pair_ordinal` does not identify a planned candidate pair.
pub fn cuda_egraph_signature_pair_rows(
    plan: &CudaEGraphSignatureBucketPlan,
    bucket_index: u32,
    pair_ordinal: u64,
) -> Result<(u32, u32), CudaEGraphKernelPlanError> {
    let Some(bucket) = plan.buckets.get(bucket_index as usize) else {
        return Err(CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
            bucket_index,
            pair_ordinal,
            candidate_pair_count: 0,
        });
    };
    if pair_ordinal >= bucket.candidate_pair_count {
        return Err(CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
            bucket_index,
            pair_ordinal,
            candidate_pair_count: bucket.candidate_pair_count,
        });
    }

    let row_count = u64::from(bucket.row_count);
    let mut lo = 0_u64;
    let mut hi = row_count - 1;
    while lo < hi {
        let mid = lo + ((hi - lo) / 2);
        let next_start = signature_pairs_before_row(mid + 1, row_count)?;
        if next_start <= pair_ordinal {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let local_left = lo;
    let row_pair_base = signature_pairs_before_row(local_left, row_count)?;
    let local_right = local_left
        .checked_add(1)
        .and_then(|value| value.checked_add(pair_ordinal - row_pair_base))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "signature pair local right row",
        })?;
    let base = bucket.first_bucket_row as usize;
    let bucket_end = base.checked_add(bucket.row_count as usize).ok_or(
        CudaEGraphKernelPlanError::CountOverflow {
            field: "signature bucket row range end",
        },
    )?;
    if bucket_end > plan.bucket_rows.len() {
        return Err(CudaEGraphKernelPlanError::SignatureBucketRowsOutOfBounds {
            bucket_index,
            first_bucket_row: base,
            row_count: bucket.row_count as usize,
            bucket_rows_len: plan.bucket_rows.len(),
        });
    }
    let left = plan.bucket_rows[base + local_left as usize];
    let right = plan.bucket_rows[base + local_right as usize];
    Ok((left, right))
}

fn validate_image_view_matches(
    image: &GpuEGraphDeviceImage,
    view: CudaEGraphDeviceKernelView,
) -> Result<(), CudaEGraphKernelPlanError> {
    if image.layout().row_count() != view.row_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "row count",
            image: image.layout().row_count(),
            view: view.row_count(),
        });
    }
    if image.layout().child_count() != view.child_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "child count",
            image: image.layout().child_count(),
            view: view.child_count(),
        });
    }
    if image.layout().eclass_group_count() != view.eclass_group_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "eclass group count",
            image: image.layout().eclass_group_count(),
            view: view.eclass_group_count(),
        });
    }
    Ok(())
}

fn packed_rows_structurally_equal(
    image: &GpuEGraphDeviceImage,
    left_row: u32,
    right_row: u32,
) -> Result<bool, CudaEGraphKernelPlanError> {
    let left = left_row as usize;
    let right = right_row as usize;
    let row_count = image.layout().row_count();
    if left >= row_count {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "rows",
            row: left_row,
            start: left,
            end: left.saturating_add(1),
            len: row_count,
        });
    }
    if right >= row_count {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "rows",
            row: right_row,
            start: right,
            end: right.saturating_add(1),
            len: row_count,
        });
    }
    if image.row_signatures()[left] != image.row_signatures()[right] {
        return Ok(false);
    }
    if image.row_language_op_ids()[left] != image.row_language_op_ids()[right] {
        return Ok(false);
    }
    if image.row_children_lens()[left] != image.row_children_lens()[right] {
        return Ok(false);
    }

    let left_children = packed_row_children(image, left_row)?;
    let right_children = packed_row_children(image, right_row)?;
    Ok(left_children == right_children)
}

fn packed_row_children(
    image: &GpuEGraphDeviceImage,
    row: u32,
) -> Result<&[u32], CudaEGraphKernelPlanError> {
    let row_index = row as usize;
    let start = image.row_children_offsets()[row_index] as usize;
    let len = image.row_children_lens()[row_index] as usize;
    let end = start
        .checked_add(len)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "packed row child span end",
        })?;
    let children = image.children();
    if end > children.len() {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "children",
            row,
            start,
            end,
            len: children.len(),
        });
    }
    Ok(&children[start..end])
}

fn append_pass_waves(
    waves: &mut Vec<CudaEGraphKernelWave>,
    total_items: &mut u64,
    total_blocks: &mut u64,
    pass: CudaEGraphKernelPass,
    item_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<(), CudaEGraphKernelPlanError> {
    if item_count == 0 {
        return Ok(());
    }
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per launch wave",
        })?;
    let mut first_item = 0_u64;
    while first_item < item_count {
        let remaining = item_count - first_item;
        let wave_items = remaining.min(items_per_wave);
        let blocks = ceil_div_u64(wave_items, u64::from(config.threads_per_block))?;
        let blocks =
            u32::try_from(blocks).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "blocks per launch wave",
            })?;
        waves.push(CudaEGraphKernelWave {
            pass,
            first_item,
            item_count: wave_items,
            blocks,
            threads_per_block: config.threads_per_block,
        });
        *total_items = total_items.checked_add(wave_items).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "total logical items",
            },
        )?;
        *total_blocks = total_blocks.checked_add(u64::from(blocks)).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "total blocks",
            },
        )?;
        first_item =
            first_item
                .checked_add(wave_items)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "next wave first item",
                })?;
    }
    Ok(())
}

fn append_signature_pair_waves(
    pair_waves: &mut Vec<CudaEGraphSignaturePairWave>,
    total_blocks: &mut u64,
    bucket_index: u32,
    pair_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<(), CudaEGraphKernelPlanError> {
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per signature pair launch wave",
        })?;
    let mut first_pair = 0_u64;
    while first_pair < pair_count {
        let remaining = pair_count - first_pair;
        let wave_pairs = remaining.min(items_per_wave);
        let blocks = ceil_div_u64(wave_pairs, u64::from(config.threads_per_block))?;
        let blocks =
            u32::try_from(blocks).map_err(|_| CudaEGraphKernelPlanError::CountOverflow {
                field: "blocks per signature pair launch wave",
            })?;
        pair_waves.push(CudaEGraphSignaturePairWave {
            bucket_index,
            first_pair,
            pair_count: wave_pairs,
            blocks,
            threads_per_block: config.threads_per_block,
        });
        *total_blocks = total_blocks.checked_add(u64::from(blocks)).ok_or(
            CudaEGraphKernelPlanError::CountOverflow {
                field: "signature pair total blocks",
            },
        )?;
        first_pair =
            first_pair
                .checked_add(wave_pairs)
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "next signature pair first item",
                })?;
    }
    Ok(())
}

fn wave_count_for(
    item_count: u64,
    config: CudaEGraphKernelLaunchConfig,
) -> Result<u64, CudaEGraphKernelPlanError> {
    if item_count == 0 {
        return Ok(0);
    }
    let items_per_wave = u64::from(config.threads_per_block)
        .checked_mul(u64::from(config.max_blocks_per_launch))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "items per launch wave",
        })?;
    ceil_div_u64(item_count, items_per_wave)
}

fn ceil_div_u64(numerator: u64, denominator: u64) -> Result<u64, CudaEGraphKernelPlanError> {
    if denominator == 0 {
        return Err(CudaEGraphKernelPlanError::CountOverflow {
            field: "ceil division denominator",
        });
    }
    if numerator == 0 {
        return Ok(0);
    }
    numerator
        .checked_add(denominator - 1)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "ceil division numerator",
        })
        .map(|value| value / denominator)
}

fn unordered_pair_count(item_count: u64) -> Result<u64, CudaEGraphKernelPlanError> {
    item_count
        .checked_mul(item_count.saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "unordered pair count",
        })
}

fn signature_pairs_before_row(
    local_row: u64,
    row_count: u64,
) -> Result<u64, CudaEGraphKernelPlanError> {
    local_row
        .checked_mul(
            row_count
                .checked_mul(2)
                .and_then(|value| value.checked_sub(local_row))
                .and_then(|value| value.checked_sub(1))
                .ok_or(CudaEGraphKernelPlanError::CountOverflow {
                    field: "signature pair row width",
                })?,
        )
        .and_then(|value| value.checked_div(2))
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "signature pairs before row",
        })
}

fn usize_to_u64(value: usize, field: &'static str) -> Result<u64, CudaEGraphKernelPlanError> {
    CUDA_NUMERIC
        .usize_to_u64(value, field)
        .map_err(|_| CudaEGraphKernelPlanError::CountOverflow { field })
}

#[cfg(test)]
mod tests;
