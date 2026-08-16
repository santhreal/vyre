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

// Inline: `egraph_kernel_plan` is `pub(crate)`, so the kernel arg packing, launch
// artifacts and signature buckets are unreachable from an integration test.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph_kernel_plan::args::{
        EGraphCanonicalRewriteKernelArgs, EGraphStructuralKernelArgs,
    };
    use crate::plan_cuda_egraph_device_upload;
    use crate::CudaEGraphDeviceKernelView;
    use vyre_foundation::optimizer::eqsat_gpu::GpuEGraphSnapshot;
    use vyre_foundation::optimizer::eqsat_gpu::{Equivalence, GpuEGraphDeviceImage};

    fn synthetic_view(rows: usize, children: usize, groups: usize) -> CudaEGraphDeviceKernelView {
        assert!(groups <= rows);
        assert!(children <= rows.saturating_mul(2));
        let mut child_storage = Vec::new();
        let mut row_specs = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = child_storage.len();
            if child_storage.len() < children && row > 0 {
                child_storage.push((row - 1) as u32);
            }
            if child_storage.len() < children && row > 1 {
                child_storage.push((row / 2) as u32);
            }
            let eclass = if groups == 0 { row } else { row % groups };
            row_specs.push((
                eclass as u32,
                if row & 1 == 0 { "lit" } else { "add" },
                start,
                child_storage.len() - start,
            ));
        }
        while child_storage.len() < children {
            child_storage.push(0);
            let last = row_specs
                .last_mut()
                .expect("Fix: synthetic child-only view requires at least one row");
            last.3 += 1;
        }
        let build_rows = row_specs
            .iter()
            .map(|&(class, op, start, len)| (class, op, &child_storage[start..start + len]))
            .collect::<Vec<_>>();
        let snapshot = GpuEGraphSnapshot::build(build_rows);
        let plan = plan_cuda_egraph_device_upload(&snapshot).expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - synthetic plan must pack");
        CudaEGraphDeviceKernelView::from_checked_parts(0x1000, plan.byte_len(), plan.byte_layout())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - synthetic view must be valid")
    }

    fn view_for_image(image: &GpuEGraphDeviceImage) -> CudaEGraphDeviceKernelView {
        let plan = crate::plan_cuda_egraph_device_upload_from_image(image.clone())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - packed egraph image must have a CUDA upload plan");
        CudaEGraphDeviceKernelView::from_checked_parts(0x4000, plan.byte_len(), plan.byte_layout())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - upload plan must resolve to a checked kernel view")
    }

    mod args_contracts {
        use super::*;

        #[test]
        fn egraph_kernel_args_into_reuses_capacity_and_preserves_abi_order() {
            let mut table = smallvec::SmallVec::<[*mut std::ffi::c_void; 8]>::new();
            let mut structural = EGraphStructuralKernelArgs {
                row_eclass_ids_ptr: 1,
                row_language_op_ids_ptr: 2,
                row_children_offsets_ptr: 3,
                row_children_lens_ptr: 4,
                row_signatures_ptr: 5,
                children_ptr: 6,
                bucket_words_ptr: 7,
                bucket_rows_ptr: 8,
                output_pairs_ptr: 9,
                output_count_ptr: 10,
                bucket_index: 11,
                first_pair: 12,
                pair_count: 13,
            };

            structural
                .write_kernel_args_into(&mut table)
                .expect("Fix: structural e-graph kernel args should build");
            let capacity = table.capacity();
            assert_eq!(table.len(), 13);
            assert_eq!(
                table[0],
                &mut structural.row_eclass_ids_ptr as *mut _ as *mut std::ffi::c_void
            );
            assert_eq!(
                table[12],
                &mut structural.pair_count as *mut _ as *mut std::ffi::c_void
            );

            let mut rewrite = EGraphCanonicalRewriteKernelArgs {
                row_eclass_ids_ptr: 21,
                children_ptr: 22,
                rewrite_words_ptr: 23,
                rewrite_count: 24,
                row_count: 25,
                child_count: 26,
                first_item: 27,
            };
            rewrite
                .write_kernel_args_into(&mut table)
                .expect("Fix: canonical rewrite e-graph kernel args should reuse table");
            assert_eq!(table.capacity(), capacity);
            assert_eq!(table.len(), 7);
            assert_eq!(
                table[0],
                &mut rewrite.row_eclass_ids_ptr as *mut _ as *mut std::ffi::c_void
            );
            assert_eq!(
                table[6],
                &mut rewrite.first_item as *mut _ as *mut std::ffi::c_void
            );
        }
    }

    mod launch_artifact_contracts {
        use super::*;

        #[test]
        fn consuming_launch_artifact_matches_borrowed_artifact_without_plan_clone_contract() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "add", &[0u32, 1u32][..]),
                (3u32, "add", &[0u32, 1u32][..]),
                (4u32, "mul", &[0u32, 1u32][..]),
                (5u32, "mul", &[0u32, 1u32][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 8,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            let borrowed = plan_cuda_egraph_structural_equivalence_launch_artifact(&plan)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - borrowed launch artifact must build");
            let consumed = plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan(plan)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - consuming launch artifact must build");

            assert_eq!(consumed, borrowed);
        }

        #[test]
        fn resident_snapshot_try_constructors_match_infallible_snapshots() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "add", &[0u32, 1u32][..]),
                (3u32, "mul", &[1u32, 2u32][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");

            let full = CudaEGraphResidentColumnSnapshot::try_from_device_image(&image)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - fallible full snapshot should reserve");
            let infallible_full = CudaEGraphResidentColumnSnapshot::from_device_image(&image);
            assert_eq!(full, infallible_full);

            let signatures = CudaEGraphResidentSignatureSnapshot::try_from_device_image(&image)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - fallible signature snapshot should reserve");
            let from_full = CudaEGraphResidentSignatureSnapshot::try_from_column_snapshot(&full)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - fallible signature snapshot from full columns should reserve");
            assert_eq!(signatures, from_full);
            assert_eq!(
                signatures,
                CudaEGraphResidentSignatureSnapshot::from_device_image(&image)
            );
        }

        #[test]
        fn union_compaction_uses_reserved_eclass_index_for_generated_large_components() {
            let edge_count = 1024_u32;
            let mut equivalences = Vec::with_capacity((edge_count as usize) * 3);
            let mut expected_self_pairs = 0_u64;
            for edge in 0..edge_count {
                equivalences.push(Equivalence {
                    left: edge + 1,
                    right: edge,
                });
                equivalences.push(Equivalence {
                    left: edge,
                    right: edge + 1,
                });
                if edge % 7 == 0 {
                    expected_self_pairs += 1;
                    equivalences.push(Equivalence {
                        left: edge,
                        right: edge,
                    });
                }
            }

            let plan = plan_cuda_egraph_union_compaction(
                &equivalences,
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 128,
                    max_blocks_per_launch: 16,
                },
            )
            .expect("Fix: generated CUDA e-graph union compaction plan should fit");

            assert_eq!(plan.canonical_pairs.len(), edge_count as usize);
            assert_eq!(plan.duplicate_pair_count, edge_count as u64);
            assert_eq!(plan.ignored_self_pair_count, expected_self_pairs);
            assert_eq!(plan.affected_eclasses.len(), edge_count as usize + 1);
            assert_eq!(plan.canonical_rewrites.len(), edge_count as usize);
            assert!(plan
                .canonical_rewrites
                .iter()
                .all(|rewrite| rewrite.representative == 0 && rewrite.eclass_id != 0));
        }
    }

    mod planner_contracts {
        use super::*;

        #[test]
        fn planner_emits_passes_in_row_child_group_order() {
            let view = synthetic_view(3, 2, 2);
            let plan = plan_cuda_egraph_kernel_work(
                view,
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 4,
                    max_blocks_per_launch: 2,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph kernel plan");

            assert_eq!(plan.waves.len(), 3);
            assert_eq!(plan.waves[0].pass, CudaEGraphKernelPass::RowScan);
            assert_eq!(plan.waves[0].item_count, 3);
            assert_eq!(plan.waves[0].blocks, 1);
            assert_eq!(plan.waves[1].pass, CudaEGraphKernelPass::ChildEdgeScan);
            assert_eq!(plan.waves[1].item_count, 2);
            assert_eq!(plan.waves[2].pass, CudaEGraphKernelPass::EclassGroupScan);
            assert_eq!(plan.waves[2].item_count, 2);
            assert_eq!(plan.total_items, 7);
            assert_eq!(plan.total_blocks, 3);
        }

        #[test]
        fn planner_splits_large_passes_into_bounded_waves() {
            let view = synthetic_view(19, 0, 0);
            let plan = plan_cuda_egraph_kernel_work(
                view,
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 4,
                    max_blocks_per_launch: 2,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid split egraph kernel plan");

            let items = plan
                .waves
                .iter()
                .map(|wave| (wave.first_item, wave.item_count, wave.blocks))
                .collect::<Vec<_>>();
            assert_eq!(
                items,
                vec![
                    (0, 8, 2),
                    (8, 8, 2),
                    (16, 3, 1),
                    (0, 8, 2),
                    (8, 8, 2),
                    (16, 3, 1),
                ]
            );
            assert_eq!(plan.total_items, 38);
            assert_eq!(plan.total_blocks, 10);
        }

        #[test]
        fn planner_rejects_zero_launch_dimensions() {
            let view = synthetic_view(1, 0, 0);
            assert_eq!(
                plan_cuda_egraph_kernel_work(
                    view,
                    CudaEGraphKernelLaunchConfig {
                        threads_per_block: 0,
                        max_blocks_per_launch: 1,
                    },
                )
                .expect_err("zero threads must be rejected"),
                CudaEGraphKernelPlanError::ZeroThreadsPerBlock
            );
            assert_eq!(
                plan_cuda_egraph_kernel_work(
                    view,
                    CudaEGraphKernelLaunchConfig {
                        threads_per_block: 1,
                        max_blocks_per_launch: 0,
                    },
                )
                .expect_err("zero max blocks must be rejected"),
                CudaEGraphKernelPlanError::ZeroMaxBlocksPerLaunch
            );
        }

        #[test]
        fn signature_bucket_planner_groups_only_candidate_duplicate_rows() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "add", &[0u32, 1u32][..]),
                (3u32, "add", &[0u32, 1u32][..]),
                (4u32, "add", &[1u32, 0u32][..]),
                (5u32, "mul", &[0u32, 1u32][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 8,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            let grouped_rows = plan
                .buckets
                .iter()
                .map(|bucket| {
                    let start = bucket.first_bucket_row as usize;
                    let end = start + bucket.row_count as usize;
                    plan.bucket_rows[start..end].to_vec()
                })
                .collect::<Vec<_>>();

            assert_eq!(grouped_rows.len(), 2);
            assert!(grouped_rows.contains(&vec![0, 1]));
            assert!(grouped_rows.contains(&vec![2, 3]));
            assert_eq!(plan.candidate_pair_count, 2);
            assert_eq!(plan.pair_waves.len(), 2);
            assert!(plan
                .pair_waves
                .iter()
                .all(|wave| wave.pair_count == 1 && wave.blocks == 1));
        }

        #[test]
        fn structural_equivalence_planner_rejects_divergent_language_op_ids() {
            let snapshot =
                GpuEGraphSnapshot::build([(10u32, "lit", &[][..]), (20u32, "opaque", &[][..])]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: valid divergent-op egraph image must pack");

            assert_ne!(
                image.row_language_op_ids()[0],
                image.row_language_op_ids()[1]
            );

            let plan = plan_cuda_egraph_structural_equivalences(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 8,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: divergent-op egraph image must remain plannable");

            assert!(plan.signature_plan.buckets.is_empty());
            assert_eq!(plan.signature_plan.candidate_pair_count, 0);
            assert!(plan.equivalences.is_empty());
            assert_eq!(plan.exact_pair_count, 0);
            assert_eq!(plan.rejected_candidate_pair_count, 0);
        }
    }

    mod ptx_contracts {
        use super::*;

        #[test]
        fn structural_equivalence_kernel_ptx_rejects_invalid_sm_target() {
            assert_eq!(
                cuda_egraph_structural_equivalence_kernel_ptx(0)
                    .expect_err("sm_0 is not a valid CUDA PTX target"),
                CudaEGraphKernelPlanError::InvalidPtxTarget { target_sm: 0 }
            );
        }

        #[test]

        fn signature_bucket_planner_rejects_mismatched_image_and_view() {
            let image = GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "lit", &[][..])])
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let mismatched_view = synthetic_view(1, 0, 1);

            assert_eq!(
                plan_cuda_egraph_signature_buckets(
                    &image,
                    mismatched_view,
                    CudaEGraphKernelLaunchConfig::default(),
                )
                .expect_err("image/view row mismatch must be rejected"),
                CudaEGraphKernelPlanError::ImageViewMismatch {
                    field: "row count",
                    image: 2,
                    view: 1,
                }
            );
        }
    }

    mod signature_bucket_contracts {
        use super::*;

        #[test]
        fn signature_bucket_planner_splits_large_candidate_bucket() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "lit", &[][..]),
                (3u32, "lit", &[][..]),
                (4u32, "lit", &[][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 2,
                    max_blocks_per_launch: 2,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            assert_eq!(plan.buckets.len(), 1);
            assert_eq!(plan.buckets[0].row_count, 5);
            assert_eq!(plan.candidate_pair_count, 10);
            assert_eq!(plan.bucket_rows, vec![0, 1, 2, 3, 4]);
            assert_eq!(
                plan.pair_waves,
                vec![
                    CudaEGraphSignaturePairWave {
                        bucket_index: 0,
                        first_pair: 0,
                        pair_count: 4,
                        blocks: 2,
                        threads_per_block: 2,
                    },
                    CudaEGraphSignaturePairWave {
                        bucket_index: 0,
                        first_pair: 4,
                        pair_count: 4,
                        blocks: 2,
                        threads_per_block: 2,
                    },
                    CudaEGraphSignaturePairWave {
                        bucket_index: 0,
                        first_pair: 8,
                        pair_count: 2,
                        blocks: 1,
                        threads_per_block: 2,
                    },
                ]
            );
            assert_eq!(plan.total_blocks, 5);
        }

        #[test]
        fn signature_pair_ordinals_decode_to_row_pairs_without_materialized_pairs() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "lit", &[][..]),
                (3u32, "lit", &[][..]),
                (4u32, "lit", &[][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 4,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            let decoded = (0..plan.candidate_pair_count)
                .map(|ordinal| cuda_egraph_signature_pair_rows(&plan, 0, ordinal).unwrap())
                .collect::<Vec<_>>();

            assert_eq!(
                decoded,
                vec![
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 3),
                    (2, 4),
                    (3, 4),
                ]
            );
        }

        #[test]
        fn signature_pair_decoder_rejects_out_of_bounds_ordinals() {
            let snapshot =
                GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "lit", &[][..])]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig::default(),
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            assert_eq!(
                cuda_egraph_signature_pair_rows(&plan, 0, 1)
                    .expect_err("one two-row bucket has exactly one pair"),
                CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
                    bucket_index: 0,
                    pair_ordinal: 1,
                    candidate_pair_count: 1,
                }
            );
            assert_eq!(
                cuda_egraph_signature_pair_rows(&plan, 7, 0)
                    .expect_err("missing bucket must be rejected"),
                CudaEGraphKernelPlanError::SignaturePairOrdinalOutOfBounds {
                    bucket_index: 7,
                    pair_ordinal: 0,
                    candidate_pair_count: 0,
                }
            );
        }

        #[test]
        fn signature_pair_decoder_rejects_malformed_bucket_row_ranges() {
            let snapshot =
                GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "lit", &[][..])]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let plan = CudaEGraphSignatureBucketPlan {
                view: view_for_image(&image),
                buckets: vec![CudaEGraphSignatureBucket {
                    signature: image.row_signatures()[0],
                    first_bucket_row: 1,
                    row_count: 2,
                    candidate_pair_count: 1,
                }],
                bucket_rows: vec![0, 1],
                pair_waves: Vec::new(),
                candidate_pair_count: 1,
                total_blocks: 0,
            };

            assert_eq!(
                cuda_egraph_signature_pair_rows(&plan, 0, 0)
                    .expect_err("malformed bucket row range must be rejected"),
                CudaEGraphKernelPlanError::SignatureBucketRowsOutOfBounds {
                    bucket_index: 0,
                    first_bucket_row: 1,
                    row_count: 2,
                    bucket_rows_len: 2,
                }
            );
        }
    }

    mod structural_equivalence_contracts {
        use super::*;

        #[test]
        fn structural_equivalence_plan_emits_unique_exact_eclass_merges() {
            let snapshot = GpuEGraphSnapshot::build([
                (10u32, "lit", &[][..]),
                (20u32, "lit", &[][..]),
                (30u32, "add", &[10u32, 20u32][..]),
                (40u32, "add", &[10u32, 20u32][..]),
                (50u32, "add", &[20u32, 10u32][..]),
                (30u32, "add", &[10u32, 20u32][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");

            let plan = plan_cuda_egraph_structural_equivalences(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 8,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - structural equivalence plan must build");

            assert_eq!(
                plan.equivalences,
                vec![
                    Equivalence {
                        left: 10,
                        right: 20,
                    },
                    Equivalence {
                        left: 30,
                        right: 40,
                    },
                ]
            );
            assert_eq!(plan.exact_pair_count, 4);
            assert_eq!(plan.redundant_pair_count, 1);
            assert_eq!(plan.rejected_candidate_pair_count, 0);
            assert_eq!(plan.equivalence_output_words, 4);
        }

        #[test]
        fn structural_equivalence_collection_filters_signature_collision_bucket() {
            let snapshot =
                GpuEGraphSnapshot::build([(0u32, "lit", &[][..]), (1u32, "add", &[0u32][..])]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let signature_plan = CudaEGraphSignatureBucketPlan {
                view: view_for_image(&image),
                buckets: vec![CudaEGraphSignatureBucket {
                    signature: image.row_signatures()[0],
                    first_bucket_row: 0,
                    row_count: 2,
                    candidate_pair_count: 1,
                }],
                bucket_rows: vec![0, 1],
                pair_waves: vec![CudaEGraphSignaturePairWave {
                    bucket_index: 0,
                    first_pair: 0,
                    pair_count: 1,
                    blocks: 1,
                    threads_per_block: 1,
                }],
                candidate_pair_count: 1,
                total_blocks: 1,
            };

            let plan = collect_cuda_egraph_structural_equivalences(&image, signature_plan)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - collision-safe structural collection must complete");

            assert!(plan.equivalences.is_empty());
            assert_eq!(plan.exact_pair_count, 0);
            assert_eq!(plan.redundant_pair_count, 0);
            assert_eq!(plan.rejected_candidate_pair_count, 1);
            assert_eq!(plan.equivalence_output_words, 0);
        }

        #[test]
        fn signature_bucket_device_image_packs_fixed_width_records() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "lit", &[][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let signature_plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 2,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            let device_image = pack_cuda_egraph_signature_bucket_device_image(&signature_plan)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket device image must pack");

            assert_eq!(device_image.bucket_count, 1);
            assert_eq!(device_image.bucket_record_words, 5);
            assert_eq!(device_image.bucket_rows, vec![0, 1, 2]);
            assert_eq!(
                device_image.bucket_words,
                vec![image.row_signatures()[0], 0, 3, 3, 0,]
            );
            assert_eq!(device_image.candidate_pair_count, 3);
        }

        #[test]
        fn structural_equivalence_launch_artifact_sizes_worst_case_output() {
            let snapshot = GpuEGraphSnapshot::build([
                (0u32, "lit", &[][..]),
                (1u32, "lit", &[][..]),
                (2u32, "lit", &[][..]),
                (3u32, "lit", &[][..]),
            ]);
            let image = snapshot
                .try_pack_device_image()
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - valid egraph image must pack");
            let signature_plan = plan_cuda_egraph_signature_buckets(
                &image,
                view_for_image(&image),
                CudaEGraphKernelLaunchConfig {
                    threads_per_block: 4,
                    max_blocks_per_launch: 1,
                },
            )
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - signature bucket plan must build");

            let artifact = plan_cuda_egraph_structural_equivalence_launch_artifact(&signature_plan)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - structural equivalence launch artifact must build");

            assert_eq!(artifact.bucket_image.bucket_count, 1);
            assert_eq!(artifact.output.max_equivalences, 6);
            assert_eq!(artifact.output.output_pair_words, 12);
            assert_eq!(artifact.output.output_pair_bytes, 48);
            assert_eq!(artifact.output.output_counter_words, 2);
            assert_eq!(artifact.output.output_counter_bytes, 8);
            assert_eq!(artifact.pair_waves.len(), 2);
        }
    }
}
