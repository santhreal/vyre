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
