//! Contracts for `vyre_driver::launch_fusion`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::launch_fusion::{
    plan_launch_fusion, plan_launch_fusion_with_scratch, LaunchFusionError, LaunchFusionPlan,
    LaunchFusionScratch, LaunchFusionStage,
};

#[test]
fn launch_fusion_groups_adjacent_compatible_stages() {
    let plan = plan_launch_fusion(
        &[
            stage(1, 7, 64, 32, 8, false),
            stage(2, 7, 32, 48, 8, false),
            stage(3, 7, 48, 16, 8, false),
        ],
        256,
    )
    .expect("Fix: compatible stages should fuse");

    assert_eq!(plan.launch_count, 1);
    assert_eq!(plan.avoided_launches, 2);
    assert_eq!(plan.groups[0].stage_ids, vec![1, 2, 3]);
    assert_eq!(plan.avoided_intermediate_bytes, 80);
}

#[test]
fn launch_fusion_splits_on_layout_host_boundary_and_budget() {
    let plan = plan_launch_fusion(
        &[
            stage(1, 7, 64, 32, 8, false),
            stage(2, 8, 32, 48, 8, false),
            stage(3, 8, 48, 16, 8, true),
            stage(4, 9, 16, 16, 8, false),
        ],
        128,
    )
    .expect("Fix: incompatible stages should split deterministically");

    assert_eq!(plan.launch_count, 4);
    assert_eq!(plan.avoided_launches, 0);
    assert_eq!(plan.groups[0].stage_ids, vec![1]);
    assert_eq!(plan.groups[1].stage_ids, vec![2]);
    assert_eq!(plan.groups[2].stage_ids, vec![3]);
    assert_eq!(plan.groups[3].stage_ids, vec![4]);
}

#[test]
fn launch_fusion_rejects_invalid_inputs() {
    assert_eq!(
        plan_launch_fusion(&[stage(1, 7, 1, 1, 1, false)], 0).expect_err("zero budget should fail"),
        LaunchFusionError::ZeroBudget
    );
    assert_eq!(
        plan_launch_fusion(
            &[stage(1, 7, 1, 1, 1, false), stage(1, 7, 1, 1, 1, false),],
            128,
        )
        .expect_err("duplicate stages should fail"),
        LaunchFusionError::DuplicateStage { id: 1 }
    );
    assert_eq!(
        plan_launch_fusion(&[stage(9, 7, 64, 32, 64, false)], 128)
            .expect_err("single over-budget stage should fail"),
        LaunchFusionError::StageOverBudget {
            id: 9,
            required_bytes: 160,
            budget_bytes: 128,
        }
    );
}

#[test]
fn generated_launch_fusion_preserves_budget_and_order_contract() {
    for seed in 0..4096_u64 {
        let stages = generated_stages(seed);
        let budget = 96 + (seed % 512);
        let plan = plan_launch_fusion(&stages, budget)
            .or_else(|error| match error {
                LaunchFusionError::StageOverBudget { .. } => Ok(LaunchFusionPlan {
                    groups: Vec::new(),
                    launch_count: 0,
                    avoided_launches: 0,
                    avoided_intermediate_bytes: 0,
                }),
                other => Err(other),
            })
            .expect("Fix: generated launch fusion should only reject singleton over-budget stages");
        if plan.groups.is_empty() {
            continue;
        }

        let flattened = plan
            .groups
            .iter()
            .flat_map(|group| group.stage_ids.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(
            flattened,
            stages.iter().map(|stage| stage.id).collect::<Vec<_>>(),
            "Fix: launch fusion must preserve original stage order for seed {seed}."
        );
        assert_eq!(
            usize::try_from(plan.launch_count).expect("Fix: plan launch_count must fit usize on this platform; reject oversized plans upstream - launch_count fits usize"),
            plan.groups.len(),
            "Fix: launch_count must match group count for seed {seed}."
        );
        assert_eq!(
            usize::try_from(plan.avoided_launches).expect("Fix: avoided_launches must fit usize; clamp or reject plan before fusion stats - avoided_launches fits usize"),
            stages.len() - plan.groups.len(),
            "Fix: avoided_launches must match fused group reduction for seed {seed}."
        );
        for group in &plan.groups {
            assert!(
                group.required_bytes <= budget,
                "Fix: fused group exceeded explicit budget for seed {seed}."
            );
        }
    }
}

#[test]
fn launch_fusion_reuses_caller_owned_duplicate_detection_scratch() {
    let mut scratch =
        LaunchFusionScratch::try_with_capacity(64).expect("Fix: fusion scratch should reserve");
    let wide = (0..64)
        .map(|id| stage(id, 7, 16, 16, 4, false))
        .collect::<Vec<_>>();
    let first = plan_launch_fusion_with_scratch(&wide, 8_192, &mut scratch)
        .expect("Fix: wide compatible stages should fuse");
    let id_capacity = scratch.id_capacity();

    assert_eq!(first.launch_count, 1);
    assert_eq!(first.avoided_launches, 63);

    let second = plan_launch_fusion_with_scratch(
        &[
            stage(10, 7, 64, 32, 8, false),
            stage(11, 8, 32, 48, 8, false),
        ],
        512,
        &mut scratch,
    )
    .expect("Fix: smaller incompatible stages should reuse duplicate-detection scratch");

    assert_eq!(second.launch_count, 2);
    assert!(scratch.id_capacity() >= id_capacity);
}

fn generated_stages(seed: u64) -> Vec<LaunchFusionStage> {
    let count = 1 + (seed as usize % 24);
    let mut stages = Vec::with_capacity(count);
    let mut state = seed ^ 0xF051_1A4A_7E57_0001;
    for index in 0..count {
        stages.push(stage(
            index as u32,
            next_u64(&mut state) % 5,
            1 + (next_u64(&mut state) % 48),
            1 + (next_u64(&mut state) % 48),
            next_u64(&mut state) % 24,
            next_u64(&mut state) % 11 == 0,
        ));
    }
    stages
}

fn stage(
    id: u32,
    layout_hash: u64,
    input_bytes: u64,
    output_bytes: u64,
    scratch_bytes: u64,
    requires_host_materialization: bool,
) -> LaunchFusionStage {
    LaunchFusionStage {
        id,
        layout_hash,
        input_bytes,
        output_bytes,
        scratch_bytes,
        requires_host_materialization,
    }
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
