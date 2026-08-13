use super::*;

#[test]
fn cuda_compiled_pipeline_identity_uses_shared_domain_separated_contract() {
    for seed in 0_u32..2048 {
        let ptx_key = generated_pipeline_identity_key(seed, 0x5054_5820);
        let module_key = generated_pipeline_identity_key(seed, 0x4D4F_4420);
        let launch = generated_pipeline_identity_launch(seed);

        let key = cuda_compiled_pipeline_identity_key(&ptx_key, &module_key, &launch)
            .expect("Fix: generated CUDA compiled pipeline key must fit");
        let changed_ptx = cuda_compiled_pipeline_identity_key(
            &generated_pipeline_identity_key(seed ^ 1, 0x5054_5820),
            &module_key,
            &launch,
        )
        .expect("Fix: generated CUDA compiled pipeline PTX variant must fit");
        let changed_module = cuda_compiled_pipeline_identity_key(
            &ptx_key,
            &generated_pipeline_identity_key(seed ^ 1, 0x4D4F_4420),
            &launch,
        )
        .expect("Fix: generated CUDA compiled pipeline module variant must fit");
        let mut changed_launch = launch.clone();
        changed_launch.grid[0] = changed_launch.grid[0].wrapping_add(1);
        let changed_launch_key =
            cuda_compiled_pipeline_identity_key(&ptx_key, &module_key, &changed_launch)
                .expect("Fix: generated CUDA compiled pipeline launch variant must fit");

        assert_ne!(key, changed_ptx);
        assert_ne!(key, changed_module);
        assert_ne!(key, changed_launch_key);
    }
}

#[test]
fn cuda_pipeline_dynamic_dispatch_reuses_existing_output_slots() {
    let mut outputs = vec![Vec::with_capacity(8), Vec::with_capacity(4)];
    let outputs_addr = outputs.as_ptr() as usize;
    let first_slot_addr = outputs[0].as_ptr() as usize;
    let second_slot_addr = outputs[1].as_ptr() as usize;

    replace_output_buffers_preserving_slots(vec![vec![1, 2, 3], vec![4]], &mut outputs);

    assert_eq!(outputs, vec![vec![1, 2, 3], vec![4]]);
    assert_eq!(outputs.as_ptr() as usize, outputs_addr);
    assert_eq!(outputs[0].as_ptr() as usize, first_slot_addr);
    assert_eq!(outputs[1].as_ptr() as usize, second_slot_addr);
}

#[test]
fn cuda_graph_lane_planner_scales_past_legacy_four_lane_cap() {
    let caps = synthetic_sm120_envelope(32 * 1024 * 1024 * 1024);
    let plan = single_input_output_plan(1024);
    let input = vec![7_u8; 1024];
    let row = [input.as_slice()];
    let batches: Vec<&[&[u8]]> = vec![row.as_slice(); 64];

    let lanes = cuda_graph_lane_count_for_batch(&caps, &plan, &batches)
        .expect("Fix: graph replay lane planning should fit");

    assert!(lanes > 4);
    assert_eq!(lanes, 22);
}

#[test]
fn cuda_graph_lane_planner_caps_large_graphs_by_vram_budget() {
    let caps = synthetic_sm120_envelope(512 * 1024 * 1024);
    let plan = single_input_output_plan(64 * 1024 * 1024);
    let input = vec![1_u8; 64 * 1024 * 1024];
    let row = [input.as_slice()];
    let batches: Vec<&[&[u8]]> = vec![row.as_slice(); 64];

    let lanes = cuda_graph_lane_count_for_batch(&caps, &plan, &batches)
        .expect("Fix: graph replay lane planning should fit");

    assert_eq!(lanes, 1);
}
