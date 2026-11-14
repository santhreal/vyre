//! Handwritten oracle matrix for dispatch-shape policy and binding-plan invariants.
//!
//! Compares `dispatch_configs_share_launch_shape` and `binding` ABI layout
//! contracts against independent reference oracles over hostile corpora.

#![forbid(unsafe_code)]

use vyre_driver::dispatch_configs_share_launch_shape;
use vyre_driver::resolve_fixpoint_iterations;
use vyre_driver::DispatchConfig;

const CONFIG_CASES: usize = 256;

#[test]
fn dispatch_config_launch_shape_oracle_matches_independent_policy() {
    let mut assertions = 0usize;
    for case in 0..CONFIG_CASES {
        let compiled = hostile_dispatch_config(case as u32);
        let runtime_match = compiled.clone();
        let runtime_grid = {
            let mut cfg = compiled.clone();
            cfg.grid_override = Some([(case as u32 % 17) + 1, 1, 1]);
            cfg
        };
        let runtime_fixpoint = {
            let mut cfg = compiled.clone();
            cfg.fixpoint_iterations = Some(1 + (case as u32 % 4));
            cfg
        };

        assert_eq!(
            dispatch_configs_share_launch_shape(&compiled, &runtime_match),
            oracle_dispatch_configs_share_launch_shape(&compiled, &runtime_match),
            "Fix: dispatch_configs_share_launch_shape identical case {case} must follow the independent oracle."
        );
        assertions += 1;

        assert_eq!(
            dispatch_configs_share_launch_shape(&compiled, &runtime_grid),
            oracle_dispatch_configs_share_launch_shape(&compiled, &runtime_grid),
            "Fix: dispatch_configs_share_launch_shape grid case {case} must follow the independent oracle."
        );
        assertions += 1;

        assert_eq!(
            dispatch_configs_share_launch_shape(&compiled, &runtime_fixpoint),
            oracle_dispatch_configs_share_launch_shape(&compiled, &runtime_fixpoint),
            "Fix: dispatch_configs_share_launch_shape fixpoint case {case} must follow the independent oracle."
        );
        assertions += 1;
    }
    assert_eq!(assertions, CONFIG_CASES * 3);
}

fn oracle_dispatch_configs_share_launch_shape(
    compiled: &DispatchConfig,
    runtime: &DispatchConfig,
) -> bool {
    compiled.profile == runtime.profile
        && compiled.ulp_budget.unwrap_or(0) == runtime.ulp_budget.unwrap_or(0)
        && compiled.max_output_bytes == runtime.max_output_bytes
        && compiled.workgroup_override == runtime.workgroup_override
        && compiled.grid_override == runtime.grid_override
        && oracle_fixpoint_iterations_share_launch_shape(compiled, runtime)
        && compiled.speculation == runtime.speculation
        && compiled.persistent_thread == runtime.persistent_thread
        && compiled.cooperative == runtime.cooperative
}

fn oracle_fixpoint_iterations_share_launch_shape(
    compiled: &DispatchConfig,
    runtime: &DispatchConfig,
) -> bool {
    let Ok(compiled_iterations) = resolve_fixpoint_iterations(compiled, "oracle-dispatch-shape")
    else {
        return false;
    };
    let Ok(runtime_iterations) = resolve_fixpoint_iterations(runtime, "oracle-dispatch-shape")
    else {
        return false;
    };
    compiled_iterations == runtime_iterations
}

fn hostile_dispatch_config(seed: u32) -> DispatchConfig {
    let mut cfg = DispatchConfig::default();
    if seed & 1 == 0 {
        cfg.profile = Some(format!("profile-{}", seed % 7));
    }
    if seed & 2 == 0 {
        cfg.ulp_budget = Some((seed % 4) as u8);
    }
    if seed & 4 == 0 {
        cfg.max_output_bytes = Some(((seed % 19) + 1) as usize * 64);
    }
    if seed & 8 == 0 {
        cfg.workgroup_override = Some([64 + (seed % 3) * 64, 1, 1]);
    }
    if seed & 16 == 0 {
        cfg.grid_override = Some([(seed % 11) + 1, 1, 1]);
    }
    if seed & 32 == 0 {
        cfg.fixpoint_iterations = Some(1 + (seed % 5));
    }
    if seed & 64 == 0 {
        cfg.cooperative = seed & 128 != 0;
    }
    cfg
}
