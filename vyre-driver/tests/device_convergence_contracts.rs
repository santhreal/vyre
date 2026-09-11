//! Contracts for `vyre_driver::device_convergence`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::device_convergence::{
    plan_device_convergence, ConvergenceReadbackPolicy, DeviceConvergencePlanError,
};

#[test]
fn convergence_plan_reads_final_flag_once() {
    let plan = plan_device_convergence(128, 4, 0).expect("Fix: valid plan should build");

    assert_eq!(plan.max_device_iterations, 128);
    assert_eq!(plan.host_sync_points, 1);
    assert_eq!(plan.changed_flag_readback_bytes, 4);
    assert_eq!(plan.host_iteration_polls, 0);
    assert_eq!(
        plan.readback_policy,
        ConvergenceReadbackPolicy::FinalFlagOnly
    );
}

#[test]
fn convergence_plan_rejects_empty_iteration_budget() {
    let err = plan_device_convergence(0, 4, 0).expect_err("zero iterations cannot converge");

    assert_eq!(err, DeviceConvergencePlanError::EmptyIterationBudget);
    assert!(err.to_string().contains("at least one device iteration"));
}

#[test]
fn convergence_plan_rejects_wrong_changed_flag_width() {
    let err = plan_device_convergence(8, 1, 0).expect_err("changed flag must be a u32");

    assert_eq!(
        err,
        DeviceConvergencePlanError::InvalidChangedFlagWidth { bytes: 1 }
    );
    assert!(err.to_string().contains("4-byte device u32 changed flag"));
}

#[test]
fn convergence_plan_rejects_host_polled_iterations() {
    let err =
        plan_device_convergence(8, 4, 8).expect_err("host polling every iteration is forbidden");

    assert_eq!(
        err,
        DeviceConvergencePlanError::HostPolledConvergence { polls: 8 }
    );
    assert!(err.to_string().contains("read only the final changed flag"));
}

#[test]
fn generated_convergence_iteration_budgets_preserve_final_only_contract() {
    for max_device_iterations in 1..=4_096 {
        let plan = plan_device_convergence(max_device_iterations, 4, 0)
            .expect("Fix: generated nonzero iteration budgets should plan");
        assert_eq!(plan.max_device_iterations, max_device_iterations);
        assert_eq!(plan.host_sync_points, 1);
        assert_eq!(plan.changed_flag_readback_bytes, 4);
        assert_eq!(plan.host_iteration_polls, 0);
        assert_eq!(
            plan.readback_policy,
            ConvergenceReadbackPolicy::FinalFlagOnly
        );
    }
}
