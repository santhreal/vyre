//! Tests for natural-gradient launch tuning.

use super::*;
use crate::binding::{Binding, BindingRole};
use crate::launch::{effective_launch_workgroup_for_mode, LaunchPlan};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn natural_gradient_launch_tunes_safe_1d_storage_program() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
        [32, 1, 1],
        vec![],
    );
    let bindings = vec![Binding {
        name: std::sync::Arc::from("out"),
        binding: 0,
        buffer_index: 0,
        role: BindingRole::Output,
        element_size: 4,
        preferred_alignment: 128,
        element_count: 4096,
        static_byte_len: Some(16_384),
        input_index: None,
        output_index: Some(0),
    }];
    let limits = LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };
    let mut plan = LaunchPlan::new();

    plan.prepare_into_for_mode(
        &program,
        &bindings,
        &DispatchConfig::default(),
        limits,
        Mode::NaturalGradient,
    )
    .expect("Fix: safe 1D storage launch should accept natural-gradient cold start");

    assert_eq!(
        plan.workgroup,
        [512, 1, 1],
        "Fix: cold start must pick the widest width that keeps every resident thread slot usable. Was [1024,1,1], which is 1536/1024 = 1 block per SM and 512 stranded slots on every SM."
    );
    assert_eq!(
        limits.resident_threads_per_compute_unit(plan.workgroup[0]),
        Some(1536),
        "Fix: the chosen width must strand no per-SM thread slot when a candidate dividing 1536 evenly exists."
    );
    assert_eq!(plan.grid, [8, 1, 1]);
    assert_eq!(plan.element_count, 4096);
}

#[test]
fn natural_gradient_launch_preserves_declared_shape_for_local_workgroup_ids() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out_local_ids", 0, DataType::U32).with_count(4096)],
        [1024, 1, 1],
        vec![
            Node::let_bind("lane", Expr::LocalId { axis: 0 }),
            Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
            Node::let_bind(
                "global",
                Expr::add(
                    Expr::mul(Expr::var("block"), Expr::u32(1024)),
                    Expr::var("lane"),
                ),
            ),
            Node::store("out_local_ids", Expr::var("global"), Expr::var("lane")),
        ],
    );
    let bindings = vec![Binding {
        name: std::sync::Arc::from("out_local_ids"),
        binding: 0,
        buffer_index: 0,
        role: BindingRole::Output,
        element_size: 4,
        preferred_alignment: 128,
        element_count: 4096,
        static_byte_len: Some(16_384),
        input_index: None,
        output_index: Some(0),
    }];
    let limits = LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };

    assert_eq!(
        effective_launch_workgroup_for_mode(
            &program,
            &bindings,
            &DispatchConfig::default(),
            limits,
            Mode::NaturalGradient,
        ),
        [1024, 1, 1],
        "Fix: automatic launch tuning must not change kernels whose LocalId/WorkgroupId arithmetic makes workgroup shape semantic."
    );
}

#[test]
fn measured_launch_feedback_overrides_heuristic_cold_start() {
    let dir = tempfile::tempdir()
        .expect("Fix: measured launch feedback test needs an isolated tuner cache");
    let path = dir.path().join("launch-feedback.toml");
    let program = Program::wrapped(
        vec![BufferDecl::output("out_feedback_isolated", 0, DataType::U32).with_count(8192)],
        [32, 1, 1],
        vec![],
    );
    let config = DispatchConfig::default();
    let limits = LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };
    let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 8192, limits);
    natural_launch_cache_remove(key);

    assert_eq!(
        natural_gradient_cold_start_workgroup_with_store(
            &program,
            [32, 1, 1],
            8192,
            limits,
            Some(&path),
        ),
        Some([512, 1, 1]),
        "Fix: this pins the cold-start selector's output, not a required constant. It was [1024,1,1] under a heuristic with no occupancy term at all, so the old message's claim of an occupancy-efficient shape described the opposite of what it selected."
    );
    assert!(
        record_launch_measurement_for_mode_with_store(
            &program,
            &config,
            limits,
            8192,
            [64, 1, 1],
            1,
            Mode::NaturalGradient,
            Some(&path),
        ),
        "Fix: natural-gradient resolver must accept measured backend timing for safe 1D launches."
    );
    assert_eq!(
        natural_gradient_cold_start_workgroup_with_store(
            &program,
            [32, 1, 1],
            8192,
            limits,
            Some(&path),
        ),
        Some([64, 1, 1]),
        "Fix: measured launch feedback must steer future automatic launch choices."
    );
}

#[test]
fn persisted_launch_feedback_rehydrates_measured_selection() {
    let dir = tempfile::tempdir()
        .expect("Fix: launch feedback persistence test needs a temporary cache directory");
    let path = dir.path().join("launch-feedback.toml");
    let program = Program::wrapped(
        vec![BufferDecl::output("out_persisted", 0, DataType::U32).with_count(16_384)],
        [32, 1, 1],
        vec![],
    );
    let limits = LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };
    let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 16_384, limits);
    natural_launch_cache_remove(key);

    persist_natural_launch_selection_to_path(key, [64, 1, 1], &path)
        .expect("Fix: measured launch feedback should persist through the tuner cache format");

    assert_eq!(
        natural_gradient_cold_start_workgroup_with_store(
            &program,
            [32, 1, 1],
            16_384,
            limits,
            Some(&path),
        ),
        Some([64, 1, 1]),
        "Fix: automatic launch resolution must rehydrate measured feedback from the bounded tuner cache before falling back to heuristics."
    );
}

#[test]
fn natural_gradient_launch_preserves_explicit_and_shared_memory_shapes() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(4096),
            BufferDecl::workgroup("scratch", 64, DataType::U32),
        ],
        [64, 1, 1],
        vec![],
    );
    let bindings = vec![Binding {
        name: std::sync::Arc::from("out"),
        binding: 0,
        buffer_index: 0,
        role: BindingRole::Output,
        element_size: 4,
        preferred_alignment: 128,
        element_count: 4096,
        static_byte_len: Some(16_384),
        input_index: None,
        output_index: Some(0),
    }];
    let limits = LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };
    let mut config = DispatchConfig::default();
    config.workgroup_override = Some([256, 1, 1]);

    assert_eq!(
        effective_launch_workgroup_for_mode(
            &program,
            &bindings,
            &config,
            limits,
            Mode::NaturalGradient,
        ),
        [256, 1, 1],
        "Fix: explicit dispatch workgroup overrides must remain authoritative."
    );

    let default_config = DispatchConfig::default();
    assert_eq!(
        effective_launch_workgroup_for_mode(
            &program,
            &bindings,
            &default_config,
            limits,
            Mode::NaturalGradient,
        ),
        [64, 1, 1],
        "Fix: workgroup-local scratch kernels must keep their declared shape."
    );
}

#[test]
fn record_launch_measurement_starts_fresh_only_when_no_prior_history_exists() {
    let dir = tempfile::tempdir()
        .expect("Fix: measurement history test needs a temporary cache directory");
    let path = dir.path().join("measurements-test.toml");
    let program = Program::wrapped(
        vec![BufferDecl::output("out_meas_history", 0, DataType::U32).with_count(4096)],
        [32, 1, 1],
        vec![],
    );
    let config = DispatchConfig::default();
    let limits = LaunchGeometryLimits {
        backend: "test-measurements",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 0,
    };
    let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 4096, limits);
    natural_launch_cache_remove(key);

    assert!(
        record_launch_measurement_for_mode_with_store(
            &program,
            &config,
            limits,
            4096,
            [256, 1, 1],
            100,
            Mode::NaturalGradient,
            Some(&path),
        ),
        "Fix: first measurement must be accepted into the cache"
    );

    let after_first = natural_launch_cache_get(key);
    assert!(
        after_first.is_some(),
        "Fix: cache must hold a selection after the first measurement"
    );

    assert!(
        record_launch_measurement_for_mode_with_store(
            &program,
            &config,
            limits,
            4096,
            [128, 1, 1],
            50,
            Mode::NaturalGradient,
            Some(&path),
        ),
        "Fix: second measurement must be accepted into the cache"
    );

    let measurements = natural_launch_cache_measurements(key)
        .expect("Fix: cache must hold measurements after two records");
    assert!(
        measurements.len() >= 2,
        "Fix: measurement history must accumulate across calls, got {} entries, expected >= 2",
        measurements.len()
    );
    assert_eq!(
        measurements.get(&[256, 1, 1]),
        Some(&100),
        "Fix: first measurement (workgroup=[256,1,1], 100ns) must be retained in history"
    );
    assert_eq!(
        measurements.get(&[128, 1, 1]),
        Some(&50),
        "Fix: second measurement (workgroup=[128,1,1], 50ns) must be present in history"
    );
}

const REGRESSION_170_SM_COUNT: u32 = 170;

fn regression_1536_thread_sm_limits() -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend: "regression-1536-thread-sm-test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    }
}

fn tunable_1d_program(output: &'static str, element_count: u32, declared: [u32; 3]) -> Program {
    Program::wrapped(
        vec![BufferDecl::output(output, 0, DataType::U32).with_count(element_count)],
        declared,
        vec![],
    )
}

#[test]
fn cold_start_never_strands_resident_thread_slots_when_an_even_divisor_exists() {
    let limits = regression_1536_thread_sm_limits();
    for (output, element_count) in [
        ("out_no_strand_1k", 1024u32),
        ("out_no_strand_4k", 4096),
        ("out_no_strand_64k", 65_536),
        ("out_no_strand_tail", 4097),
        ("out_no_strand_100k", 100_000),
    ] {
        let program = tunable_1d_program(output, element_count, [32, 1, 1]);
        let resolved = resolve_launch_workgroup_for_mode(
            &program,
            &DispatchConfig::default(),
            limits,
            element_count,
            Mode::NaturalGradient,
        );
        let resident = limits.resident_threads_per_compute_unit(resolved[0]);
        assert_eq!(
            resident,
            Some(1536),
            "Fix: cold start chose {resolved:?} for {element_count} elements, leaving {} of every SM's 1536 thread slots unusable. Prefer a width that divides the per-SM budget evenly.",
            1536 - resident.unwrap_or(1536)
        );
    }
}

#[test]
fn cold_start_still_admits_1024_where_the_per_sm_budget_divides_evenly() {
    let limits = LaunchGeometryLimits {
        backend: "even-divisor-test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 2048,
    };
    let program = tunable_1d_program("out_even_divisor", 65_536, [32, 1, 1]);

    assert_eq!(
        limits.resident_threads_per_compute_unit(1024),
        Some(2048),
        "Fix: 1024 must reach every thread slot on a 2048-thread SM, otherwise this test's premise is wrong."
    );
    assert_eq!(
        resolve_launch_workgroup_for_mode(
            &program,
            &DispatchConfig::default(),
            limits,
            65_536,
            Mode::NaturalGradient,
        ),
        [1024, 1, 1],
        "Fix: residency-aware cold start must stay a residency rule. A width that strands nothing has to remain selectable on every device."
    );
}

#[test]
fn unreported_per_sm_budget_leaves_cold_start_byte_identical() {
    let limits = LaunchGeometryLimits {
        backend: "unreported-residency-test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 0,
    };
    assert_eq!(
        limits.resident_threads_per_compute_unit(1024),
        None,
        "Fix: an unreported per-SM budget must answer `unknown`, never a guessed number."
    );

    for (output, element_count) in [
        ("out_inert_1k", 1024u32),
        ("out_inert_4k", 4096),
        ("out_inert_64k", 65_536),
        ("out_inert_1000", 1_000),
        ("out_inert_4097", 4097),
        ("out_inert_100k", 100_000),
    ] {
        let program = tunable_1d_program(output, element_count, [32, 1, 1]);
        assert_eq!(
            resolve_launch_workgroup_for_mode(
                &program,
                &DispatchConfig::default(),
                limits,
                element_count,
                Mode::NaturalGradient,
            ),
            [1024, 1, 1],
            "Fix: residency-aware cold start must be inert for a backend that reports no per-SM budget. {element_count} elements resolved differently than they did before residency entered this decision."
        );
    }
}

#[test]
fn explicit_geometry_pins_outrank_residency_aware_cold_start() {
    let limits = regression_1536_thread_sm_limits();
    let declared = [256, 1, 1];
    let program = tunable_1d_program("out_pinned_geometry", 262_144, declared);

    let mut pinned_workgroup = DispatchConfig::default();
    pinned_workgroup.workgroup_override = Some([64, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup_for_mode(
            &program,
            &pinned_workgroup,
            limits,
            262_144,
            Mode::NaturalGradient,
        ),
        [64, 1, 1],
        "Fix: an explicit workgroup override stays authoritative even when residency prefers another width."
    );

    let mut pinned_grid = DispatchConfig::default();
    pinned_grid.grid_override = Some([1024, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup_for_mode(
            &program,
            &pinned_grid,
            limits,
            262_144,
            Mode::NaturalGradient,
        ),
        declared,
        "Fix: an explicit grid override must keep the declared workgroup, since the caller sized the grid against it."
    );
}

#[test]
fn cooperative_lane_ceiling_follows_the_resolved_width_not_the_declared_one() {
    let limits = regression_1536_thread_sm_limits();
    let declared = [256, 1, 1];
    let program = tunable_1d_program("out_resolved_ceiling", 262_144, declared);
    let lane_ceiling = |width: u32| -> u64 {
        u64::from(
            limits
                .resident_threads_per_compute_unit(width)
                .expect("Fix: this device model reports a per-SM thread budget"),
        ) * u64::from(REGRESSION_170_SM_COUNT)
    };

    let resolved = resolve_launch_workgroup_for_mode(
        &program,
        &DispatchConfig::default(),
        limits,
        262_144,
        Mode::NaturalGradient,
    );
    assert_ne!(
        resolved, declared,
        "Fix: this program is tunable, so a bound taken from the declared width would bound a width nothing launches."
    );
    assert_eq!(
        lane_ceiling(1024),
        174_080,
        "Fix: 1024 wide is 1 block/SM x 170 SMs x 1024 lanes. This is the ceiling the defect produced."
    );
    assert_eq!(
        lane_ceiling(resolved[0]),
        261_120,
        "Fix: the resolved width must reach the device's full cooperative capacity, 1536 resident threads x 170 SMs. Seeing 174,080 here means the tuner resolved 1024 again."
    );

    let mut pinned = DispatchConfig::default();
    pinned.workgroup_override = Some(declared);
    assert_eq!(
        resolve_launch_workgroup_for_mode(
            &program,
            &pinned,
            limits,
            262_144,
            Mode::NaturalGradient,
        ),
        declared,
        "Fix: a pinned width must resolve to itself so the declared and resolved bounds coincide."
    );
    assert_eq!(lane_ceiling(declared[0]), 261_120);
}

#[test]
fn measured_feedback_can_still_select_a_width_cold_start_would_reject() {
    let dir =
        tempfile::tempdir().expect("Fix: measured feedback test needs an isolated tuner cache");
    let path = dir.path().join("residency-feedback.toml");
    let limits = regression_1536_thread_sm_limits();
    let declared = [32, 1, 1];
    let program = tunable_1d_program("out_measured_beats_residency", 65_536, declared);
    let key = NaturalLaunchCacheKey::new(&program, declared, 65_536, limits);
    natural_launch_cache_remove(key);

    assert_eq!(
        natural_gradient_cold_start_workgroup_with_store(
            &program,
            declared,
            65_536,
            limits,
            Some(&path),
        ),
        Some([512, 1, 1]),
        "Fix: with no measurements the residency preference decides."
    );
    natural_launch_cache_remove(key);
    assert!(
        record_launch_measurement_for_mode_with_store(
            &program,
            &DispatchConfig::default(),
            limits,
            65_536,
            [1024, 1, 1],
            1,
            Mode::NaturalGradient,
            Some(&path),
        ),
        "Fix: a real timing for a residency-poor width must still be accepted."
    );
    assert_eq!(
        natural_gradient_cold_start_workgroup_with_store(
            &program,
            declared,
            65_536,
            limits,
            Some(&path),
        ),
        Some([1024, 1, 1]),
        "Fix: residency governs the cold start only. Measured feedback must remain able to choose a width cold start would never propose."
    );
}
