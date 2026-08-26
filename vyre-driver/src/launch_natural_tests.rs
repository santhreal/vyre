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
        vec![Node::store("out", Expr::logical_index(0), Expr::u32(1))],
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

/// A 1D storage program whose lane space is `count` and whose body reads only
/// the global invocation id, so its result does not depend on the block width.
fn width_agnostic_program(count: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(count)],
        [32, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::u32(7))],
    )
}

fn ceiling_limits(max_blocks_per_axis: u32) -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend: "test",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [max_blocks_per_axis; 3],
        max_threads_per_sm: 0,
    }
}

/// WHY: a block width nobody chose must not decide whether a launch runs. A
/// program that declares one output element per lane asks for one workgroup per
/// block of lanes, so a large launch reaches a graphics-derived per-axis ceiling
/// while the same lane space in wider blocks launches. Refusing it prices
/// legality into a ranking decision and takes every large launch off the
/// backend.
///
/// The ceilings swept here are the candidate widths themselves, read from
/// `WORKGROUP_CANDIDATES`, so adding a candidate widens this case rather than
/// leaving it behind. Where no candidate can fit, the declared width stays and
/// the dispatch is refused downstream by name.
#[test]
fn a_launch_past_the_grid_ceiling_widens_in_every_tuner_mode() {
    let element_count = 65_536;
    let program = width_agnostic_program(element_count);
    let widest = WORKGROUP_CANDIDATES
        .iter()
        .copied()
        .max()
        .expect("Fix: the tuner must publish at least one candidate width.");
    for mode in [Mode::NaturalGradient, Mode::On, Mode::OffUseDefault] {
        for &ceiling in WORKGROUP_CANDIDATES {
            let limits = ceiling_limits(ceiling);
            let selected = resolve_launch_workgroup_for_geometry(
                &program,
                &DispatchConfig::default(),
                limits,
                element_count,
                mode,
                LaunchGeometry::Untracked,
            );
            let reachable = element_count.div_ceil(widest) <= ceiling;
            assert_eq!(
                candidate_grid_fits(selected[0], element_count, limits),
                reachable,
                "Fix: {mode:?} resolved {selected:?} for {element_count} lanes under a {ceiling}-workgroup ceiling. A width the ceiling admits exists whenever {widest} lanes per block fit."
            );
            assert!(
                selected[0] <= widest && selected[1] == 1 && selected[2] == 1,
                "Fix: widening must stay inside the candidate widths, got {selected:?}."
            );
        }
    }
}

/// WHY: widening is only sound for a program whose result does not depend on the
/// block width. One that reads its workgroup or local id observes the width
/// directly, so a wider block would change what every lane computes. Such a
/// launch keeps its declared width and is refused by the grid check, which names
/// the ceiling and the reshape.
#[test]
fn a_launch_whose_program_reads_its_block_keeps_the_declared_width() {
    let element_count = 65_536;
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(element_count)],
        [32, 1, 1],
        vec![Node::store(
            "out",
            Expr::add(
                Expr::mul(Expr::WorkgroupId { axis: 0 }, Expr::u32(32)),
                Expr::LocalId { axis: 0 },
            ),
            Expr::u32(7),
        )],
    );
    let limits = ceiling_limits(64);
    assert_eq!(
        resolve_launch_workgroup_for_geometry(
            &program,
            &DispatchConfig::default(),
            limits,
            element_count,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        ),
        [32, 1, 1],
        "Fix: a program that reads its block id must keep the width it declared, whatever the grid ceiling costs it."
    );
}

/// WHY: widening reshapes a launch the caller left free. A caller who pinned the
/// block asked for that block, and one who pinned the grid computed it against
/// the declared block, so changing either behind them would launch a shape
/// nobody asked for.
#[test]
fn a_pinned_block_or_grid_is_never_widened() {
    let element_count = 65_536;
    let program = width_agnostic_program(element_count);
    let limits = ceiling_limits(64);
    let mut pinned_block = DispatchConfig::default();
    pinned_block.workgroup_override = Some([32, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup_for_geometry(
            &program,
            &pinned_block,
            limits,
            element_count,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        ),
        [32, 1, 1],
        "Fix: a caller-pinned block is authoritative."
    );
    let mut pinned_grid = DispatchConfig::default();
    pinned_grid.grid_override = Some([1, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup_for_geometry(
            &program,
            &pinned_grid,
            limits,
            element_count,
            Mode::NaturalGradient,
            LaunchGeometry::Untracked,
        ),
        program.workgroup_size(),
        "Fix: a caller-pinned grid was computed against the declared block."
    );
}

/// WHY: the tuner ranks by estimated latency, and a block whose grid the target
/// refuses has no latency to rank. Leaving it in the sample set lets a policy
/// step select a launch that cannot run, which is how legality gets priced
/// instead of decided.
#[test]
fn the_tuner_never_ranks_a_block_whose_grid_the_target_refuses() {
    let element_count = 65_536;
    for &ceiling in WORKGROUP_CANDIDATES {
        let limits = ceiling_limits(ceiling);
        if let Some(selected) =
            select_natural_launch_workgroup([32, 1, 1], element_count, limits, None)
        {
            assert!(
                candidate_grid_fits(selected[0], element_count, limits),
                "Fix: the tuner ranked {selected:?} for {element_count} lanes under a {ceiling}-workgroup ceiling, a grid the target refuses."
            );
        }
    }
}
