//! Tests for launch realization and measured launch facts.
//!
//! WHY: the class closed here is a driver that chooses a launch shape. Every
//! case asserts that the width a dispatch runs at comes from the artifact, the
//! caller, or the program, and that a measurement changes the fact table and
//! nothing else. Legality is the one width change left: a grid the target
//! refuses is not a slow launch, so widening into the per-axis ceiling is
//! asserted as arithmetic on the ceiling rather than a ranked choice.
//!
//! What these cases do not catch: whether a selector consumes the reported
//! facts well. That is `vyre-megakernel`'s contract, measured there.

use super::*;
use crate::binding::{Binding, BindingRole};
use crate::launch::{effective_launch_workgroup, LaunchPlan};
use crate::launch_fixtures::wide_limits;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// A 1D storage program whose body reads only the global invocation id, so its
/// result does not depend on the block width.
fn width_free_program(output: &'static str, count: u32, declared: [u32; 3]) -> Program {
    Program::wrapped(
        vec![BufferDecl::output(output, 0, DataType::U32).with_count(count)],
        declared,
        vec![Node::store(output, Expr::gid_x(), Expr::u32(7))],
    )
}

/// A program whose lane arithmetic observes the block width.
fn width_semantic_program(count: u32, declared: [u32; 3]) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out_local_ids", 0, DataType::U32).with_count(count)],
        declared,
        vec![
            Node::let_bind("lane", Expr::LocalId { axis: 0 }),
            Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
            Node::let_bind(
                "global",
                Expr::add(
                    Expr::mul(Expr::var("block"), Expr::u32(declared[0])),
                    Expr::var("lane"),
                ),
            ),
            Node::store("out_local_ids", Expr::var("global"), Expr::var("lane")),
        ],
    )
}

fn output_binding(name: &'static str, element_count: u32) -> Binding {
    Binding {
        name: std::sync::Arc::from(name),
        binding: 0,
        buffer_index: 0,
        role: BindingRole::Output,
        element_size: 4,
        preferred_alignment: 128,
        element_count,
        static_byte_len: Some(element_count as usize * 4),
        input_index: None,
        output_index: Some(0),
    }
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

#[test]
fn an_untracked_launch_runs_at_the_width_its_program_declares() {
    let limits = wide_limits("test", 1536);
    for declared_x in [32u32, 64, 256, 1024] {
        let program = width_free_program("out_declared", 4096, [declared_x, 1, 1]);
        assert_eq!(
            resolve_launch_workgroup(&program, &DispatchConfig::default(), limits, 4096),
            [declared_x, 1, 1],
            "Fix: a launch with no recorded geometry runs the width the program declares. \
             A driver that picks another width runs a shape no artifact identity covers."
        );
    }
}

#[test]
fn a_recorded_geometry_outranks_the_declared_width_and_every_override() {
    let limits = wide_limits("test", 1536);
    let program = width_free_program("out_recorded", 4096, [32, 1, 1]);
    let mut pinned = DispatchConfig::default();
    pinned.workgroup_override = Some([64, 1, 1]);

    for config in [DispatchConfig::default(), pinned] {
        assert_eq!(
            resolve_launch_workgroup_for_geometry(
                &program,
                &config,
                limits,
                4096,
                LaunchGeometry::Compiled([128, 1, 1]),
            ),
            [128, 1, 1],
            "Fix: the emitted module declares the recorded shape, so it outranks everything."
        );
    }
}

#[test]
fn a_caller_override_outranks_the_declared_width_and_a_grid_override_keeps_it() {
    let limits = wide_limits("test", 1536);
    let program = width_free_program("out_pinned", 262_144, [64, 1, 1]);

    let mut pinned_workgroup = DispatchConfig::default();
    pinned_workgroup.workgroup_override = Some([256, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup(&program, &pinned_workgroup, limits, 262_144),
        [256, 1, 1],
        "Fix: an explicit workgroup override stays authoritative."
    );

    let mut pinned_grid = DispatchConfig::default();
    pinned_grid.grid_override = Some([1024, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup(&program, &pinned_grid, limits, 262_144),
        [64, 1, 1],
        "Fix: an explicit grid override keeps the declared workgroup, since the caller sized \
         the grid against it."
    );
}

#[test]
fn a_launch_plan_prepares_the_declared_width_and_the_grid_it_implies() {
    let program = width_free_program("out_plan", 4096, [32, 1, 1]);
    let bindings = vec![output_binding("out_plan", 4096)];
    let mut plan = LaunchPlan::new();

    plan.prepare_into(
        &program,
        &bindings,
        &DispatchConfig::default(),
        wide_limits("test", 1536),
    )
    .expect("Fix: a width-free 1D storage launch must prepare without a device");

    assert_eq!(plan.workgroup, [32, 1, 1]);
    assert_eq!(plan.grid, [128, 1, 1]);
    assert_eq!(plan.element_count, 4096);
    assert_eq!(
        effective_launch_workgroup(
            &program,
            &bindings,
            &DispatchConfig::default(),
            wide_limits("test", 1536)
        ),
        [32, 1, 1],
        "Fix: preparation and the effective width must agree, or a pipeline compiles one \
         shape and the dispatch runs another."
    );
}

#[test]
fn a_grid_past_the_ceiling_widens_to_the_smallest_power_of_two_that_fits() {
    let element_count = 16_777_216u32;
    for ceiling in [64u32, 1024, 16_384, 65_536] {
        let limits = ceiling_limits(ceiling);
        let program = width_free_program("out_widen", element_count, [32, 1, 1]);
        let selected = resolve_launch_workgroup(&program, &DispatchConfig::default(), limits, 16);
        assert_eq!(
            selected,
            [32, 1, 1],
            "Fix: a launch whose grid already fits keeps its declared width."
        );

        let selected =
            resolve_launch_workgroup(&program, &DispatchConfig::default(), limits, element_count);
        let needed = element_count.div_ceil(ceiling);
        let expected = needed
            .checked_next_power_of_two()
            .filter(|width| *width <= 1024)
            .map_or([32, 1, 1], |width| [width.max(32), 1, 1]);
        assert_eq!(
            selected, expected,
            "Fix: widening is arithmetic on the ceiling: {element_count} lanes under a \
             {ceiling}-block ceiling need blocks of {needed}."
        );
        assert!(
            expected[0] != 32 || needed > 1024,
            "Fix: the sweep must exercise a real widening for ceiling {ceiling}."
        );
    }
}

#[test]
fn a_width_semantic_program_is_never_widened() {
    let element_count = 16_777_216u32;
    let program = width_semantic_program(element_count, [32, 1, 1]);
    assert_eq!(
        resolve_launch_workgroup(
            &program,
            &DispatchConfig::default(),
            ceiling_limits(1024),
            element_count
        ),
        [32, 1, 1],
        "Fix: a program whose lane arithmetic observes the block width keeps its shape and is \
         refused downstream by name, rather than silently computing something else."
    );
}

#[test]
fn a_measurement_enters_the_fact_table_and_never_changes_the_launch() {
    let limits = wide_limits("measured", 1536);
    let program = width_free_program("out_measured", 8192, [32, 1, 1]);
    let config = DispatchConfig::default();
    forget_launch_measurements(&program, limits, 8192);

    assert!(
        record_launch_measurement(&program, &config, limits, 8192, [256, 1, 1], 100),
        "Fix: a real timing for an admissible width is a fact the driver reports."
    );
    assert!(record_launch_measurement(
        &program,
        &config,
        limits,
        8192,
        [512, 1, 1],
        40
    ));
    assert!(
        record_launch_measurement(&program, &config, limits, 8192, [256, 1, 1], 60),
        "Fix: a second timing for a known width is still a fact."
    );

    let facts = launch_width_measurements(&program, limits, 8192);
    assert_eq!(
        facts.get(&[256, 1, 1]),
        Some(&60),
        "Fix: the table reports the fastest observation for each width."
    );
    assert_eq!(facts.get(&[512, 1, 1]), Some(&40));
    assert_eq!(
        resolve_launch_workgroup(&program, &config, limits, 8192),
        [32, 1, 1],
        "Fix: a measured figure is a fact for the compiler to rank, never a width the driver \
         switches to. A launch that changes shape because of a timing runs bytes no artifact \
         authorized."
    );
    forget_launch_measurements(&program, limits, 8192);
}

#[test]
fn a_timing_no_selector_could_rank_is_refused() {
    let limits = wide_limits("refused", 1536);
    let program = width_free_program("out_refused", 4096, [32, 1, 1]);
    let config = DispatchConfig::default();
    forget_launch_measurements(&program, limits, 4096);

    assert!(
        !record_launch_measurement(&program, &config, limits, 4096, [64, 1, 1], 0),
        "Fix: a zero timing measures nothing."
    );
    assert!(
        !record_launch_measurement(&program, &config, limits, 4096, [2048, 1, 1], 10),
        "Fix: a width past the target's block ceiling is not a launch that ran."
    );
    assert!(
        !record_launch_measurement(&program, &config, limits, 4096, [64, 2, 1], 10),
        "Fix: the table holds 1D widths, since only those are comparable."
    );

    let mut pinned = DispatchConfig::default();
    pinned.workgroup_override = Some([64, 1, 1]);
    assert!(
        !record_launch_measurement(&program, &pinned, limits, 4096, [64, 1, 1], 10),
        "Fix: a pinned launch measures the caller's shape, not one a selector may compare."
    );

    let semantic = width_semantic_program(4096, [32, 1, 1]);
    assert!(
        !record_launch_measurement(&semantic, &config, limits, 4096, [64, 1, 1], 10),
        "Fix: timings at different widths of a width-semantic program describe different \
         computations."
    );

    assert!(
        launch_width_measurements(&program, limits, 4096).is_empty(),
        "Fix: a refused timing must leave no fact behind."
    );
    forget_launch_measurements(&program, limits, 4096);
}

#[test]
fn facts_are_keyed_by_program_launch_size_and_target() {
    let limits = wide_limits("keyed", 1536);
    let other_limits = wide_limits("keyed", 2048);
    let program = width_free_program("out_keyed", 4096, [32, 1, 1]);
    let other = width_free_program("out_keyed_other", 4096, [32, 1, 1]);
    let config = DispatchConfig::default();
    for (target, count) in [(limits, 4096u32), (other_limits, 4096), (limits, 2048)] {
        forget_launch_measurements(&program, target, count);
    }
    forget_launch_measurements(&other, limits, 4096);

    assert!(record_launch_measurement(
        &program,
        &config,
        limits,
        4096,
        [64, 1, 1],
        25
    ));

    assert!(
        launch_width_measurements(&other, limits, 4096).is_empty(),
        "Fix: another program's timings are not this program's facts."
    );
    assert!(
        launch_width_measurements(&program, other_limits, 4096).is_empty(),
        "Fix: a timing taken under other target limits is not a fact about this target."
    );
    assert!(
        launch_width_measurements(&program, limits, 2048).is_empty(),
        "Fix: a timing at another launch size is not a fact about this one."
    );
    assert_eq!(
        launch_width_measurements(&program, limits, 4096).get(&[64, 1, 1]),
        Some(&25)
    );
    forget_launch_measurements(&program, limits, 4096);
}
