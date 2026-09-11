//! Omitted launch geometry never resolves to one workgroup.
//!
//! WHY: the retired operation-facing dispatch API let a caller omit geometry and
//! took a single workgroup as the answer, so a program guarded over a million
//! elements ran 256 lanes and reported the untouched tail as a result. Geometry is
//! now derived below admission from the program's own guard, and the only case
//! with no derivable answer is rejected instead of defaulted.
//!
//! Does NOT catch a wrong derived extent: `logical_span_contracts` owns which span
//! a guard admits, and this file owns that the span reaches the launch.

use vyre_driver::validation::LaunchGeometryLimits;
use vyre_driver::{BindingPlan, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Limits wide enough that nothing here fails on a backend ceiling.
const LIMITS: LaunchGeometryLimits = LaunchGeometryLimits {
    backend: "omitted-geometry-test",
    max_threads_per_block: 1024,
    max_block_dim: [1024, 1024, 64],
    max_grid_dim: [u32::MAX, 65_535, 65_535],
    max_threads_per_sm: 2048,
};

/// A program that stores one element per guarded axis-0 lane.
fn guarded_store(elements: u32, workgroup: [u32; 3]) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(elements)],
        workgroup,
        vec![Node::if_then(
            Expr::lt(Expr::logical_index(0), Expr::u32(elements)),
            vec![Node::store("out", Expr::logical_index(0), Expr::u32(1))],
        )],
    )
}

fn plan_for(program: &Program, config: &DispatchConfig) -> LaunchPlan {
    let bindings = BindingPlan::build(program)
        .expect("Fix: a single-output program must produce a binding plan.")
        .bindings;
    LaunchPlan::from_bindings(program, &bindings, config, LIMITS)
        .expect("Fix: an omitted grid over a 1-D workgroup must derive a launch.")
}

/// Total lanes a prepared plan launches.
fn lanes(plan: &LaunchPlan) -> u64 {
    plan.grid
        .iter()
        .chain(plan.workgroup.iter())
        .fold(1u64, |product, &extent| product * u64::from(extent))
}

/// An omitted grid covers every guarded element, at every declared width.
///
/// The widths and element counts are generated rather than listed so a width
/// that divides the domain evenly, one that leaves a partial group, and one
/// wider than the whole domain are all exercised against the same derivation.
/// The launch is judged on the workgroup the driver resolved, not the declared
/// one: the tuner may widen a 1-D workgroup, and a wider group with a smaller
/// grid covers the same domain.
#[test]
fn an_omitted_grid_covers_every_guarded_element() {
    for width in [1u32, 2, 32, 64, 256, 1024] {
        for elements in [
            1u32,
            width.saturating_sub(1).max(1),
            width,
            width + 1,
            100_000,
        ] {
            let program = guarded_store(elements, [width, 1, 1]);
            let plan = plan_for(&program, &DispatchConfig::default());

            assert!(
                lanes(&plan) >= u64::from(elements),
                "Fix: a declared {width}-wide workgroup with no grid_override prepared grid {:?} over workgroup {:?}, which covers {} of {elements} guarded elements. The uncovered tail reads back as its initial value and passes a parity check that never inspects it.",
                plan.grid,
                plan.workgroup,
                lanes(&plan)
            );
            assert_eq!(
                plan.grid,
                [elements.div_ceil(plan.workgroup[0]), 1, 1],
                "Fix: the derived launch must be exactly the groups the resolved {:?} workgroup needs for {elements} elements, not a padded or truncated shape.",
                plan.workgroup
            );
        }
    }
}

/// A domain past one workgroup never resolves to a single workgroup of lanes.
///
/// This is the retired default stated directly: the seed `LaunchPlan::new()`
/// carries is one group of one lane, so a preparation path that returned early,
/// skipped the derivation, or left the field alone reads as a legal launch that
/// runs one lane over the whole domain.
#[test]
fn a_domain_past_one_workgroup_is_never_prepared_as_one_workgroup() {
    let seed = LaunchPlan::new();
    assert_eq!(
        (seed.grid, seed.workgroup),
        ([1, 1, 1], [1, 1, 1]),
        "Fix: this contract watches the seed an unprepared plan holds; if the seed changes, restate what preparation has to overwrite."
    );

    for (elements, width) in [(257u32, 256u32), (1_000_000, 256), (2, 1), (65, 64)] {
        let program = guarded_store(elements, [width, 1, 1]);
        let plan = plan_for(&program, &DispatchConfig::default());
        assert!(
            lanes(&plan) > 1,
            "Fix: {elements} guarded elements over a declared {width}-wide workgroup prepared the unprepared seed grid {:?} and workgroup {:?}, so the derivation never ran.",
            plan.grid,
            plan.workgroup
        );
        assert!(
            lanes(&plan) >= u64::from(elements),
            "Fix: {elements} elements need more lanes than grid {:?} over workgroup {:?} launches.",
            plan.grid,
            plan.workgroup
        );
    }
}

/// A workgroup with no unambiguous default grid is rejected, not defaulted.
///
/// A 2-D or 3-D thread tile has no single mapping from a logical element count,
/// so the omission has no answer. Returning one silently picks a mapping the
/// caller never chose; the previous API picked `[1, 1, 1]`.
#[test]
fn a_multi_dimensional_workgroup_rejects_an_omitted_grid() {
    for workgroup in [
        [16u32, 16, 1],
        [8, 8, 4],
        [1, 256, 1],
        [1, 1, 64],
        [64, 1, 4],
    ] {
        let program = guarded_store(4096, workgroup);
        let bindings = BindingPlan::build(&program)
            .expect("Fix: a single-output program must produce a binding plan.")
            .bindings;
        let error = LaunchPlan::from_bindings(&program, &bindings, &DispatchConfig::default(), LIMITS)
            .expect_err(
                "Fix: a multi-dimensional workgroup has no unambiguous default grid, so omitting one must fail rather than launch a shape nobody chose.",
            );
        let message = error.to_string();
        assert!(
            message.contains("Fix:"),
            "Fix: the rejection must state the corrective action; got: {message}"
        );
        assert!(
            message.contains("grid_override"),
            "Fix: the rejection must name the field the caller has to supply; got: {message}"
        );
        assert!(
            message.contains(&format!("{workgroup:?}")),
            "Fix: the rejection must report the workgroup that has no default; got: {message}"
        );
    }
}

/// An unguarded program launches over its declared resources, not one group.
///
/// A program that proves no bound on its logical index has no guard to narrow to,
/// which is the case a one-group default was most defensible for and is still
/// wrong: the declared output is the domain the kernel writes.
#[test]
fn an_unguarded_program_launches_over_its_declared_resources() {
    let elements = 4096u32;
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(elements)],
        [256, 1, 1],
        vec![Node::store("out", Expr::logical_index(0), Expr::u32(1))],
    );
    let plan = plan_for(&program, &DispatchConfig::default());
    assert_eq!(
        plan.grid,
        [elements.div_ceil(plan.workgroup[0]), 1, 1],
        "Fix: an unguarded store over {elements} declared elements must launch over all of them; the resolved workgroup is {:?}.",
        plan.workgroup
    );
    assert_eq!(lanes(&plan), u64::from(elements));
    assert_eq!(plan.element_count, elements);
}
