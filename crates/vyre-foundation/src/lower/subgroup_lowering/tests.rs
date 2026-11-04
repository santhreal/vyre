//! Lowering contracts for workgroup reductions onto subgroup intrinsics.

use super::*;
use crate::ir::{BufferDecl, DataType, Expr, Node, Program};
use crate::visit::try_for_each_expr;
use core::ops::ControlFlow;

fn caps_with_subgroup(size: u32) -> AdapterCaps {
    AdapterCaps {
        supports_subgroup_ops: true,
        subgroup_size: size,
        ..AdapterCaps::default()
    }
}

#[test]
fn does_not_replace_full_standalone_workgroup_sum_region() {
    let program = Program::wrapped(
        vec![
            BufferDecl::workgroup("scratch", 4, DataType::F32),
            BufferDecl::output("out", 0, DataType::F32).with_count(1),
        ],
        [4, 1, 1],
        vec![Node::Region {
            generator: "vyre-libs::reduce::workgroup_sum_f32".into(),
            source_region: None,
            body: Arc::new(vec![
                Node::let_bind("local", Expr::LocalId { axis: 0 }),
                Node::store("scratch", Expr::var("local"), Expr::f32(1.0)),
                Node::barrier(),
                Node::store("out", Expr::u32(0), Expr::load("scratch", Expr::u32(0))),
            ]),
        }],
    );

    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));
    let [Node::Region { body, .. }] = lowered.entry() else {
        panic!("Fix: standalone workgroup sum must remain wrapped in one region.");
    };

    assert!(
        has_standalone_reduction_preamble(body),
        "Fix: subgroup lowering must not drop the standalone local-id preamble."
    );
    assert!(
        body.iter()
            .any(|node| matches!(node, Node::Store { buffer, .. } if buffer.as_str() == "out")),
        "Fix: subgroup lowering must not drop the standalone final output store."
    );
}

#[test]
fn u32_two_level_workgroup_sum_uses_u32_neutral() {
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 64, DataType::U32)],
        [64, 1, 1],
        vec![Node::Region {
            generator: "vyre-libs::reduce::workgroup_sum_u32".into(),
            source_region: None,
            body: Arc::new(vec![
                Node::store(
                    "scratch",
                    Expr::var("local"),
                    Expr::load("scratch", Expr::var("local")),
                ),
                Node::barrier(),
            ]),
        }],
    );

    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));
    let [Node::Region { body, .. }] = lowered.entry() else {
        panic!("Fix: u32 workgroup sum must remain wrapped in one region.");
    };

    assert!(
        nodes_contain_select_false_u32_zero(body),
        "Fix: u32 two-level subgroup lowering must use a u32 zero neutral."
    );
    assert!(
        !nodes_contain_select_false_f32_zero(body),
        "Fix: u32 two-level subgroup lowering must not emit a f32 zero neutral into a u32 select."
    );
}

fn nodes_contain_select_false_u32_zero(nodes: &[Node]) -> bool {
    nodes_contain_select_false(nodes, |expr| matches!(expr, Expr::LitU32(0)))
}

fn nodes_contain_select_false_f32_zero(nodes: &[Node]) -> bool {
    nodes_contain_select_false(
        nodes,
        |expr| matches!(expr, Expr::LitF32(value) if *value == 0.0),
    )
}

/// True when some `Select` under `nodes` has a `false_val` matching
/// `predicate`.
///
/// A pair of hand-written descents used to stand here, one over `Node` and
/// one over `Expr`, together 90 lines and both ending in `_ => false`. As a
/// TEST helper that is worse than in production code: the assertion built on
/// it is `!contains(...)`, so a position the walk failed to reach reads as
/// proof that the emitted neutral is absent, and it would have gone on
/// passing after the lowering moved a select into a position neither list
/// named.
fn nodes_contain_select_false(nodes: &[Node], predicate: fn(&Expr) -> bool) -> bool {
    any_expr_matching(
        nodes,
        &|expr| matches!(expr, Expr::Select { false_val, .. } if predicate(false_val)),
    )
}

/// True when some expression anywhere under `nodes` satisfies `predicate`.
///
/// `try_for_each_expr` owns which positions exist: every operand of every
/// node and every sub-expression of every operand. The predicate is shallow,
/// so a variant that gains an operand is reached without editing anything
/// here.
fn any_expr_matching(nodes: &[Node], predicate: &dyn Fn(&Expr) -> bool) -> bool {
    try_for_each_expr(nodes, |expr| {
        if predicate(expr) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

fn workgroup_sum_region(scratch: &str, scope: ReductionScope) -> Node {
    let body = if scope == ReductionScope::FirstWorkgroup {
        vec![
            Node::if_then(
                Expr::and(
                    Expr::is_first_workgroup(),
                    Expr::lt(Expr::var("local"), Expr::u32(2)),
                ),
                vec![Node::Store {
                    buffer: scratch.into(),
                    index: Expr::var("local"),
                    value: Expr::add(
                        Expr::load(scratch, Expr::var("local")),
                        Expr::load(scratch, Expr::add(Expr::var("local"), Expr::u32(2))),
                    ),
                }],
            ),
            Node::barrier(),
            Node::if_then(
                Expr::and(
                    Expr::is_first_workgroup(),
                    Expr::lt(Expr::var("local"), Expr::u32(1)),
                ),
                vec![Node::Store {
                    buffer: scratch.into(),
                    index: Expr::var("local"),
                    value: Expr::add(
                        Expr::load(scratch, Expr::var("local")),
                        Expr::load(scratch, Expr::add(Expr::var("local"), Expr::u32(1))),
                    ),
                }],
            ),
            Node::barrier(),
        ]
    } else {
        vec![
            Node::if_then(
                Expr::lt(Expr::var("local"), Expr::u32(2)),
                vec![Node::Store {
                    buffer: scratch.into(),
                    index: Expr::var("local"),
                    value: Expr::add(
                        Expr::load(scratch, Expr::var("local")),
                        Expr::load(scratch, Expr::add(Expr::var("local"), Expr::u32(2))),
                    ),
                }],
            ),
            Node::barrier(),
            Node::if_then(
                Expr::lt(Expr::var("local"), Expr::u32(1)),
                vec![Node::Store {
                    buffer: scratch.into(),
                    index: Expr::var("local"),
                    value: Expr::add(
                        Expr::load(scratch, Expr::var("local")),
                        Expr::load(scratch, Expr::add(Expr::var("local"), Expr::u32(1))),
                    ),
                }],
            ),
            Node::barrier(),
        ]
    };
    Node::Region {
        generator: "vyre-libs::reduce::workgroup_sum_f32".into(),
        source_region: None,
        body: Arc::new(body),
    }
}

#[test]
fn no_change_when_subgroup_not_supported() {
    let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let caps = AdapterCaps::default();
    let lowered = lower_subgroup_reductions(Clone::clone(&program), &caps);
    assert_eq!(lowered, program);
}

#[test]
fn no_change_when_workgroup_larger_than_subgroup() {
    let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 2048, DataType::F32)],
        [2048, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(Clone::clone(&program), &caps);
    assert_eq!(lowered, program);
}

/// Splits a lowered reduction body into the lane binding it opens with and
/// the reduction nodes that follow.
///
/// The replacement binds its own lane index instead of reading one the
/// enclosing scope declared. A body that reads a caller-declared name is
/// broken by any transform that renames the enclosing scope's bindings, and
/// fusion renames every arm binding it copies.
fn lane_bound_body(body: &[Node]) -> &[Node] {
    let Node::Let { name, value, .. } = &body[0] else {
        panic!(
            "a lowered reduction must open by binding its own lane, got {:?}",
            body[0]
        );
    };
    assert_eq!(name.as_str(), SUBGROUP_LANE);
    assert!(
        matches!(value, Expr::LogicalWithinTileId { axis: 0 }),
        "the lane binding must come from the tile index, got {value:?}"
    );
    &body[1..]
}

/// Appends the name of every `Let` under `nodes`, at any depth.
fn collect_let_names(nodes: &[Node], out: &mut Vec<String>) {
    for node in nodes {
        if let Node::Let { name, .. } = node {
            out.push(name.as_str().to_string());
        }
        for child in crate::visit::child_bodies(node) {
            collect_let_names(child, out);
        }
    }
}

#[test]
fn lowered_reduction_body_is_closed_over_the_lanes_it_reads() {
    // The emitted body must bind every variable it reads. A body that reads
    // a name the enclosing scope declared holds only while that name
    // survives, and fusion alpha-renames every arm binding it copies, which
    // leaves the read dangling and refuses the whole program at physical
    // lowering with "variable is referenced before binding".
    for workgroup in [4u32, 256] {
        let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
        let program = Program::wrapped(
            vec![BufferDecl::workgroup("scratch", workgroup, DataType::F32)],
            [workgroup, 1, 1],
            vec![region],
        );
        let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));
        let Node::Region { body, .. } = &lowered.entry()[0] else {
            panic!("expected Region");
        };
        let mut bound: Vec<String> = Vec::new();
        collect_let_names(body, &mut bound);
        let reads_a_free_variable = any_expr_matching(
            body,
            &|expr| matches!(expr, Expr::Var(v) if !bound.iter().any(|name| name == v.as_str())),
        );
        assert!(
            !reads_a_free_variable,
            "workgroup {workgroup}: the lowered body reads a variable it does not bind: {body:?}"
        );
    }
}

#[test]
fn lowers_every_workgroup_sum_to_subgroup_add() {
    let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(program, &caps);

    let entry = lowered.entry();
    assert_eq!(entry.len(), 1);
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    // Should be: let lane = tile index; store(scratch, lane, subgroup_add(load(scratch, lane))); barrier
    let body = lane_bound_body(body);
    assert_eq!(body.len(), 2);
    assert!(
        matches!(&body[0], Node::Store { buffer, index, value } if
            buffer.as_str() == "scratch" &&
            matches!(index, Expr::Var(v) if v.as_str() == SUBGROUP_LANE) &&
            matches!(value, Expr::SubgroupReduce { .. })
        ),
        "expected subgroup_add store, got {:?}",
        body[0]
    );
    assert!(matches!(&body[1], Node::Barrier { .. }));
}

#[test]
fn lowers_every_workgroup_max_to_subgroup_reduce_max() {
    // workgroup_max_f32 must now lower to the native subgroup Max reduction
    // instead of being kept as the slow shared-memory tree.
    let Node::Region { body, .. } = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
    else {
        panic!("workgroup_sum_region must build a Region");
    };
    let region = Node::Region {
        generator: "vyre-libs::reduce::workgroup_max_f32".into(),
        source_region: None,
        body,
    };
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));

    let entry = lowered.entry();
    assert_eq!(entry.len(), 1);
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    assert_eq!(
        body.len(),
        2,
        "single-subgroup max lowers to store+barrier, got {body:?}"
    );
    let Node::Store { buffer, value, .. } = &body[0] else {
        panic!("expected a store, got {:?}", body[0]);
    };
    assert_eq!(buffer.as_str(), "scratch");
    assert!(
        matches!(
            value,
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Max,
                ..
            }
        ),
        "workgroup_max must lower to subgroup_reduce(Max), got {value:?}"
    );
    assert!(matches!(&body[1], Node::Barrier { .. }));
}

#[test]
fn lowers_workgroup_max_u32_to_subgroup_reduce_max() {
    // The u32 twin: workgroup_max_u32 must ALSO lower to subgroup_reduce(Max).
    // The lowering recognizes the `workgroup_max_` prefix with a u32 value
    // type, so the new primitive gets the fast subgroup-reduction path for free.
    let Node::Region { body, .. } = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
    else {
        panic!("workgroup_sum_region must build a Region");
    };
    let region = Node::Region {
        generator: "vyre-libs::reduce::workgroup_max_u32".into(),
        source_region: None,
        body,
    };
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::U32)],
        [4, 1, 1],
        vec![region],
    );
    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));

    let entry = lowered.entry();
    assert_eq!(entry.len(), 1);
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    let Node::Store { value, .. } = &body[0] else {
        panic!("expected a store, got {:?}", body[0]);
    };
    assert!(
        matches!(
            value,
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Max,
                ..
            }
        ),
        "workgroup_max_u32 must lower to subgroup_reduce(Max), got {value:?}"
    );
}

#[test]
fn lowers_workgroup_min_f32_to_subgroup_reduce_min() {
    // workgroup_min_f32 must lower to subgroup_reduce(Min), the subgroup-reduce
    // fast path. A missing Min prefix arm would leave the slow shared-memory
    // tree in place (correct, but pessimal. Law 7).
    let Node::Region { body, .. } = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
    else {
        panic!("workgroup_sum_region must build a Region");
    };
    let region = Node::Region {
        generator: "vyre-libs::reduce::workgroup_min_f32".into(),
        source_region: None,
        body,
    };
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));

    let Node::Region { body, .. } = &lowered.entry()[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    let Node::Store { value, .. } = &body[0] else {
        panic!("expected a store, got {:?}", body[0]);
    };
    assert!(
        matches!(
            value,
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Min,
                ..
            }
        ),
        "workgroup_min_f32 must lower to subgroup_reduce(Min), got {value:?}"
    );
}

#[test]
fn lowers_workgroup_min_u32_to_subgroup_reduce_min() {
    // The u32 twin of the Min lowering, exercises the unsigned value-type
    // branch of workgroup_min_value_type.
    let Node::Region { body, .. } = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
    else {
        panic!("workgroup_sum_region must build a Region");
    };
    let region = Node::Region {
        generator: "vyre-libs::reduce::workgroup_min_u32".into(),
        source_region: None,
        body,
    };
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::U32)],
        [4, 1, 1],
        vec![region],
    );
    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));

    let Node::Region { body, .. } = &lowered.entry()[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    let Node::Store { value, .. } = &body[0] else {
        panic!("expected a store, got {:?}", body[0]);
    };
    assert!(
        matches!(
            value,
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Min,
                ..
            }
        ),
        "workgroup_min_u32 must lower to subgroup_reduce(Min), got {value:?}"
    );
}

#[test]
fn lowers_two_level_workgroup_max_uses_neg_inf_neutral() {
    // The two-level max reduction must fill out-of-range lanes with the
    // max identity (-inf), not 0, a 0 fill would clobber all-negative
    // inputs. This is the op-aware neutral.
    let Node::Region { body, .. } = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
    else {
        panic!("workgroup_sum_region must build a Region");
    };
    let region = Node::Region {
        generator: "vyre-libs::reduce::workgroup_max_f32".into(),
        source_region: None,
        body,
    };
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 256, DataType::F32)],
        [256, 1, 1],
        vec![region],
    );
    let lowered = lower_subgroup_reductions(program, &caps_with_subgroup(32));
    let entry = lowered.entry();
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    assert!(
        nodes_contain_subgroup_reduce_max(&body[0..1])
            && nodes_contain_subgroup_reduce_max(&body[4..5]),
        "both levels of the 256-lane max reduction must use subgroup_reduce(Max): {body:?}"
    );
    assert!(
        nodes_contain_neg_inf_select_neutral(body),
        "two-level max must use a -inf select neutral for out-of-range lanes: {body:?}"
    );
}

/// True when some expression under `nodes` is a Max subgroup reduce.
///
/// The predicate is shallow; `any_expr_matching` owns which positions
/// exist, so a variant that gains an operand is reached without editing
/// this.
fn nodes_contain_subgroup_reduce_max(nodes: &[Node]) -> bool {
    any_expr_matching(nodes, &|expr| {
        matches!(
            expr,
            Expr::SubgroupReduce {
                op: SubgroupReduceOp::Max,
                ..
            }
        )
    })
}

/// True when some select under `nodes` uses -inf as its false arm.
fn nodes_contain_neg_inf_select_neutral(nodes: &[Node]) -> bool {
    any_expr_matching(nodes, &|expr| {
        matches!(expr, Expr::Select { false_val, .. }
            if matches!(false_val.as_ref(), Expr::LitF32(v) if *v == f32::NEG_INFINITY))
    })
}

#[test]
fn lowers_two_level_workgroup_sum_for_large_workgroups() {
    let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 256, DataType::F32)],
        [256, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(program, &caps);

    let entry = lowered.entry();
    assert_eq!(entry.len(), 1);
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    let body = lane_bound_body(body);
    assert_eq!(
        body.len(),
        7,
        "Fix: two-level subgroup lowering should emit first-level subgroup work, a fence before the head store overwrites the tile that level just read, the head store, a barrier, full-subgroup second-level subgroup work, and a final barrier."
    );
    assert!(
        nodes_contain_subgroup_add(&body[0..1]) && nodes_contain_subgroup_add(&body[4..5]),
        "Fix: both levels of the 256-lane reduction must use subgroup_add instead of the shared-memory tree: {body:?}"
    );
    assert!(
        matches!(&body[1], Node::Barrier { .. }),
        "Fix: the first level must fence its tile-wide read against the head store that writes back into that tile: {body:?}"
    );
    assert!(matches!(&body[3], Node::Barrier { .. }));
    assert!(matches!(&body[6], Node::Barrier { .. }));
}

#[test]
fn lowers_first_workgroup_sum_with_guard() {
    let region = workgroup_sum_region("scratch", ReductionScope::FirstWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(program, &caps);

    let entry = lowered.entry();
    assert_eq!(entry.len(), 1);
    let Node::Region { body, .. } = &entry[0] else {
        panic!("expected Region");
    };
    // Should be: if (workgroup_id.x == 0) { store(...) } barrier
    let body = lane_bound_body(body);
    assert_eq!(body.len(), 2);
    let Node::If { cond, then, .. } = &body[0] else {
        panic!("expected If guard");
    };
    assert!(
        matches!(cond, Expr::BinOp { op: crate::ir::BinOp::Eq, left, right } if
            matches!(left.as_ref(), Expr::WorkgroupId { axis: 0 }) &&
            matches!(right.as_ref(), Expr::LitU32(0))
        )
    );
    assert_eq!(then.len(), 1);
    assert!(matches!(&then[0], Node::Store { buffer, .. } if buffer.as_str() == "scratch"));
    assert!(matches!(&body[1], Node::Barrier { .. }));
}

#[test]
fn non_reduction_regions_are_unchanged() {
    let region = Node::Region {
        generator: "vyre-libs::math::dot".into(),
        source_region: None,
        body: Arc::new(vec![Node::store("out", Expr::u32(0), Expr::u32(1))]),
    };
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(Clone::clone(&program), &caps);
    assert_eq!(lowered, program);
}

#[test]
fn stats_flag_subgroup_ops_after_lowering() {
    let region = workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup);
    let program = Program::wrapped(
        vec![BufferDecl::workgroup("scratch", 4, DataType::F32)],
        [4, 1, 1],
        vec![region],
    );
    let caps = caps_with_subgroup(32);
    let lowered = lower_subgroup_reductions(program, &caps);
    assert!(
        lowered.stats().subgroup_ops(),
        "lowering must set the subgroup_ops capability bit"
    );
}

/// True when some expression under `nodes` is a subgroup reduce.
fn nodes_contain_subgroup_add(nodes: &[Node]) -> bool {
    any_expr_matching(nodes, &|expr| matches!(expr, Expr::SubgroupReduce { .. }))
}
