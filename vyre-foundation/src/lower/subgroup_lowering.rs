//! Subgroup-first lowering pass (Phase 2.3).
//!
//! Converts workgroup-tree reductions over shared memory into
//! `subgroup_add` / `subgroup_shuffle` operations when the backend
//! reports native subgroup support and the workgroup shape fits the
//! subgroup size.

use crate::ir::{Expr, Node, Program, SubgroupReduceOp};
use crate::optimizer::ctx::AdapterCaps;
use crate::optimizer::rewrite::rewrite_node_slices;
use crate::visit::map_bodies_cow;
use std::borrow::Cow;
use std::sync::Arc;

/// Canonical generator prefixes emitted by `vyre-libs::reduce::workgroup_tree`.
const WORKGROUP_SUM_PREFIX: &str = "vyre-libs::reduce::workgroup_sum_";
const WORKGROUP_MAX_PREFIX: &str = "vyre-libs::reduce::workgroup_max_";
const WORKGROUP_MIN_PREFIX: &str = "vyre-libs::reduce::workgroup_min_";

/// Name the replacement body binds its lane index to.
///
/// The body emitted here indexes `scratch` by the invocation's index inside
/// the workgroup. Reading that index from a caller-declared name, which the
/// shipped reduction builders all spell `local`, makes the replacement depend
/// on a binding it does not own. Fusion alpha-renames an arm's locals to
/// `__vyre_fuse_a{arm}_local`, so the emitted body referenced a name that was
/// no longer in scope and physical lowering refused the program as a variable
/// referenced before binding. Binding the lane here puts it out of reach of
/// any renaming transform and of any caller's naming choice.
const SUBGROUP_LANE: &str = "vyre_subgroup_lane";

/// The lane index the replacement body reads.
fn lane() -> Expr {
    Expr::var(SUBGROUP_LANE)
}

/// Bind the lane index the replacement body reads.
fn bind_lane() -> Node {
    Node::let_bind(SUBGROUP_LANE, Expr::LogicalWithinTileId { axis: 0 })
}

/// Scope deduced from a workgroup reduction region body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReductionScope {
    EveryWorkgroup,
    FirstWorkgroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReductionValueType {
    F32,
    U32,
}

impl ReductionValueType {
    /// Identity element for `op` at this value type. Used as the second-level
    /// `select` fill for out-of-range lanes so they cannot perturb the result:
    /// `0` for `Add` (sum), `-inf`/`0` for `Max`, etc.
    ///
    /// `None` for an op this table has no identity for. `SubgroupReduceOp` is
    /// `#[non_exhaustive]`, so a variant added upstream reaches here, and any
    /// fill this table guessed for it would be a wrong reduction result rather
    /// than a slow one. Refusing leaves the portable workgroup tree in place.
    fn neutral(self, op: SubgroupReduceOp) -> Option<Expr> {
        match (op, self) {
            (SubgroupReduceOp::Add, Self::F32) => Some(Expr::f32(0.0)),
            (SubgroupReduceOp::Add, Self::U32) => Some(Expr::u32(0)),
            (SubgroupReduceOp::Mul, Self::F32) => Some(Expr::f32(1.0)),
            (SubgroupReduceOp::Mul, Self::U32) => Some(Expr::u32(1)),
            (SubgroupReduceOp::Max, Self::F32) => Some(Expr::f32(f32::NEG_INFINITY)),
            (SubgroupReduceOp::Max, Self::U32) => Some(Expr::u32(0)),
            (SubgroupReduceOp::Min, Self::F32) => Some(Expr::f32(f32::INFINITY)),
            (SubgroupReduceOp::Min, Self::U32) => Some(Expr::u32(u32::MAX)),
            (SubgroupReduceOp::And, _) => Some(Expr::u32(u32::MAX)),
            (SubgroupReduceOp::Or | SubgroupReduceOp::Xor, _) => Some(Expr::u32(0)),
            _ => None,
        }
    }
}

/// Lower workgroup-tree reductions to subgroup ops when the adapter supports it.
///
/// The pass is gated by `caps.supports_subgroup_ops`. A workgroup that fits
/// in one subgroup lowers to one `subgroup_add`. A larger workgroup lowers to
/// a subgroup-then-shared reduction when its subgroup count fits in one
/// subgroup.
#[must_use]
pub fn lower_subgroup_reductions(program: Program, caps: &AdapterCaps) -> Program {
    if !caps.supports_subgroup_ops || caps.subgroup_size == 0 {
        return program;
    }

    let workgroup_total = program.workgroup_size()[0]
        .saturating_mul(program.workgroup_size()[1])
        .saturating_mul(program.workgroup_size()[2]);

    if workgroup_total > subgroup_reduce_lane_limit(caps.subgroup_size) {
        return program;
    }

    let plan = SubgroupReductionPlan {
        subgroup_size: caps.subgroup_size,
        workgroup_total,
    };
    match rewrite_nodes(program.entry(), plan) {
        Cow::Borrowed(_) => program,
        Cow::Owned(entry) => program.with_rewritten_entry(entry),
    }
}

#[derive(Clone, Copy)]
struct SubgroupReductionPlan {
    subgroup_size: u32,
    workgroup_total: u32,
}

fn subgroup_reduce_lane_limit(subgroup_size: u32) -> u32 {
    subgroup_size.saturating_mul(subgroup_size)
}

fn rewrite_nodes(nodes: &[Node], plan: SubgroupReductionPlan) -> Cow<'_, [Node]> {
    rewrite_node_slices(nodes, |node| rewrite_node(node, plan))
}

fn rewrite_node(node: &Node, plan: SubgroupReductionPlan) -> Cow<'_, [Node]> {
    match node {
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            let generator_name = generator.as_str();
            if let Some(lowered) = try_lower_workgroup_reduction(generator_name, body, plan) {
                return Cow::Owned(vec![Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(lowered),
                }]);
            }
            match rewrite_nodes(body, plan) {
                Cow::Borrowed(_) => Cow::Borrowed(std::slice::from_ref(node)),
                Cow::Owned(new_body) => Cow::Owned(vec![Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(new_body),
                }]),
            }
        }
        // Every other variant recurses through the one owner of which variants
        // nest bodies. A leaf has no body slot, the map hands it straight back
        // borrowed, and the walk stops. A body-bearing variant added tomorrow
        // is walked here rather than reaching a backend with the reduction
        // region inside it still on the portable shared-memory tree while the
        // rest of the program lowered.
        other => match map_bodies_cow(other, &mut |body| rewrite_nodes(body, plan)) {
            Cow::Borrowed(_) => Cow::Borrowed(std::slice::from_ref(node)),
            Cow::Owned(rewritten) => Cow::Owned(vec![rewritten]),
        },
    }
}

/// Attempt to lower a workgroup reduction region body to subgroup ops.
fn try_lower_workgroup_reduction(
    generator: &str,
    body: &[Node],
    plan: SubgroupReductionPlan,
) -> Option<Vec<Node>> {
    if has_standalone_reduction_preamble(body) {
        return None;
    }
    let scratch = extract_scratch_buffer(body)?;
    let scope = detect_scope(body)?;

    if let Some(value_type) = workgroup_sum_value_type(generator) {
        subgroup_reduce_body(SubgroupReduceOp::Add, &scratch, scope, plan, value_type)
    } else if let Some(value_type) = workgroup_max_value_type(generator) {
        // Max reductions lower to `subgroup_reduce(Max, ...)`, mirroring the
        // sum path but with the max identity (`-inf`) filling out-of-range
        // lanes in the two-level reduction. Backends emit the native
        // `subgroupMax` / `redux.sync.max` instead of the slow shared tree.
        subgroup_reduce_body(SubgroupReduceOp::Max, &scratch, scope, plan, value_type)
    } else if let Some(value_type) = workgroup_min_value_type(generator) {
        // Min reductions lower to `subgroup_reduce(Min, ...)`, with the min
        // identity (`+inf` for f32, `u32::MAX` for u32) filling out-of-range
        // lanes. Backends emit the native `subgroupMin` / `redux.sync.min`.
        subgroup_reduce_body(SubgroupReduceOp::Min, &scratch, scope, plan, value_type)
    } else {
        None
    }
}

fn workgroup_sum_value_type(generator: &str) -> Option<ReductionValueType> {
    reduction_value_type(generator.strip_prefix(WORKGROUP_SUM_PREFIX)?)
}

fn workgroup_max_value_type(generator: &str) -> Option<ReductionValueType> {
    reduction_value_type(generator.strip_prefix(WORKGROUP_MAX_PREFIX)?)
}

fn workgroup_min_value_type(generator: &str) -> Option<ReductionValueType> {
    reduction_value_type(generator.strip_prefix(WORKGROUP_MIN_PREFIX)?)
}

fn reduction_value_type(suffix: &str) -> Option<ReductionValueType> {
    if suffix.starts_with("f32") {
        Some(ReductionValueType::F32)
    } else if suffix.starts_with("u32") {
        Some(ReductionValueType::U32)
    } else {
        None
    }
}

fn has_standalone_reduction_preamble(body: &[Node]) -> bool {
    matches!(
        body.first(),
        Some(Node::Let {
            name,
            value: Expr::LocalId { axis: 0 }
        }) if name.as_str() == "local"
    )
}

/// Extract the scratch buffer name from the first `Store` in the body.
fn extract_scratch_buffer(body: &[Node]) -> Option<String> {
    for node in body {
        if let Node::Store { buffer, .. } = node {
            return Some(buffer.as_str().to_owned());
        }
        if let Node::If { then, .. } = node {
            for child in then {
                if let Node::Store { buffer, .. } = child {
                    return Some(buffer.as_str().to_owned());
                }
                if let Node::If {
                    then: inner_then, ..
                } = child
                {
                    for inner in inner_then {
                        if let Node::Store { buffer, .. } = inner {
                            return Some(buffer.as_str().to_owned());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Detect the reduction scope by looking for a `workgroup_id.x == 0` guard.
fn detect_scope(body: &[Node]) -> Option<ReductionScope> {
    let first = body.first()?;
    let Node::If { cond, .. } = first else {
        return Some(ReductionScope::EveryWorkgroup);
    };
    if contains_workgroup_zero_guard(cond) {
        Some(ReductionScope::FirstWorkgroup)
    } else {
        Some(ReductionScope::EveryWorkgroup)
    }
}

fn contains_workgroup_zero_guard(expr: &Expr) -> bool {
    match expr {
        Expr::BinOp {
            op: crate::ir::BinOp::And,
            left,
            right,
        } => contains_workgroup_zero_guard(left) || contains_workgroup_zero_guard(right),
        Expr::BinOp {
            op: crate::ir::BinOp::Eq,
            left,
            right,
        } => {
            matches!(left.as_ref(), Expr::WorkgroupId { axis: 0 })
                && matches!(right.as_ref(), Expr::LitU32(0))
                || matches!(right.as_ref(), Expr::WorkgroupId { axis: 0 })
                    && matches!(left.as_ref(), Expr::LitU32(0))
        }
        _ => false,
    }
}

/// Body that replaces the portable workgroup reduction, or `None` when the
/// two-level form has no identity for `op` to fill its out-of-range lanes
/// with. The single-subgroup form reads every lane, so it needs no identity
/// and is always available.
fn subgroup_reduce_body(
    op: SubgroupReduceOp,
    scratch: &str,
    scope: ReductionScope,
    plan: SubgroupReductionPlan,
    value_type: ReductionValueType,
) -> Option<Vec<Node>> {
    if plan.workgroup_total <= plan.subgroup_size {
        return Some(single_subgroup_reduce_body(op, scratch, scope));
    }
    two_level_subgroup_reduce_body(op, scratch, scope, plan, value_type)
}

fn single_subgroup_reduce_body(
    op: SubgroupReduceOp,
    scratch: &str,
    scope: ReductionScope,
) -> Vec<Node> {
    let load_expr = Expr::load(scratch, lane());
    let subgroup_expr = Expr::subgroup_reduce(op, load_expr);
    let store_node = Node::store(scratch, lane(), subgroup_expr);

    match scope {
        ReductionScope::EveryWorkgroup => vec![bind_lane(), store_node, Node::barrier()],
        ReductionScope::FirstWorkgroup => vec![
            bind_lane(),
            Node::if_then(Expr::is_first_workgroup(), vec![store_node]),
            Node::barrier(),
        ],
    }
}

fn two_level_subgroup_reduce_body(
    op: SubgroupReduceOp,
    scratch: &str,
    scope: ReductionScope,
    plan: SubgroupReductionPlan,
    value_type: ReductionValueType,
) -> Option<Vec<Node>> {
    let subgroup_count = plan.workgroup_total.div_ceil(plan.subgroup_size);
    let subgroup_slot = Expr::div(lane(), Expr::u32(plan.subgroup_size));
    let subgroup_sum = Expr::subgroup_reduce(op, Expr::load(scratch, lane()));
    let subgroup_head = Expr::eq(Expr::subgroup_local_id(), Expr::u32(0));
    let first_level = vec![
        Node::let_bind("vyre_subgroup_sum", subgroup_sum),
        // Every subgroup reads its own span of the tile above, and the head
        // store below writes back into the low slots of that same tile. One
        // subgroup's store lands in a slot another subgroup has not read yet,
        // so without this fence the second subgroup sums a partial in place of
        // a lane value and the workgroup total comes out high by whatever the
        // partial exceeded it. That is a wrong answer, not a slow one, and it
        // varies run to run with subgroup scheduling.
        Node::barrier(),
        Node::if_then(
            subgroup_head,
            vec![Node::store(
                scratch,
                subgroup_slot,
                Expr::var("vyre_subgroup_sum"),
            )],
        ),
    ];
    let second_level_sum = Expr::subgroup_reduce(
        op,
        Expr::select(
            Expr::lt(lane(), Expr::u32(subgroup_count)),
            Expr::load(scratch, lane()),
            value_type.neutral(op)?,
        ),
    );
    let second_level = vec![
        Node::let_bind("vyre_workgroup_sum", second_level_sum),
        Node::if_then(
            Expr::eq(lane(), Expr::u32(0)),
            vec![Node::store(
                scratch,
                Expr::u32(0),
                Expr::var("vyre_workgroup_sum"),
            )],
        ),
    ];

    Some(match scope {
        ReductionScope::EveryWorkgroup => {
            let mut nodes = vec![bind_lane()];
            nodes.extend(first_level);
            nodes.push(Node::barrier());
            nodes.extend(second_level);
            nodes.push(Node::barrier());
            nodes
        }
        ReductionScope::FirstWorkgroup => vec![
            bind_lane(),
            Node::if_then(Expr::is_first_workgroup(), first_level),
            Node::barrier(),
            Node::if_then(Expr::is_first_workgroup(), second_level),
            Node::barrier(),
        ],
    })
}

#[cfg(test)]
mod tests {
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
            let reads_a_free_variable = any_expr_matching(body, &|expr| {
                matches!(expr, Expr::Var(v) if !bound.iter().any(|name| name == v.as_str()))
            });
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
        let Node::Region { body, .. } =
            workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
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
        let Node::Region { body, .. } =
            workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
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
        let Node::Region { body, .. } =
            workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
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
        let Node::Region { body, .. } =
            workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
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
        let Node::Region { body, .. } =
            workgroup_sum_region("scratch", ReductionScope::EveryWorkgroup)
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
}
