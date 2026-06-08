//! Subgroup-first lowering pass (Phase 2.3).
//!
//! Converts workgroup-tree reductions over shared memory into
//! `subgroup_add` / `subgroup_shuffle` warp operations when the backend
//! reports native subgroup support and the workgroup shape fits the
//! subgroup size.

use crate::ir::{Expr, Node, Program};
use crate::optimizer::ctx::AdapterCaps;
use std::borrow::Cow;
use std::sync::Arc;

/// Canonical generator prefixes emitted by `vyre-primitives::reduce::workgroup_tree`.
const WORKGROUP_SUM_PREFIX: &str = "vyre-primitives::reduce::workgroup_sum_";
const WORKGROUP_MAX_PREFIX: &str = "vyre-primitives::reduce::workgroup_max_";

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
    fn neutral(self) -> Expr {
        match self {
            Self::F32 => Expr::f32(0.0),
            Self::U32 => Expr::u32(0),
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
    let mut rewritten: Option<Vec<Node>> = None;
    for (index, node) in nodes.iter().enumerate() {
        match rewrite_node(node, plan) {
            Cow::Borrowed(_) if rewritten.is_none() => {}
            Cow::Borrowed(borrowed) => {
                if let Some(out) = rewritten.as_mut() {
                    out.extend_from_slice(borrowed);
                }
            }
            Cow::Owned(owned) => {
                let out = rewritten.get_or_insert_with(|| nodes[..index].to_vec());
                out.extend(owned);
            }
        }
    }
    rewritten.map_or(Cow::Borrowed(nodes), Cow::Owned)
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
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let t = rewrite_nodes(then, plan);
            let o = rewrite_nodes(otherwise, plan);
            if matches!((&t, &o), (Cow::Borrowed(_), Cow::Borrowed(_))) {
                Cow::Borrowed(std::slice::from_ref(node))
            } else {
                Cow::Owned(vec![Node::if_then_else(
                    cond.clone(),
                    t.into_owned(),
                    o.into_owned(),
                )])
            }
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => {
            let b = rewrite_nodes(body, plan);
            if matches!(b, Cow::Borrowed(_)) {
                Cow::Borrowed(std::slice::from_ref(node))
            } else {
                Cow::Owned(vec![Node::loop_for(
                    var.clone(),
                    from.clone(),
                    to.clone(),
                    b.into_owned(),
                )])
            }
        }
        Node::Block(body) => match rewrite_nodes(body, plan) {
            Cow::Borrowed(_) => Cow::Borrowed(std::slice::from_ref(node)),
            Cow::Owned(b) => Cow::Owned(vec![Node::block(b)]),
        },
        _ => Cow::Borrowed(std::slice::from_ref(node)),
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
        Some(subgroup_sum_body(&scratch, scope, plan, value_type))
    } else if generator.starts_with(WORKGROUP_MAX_PREFIX) {
        // Max reductions are lowered via a shuffle-based tree using
        // subgroup_shuffle.  For simplicity we emit the same load+
        // shuffle-tree pattern, but here we just keep the original
        // body  -  the task focuses on sum reductions (attention softmax,
        // MoE routing, KV cache compaction).
        None
    } else {
        None
    }
}

fn workgroup_sum_value_type(generator: &str) -> Option<ReductionValueType> {
    let suffix = generator.strip_prefix(WORKGROUP_SUM_PREFIX)?;
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

fn subgroup_sum_body(
    scratch: &str,
    scope: ReductionScope,
    plan: SubgroupReductionPlan,
    value_type: ReductionValueType,
) -> Vec<Node> {
    if plan.workgroup_total <= plan.subgroup_size {
        return single_subgroup_sum_body(scratch, scope);
    }
    two_level_subgroup_sum_body(scratch, scope, plan, value_type)
}

fn single_subgroup_sum_body(scratch: &str, scope: ReductionScope) -> Vec<Node> {
    let load_expr = Expr::load(scratch, Expr::var("local"));
    let subgroup_expr = Expr::subgroup_add(load_expr);
    let store_node = Node::store(scratch, Expr::var("local"), subgroup_expr);

    match scope {
        ReductionScope::EveryWorkgroup => vec![store_node, Node::barrier()],
        ReductionScope::FirstWorkgroup => vec![
            Node::if_then(
                Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)),
                vec![store_node],
            ),
            Node::barrier(),
        ],
    }
}

fn two_level_subgroup_sum_body(
    scratch: &str,
    scope: ReductionScope,
    plan: SubgroupReductionPlan,
    value_type: ReductionValueType,
) -> Vec<Node> {
    let subgroup_count = plan.workgroup_total.div_ceil(plan.subgroup_size);
    let subgroup_slot = Expr::div(Expr::var("local"), Expr::u32(plan.subgroup_size));
    let subgroup_sum = Expr::subgroup_add(Expr::load(scratch, Expr::var("local")));
    let subgroup_head = Expr::eq(Expr::subgroup_local_id(), Expr::u32(0));
    let first_level = vec![
        Node::let_bind("vyre_subgroup_sum", subgroup_sum),
        Node::if_then(
            subgroup_head,
            vec![Node::store(
                scratch,
                subgroup_slot,
                Expr::var("vyre_subgroup_sum"),
            )],
        ),
    ];
    let second_level_sum = Expr::subgroup_add(Expr::select(
        Expr::lt(Expr::var("local"), Expr::u32(subgroup_count)),
        Expr::load(scratch, Expr::var("local")),
        value_type.neutral(),
    ));
    let second_level = vec![
        Node::let_bind("vyre_workgroup_sum", second_level_sum),
        Node::if_then(
            Expr::eq(Expr::var("local"), Expr::u32(0)),
            vec![Node::store(
                scratch,
                Expr::u32(0),
                Expr::var("vyre_workgroup_sum"),
            )],
        ),
    ];

    match scope {
        ReductionScope::EveryWorkgroup => {
            let mut nodes = first_level;
            nodes.push(Node::barrier());
            nodes.extend(second_level);
            nodes.push(Node::barrier());
            nodes
        }
        ReductionScope::FirstWorkgroup => vec![
            Node::if_then(
                Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)),
                first_level,
            ),
            Node::barrier(),
            Node::if_then(
                Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)),
                second_level,
            ),
            Node::barrier(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType, Expr, Node, Program};

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
                generator: "vyre-primitives::reduce::workgroup_sum_f32".into(),
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
                generator: "vyre-primitives::reduce::workgroup_sum_u32".into(),
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
        nodes_contain_select_false(nodes, |expr| matches!(expr, Expr::LitF32(value) if *value == 0.0))
    }

    fn nodes_contain_select_false(
        nodes: &[Node],
        predicate: fn(&Expr) -> bool,
    ) -> bool {
        nodes
            .iter()
            .any(|node| node_contains_select_false(node, predicate))
    }

    fn node_contains_select_false(node: &Node, predicate: fn(&Expr) -> bool) -> bool {
        match node {
            Node::Let { value, .. } | Node::Assign { value, .. } => {
                expr_contains_select_false(value, predicate)
            }
            Node::Store { index, value, .. } => {
                expr_contains_select_false(index, predicate)
                    || expr_contains_select_false(value, predicate)
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                expr_contains_select_false(cond, predicate)
                    || nodes_contain_select_false(then, predicate)
                    || nodes_contain_select_false(otherwise, predicate)
            }
            Node::Loop { from, to, body, .. } => {
                expr_contains_select_false(from, predicate)
                    || expr_contains_select_false(to, predicate)
                    || nodes_contain_select_false(body, predicate)
            }
            Node::Block(body) => nodes_contain_select_false(body, predicate),
            Node::Region { body, .. } => nodes_contain_select_false(body, predicate),
            Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
                expr_contains_select_false(offset, predicate)
                    || expr_contains_select_false(size, predicate)
            }
            Node::Trap { address, .. } => expr_contains_select_false(address, predicate),
            _ => false,
        }
    }

    fn expr_contains_select_false(expr: &Expr, predicate: fn(&Expr) -> bool) -> bool {
        match expr {
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                predicate(false_val)
                    || expr_contains_select_false(cond, predicate)
                    || expr_contains_select_false(true_val, predicate)
                    || expr_contains_select_false(false_val, predicate)
            }
            Expr::Load { index, .. }
            | Expr::UnOp { operand: index, .. }
            | Expr::Cast { value: index, .. }
            | Expr::SubgroupBallot { cond: index }
            | Expr::SubgroupAdd { value: index } => expr_contains_select_false(index, predicate),
            Expr::BinOp { left, right, .. } | Expr::SubgroupShuffle { value: left, lane: right } => {
                expr_contains_select_false(left, predicate)
                    || expr_contains_select_false(right, predicate)
            }
            Expr::Call { args, .. } => args
                .iter()
                .any(|arg| expr_contains_select_false(arg, predicate)),
            Expr::Fma { a, b, c } => {
                expr_contains_select_false(a, predicate)
                    || expr_contains_select_false(b, predicate)
                    || expr_contains_select_false(c, predicate)
            }
            Expr::Atomic {
                index,
                expected,
                value,
                ..
            } => {
                expr_contains_select_false(index, predicate)
                    || expected
                        .as_ref()
                        .is_some_and(|expected| expr_contains_select_false(expected, predicate))
                    || expr_contains_select_false(value, predicate)
            }
            _ => false,
        }
    }

    fn workgroup_sum_region(scratch: &str, scope: ReductionScope) -> Node {
        let body = if scope == ReductionScope::FirstWorkgroup {
            vec![
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)),
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
                        Expr::eq(Expr::WorkgroupId { axis: 0 }, Expr::u32(0)),
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
            generator: "vyre-primitives::reduce::workgroup_sum_f32".into(),
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
        // Should be: store(scratch, local, subgroup_add(load(scratch, local))); barrier
        assert_eq!(body.len(), 2);
        assert!(
            matches!(&body[0], Node::Store { buffer, index, value } if
                buffer.as_str() == "scratch" &&
                matches!(index, Expr::Var(v) if v.as_str() == "local") &&
                matches!(value, Expr::SubgroupAdd { .. })
            ),
            "expected subgroup_add store, got {:?}",
            body[0]
        );
        assert!(matches!(&body[1], Node::Barrier { .. }));
    }

    #[test]
    fn lowers_two_level_workgroup_sum_for_large_cuda_blocks() {
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
        assert_eq!(
            body.len(),
            6,
            "Fix: two-level subgroup lowering should emit first-level subgroup work, a barrier, full-warp second-level subgroup work, and a final barrier."
        );
        assert!(
            node_contains_subgroup_add(&body[0]) && node_contains_subgroup_add(&body[3]),
            "Fix: both levels of the 256-lane reduction must use subgroup_add instead of the shared-memory tree: {body:?}"
        );
        assert!(matches!(&body[2], Node::Barrier { .. }));
        assert!(matches!(&body[5], Node::Barrier { .. }));
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

    fn node_contains_subgroup_add(node: &Node) -> bool {
        match node {
            Node::Let { value, .. } | Node::Assign { value, .. } => {
                expr_contains_subgroup_add(value)
            }
            Node::Store { index, value, .. } => {
                expr_contains_subgroup_add(index) || expr_contains_subgroup_add(value)
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                expr_contains_subgroup_add(cond)
                    || then.iter().any(node_contains_subgroup_add)
                    || otherwise.iter().any(node_contains_subgroup_add)
            }
            Node::Loop { from, to, body, .. } => {
                expr_contains_subgroup_add(from)
                    || expr_contains_subgroup_add(to)
                    || body.iter().any(node_contains_subgroup_add)
            }
            Node::Block(body) => body.iter().any(node_contains_subgroup_add),
            Node::Region { body, .. } => body.iter().any(node_contains_subgroup_add),
            Node::Barrier { .. }
            | Node::IndirectDispatch { .. }
            | Node::AsyncWait { .. }
            | Node::Trap { .. }
            | Node::Resume { .. }
            | Node::AllReduce { .. }
            | Node::AllGather { .. }
            | Node::ReduceScatter { .. }
            | Node::Broadcast { .. }
            | Node::Opaque(_)
            | Node::Return => false,
            Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
                expr_contains_subgroup_add(offset) || expr_contains_subgroup_add(size)
            }
        }
    }

    fn expr_contains_subgroup_add(expr: &Expr) -> bool {
        match expr {
            Expr::SubgroupAdd { .. } => true,
            Expr::Load { index, .. }
            | Expr::Cast { value: index, .. }
            | Expr::SubgroupShuffle { value: index, .. }
            | Expr::SubgroupBallot { cond: index } => expr_contains_subgroup_add(index),
            Expr::BinOp { left, right, .. } => {
                expr_contains_subgroup_add(left) || expr_contains_subgroup_add(right)
            }
            Expr::UnOp { operand, .. } => expr_contains_subgroup_add(operand),
            Expr::Call { args, .. } => args.iter().any(expr_contains_subgroup_add),
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                expr_contains_subgroup_add(cond)
                    || expr_contains_subgroup_add(true_val)
                    || expr_contains_subgroup_add(false_val)
            }
            Expr::Fma { a, b, c } => {
                expr_contains_subgroup_add(a)
                    || expr_contains_subgroup_add(b)
                    || expr_contains_subgroup_add(c)
            }
            Expr::Atomic {
                index,
                expected,
                value,
                ..
            } => {
                expr_contains_subgroup_add(index)
                    || expected
                        .as_ref()
                        .is_some_and(|expr| expr_contains_subgroup_add(expr))
                    || expr_contains_subgroup_add(value)
            }
            Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF32(_)
            | Expr::LitBool(_)
            | Expr::Var(_)
            | Expr::InvocationId { .. }
            | Expr::WorkgroupId { .. }
            | Expr::LocalId { .. }
            | Expr::BufLen { .. }
            | Expr::SubgroupLocalId
            | Expr::SubgroupSize
            | Expr::Opaque(_) => false,
        }
    }
}
