//! Subgroup-first lowering pass (Phase 2.3).
//!
//! Converts workgroup-tree reductions over shared memory into
//! `subgroup_add` / `subgroup_shuffle` operations when the backend
//! reports native subgroup support and the workgroup shape fits the
//! subgroup size.

use crate::composition::bounded_index;
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
    // Every lane in the workgroup reaches the second-level reduce, and the scratch
    // slab holds one slot per subgroup. A select evaluates both arms, so a lane past
    // the subgroup count reads the slab too: its index is folded inside the slab and
    // the same select replaces the value with the reduction's neutral element.
    let second_level_sum = Expr::subgroup_reduce(
        op,
        Expr::select(
            Expr::lt(lane(), Expr::u32(subgroup_count)),
            Expr::load(scratch, bounded_index(lane(), Expr::u32(subgroup_count))),
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
mod tests;
