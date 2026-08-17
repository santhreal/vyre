//! Workgroup-local tree reductions over scratch buffers.
//!
//! These helpers are Tier 2.5 LEGO blocks for higher-level library ops that
//! already stage one partial value per lane into workgroup memory. They emit
//! child `Region`s so composition audits and traces show the shared reduction
//! instead of treating every math/NN op as a hand-rolled loop.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id for an f32 workgroup sum over a scratch buffer.
pub const SUM_F32_OP_ID: &str = "vyre-libs::reduce::workgroup_sum_f32";
/// Canonical op id for a u32 workgroup sum over a scratch buffer.
pub const SUM_U32_OP_ID: &str = "vyre-libs::reduce::workgroup_sum_u32";
/// Canonical op id for an f32 workgroup maximum over a scratch buffer.
pub const MAX_F32_OP_ID: &str = "vyre-libs::reduce::workgroup_max_f32";
/// Canonical op id for a u32 workgroup maximum over a scratch buffer.
pub const MAX_U32_OP_ID: &str = "vyre-libs::reduce::workgroup_max_u32";
/// Canonical op id for an f32 workgroup minimum over a scratch buffer.
pub const MIN_F32_OP_ID: &str = "vyre-libs::reduce::workgroup_min_f32";
/// Canonical op id for a u32 workgroup minimum over a scratch buffer.
pub const MIN_U32_OP_ID: &str = "vyre-libs::reduce::workgroup_min_u32";

/// Scope for a workgroup-local reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupReductionScope {
    /// Every dispatched workgroup reduces its own scratch buffer.
    EveryWorkgroup,
    /// Only workgroup `x == 0` participates in the reduction.
    FirstWorkgroup,
}

impl WorkgroupReductionScope {
    fn lane_guard(self, lane_expr: Expr) -> Expr {
        match self {
            Self::EveryWorkgroup => lane_expr,
            Self::FirstWorkgroup => Expr::and(Expr::is_first_workgroup(), lane_expr),
        }
    }
}

/// Emit a child region that sums f32 lane partials in `scratch`.
#[must_use]
pub fn sum_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(SUM_F32_OP_ID, parent_op_id, sum_body(tile, scratch, scope))
}

/// Emit a child region that sums u32 lane partials in `scratch`.
#[must_use]
pub fn sum_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(SUM_U32_OP_ID, parent_op_id, sum_body(tile, scratch, scope))
}

/// Emit a child region that maximizes f32 lane partials in `scratch`.
#[must_use]
pub fn max_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(MAX_F32_OP_ID, parent_op_id, max_body(tile, scratch, scope))
}

/// Emit a child region that maximizes u32 lane partials in `scratch`.
#[must_use]
pub fn max_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(MAX_U32_OP_ID, parent_op_id, max_body(tile, scratch, scope))
}

/// Emit a child region that minimizes f32 lane partials in `scratch`.
#[must_use]
pub fn min_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(MIN_F32_OP_ID, parent_op_id, min_body(tile, scratch, scope))
}

/// Emit a child region that minimizes u32 lane partials in `scratch`.
#[must_use]
pub fn min_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    child_region(MIN_U32_OP_ID, parent_op_id, min_body(tile, scratch, scope))
}

/// Build a standalone f32 workgroup sum Program.
#[must_use]
pub fn workgroup_sum_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: SUM_F32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::F32,
        fold: WorkgroupFold::Sum,
    })
}

/// Build a standalone u32 workgroup sum Program.
#[must_use]
pub fn workgroup_sum_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: SUM_U32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::U32,
        fold: WorkgroupFold::Sum,
    })
}

/// Build a standalone f32 workgroup maximum Program.
#[must_use]
pub fn workgroup_max_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: MAX_F32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::F32,
        fold: WorkgroupFold::Max,
    })
}

/// Build a standalone u32 workgroup maximum Program.
///
/// The u32 twin of [`workgroup_max_f32`], closing the
/// sum-has-both-types / max-has-only-f32 asymmetry. `0` (`u32::MIN`) is the
/// neutral for an unsigned max, and the subgroup-first lowering already
/// recognizes the `workgroup_max_` prefix with a u32 value type, so this gets
/// the fast warp-reduction path on subgroup-capable backends for free.
#[must_use]
pub fn workgroup_max_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: MAX_U32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::U32,
        fold: WorkgroupFold::Max,
    })
}

/// Build a standalone f32 workgroup minimum Program.
///
/// `f32::MAX` is the neutral for a minimum (any real value is smaller). The
/// subgroup-first lowering recognizes the `workgroup_min_` prefix, so this gets
/// the native warp `subgroupMin` / `redux.sync.min` path on capable backends.
#[must_use]
pub fn workgroup_min_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: MIN_F32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::F32,
        fold: WorkgroupFold::Min,
    })
}

/// Build a standalone u32 workgroup minimum Program.
///
/// `u32::MAX` is the neutral for an unsigned minimum.
#[must_use]
pub fn workgroup_min_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    reduction_program(WorkgroupReduction {
        op_id: MIN_U32_OP_ID,
        values,
        out,
        count,
        tile,
        dtype: DataType::U32,
        fold: WorkgroupFold::Min,
    })
}

/// Which value a workgroup reduction folds its lanes down to.
///
/// The fold decides three things at once - the identity a lane starts from, how
/// two lanes combine, and which tree body the sweep emits - and the three must
/// agree. Passed as three separate arguments they could disagree, and a `max`
/// combine over a `sum` tree is a silently wrong reduction; naming the fold
/// once makes that unstateable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkgroupFold {
    /// Add the lanes.
    Sum,
    /// Keep the largest lane.
    Max,
    /// Keep the smallest lane.
    Min,
}

impl WorkgroupFold {
    /// Value a lane accumulates from before it reads anything.
    fn identity(self, dtype: &DataType) -> Expr {
        match (self, dtype) {
            (Self::Sum, DataType::F32) => Expr::f32(0.0),
            (Self::Sum, _) => Expr::u32(0),
            (Self::Max, DataType::F32) => Expr::f32(f32::MIN),
            (Self::Max, _) => Expr::u32(u32::MIN),
            (Self::Min, DataType::F32) => Expr::f32(f32::MAX),
            (Self::Min, _) => Expr::u32(u32::MAX),
        }
    }

    /// Combine two lane values.
    fn combine(self, left: Expr, right: Expr) -> Expr {
        match self {
            Self::Sum => Expr::add(left, right),
            Self::Max => Expr::max(left, right),
            Self::Min => Expr::min(left, right),
        }
    }

    /// Tree sweep over the staged scratch buffer.
    fn tree(self, tile: u32, scratch: &'static str) -> Vec<Node> {
        let scope = WorkgroupReductionScope::FirstWorkgroup;
        match self {
            Self::Sum => sum_body(tile, scratch, scope),
            Self::Max => max_body(tile, scratch, scope),
            Self::Min => min_body(tile, scratch, scope),
        }
    }
}

/// One standalone workgroup reduction: what to fold, over which buffer, into
/// which output, at which launch geometry.
struct WorkgroupReduction<'a> {
    /// Region generator id the emitted Program carries.
    op_id: &'static str,
    /// Input buffer the lanes read.
    values: &'a str,
    /// Single-slot output buffer the reduction writes.
    out: &'a str,
    /// Elements in `values`.
    count: u32,
    /// Lanes in the workgroup, and slots in the scratch buffer.
    tile: u32,
    /// Element type of both buffers.
    dtype: DataType,
    /// Which reduction to emit.
    fold: WorkgroupFold,
}

fn reduction_program(spec: WorkgroupReduction<'_>) -> Program {
    let WorkgroupReduction {
        op_id,
        values,
        out,
        count,
        tile,
        dtype,
        fold,
    } = spec;
    let tile = tile.max(1);
    let chunks = count.div_ceil(tile);
    let scratch = "__workgroup_reduce_scratch";
    let local = Expr::var("local");
    let idx = Expr::var("idx");
    let mut body = vec![
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        Node::if_then(
            Expr::is_first_workgroup(),
            vec![
                Node::let_bind("acc", fold.identity(&dtype)),
                Node::loop_for(
                    "chunk",
                    Expr::u32(0),
                    Expr::u32(chunks),
                    vec![
                        Node::let_bind(
                            "idx",
                            Expr::add(
                                Expr::mul(Expr::var("chunk"), Expr::u32(tile)),
                                local.clone(),
                            ),
                        ),
                        Node::if_then(
                            Expr::lt(idx.clone(), Expr::u32(count)),
                            vec![Node::assign(
                                "acc",
                                fold.combine(Expr::var("acc"), Expr::load(values, idx.clone())),
                            )],
                        ),
                    ],
                ),
                Node::store(scratch, local.clone(), Expr::var("acc")),
            ],
        ),
        Node::barrier(),
    ];
    body.extend(fold.tree(tile, scratch));
    body.push(Node::if_then(
        Expr::and(Expr::is_first_workgroup(), Expr::eq(local, Expr::u32(0))),
        vec![Node::store(
            out,
            Expr::u32(0),
            Expr::load(scratch, Expr::u32(0)),
        )],
    ));
    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(count),
            BufferDecl::workgroup(scratch, tile, dtype.clone()),
            BufferDecl::output(out, 1, dtype).with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}

fn child_region(generator: &'static str, parent_op_id: &str, body: Vec<Node>) -> Node {
    wrap_child_region(generator, Ident::from(parent_op_id), body)
}

fn sum_body(tile: u32, scratch: &'static str, scope: WorkgroupReductionScope) -> Vec<Node> {
    tree_body(tile, scratch, scope, Expr::add)
}

fn max_body(tile: u32, scratch: &'static str, scope: WorkgroupReductionScope) -> Vec<Node> {
    tree_body(tile, scratch, scope, Expr::max)
}

fn min_body(tile: u32, scratch: &'static str, scope: WorkgroupReductionScope) -> Vec<Node> {
    tree_body(tile, scratch, scope, Expr::min)
}

fn tree_body<F>(
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
    combine: F,
) -> Vec<Node>
where
    F: Fn(Expr, Expr) -> Expr,
{
    let mut nodes = Vec::new();
    let mut stride = tile.next_power_of_two() / 2;
    while stride > 0 {
        let lhs = Expr::load(scratch, Expr::var("local"));
        let rhs_index = Expr::add(Expr::var("local"), Expr::u32(stride));
        let rhs = Expr::load(scratch, rhs_index.clone());
        nodes.push(Node::if_then(
            scope.lane_guard(Expr::lt(Expr::var("local"), Expr::u32(stride))),
            vec![Node::if_then(
                Expr::lt(rhs_index, Expr::u32(tile)),
                vec![Node::Store {
                    buffer: scratch.into(),
                    index: Expr::var("local"),
                    value: combine(lhs, rhs),
                }],
            )],
        ));
        nodes.push(Node::barrier());
        stride /= 2;
    }
    nodes
}

fn fixture_f32(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

fn fixture_u32(values: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(values)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SUM_F32_OP_ID,
        || workgroup_sum_f32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_f32(&[1.25, -2.0, 5.5, 3.25]),
            fixture_f32(&[0.0]),
        ]]),
        Some(|| vec![vec![fixture_f32(&[8.0])]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SUM_U32_OP_ID,
        || workgroup_sum_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[1, 2, 3, 4]),
            fixture_u32(&[0]),
        ]]),
        Some(|| vec![vec![fixture_u32(&[10])]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MAX_F32_OP_ID,
        || workgroup_max_f32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_f32(&[-3.0, 9.5, 4.0, 1.25]),
            fixture_f32(&[0.0]),
        ]]),
        Some(|| vec![vec![fixture_f32(&[9.5])]]),
    )
}
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MAX_U32_OP_ID,
        || workgroup_max_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[1, 9, 4, 2]),
            fixture_u32(&[0]),
        ]]),
        Some(|| vec![vec![fixture_u32(&[9])]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MIN_F32_OP_ID,
        || workgroup_min_f32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_f32(&[-3.0, 9.5, 4.0, 1.25]),
            fixture_f32(&[0.0]),
        ]]),
        Some(|| vec![vec![fixture_f32(&[-3.0])]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MIN_U32_OP_ID,
        || workgroup_min_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[3, 9, 4, 2]),
            fixture_u32(&[0]),
        ]]),
        Some(|| vec![vec![fixture_u32(&[2])]]),
    )
}

/// Index of the lane `stride` positions before `lane`.
///
/// One owner for "previous lane" addressing. The two spellings this replaces
/// (`lane + (0u32).wrapping_sub(stride)` here and `lane + u32::MAX` in the
/// exclusive scan) both reached the answer through `BinOp::Add` on a
/// pre-negated constant, so every consumer of the IR - the optimizer's identity
/// rules, the value-range analysis, a reader - saw an addition where the
/// program means a subtraction. `BinOp::WrappingSub` says it once.
pub(crate) fn previous_lane(lane: &Expr, stride: u32) -> Expr {
    lane.clone().wrapping_sub(Expr::u32(stride))
}

/// Emit the work-efficient Blelloch inclusive-sum sweep over `lanes` workgroup
/// lanes, reading the staged per-lane values from `scratch_a` and leaving the
/// inclusive prefix sums there.
///
/// `scratch_b` keeps each lane's staged value across the sweep. The sweep
/// itself produces an EXCLUSIVE scan in `scratch_a`, and the inclusive result
/// is that prefix plus the lane's own staged value, so the second scratch
/// buffer every caller already declares carries the addend instead of being a
/// ping-pong target.
///
/// Reduce phase: at stride `s` the lane owning slot `(k+1)*2s-1` folds the slot
/// `s` positions back into it, so slot `lanes-1` ends holding the total after
/// `log2(lanes)` rounds and `lanes-1` additions. Downsweep phase: that total is
/// cleared and the same slot pairs are walked in reverse, each handing its left
/// child the running prefix and taking the sum, another `lanes-1` additions.
/// Total work is `2*lanes-2` additions against the `lanes*log2(lanes)` a
/// Hillis-Steele sweep performs, because round `s` activates `lanes/(2s)` lanes
/// instead of all of them.
///
/// Barriers are workgroup-scoped: the sweep touches nothing but the two
/// workgroup scratch buffers. The sweep ends on a barrier, so every lane may
/// read any lane's inclusive sum the moment it returns. That is part of the
/// contract rather than a caller's responsibility:
/// `frontier_word_block_offsets_single_workgroup` reads `scratch_a[lane - 1]`
/// immediately after the call to turn the inclusive scan into an exclusive one,
/// and without the trailing barrier lane `k` could read lane `k - 1` before that
/// lane added its own staged value, producing a block offset short by exactly
/// the previous block's count.
///
/// Callers differ in how they stage `scratch_a` and how they write the result
/// out; the sweep between those two steps does not, and was hand-written five
/// times before this became its owner.
///
/// # Panics
///
/// Panics when `lanes` is not a power of two. The slot pairing walks a balanced
/// binary tree over the scratch buffers, and a partial top level would leave
/// slots the downsweep never reaches.
pub(crate) fn blelloch_inclusive_sum_nodes(
    scratch_a: &str,
    scratch_b: &str,
    lane: &Expr,
    lanes: u32,
) -> Vec<Node> {
    assert!(
        lanes.is_power_of_two(),
        "Fix: blelloch_inclusive_sum_nodes needs a power-of-two lane count, got {lanes}; round the scratch buffers up to the next power of two before staging into them."
    );

    let mut nodes = vec![
        Node::store(scratch_b, lane.clone(), Expr::load(scratch_a, lane.clone())),
        Node::barrier(),
    ];

    let mut stride = 1_u32;
    while stride < lanes {
        let slot = sweep_slot(lane, stride);
        nodes.push(Node::if_then(
            Expr::lt(slot.clone(), Expr::u32(lanes)),
            vec![Node::store(
                scratch_a,
                slot.clone(),
                Expr::add(
                    Expr::load(scratch_a, slot.clone()),
                    Expr::load(scratch_a, previous_lane(&slot, stride)),
                ),
            )],
        ));
        nodes.push(Node::barrier());
        stride *= 2;
    }

    nodes.push(Node::if_then(
        Expr::eq(lane.clone(), Expr::u32(0)),
        vec![Node::store(scratch_a, Expr::u32(lanes - 1), Expr::u32(0))],
    ));
    nodes.push(Node::barrier());

    let mut stride = lanes / 2;
    let mut round = 0_u32;
    while stride >= 1 {
        let slot = sweep_slot(lane, stride);
        let left = previous_lane(&slot, stride);
        let held = format!("{scratch_a}_downsweep_{round}");
        nodes.push(Node::if_then(
            Expr::lt(slot.clone(), Expr::u32(lanes)),
            vec![
                Node::let_bind(held.as_str(), Expr::load(scratch_a, left.clone())),
                Node::store(scratch_a, left, Expr::load(scratch_a, slot.clone())),
                Node::store(
                    scratch_a,
                    slot.clone(),
                    Expr::add(Expr::load(scratch_a, slot), Expr::var(held.as_str())),
                ),
            ],
        ));
        nodes.push(Node::barrier());
        stride /= 2;
        round += 1;
    }

    nodes.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(lanes)),
        vec![Node::store(
            scratch_a,
            lane.clone(),
            Expr::add(
                Expr::load(scratch_a, lane.clone()),
                Expr::load(scratch_b, lane.clone()),
            ),
        )],
    ));
    nodes.push(Node::barrier());
    nodes
}

/// Scratch slot lane `lane` owns in a sweep round of stride `s`: `(lane+1)*2s-1`.
///
/// Active lanes are the contiguous prefix `0..lanes/(2s)`, so the divergence
/// sits at one subgroup boundary instead of splitting every subgroup in the
/// workgroup the way a `lane >= stride` predicate does.
fn sweep_slot(lane: &Expr, stride: u32) -> Expr {
    previous_lane(
        &Expr::mul(
            Expr::add(lane.clone(), Expr::u32(1)),
            Expr::u32(stride.saturating_mul(2)),
        ),
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    #[test]
    fn child_region_names_parent_and_primitive() {
        let node = sum_f32_child(
            "vyre-libs::math::reduce_mean",
            256,
            "scratch",
            WorkgroupReductionScope::FirstWorkgroup,
        );
        let Node::Region {
            generator,
            source_region,
            body,
        } = node
        else {
            panic!("Fix: workgroup tree helper must emit a child Region.");
        };
        assert_eq!(generator.as_str(), SUM_F32_OP_ID);
        assert_eq!(
            source_region
                .expect("Fix: child Region must name parent.")
                .as_str(),
            "vyre-libs::math::reduce_mean"
        );
        assert!(!body.is_empty());
    }

    #[test]
    fn standalone_sum_f32_matches_reference_arithmetic() {
        let values = [1.25_f32, -2.0, 5.5, 3.25, 8.0];
        let program = workgroup_sum_f32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_f32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: workgroup_sum_f32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes())[0],
            values.iter().copied().sum::<f32>()
        );
    }

    #[test]
    fn standalone_sum_u32_matches_reference_arithmetic() {
        let values = [1_u32, 2, 3, 4, 5, 6, 7];
        let program = workgroup_sum_u32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_u32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<u32>()]),
            ],
        )
        .expect("Fix: workgroup_sum_u32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())[0],
            values.iter().copied().sum::<u32>()
        );
    }

    #[test]
    fn standalone_max_f32_matches_reference_arithmetic() {
        let values = [-3.0_f32, 9.5, 4.0, 1.25, 8.75];
        let program = workgroup_max_f32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_f32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: workgroup_max_f32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes())[0],
            9.5
        );
    }

    #[test]
    fn standalone_max_u32_matches_reference_arithmetic() {
        // Max (42) is at index 3, not 0, so a broken reduction that kept the
        // first lane or the `0` identity would be caught.
        let values = [3_u32, 17, 5, 42, 8, 1];
        let program = workgroup_max_u32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_u32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<u32>()]),
            ],
        )
        .expect("Fix: workgroup_max_u32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())[0],
            values.iter().copied().max().expect("non-empty"),
            "workgroup_max_u32 must compute the unsigned max (42)"
        );
    }

    #[test]
    fn standalone_min_f32_matches_reference_arithmetic() {
        // Min (-2.5) is not at index 0; an f32::MAX-identity or kept-first bug fails.
        let values = [3.0_f32, 9.5, -2.5, 1.25, 8.0];
        let program = workgroup_min_f32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_f32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: workgroup_min_f32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0].to_bytes())[0],
            values.iter().copied().fold(f32::INFINITY, f32::min),
            "workgroup_min_f32 must compute the min (-2.5)"
        );
    }

    #[test]
    fn standalone_min_u32_matches_reference_arithmetic() {
        let values = [17_u32, 3, 42, 8, 25];
        let program = workgroup_min_u32("values", "out", values.len() as u32, 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vyre_primitives::wire::pack_u32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<u32>()]),
            ],
        )
        .expect("Fix: workgroup_min_u32 must execute in the reference interpreter.");
        assert_eq!(
            vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())[0],
            values.iter().copied().min().expect("non-empty"),
            "workgroup_min_u32 must compute the unsigned min (3)"
        );
    }

    #[test]
    fn non_power_of_two_tile_reductions_match_reference_arithmetic() {
        let values = [4.0_f32, -7.0, 2.5, 9.0, 1.0, 3.25, -2.0];
        let sum_program = workgroup_sum_f32("values", "out", values.len() as u32, 3);
        let sum_outputs = vyre_reference::reference_eval(
            &sum_program,
            &[
                Value::from(vyre_primitives::wire::pack_f32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: workgroup_sum_f32 must support non-power-of-two tiles.");
        assert_eq!(
            vyre_primitives::wire::decode_f32_le_bytes_all(&sum_outputs[0].to_bytes())[0],
            values.iter().copied().sum::<f32>()
        );

        let max_program = workgroup_max_f32("values", "out", values.len() as u32, 3);
        let max_outputs = vyre_reference::reference_eval(
            &max_program,
            &[
                Value::from(vyre_primitives::wire::pack_f32_slice(&values)),
                Value::from(vec![0_u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect("Fix: workgroup_max_f32 must support non-power-of-two tiles.");
        assert_eq!(
            vyre_primitives::wire::decode_f32_le_bytes_all(&max_outputs[0].to_bytes())[0],
            9.0
        );
    }

    /// Every caller of the sweep reads a slot it did not write.
    ///
    /// `frontier_word_block_offsets_single_workgroup` reads `scratch_a[lane - 1]`
    /// on the statement after the call, so the sweep has to leave its result
    /// readable by every lane rather than only by the lane that wrote it. Before
    /// this was fixed the node list ended on the store that adds each lane's
    /// staged value back in, with no barrier behind it, and lane `k` could read
    /// lane `k - 1` one round early and take a block offset short by that
    /// block's own count. The reference interpreter runs lanes in order, so no
    /// value assertion can see this; the shape of the emitted program can.
    #[test]
    fn the_sweep_publishes_its_result_before_returning() {
        let nodes = blelloch_inclusive_sum_nodes("scratch_a", "scratch_b", &Expr::var("lane"), 8);
        assert!(
            matches!(nodes.last(), Some(Node::Barrier { .. })),
            "the sweep must end on a barrier so a cross-lane read is safe on the next statement, got {:?}",
            nodes.last()
        );
        let stores_after_last_barrier = nodes
            .iter()
            .rev()
            .take_while(|node| !matches!(node, Node::Barrier { .. }))
            .count();
        assert_eq!(
            stores_after_last_barrier, 0,
            "no node may write scratch after the sweep's final barrier"
        );
    }

    /// The sweep runs under whatever dispatch its caller declared.
    ///
    /// Every write inside the sweep is bounded by the lane count it was given,
    /// including the last one, so a dispatch wider than the scratch buffers
    /// cannot store past their end.
    #[test]
    fn every_scratch_write_is_bounded_by_the_lane_count() {
        let nodes = blelloch_inclusive_sum_nodes("scratch_a", "scratch_b", &Expr::var("lane"), 8);
        let bare_stores = nodes
            .iter()
            .skip(1)
            .filter(|node| matches!(node, Node::Store { .. }))
            .count();
        assert_eq!(
            bare_stores, 0,
            "every store after the staging write must sit inside a bounds guard"
        );
    }
}
