//! Workgroup-local tree reductions over scratch buffers.
//!
//! These helpers are shared blocks for higher-level library ops that
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

/// Emit a child region for a workgroup reduction parameterized by combine op.
#[must_use]
pub fn workgroup_reduction_child<F>(
    op_id: &'static str,
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
    combine: F,
) -> Node
where
    F: Fn(Expr, Expr) -> Expr,
{
    child_region(
        op_id,
        parent_op_id,
        tree_body(tile, scratch, scope, combine),
    )
}

/// Emit a child region that sums f32 lane partials in `scratch`.
#[must_use]
pub fn sum_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(SUM_F32_OP_ID, parent_op_id, tile, scratch, scope, Expr::add)
}

/// Emit a child region that sums u32 lane partials in `scratch`.
#[must_use]
pub fn sum_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(SUM_U32_OP_ID, parent_op_id, tile, scratch, scope, Expr::add)
}

/// Emit a child region that maximizes f32 lane partials in `scratch`.
#[must_use]
pub fn max_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(MAX_F32_OP_ID, parent_op_id, tile, scratch, scope, Expr::max)
}

/// Emit a child region that maximizes u32 lane partials in `scratch`.
#[must_use]
pub fn max_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(MAX_U32_OP_ID, parent_op_id, tile, scratch, scope, Expr::max)
}

/// Emit a child region that minimizes f32 lane partials in `scratch`.
#[must_use]
pub fn min_f32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(MIN_F32_OP_ID, parent_op_id, tile, scratch, scope, Expr::min)
}

/// Emit a child region that minimizes u32 lane partials in `scratch`.
#[must_use]
pub fn min_u32_child(
    parent_op_id: &str,
    tile: u32,
    scratch: &'static str,
    scope: WorkgroupReductionScope,
) -> Node {
    workgroup_reduction_child(MIN_U32_OP_ID, parent_op_id, tile, scratch, scope, Expr::min)
}

/// Build a standalone f32 workgroup sum Program.
#[must_use]
pub fn workgroup_sum_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    WorkgroupReductionBuilder::new(
        SUM_F32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::F32,
        WorkgroupFold::Sum,
    )
    .build()
}

/// Build a standalone u32 workgroup sum Program.
#[must_use]
pub fn workgroup_sum_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    WorkgroupReductionBuilder::new(
        SUM_U32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::U32,
        WorkgroupFold::Sum,
    )
    .build()
}

/// Build a standalone f32 workgroup maximum Program.
#[must_use]
pub fn workgroup_max_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    WorkgroupReductionBuilder::new(
        MAX_F32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::F32,
        WorkgroupFold::Max,
    )
    .build()
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
    WorkgroupReductionBuilder::new(
        MAX_U32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::U32,
        WorkgroupFold::Max,
    )
    .build()
}

/// Build a standalone f32 workgroup minimum Program.
///
/// `f32::MAX` is the neutral for a minimum (any real value is smaller). The
/// subgroup-first lowering recognizes the `workgroup_min_` prefix, so this gets
/// the native warp `subgroupMin` / `redux.sync.min` path on capable backends.
#[must_use]
pub fn workgroup_min_f32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    WorkgroupReductionBuilder::new(
        MIN_F32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::F32,
        WorkgroupFold::Min,
    )
    .build()
}

/// Build a standalone u32 workgroup minimum Program.
///
/// `u32::MAX` is the neutral for an unsigned minimum.
#[must_use]
pub fn workgroup_min_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    WorkgroupReductionBuilder::new(
        MIN_U32_OP_ID,
        values,
        out,
        count,
        tile,
        DataType::U32,
        WorkgroupFold::Min,
    )
    .build()
}

/// Which value a workgroup reduction folds its lanes down to.
///
/// The fold decides three things at once - the identity a lane starts from, how
/// two lanes combine, and which tree body the sweep emits - and the three must
/// agree. Passed as three separate arguments they could disagree, and a `max`
/// combine over a `sum` tree is a silently wrong reduction; naming the fold
/// once makes that unstateable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupFold {
    /// Add the lanes.
    Sum,
    /// Keep the largest lane.
    Max,
    /// Keep the smallest lane.
    Min,
}

impl WorkgroupFold {
    /// Value a lane accumulates from before it reads anything.
    #[must_use]
    pub fn identity(self, dtype: &DataType) -> Expr {
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
    #[must_use]
    pub fn combine(self, left: Expr, right: Expr) -> Expr {
        match self {
            Self::Sum => Expr::add(left, right),
            Self::Max => Expr::max(left, right),
            Self::Min => Expr::min(left, right),
        }
    }

    /// Tree sweep over the staged scratch buffer.
    #[must_use]
    pub fn tree(self, tile: u32, scratch: &'static str) -> Vec<Node> {
        let scope = WorkgroupReductionScope::FirstWorkgroup;
        match self {
            Self::Sum => sum_body(tile, scratch, scope),
            Self::Max => max_body(tile, scratch, scope),
            Self::Min => min_body(tile, scratch, scope),
        }
    }

    /// Algebraic laws satisfied by this fold kind.
    #[must_use]
    pub const fn laws(self) -> &'static [&'static str] {
        match self {
            Self::Sum => &["associative", "commutative", "identity"],
            Self::Max => &[
                "absorbing",
                "associative",
                "commutative",
                "idempotent",
                "identity",
            ],
            Self::Min => &["absorbing", "associative", "commutative", "idempotent"],
        }
    }
}

/// Typed builder for standalone workgroup reductions parameterized by DataType and WorkgroupFold.
#[derive(Debug, Clone)]
pub struct WorkgroupReductionBuilder<'a> {
    /// Region generator id the emitted Program carries.
    pub op_id: &'static str,
    /// Input buffer the lanes read.
    pub values: &'a str,
    /// Single-slot output buffer the reduction writes.
    pub out: &'a str,
    /// Elements in `values`.
    pub count: u32,
    /// Lanes in the workgroup, and slots in the scratch buffer.
    pub tile: u32,
    /// Element type of both buffers.
    pub dtype: DataType,
    /// Which reduction to emit.
    pub fold: WorkgroupFold,
}

impl<'a> WorkgroupReductionBuilder<'a> {
    /// Construct a new typed workgroup reduction builder.
    #[must_use]
    pub const fn new(
        op_id: &'static str,
        values: &'a str,
        out: &'a str,
        count: u32,
        tile: u32,
        dtype: DataType,
        fold: WorkgroupFold,
    ) -> Self {
        Self {
            op_id,
            values,
            out,
            count,
            tile,
            dtype,
            fold,
        }
    }

    /// Assemble the reduction into a final [`Program`].
    #[must_use]
    pub fn build(self) -> Program {
        let Self {
            op_id,
            values,
            out,
            count,
            tile,
            dtype,
            fold,
        } = self;
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
                BufferDecl::storage(values, 0, BufferAccess::ReadOnly, dtype.clone())
                    .with_count(count),
                BufferDecl::workgroup(scratch, tile, dtype.clone()),
                BufferDecl::output(out, 1, dtype).with_count(1),
            ],
            [tile, 1, 1],
            vec![wrap_anonymous_region(op_id, body)],
        )
    }
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
        ]]),
        Some(|| vec![vec![vec![0x00, 0x00, 0x00, 0x41]]]), // 8.0f32
    )
    .with_laws(WorkgroupFold::Sum.laws())
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SUM_U32_OP_ID,
        || workgroup_sum_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[1, 2, 3, 4]),
        ]]),
        Some(|| vec![vec![vec![0x0a, 0x00, 0x00, 0x00]]]), // 10u32
    )
    .with_laws(WorkgroupFold::Sum.laws())
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MAX_F32_OP_ID,
        || workgroup_max_f32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_f32(&[-3.0, 9.5, 4.0, 1.25]),
        ]]),
        Some(|| vec![vec![vec![0x00, 0x00, 0x18, 0x41]]]), // 9.5f32
    )
    .with_laws(WorkgroupFold::Max.laws())
}
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MAX_U32_OP_ID,
        || workgroup_max_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[1, 9, 4, 2]),
        ]]),
        Some(|| vec![vec![vec![0x09, 0x00, 0x00, 0x00]]]), // 9u32
    )
    .with_laws(WorkgroupFold::Max.laws())
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MIN_F32_OP_ID,
        || workgroup_min_f32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_f32(&[-3.0, 9.5, 4.0, 1.25]),
        ]]),
        Some(|| vec![vec![vec![0x00, 0x00, 0x40, 0xc0]]]), // -3.0f32
    )
    .with_laws(WorkgroupFold::Min.laws())
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        MIN_U32_OP_ID,
        || workgroup_min_u32("values", "out", 4, 4),
        Some(|| vec![vec![
            fixture_u32(&[3, 9, 4, 2]),
        ]]),
        Some(|| vec![vec![vec![0x02, 0x00, 0x00, 0x00]]]), // 2u32
    )
    .with_laws(WorkgroupFold::Min.laws())
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

    #[test]
    fn typed_builder_constructs_all_six_workgroup_reductions() {
        let sum_f = WorkgroupReductionBuilder::new(
            SUM_F32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::F32,
            WorkgroupFold::Sum,
        )
        .build();
        assert_eq!(sum_f.workgroup_size(), [4, 1, 1]);
        assert_eq!(
            WorkgroupFold::Sum.laws(),
            &["associative", "commutative", "identity"]
        );

        let sum_u = WorkgroupReductionBuilder::new(
            SUM_U32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::U32,
            WorkgroupFold::Sum,
        )
        .build();
        assert_eq!(sum_u.workgroup_size(), [4, 1, 1]);

        let max_f = WorkgroupReductionBuilder::new(
            MAX_F32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::F32,
            WorkgroupFold::Max,
        )
        .build();
        assert_eq!(max_f.workgroup_size(), [4, 1, 1]);
        assert_eq!(
            WorkgroupFold::Max.laws(),
            &[
                "absorbing",
                "associative",
                "commutative",
                "idempotent",
                "identity"
            ]
        );

        let max_u = WorkgroupReductionBuilder::new(
            MAX_U32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::U32,
            WorkgroupFold::Max,
        )
        .build();
        assert_eq!(max_u.workgroup_size(), [4, 1, 1]);

        let min_f = WorkgroupReductionBuilder::new(
            MIN_F32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::F32,
            WorkgroupFold::Min,
        )
        .build();
        assert_eq!(min_f.workgroup_size(), [4, 1, 1]);
        assert_eq!(
            WorkgroupFold::Min.laws(),
            &["absorbing", "associative", "commutative", "idempotent"]
        );

        let min_u = WorkgroupReductionBuilder::new(
            MIN_U32_OP_ID,
            "values",
            "out",
            16,
            4,
            DataType::U32,
            WorkgroupFold::Min,
        )
        .build();
        assert_eq!(min_u.workgroup_size(), [4, 1, 1]);
    }
}
