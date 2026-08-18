//! Single-workgroup atomic scalar reductions.
//!
//! Every reduction here (`Sum`, `Min`, `Max`, `CountNonZero`, `PopcountSum`, `AnyNonZero`,
//! `AllNonZero`) folds a u32 `ValueSet` into ONE output slot via a grid-stride loop and a single
//! atomic accumulator. The kernel is single-workgroup by construction: the `lane == 0` identity
//! init and the WORKGROUP-scoped `SeqCst` barrier only synchronize within one workgroup, so exactly
//! one workgroup (a `[1, 1, 1]` dispatch of `WORKGROUP_SIZE` lanes) is meant to run it.
//!
//! To stay correct even when a caller (or a shape-inferred grid) fires extra workgroups, the whole
//! accumulate loop is gated on `WorkgroupId == 0` (the canonical "first workgroup" predicate, shared
//! with `reduce::workgroup_tree`'s `FirstWorkgroup` scope). Extra workgroups then no-op instead of
//! double-counting the non-idempotent sums. See `atomic_grid_stride_u32`.
//!
//! Performance: this path serializes every element through one global atomic on a single output
//! slot, so at large input sizes the accumulator contention dominates. It is NOT subgroup-lowered
//! (the subgroup-first pass only rewrites `workgroup_sum_`/`workgroup_max_`/`workgroup_min_`
//! generators). For large reductions prefer `reduce::workgroup_tree` (its standalone
//! `workgroup_sum_u32`/`workgroup_max_u32`/... builders), which reduce per-lane partials in
//! workgroup memory and lower to native `subgroup_add`/`subgroup_reduce` on capable backends. Use
//! this atomic path for small `ValueSet`s or where a single-atomic kernel is simpler than staging
//! scratch.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

pub(crate) const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AtomicReduceKind {
    Sum,
    Min,
    Max,
    CountNonZero,
    PopcountSum,
    AnyNonZero,
    AllNonZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AtomicBoolReduceKind {
    AnyNonZero,
    AllNonZero,
}

impl From<AtomicBoolReduceKind> for AtomicReduceKind {
    fn from(kind: AtomicBoolReduceKind) -> Self {
        match kind {
            AtomicBoolReduceKind::AnyNonZero => Self::AnyNonZero,
            AtomicBoolReduceKind::AllNonZero => Self::AllNonZero,
        }
    }
}

impl AtomicBoolReduceKind {
    pub(crate) fn identity(self) -> u32 {
        AtomicReduceKind::from(self).identity()
    }

    pub(crate) fn atomic(self, out: &str, value: Expr) -> Expr {
        AtomicReduceKind::from(self).atomic(out, value)
    }

    pub(crate) const fn laws(self) -> &'static [&'static str] {
        match self {
            Self::AnyNonZero => &[
                "absorbing",
                "associative",
                "commutative",
                "idempotent",
                "lattice-absorption",
            ],
            Self::AllNonZero => &[
                "absorbing",
                "associative",
                "commutative",
                "distributive",
                "idempotent",
            ],
        }
    }
}

impl AtomicReduceKind {
    pub(crate) fn identity(self) -> u32 {
        match self {
            Self::Sum | Self::Max | Self::CountNonZero | Self::PopcountSum | Self::AnyNonZero => 0,
            Self::Min => u32::MAX,
            Self::AllNonZero => 1,
        }
    }

    pub(crate) fn value(self, input: &str, index: Expr) -> Expr {
        let loaded = Expr::load(input, index);
        match self {
            Self::CountNonZero | Self::AnyNonZero | Self::AllNonZero => {
                Expr::select(Expr::ne(loaded, Expr::u32(0)), Expr::u32(1), Expr::u32(0))
            }
            Self::PopcountSum => Expr::UnOp {
                op: UnOp::Popcount,
                operand: Box::new(loaded),
            },
            Self::Sum | Self::Min | Self::Max => loaded,
        }
    }

    pub(crate) fn atomic(self, out: &str, value: Expr) -> Expr {
        match self {
            Self::Sum | Self::CountNonZero | Self::PopcountSum => {
                Expr::atomic_add(out, Expr::u32(0), value)
            }
            Self::Min => Expr::atomic_min(out, Expr::u32(0), value),
            Self::Max => Expr::atomic_max(out, Expr::u32(0), value),
            Self::AnyNonZero => Expr::atomic_or(out, Expr::u32(0), value),
            Self::AllNonZero => Expr::atomic_and(out, Expr::u32(0), value),
        }
    }

    pub(crate) const fn laws(self) -> &'static [&'static str] {
        match self {
            Self::Sum => &["associative", "commutative", "identity"],
            Self::Min => &["absorbing", "associative", "commutative", "idempotent"],
            Self::Max => &[
                "absorbing",
                "associative",
                "commutative",
                "idempotent",
                "identity",
            ],
            Self::CountNonZero => &["bounded", "monotonic"],
            Self::PopcountSum => &["associative", "bounded"],
            Self::AnyNonZero => &[
                "absorbing",
                "associative",
                "commutative",
                "idempotent",
                "lattice-absorption",
            ],
            Self::AllNonZero => &[
                "absorbing",
                "associative",
                "commutative",
                "distributive",
                "idempotent",
            ],
        }
    }

    pub(crate) fn reference_reduce(self, values: &[u32]) -> u32 {
        match self {
            Self::Sum => values.iter().copied().fold(0u32, |a, b| a.wrapping_add(b)),
            Self::Min => values.iter().copied().fold(u32::MAX, u32::min),
            Self::Max => values.iter().copied().fold(0u32, u32::max),
            Self::CountNonZero => values.iter().filter(|&&v| v != 0).count() as u32,
            Self::PopcountSum => values.iter().map(|&w| w.count_ones()).sum(),
            Self::AnyNonZero => u32::from(values.iter().any(|&v| v != 0)),
            Self::AllNonZero => u32::from(!values.is_empty() && values.iter().all(|&v| v != 0)),
        }
    }
}

/// Typed builder for atomic scalar reductions over a u32 input buffer.
#[derive(Debug, Clone)]
pub(crate) struct AtomicReductionBuilder<'a> {
    pub(crate) op_id: &'static str,
    pub(crate) input: &'a str,
    pub(crate) out: &'a str,
    pub(crate) count: u32,
    pub(crate) kind: AtomicReduceKind,
}

impl<'a> AtomicReductionBuilder<'a> {
    #[must_use]
    pub(crate) const fn new(
        op_id: &'static str,
        input: &'a str,
        out: &'a str,
        count: u32,
        kind: AtomicReduceKind,
    ) -> Self {
        Self {
            op_id,
            input,
            out,
            count,
            kind,
        }
    }

    #[must_use]
    pub(crate) fn build(self) -> Program {
        atomic_reduce_u32(self.input, self.out, self.count, self.kind, self.op_id)
    }
}

pub(crate) fn atomic_reduce_u32(
    input: &str,
    out: &str,
    count: u32,
    kind: AtomicReduceKind,
    op_id: &'static str,
) -> Program {
    atomic_grid_stride_u32(
        &[input],
        out,
        count,
        kind.identity(),
        |index| kind.value(input, index),
        |out, value| kind.atomic(out, value),
        op_id,
    )
}

pub(crate) fn atomic_nonzero_bool_reduce_u32(
    input: &str,
    out: &str,
    count: u32,
    kind: AtomicBoolReduceKind,
    op_id: &'static str,
) -> Program {
    atomic_reduce_u32(input, out, count, kind.into(), op_id)
}
/// Build a grid-stride loop that folds one value per element into `out[0]`
/// through a single atomic.
///
/// `inputs` are the read-only u32 buffers the value expression reads, bound in
/// order from zero, with `out` bound after them. `value` receives the element
/// index and closes over whichever inputs it reads, so a relation over two
/// buffers and a reduction over one share this one shape.
pub(crate) fn atomic_grid_stride_u32<V, A>(
    inputs: &[&str],
    out: &str,
    count: u32,
    identity: u32,
    value: V,
    atomic: A,
    op_id: &'static str,
) -> Program
where
    V: Fn(Expr) -> Expr,
    A: Fn(&str, Expr) -> Expr,
{
    let lane = Expr::InvocationId { axis: 0 };
    let chunk_count = Expr::div(
        Expr::add(Expr::u32(count), Expr::u32(WORKGROUP_SIZE - 1)),
        Expr::u32(WORKGROUP_SIZE),
    );

    let body = vec![
        Node::if_then(
            Expr::eq(lane.clone(), Expr::u32(0)),
            vec![Node::store(out, Expr::u32(0), Expr::u32(identity))],
        ),
        Node::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        },
        // Gate the ENTIRE grid-stride loop on `WorkgroupId == 0`, the canonical "first workgroup"
        // predicate this codebase already uses for single-workgroup reductions (see
        // `reduce::workgroup_tree::WorkgroupReductionScope::FirstWorkgroup` and the subgroup lowering
        // pass). The predicate is loop-invariant, so it gates the loop once rather than being
        // re-tested every chunk. This makes the reduction correct under ANY dispatch grid: only the
        // first workgroup accumulates and any extra workgroups skip the whole loop as a no-op. Absent
        // it, a caller (or the reference interpreter's buffer-shape grid inference) that fires
        // `ceil(count/256)` workgroups for `count > 256` would have every extra workgroup re-run the
        // grid-stride and DOUBLE-COUNT the non-idempotent Sum/PopcountSum/CountNonZero; idempotent
        // Max/Min/Or/And silently absorb it. The kernel is single-workgroup by construction (the
        // `lane == 0` identity init + the WORKGROUP-scoped `SeqCst` barrier), and this fails extra
        // workgroups closed. The `lane == 0` init and the barrier stay unconditional so the barrier
        // is reached uniformly by every lane in every workgroup.
        Node::if_then(
            Expr::is_first_workgroup(),
            vec![Node::loop_for(
                "chunk",
                Expr::u32(0),
                chunk_count,
                vec![
                    Node::let_bind(
                        "i",
                        Expr::add(
                            Expr::mul(Expr::var("chunk"), Expr::u32(WORKGROUP_SIZE)),
                            lane.clone(),
                        ),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("i"), Expr::u32(count)),
                        vec![Node::let_bind(
                            "_acc_prev",
                            atomic(out, value(Expr::var("i"))),
                        )],
                    ),
                ],
            )],
        ),
    ];

    let mut buffers: Vec<BufferDecl> = inputs
        .iter()
        .enumerate()
        .map(|(binding, name)| {
            BufferDecl::storage(
                name,
                u32::try_from(binding).unwrap_or(u32::MAX),
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(count)
        })
        .collect();
    buffers.push(
        BufferDecl::storage(
            out,
            u32::try_from(inputs.len()).unwrap_or(u32::MAX),
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );

    Program::wrapped(
        buffers,
        [WORKGROUP_SIZE, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}
