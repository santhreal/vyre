//! Registered reference optimizer passes.
//!
//! Layout (audit cleanup A3, 2026-04-30): passes are grouped into category
//! subdirs aligned with the Phase 4 catalog buckets so the directory scales
//! to ~250 named transforms without becoming an unreviewable flat dir:
//!
//! - `algebraic/` (Phase 4A)  -  `const_fold`, `strength_reduce`,
//!   canonicalize, `normalize_atomics`
//! - `loops/` (Phase 4B)  -  `loop_unroll`, `loop_trip_zero_eliminate`
//! - `memory/` (Phase 4C)  -  `const_buffer_fold`, `dead_buffer_elim`,
//!   `read_only_load_hoist`, `vectorization`, `decode_scan_fuse`
//! - `sync/` (Phase 4D)  -  `barrier_coalesce`
//! - `fusion_cse/`  -  fusion, `fuse_cse`, cse, dce
//! - `cleanup/`  -  `empty_block_collapse`, `region_inline`,
//!   `if_constant_branch_eliminate`, `noop_assign_eliminate`,
//!   `region_promote_singleton_block`, `buffer_decl_sort`
//! - `specialization/` (Phase 4G)  -  autotune
//!
//! Backend-specific lowering strategy code belongs in the concrete driver
//! crates. Foundation passes are math- and IR-structural rewrites that any
//! backend can inherit before target emission.
//!

use crate::ir::Ident;
use rustc_hash::FxHashSet;

pub mod algebraic;
pub mod cleanup;
pub(crate) mod driver;
pub mod fusion_cse;
pub mod loops;
pub mod memory;
pub mod specialization;
pub mod sync;

/// What a caller has already proved about the point an expression is moving to.
///
/// The rewrites that ask whether an expression may be re-evaluated somewhere
/// else differ only in which facts they hold, not in how expressions are
/// classified. Carrying the facts as one value keeps [`classify`] the single
/// owner of the classification.
#[derive(Clone, Copy)]
struct ReexecutionRules<'a> {
    /// `BufferRef` and `BufLen` read buffer metadata rather than contents, so a
    /// rewrite moving an expression within one invocation may carry them; one
    /// re-evaluating it at a new program point may not.
    allow_buffer_metadata: bool,
    /// `SubgroupLocalId` and `SubgroupSize` are invariant within a subgroup, so
    /// a rewrite that stays inside one may carry them.
    allow_subgroup_identity: bool,
    /// Buffers a `Load` may read. A read-only buffer is written by nothing in a
    /// valid program, so its load answers the same value wherever it runs.
    loadable: Option<&'a FxHashSet<Ident>>,
}

pub(crate) fn expr_is_observably_free(expr: &crate::ir::Expr) -> bool {
    classify(
        expr,
        ReexecutionRules {
            allow_buffer_metadata: true,
            allow_subgroup_identity: false,
            loadable: None,
        },
    )
}

pub(crate) fn expr_is_observably_free_for_reexecution(
    expr: &crate::ir::Expr,
    allow_subgroup_identity: bool,
) -> bool {
    classify(
        expr,
        ReexecutionRules {
            allow_buffer_metadata: false,
            allow_subgroup_identity,
            loadable: None,
        },
    )
}

/// Re-executable at a new program point, allowing a `Load` from a buffer in
/// `read_only`.
///
/// Loop-invariant hoisting is the caller: it moves a binding out of a loop
/// body, where a load from a buffer the program declares read-only reads the
/// same element it read on every iteration.
pub(crate) fn expr_is_reexecutable_over_read_only_loads(
    expr: &crate::ir::Expr,
    read_only: &FxHashSet<Ident>,
) -> bool {
    classify(
        expr,
        ReexecutionRules {
            allow_buffer_metadata: false,
            allow_subgroup_identity: false,
            loadable: Some(read_only),
        },
    )
}

/// Whether `expr` contains no `Atomic` and no `Opaque` anywhere.
///
/// A weaker question than `expr_is_observably_free_for_reexecution`, and a
/// different one: a `Load`, a `Call` or a subgroup op is allowed here, because
/// the rewrites that ask this are deleting or hoisting a whole expression
/// within one invocation rather than re-evaluating it at a new program point.
/// What they cannot do is discard a memory mutation or an extension whose
/// effects the IR does not model, which is exactly these two variants.
///
/// Exhaustive with no catch-all arm: a new `Expr` variant fails to compile here
/// rather than reading as atomic-free and letting a rewrite drop its operands.
pub fn expr_is_atomic_free(expr: &crate::ir::Expr) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::Atomic { .. } | Expr::Opaque(_) => false,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. } => true,
        Expr::BinOp { left, right, .. } => expr_is_atomic_free(left) && expr_is_atomic_free(right),
        Expr::UnOp { operand, .. } => expr_is_atomic_free(operand),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_is_atomic_free(cond)
                && expr_is_atomic_free(true_val)
                && expr_is_atomic_free(false_val)
        }
        Expr::Fma { a, b, c } => {
            expr_is_atomic_free(a) && expr_is_atomic_free(b) && expr_is_atomic_free(c)
        }
        Expr::Load { index, .. } => expr_is_atomic_free(index),
        Expr::Cast { value, .. } => expr_is_atomic_free(value),
        Expr::Call { args, .. } => args.iter().all(expr_is_atomic_free),
        Expr::SubgroupBallot { cond } => expr_is_atomic_free(cond),
        Expr::SubgroupShuffle { value, lane } => {
            expr_is_atomic_free(value) && expr_is_atomic_free(lane)
        }
        Expr::SubgroupReduce { value, .. } => expr_is_atomic_free(value),
    }
}

pub(crate) fn expr_is_observably_free_with(
    expr: &crate::ir::Expr,
    allow_buffer_metadata: bool,
    allow_subgroup_identity: bool,
) -> bool {
    classify(
        expr,
        ReexecutionRules {
            allow_buffer_metadata,
            allow_subgroup_identity,
            loadable: None,
        },
    )
}

/// The one walk that decides whether an expression may be evaluated somewhere
/// other than where it is written.
///
/// Exhaustive with no catch-all arm: a new `Expr` variant fails to compile here
/// rather than reading as free of effects and licensing a move no one judged.
fn classify(expr: &crate::ir::Expr, rules: ReexecutionRules<'_>) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::Atomic { .. }
        | Expr::Call { .. }
        | Expr::Opaque(_)
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. } => false,
        Expr::Load { buffer, index } => {
            rules
                .loadable
                .is_some_and(|allowed| allowed.contains(buffer))
                && classify(index, rules)
        }
        Expr::SubgroupLocalId | Expr::SubgroupSize => rules.allow_subgroup_identity,
        Expr::BufferRef { .. } | Expr::BufLen { .. } => rules.allow_buffer_metadata,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. } => true,
        Expr::BinOp { left, right, .. } => classify(left, rules) && classify(right, rules),
        Expr::UnOp { operand, .. } => classify(operand, rules),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => classify(cond, rules) && classify(true_val, rules) && classify(false_val, rules),
        Expr::Cast { value, .. } => classify(value, rules),
        Expr::Fma { a, b, c } => classify(a, rules) && classify(b, rules) && classify(c, rules),
    }
}
