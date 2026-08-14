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

pub mod algebraic;
pub mod cleanup;
pub(crate) mod driver;
pub mod fusion_cse;
pub mod loops;
pub mod memory;
pub mod specialization;
pub mod sync;

pub(crate) fn expr_is_observably_free(expr: &crate::ir::Expr) -> bool {
    expr_is_observably_free_with(expr, true, false)
}

pub(crate) fn expr_is_observably_free_for_reexecution(
    expr: &crate::ir::Expr,
    allow_subgroup_identity: bool,
) -> bool {
    expr_is_observably_free_with(expr, false, allow_subgroup_identity)
}

pub(crate) fn expr_is_observably_free_with(
    expr: &crate::ir::Expr,
    allow_buffer_metadata: bool,
    allow_subgroup_identity: bool,
) -> bool {
    use crate::ir::Expr;
    match expr {
        Expr::Load { .. }
        | Expr::Atomic { .. }
        | Expr::Call { .. }
        | Expr::Opaque(_)
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. } => false,
        Expr::SubgroupLocalId | Expr::SubgroupSize => allow_subgroup_identity,
        Expr::BufferRef { .. } | Expr::BufLen { .. } => allow_buffer_metadata,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. } => true,
        Expr::BinOp { left, right, .. } => {
            expr_is_observably_free_with(left, allow_buffer_metadata, allow_subgroup_identity)
                && expr_is_observably_free_with(
                    right,
                    allow_buffer_metadata,
                    allow_subgroup_identity,
                )
        }
        Expr::UnOp { operand, .. } => {
            expr_is_observably_free_with(operand, allow_buffer_metadata, allow_subgroup_identity)
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_is_observably_free_with(cond, allow_buffer_metadata, allow_subgroup_identity)
                && expr_is_observably_free_with(
                    true_val,
                    allow_buffer_metadata,
                    allow_subgroup_identity,
                )
                && expr_is_observably_free_with(
                    false_val,
                    allow_buffer_metadata,
                    allow_subgroup_identity,
                )
        }
        Expr::Cast { value, .. } => {
            expr_is_observably_free_with(value, allow_buffer_metadata, allow_subgroup_identity)
        }
        Expr::Fma { a, b, c } => {
            expr_is_observably_free_with(a, allow_buffer_metadata, allow_subgroup_identity)
                && expr_is_observably_free_with(b, allow_buffer_metadata, allow_subgroup_identity)
                && expr_is_observably_free_with(c, allow_buffer_metadata, allow_subgroup_identity)
        }
    }
}
