//! Self-hosted optimizer keystone.
//!
//! The encoder turns a `vyre_foundation::ir::Program` into the canonical
//! 5-buffer `ProgramGraph` ABI shared by every Tier 2.5 graph primitive.
//! Once the IR lives in that shape, optimizer passes are *graph primitives
//! reused as compiler passes*: DCE is `persistent_bfs` reachability, CSE is
//! `union_find` over a structural-hash key, const-fold is `level_wave`
//! bottom-up evaluation. The compiler runs on the same substrate it
//! ships to users.
//!
//! V1 scope: flat-entry Programs only (no nested `If`/`Loop`/`Block`/
//! `Region` scoping) and DCE only. Nested scopes and the CSE/const-fold
//! passes land in V2 against the same encoding.
//!
//! GPU dispatch sits one layer above this crate (driver layer). V1 uses
//! `vyre_primitives::graph::persistent_bfs::cpu_ref` so the encoding can
//! be proven sound against the existing `vyre_foundation` DCE pass before
//! any backend is wired.

pub mod canonicalize_via_encoded;
pub mod const_fold_via_encoded;
pub mod const_prop;
pub mod contracts;
pub mod cross_scope_cse;
pub mod cse_via_encoded;
pub mod dce_program;
pub mod dce_via_encoded;
pub mod dead_branch;
pub mod dispatcher;
pub mod encode;
pub mod expr_arena;
pub mod licm;
pub mod pattern_match_via_encoded;
pub mod pipeline;
pub mod pipeline_resident;
pub mod pipeline_resident_decode;
mod rewrite_walk;
pub mod validate_via_encoded;

use vyre_foundation::ir::Expr;

pub(crate) fn expr_no_atomic(expr: &Expr) -> bool {
    match expr {
        Expr::Atomic { .. } | Expr::Opaque(_) => false,
        Expr::BinOp { left, right, .. } => expr_no_atomic(left) && expr_no_atomic(right),
        Expr::UnOp { operand, .. } => expr_no_atomic(operand),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => expr_no_atomic(cond) && expr_no_atomic(true_val) && expr_no_atomic(false_val),
        Expr::Fma { a, b, c } => expr_no_atomic(a) && expr_no_atomic(b) && expr_no_atomic(c),
        Expr::Load { index, .. } => expr_no_atomic(index),
        Expr::Cast { value, .. } => expr_no_atomic(value),
        Expr::Call { args, .. } => args.iter().all(expr_no_atomic),
        Expr::SubgroupBallot { cond } => expr_no_atomic(cond),
        Expr::SubgroupShuffle { value, lane } => expr_no_atomic(value) && expr_no_atomic(lane),
        Expr::SubgroupReduce { value, .. } => expr_no_atomic(value),
        _ => true,
    }
}

fn rewrite_program_entry(
    program: &vyre_foundation::ir::Program,
    rewrite: impl FnOnce(&[vyre_foundation::ir::Node]) -> Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::Node;

    let new_entry = match program.entry() {
        [Node::Region {
            generator,
            source_region,
            body,
        }] => vec![Node::Region {
            generator: generator.clone(),
            source_region: source_region.clone(),
            body: std::sync::Arc::new(rewrite(body)),
        }],
        entry => rewrite(entry),
    };
    program.with_rewritten_entry(new_entry)
}

fn build_encoded_analysis_program(
    expr_count: u32,
    output_name: &str,
    per_expr_body: Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
    let count = expr_count.max(1);
    let buffers = vec![
        BufferDecl::storage("arena_kinds", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
        BufferDecl::storage("arena_arg0", 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
        BufferDecl::storage("arena_arg1", 2, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
        BufferDecl::storage("arena_arg2", 3, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
        BufferDecl::storage(output_name, 4, BufferAccess::ReadWrite, DataType::U32)
            .with_count(count),
    ];
    let body = vec![
        Node::let_bind("i", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
            per_expr_body,
        ),
    ];
    Program::wrapped(buffers, [256, 1, 1], body)
}

fn run_encoded_analysis_kernel(
    arena: &expr_arena::ExprArenaEncoding,
    dispatcher: &dyn dispatcher::OptimizerDispatcher,
    inputs: &mut Vec<Vec<u8>>,
    output: &mut Vec<u32>,
    build_program: fn(u32) -> vyre_foundation::ir::Program,
    stage: &str,
    output_name: &str,
) -> Result<(), dispatcher::DispatchError> {
    use crate::dispatch_buffers::{
        decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
    };
    let count = arena.expr_count;
    let words = count as usize;
    let output_bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            dispatcher::DispatchError::BadInputs(format!(
                "Fix: {stage} output byte count overflows usize for expr_count={count}."
            ))
        })?;
    ensure_input_slots(inputs, 5);
    write_u32_slice_le_bytes(&mut inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut inputs[3], &arena.arg2);
    write_zero_bytes(&mut inputs[4], output_bytes);

    let outputs = dispatcher.dispatch(
        &build_program(count),
        inputs,
        Some([count.div_ceil(256), 1, 1]),
    )?;
    if outputs.len() != 1 {
        return Err(dispatcher::DispatchError::BackendError(format!(
            "Fix: {stage} dispatch expected exactly one {output_name} output, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(
        &outputs[0],
        words,
        &format!("{stage} {output_name}"),
        output,
    )
}

pub use contracts::{
    cross_crate_perf_contracts, optimization_composition_contracts, optimization_pass_selection,
    optimization_registry, optimization_release_passes,
};
