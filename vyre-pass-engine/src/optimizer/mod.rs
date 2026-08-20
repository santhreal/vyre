//! The optimizer's passes, executed as dispatched vyre Programs.
//!
//! The encoder turns a `vyre_foundation::ir::Program` into the canonical
//! 5-buffer `ProgramGraph` ABI shared by every graph primitive.
//! Once the IR lives in that shape, optimizer passes are *graph primitives
//! reused as compiler passes*: DCE is `persistent_bfs` reachability, CSE is
//! `union_find` over a structural-hash key, const-fold is `level_wave`
//! bottom-up evaluation. The compiler runs on the primitives it ships.
//!
//! Nested scopes are in scope: the encoder walks `If`, `Loop`, `Block`, and
//! `Region` bodies in prefix DFS, each with its own scope frame, and gives
//! every visited Node a graph-node id.
//!
//! Host-side rewrites over the returned Program are IR structure, not
//! encoded-order dispatch, so they live in `vyre_foundation::transform`:
//! const propagation, dead-branch elimination, and loop-invariant code motion.
//! Cross-scope CSE stays here, in `cse_via_encoded`, because it reads the
//! canonical ids the dispatched CSE kernel produced.
//!
//! Dispatch belongs to the caller. Every `*_via_encoded` entry point takes a
//! `vyre_foundation::program_dispatch::ProgramDispatcher`, so one pass runs
//! against `vyre_driver_reference::ReferenceEvalDispatcher` in the parity suites
//! and against a backend dispatcher in production, from the same Program.

mod arena_cursor;
mod arena_kernel;
pub mod canonicalize_via_encoded;
pub mod const_fold_via_encoded;
// `cse_via_encoded` is the public path for cross-scope CSE and for the analysis
// Programs it dispatches. These two submodules exist because that file was
// split, so they stay private and the owner re-exports what they hold; a second
// public path is a second name for one item.
mod cse_cross_scope;
mod cse_programs;
pub mod cse_via_encoded;
pub mod dce_program;
pub mod dce_via_encoded;
pub mod encode;
pub mod expr_arena;
pub mod pattern_match_via_encoded;
pub mod pipeline;
pub mod pipeline_resident;
pub mod pipeline_resident_decode;
mod rewrite_walk;
pub mod validate_via_encoded;

fn build_encoded_analysis_program(
    expr_count: u32,
    output_name: &str,
    per_expr_body: Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
    let count = expr_count.max(1);
    let mut buffers = arena_kernel::arena_row_buffers(expr_count, 0);
    buffers.push(
        BufferDecl::storage(output_name, 4, BufferAccess::ReadWrite, DataType::U32)
            .with_count(count),
    );
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
    dispatcher: &dyn vyre_foundation::program_dispatch::ProgramDispatcher,
    inputs: &mut Vec<Vec<u8>>,
    output: &mut Vec<u32>,
    build_program: fn(u32) -> vyre_foundation::ir::Program,
    stage: &str,
    output_name: &str,
) -> Result<(), vyre_foundation::program_dispatch::DispatchError> {
    use vyre_libs::dispatch_buffers::{
        decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
    };
    let count = arena.expr_count;
    let words = count as usize;
    let output_bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            vyre_foundation::program_dispatch::DispatchError::BadInputs(format!(
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
        return Err(
            vyre_foundation::program_dispatch::DispatchError::BackendError(format!(
                "Fix: {stage} dispatch expected exactly one {output_name} output, got {}.",
                outputs.len()
            )),
        );
    }
    decode_u32_output_exact(
        &outputs[0],
        words,
        &format!("{stage} {output_name}"),
        output,
    )
}
