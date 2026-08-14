//! Kernel skeletons shared by every analysis pass over the encoded Expr arena.
//!
//! Each `*_via_encoded` pass dispatches a Program that reads the same four
//! canonical arena row buffers and writes its own per-Expr verdict. Const-fold
//! and structural-hash additionally share the fused single-workgroup level wave:
//! one dispatch that iterates `level` from `0..=max_depth` with a
//! workgroup-scope barrier between levels, each thread striding over the arena
//! in chunks of `WORKGROUP_X`. Only the per-Expr body and the output buffers
//! differ, so the skeleton lives here and the passes supply those two things.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Workgroup size for every per-Expr arena kernel. One value so the passes
/// stride the arena identically and a reader does not have to check.
pub(super) const WORKGROUP_X: u32 = 256;

/// The four canonical arena row buffers, read-only, at bindings
/// `first_binding ..= first_binding + 3`.
///
/// `expr_kind::*` tags live in `arena_kinds`; `arena_arg0..2` carry the
/// kind-specific payload slots. Most arena kernels bind these first
/// (`first_binding = 0`); the canonical-id kernel binds its hash table ahead of
/// them and passes a base of 3.
pub(super) fn arena_row_buffers(expr_count: u32, first_binding: u32) -> Vec<BufferDecl> {
    let count = expr_count.max(1);
    ["arena_kinds", "arena_arg0", "arena_arg1", "arena_arg2"]
        .into_iter()
        .enumerate()
        .map(|(offset, name)| {
            BufferDecl::storage(
                name,
                first_binding + offset as u32,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(count)
        })
        .collect()
}

/// Build a fused level-wave analysis Program over the encoded arena.
///
/// Buffer layout: bindings 0..=3 are the arena rows, binding 4 is
/// `arena_depths` (RO), binding 5 is `max_depth_buf` (RO, a single u32), and
/// `outputs` supplies bindings 6.. in order.
///
/// Single-workgroup design (`workgroup_size = [WORKGROUP_X, 1, 1]`, grid
/// `[1, 1, 1]`). `expr_count` may exceed `WORKGROUP_X`; each thread strides in
/// chunks of `WORKGROUP_X` per level. Because there is exactly one workgroup, a
/// workgroup-scope `SeqCst` barrier is enough to make stores from level `k`
/// visible to reads at level `k + 1`; no grid-wide sync is needed.
///
/// `max_depth_iter_cap` is the static upper bound on the outer loop in the IR.
/// The real depth is read from `max_depth_buf` at run time and the body is
/// gated when `level > max_depth`, so a generous cap costs empty iterations
/// rather than wrong answers.
pub(super) fn build_fused_level_wave_program(
    expr_count: u32,
    max_depth_iter_cap: u32,
    outputs: Vec<BufferDecl>,
    per_expr_body: Vec<Node>,
) -> Program {
    let count = expr_count.max(1);
    let mut buffers = arena_row_buffers(expr_count, 0);
    buffers.push(
        BufferDecl::storage("arena_depths", 4, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count),
    );
    buffers.push(
        BufferDecl::storage("max_depth_buf", 5, BufferAccess::ReadOnly, DataType::U32)
            .with_count(1),
    );
    buffers.extend(outputs);

    // Stride chunks needed to cover the arena with WORKGROUP_X threads. Static
    // upper bound: the kernel re-checks `i < expr_count` each iteration, so
    // over-shoot is safe.
    let chunk_cap = expr_count.div_ceil(WORKGROUP_X);

    let chunk_loop = Node::loop_for(
        "chunk",
        Expr::u32(0),
        Expr::u32(chunk_cap.max(1)),
        vec![
            Node::let_bind(
                "i",
                Expr::add(
                    Expr::gid_x(),
                    Expr::mul(Expr::var("chunk"), Expr::u32(WORKGROUP_X)),
                ),
            ),
            Node::if_then(
                Expr::lt(Expr::var("i"), Expr::u32(expr_count)),
                vec![
                    Node::let_bind("my_depth", Expr::load("arena_depths", Expr::var("i"))),
                    Node::if_then(
                        Expr::eq(Expr::var("my_depth"), Expr::var("level")),
                        per_expr_body,
                    ),
                ],
            ),
        ],
    );

    let outer = Node::loop_for(
        "level",
        Expr::u32(0),
        Expr::u32(max_depth_iter_cap.max(1)),
        vec![
            // The IR has no early loop exit, so gate the body instead. The
            // barrier still fires on skipped levels; it is cheap.
            Node::let_bind("md", Expr::load("max_depth_buf", Expr::u32(0))),
            Node::if_then(
                Expr::le(Expr::var("level"), Expr::var("md")),
                vec![chunk_loop],
            ),
            Node::Barrier {
                ordering: vyre_foundation::MemoryOrdering::SeqCst,
            },
        ],
    );

    Program::wrapped(buffers, [WORKGROUP_X, 1, 1], vec![outer])
}

/// A single-row `LitU32` arena, the minimum encoding every arena pass accepts.
///
/// Shared so the per-pass unit tests exercise one fixture rather than four
/// near-copies that drift apart on the depth fields.
#[cfg(test)]
pub(super) fn single_lit_u32_arena() -> super::expr_arena::ExprArenaEncoding {
    super::expr_arena::ExprArenaEncoding {
        expr_count: 1,
        kinds: vec![super::expr_arena::expr_kind::LIT_U32],
        arg0: vec![0],
        arg1: vec![0],
        arg2: vec![0],
        depths: vec![0],
        max_depth: 0,
        ..super::expr_arena::ExprArenaEncoding::default()
    }
}
