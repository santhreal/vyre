//! SSA dominance-frontier lookahead scan.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::ast_ops::AST_ASSIGN;

/// Stable op id for the SSA dominance scan child region.
pub const OP_ID: &str = "vyre-primitives::parsing::ssa_dominance_scan";

/// Emit the bounded lookahead scan that allocates phi nodes when rival
/// assignments to the same variable cross block headers.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ssa_dominance_scan(
    ast_opcodes: &str,
    ast_rights: &str,
    ast_vals: &str,
    block_headers: &str,
    num_nodes: Expr,
    out_phi_nodes: &str,
    out_phi_count: &str,
    t: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("var_id", Expr::load(ast_vals, t.clone())),
        Node::let_bind("blk", Expr::load(block_headers, t.clone())),
        Node::let_bind("active", Expr::bool(true)),
        Node::loop_for(
            "lookahead",
            Expr::add(t.clone(), Expr::u32(1)),
            Expr::add(t.clone(), Expr::u32(64)),
            vec![
                Node::if_then(
                    Expr::ge(Expr::var("lookahead"), num_nodes.clone()),
                    vec![Node::assign("active", Expr::bool(false))],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::var("active"),
                        Expr::lt(Expr::var("lookahead"), num_nodes.clone()),
                    ),
                    vec![
                        Node::let_bind("fwd_op", Expr::load(ast_opcodes, Expr::var("lookahead"))),
                        Node::let_bind("fwd_var", Expr::load(ast_vals, Expr::var("lookahead"))),
                        Node::let_bind(
                            "fwd_blk",
                            Expr::load(block_headers, Expr::var("lookahead")),
                        ),
                        // Combined guard: same variable + rival assignment in a different block.
                        // Merged from two nested if_thens to stay within MAX_DEPTH=6.
                        Node::if_then(
                            Expr::and(
                                Expr::and(
                                    Expr::eq(Expr::var("fwd_op"), Expr::u32(AST_ASSIGN)),
                                    Expr::eq(Expr::var("fwd_var"), Expr::var("var_id")),
                                ),
                                Expr::ne(Expr::var("fwd_blk"), Expr::var("blk")),
                            ),
                            vec![
                                Node::let_bind(
                                    "phi_idx",
                                    Expr::atomic_add(out_phi_count, Expr::u32(0), Expr::u32(4)),
                                ),
                                // Gate the 3 phi-record stores against the output capacity.
                                // `phi_idx` is a SHARED running allocation offset (atomic_add of
                                // 4 per match across ALL lanes); if total matches exceed
                                // phi_words/4 the later offsets run past `out_phi_nodes`. The
                                // reference silently DROPS OOB stores, masking the hazard, but a
                                // real GPU (CUDA does no bounds-checking) corrupts memory. We
                                // skip the store when full while STILL counting via the atomic
                                // above, the canonical GPU append-buffer overflow protocol:
                                // `out_phi_count` keeps rising past capacity so the caller
                                // detects the overflow (count > phi_words) and re-runs with a
                                // larger buffer. NOT a silent fallback (Law 10), the overflow
                                // is loud in the returned count; only the illegal write is
                                // removed. Depth 6 == MAX_DEPTH (the last available level, which
                                // is why the match condition above is merged into one `and`).
                                Node::if_then(
                                    Expr::lt(
                                        Expr::add(Expr::var("phi_idx"), Expr::u32(2)),
                                        Expr::buf_len(out_phi_nodes),
                                    ),
                                    vec![
                                        Node::store(
                                            out_phi_nodes,
                                            Expr::var("phi_idx"),
                                            Expr::var("var_id"),
                                        ),
                                        Node::store(
                                            out_phi_nodes,
                                            Expr::add(Expr::var("phi_idx"), Expr::u32(1)),
                                            Expr::load(ast_rights, t.clone()),
                                        ),
                                        Node::store(
                                            out_phi_nodes,
                                            Expr::add(Expr::var("phi_idx"), Expr::u32(2)),
                                            Expr::load(ast_rights, Expr::var("lookahead")),
                                        ),
                                    ],
                                ),
                                Node::assign("active", Expr::bool(false)),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
}

/// Build the standalone dominance-scan primitive.
#[must_use]
pub fn ssa_dominance_scan_program(num_nodes: u32, phi_words: u32) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    // Control-flow nest the lane guard: `t < num_nodes` must gate the `ast_opcodes[t]`
    // load via an `if_then`, NOT an `Expr::and`: a data-flow AND evaluates both
    // operands, so `load(ast_opcodes, t)` would execute for over-fired lanes
    // (t >= num_nodes; GPU dispatch rounds up to full workgroups), reading OOB (UB on
    // CUDA; silently zero-filled on the reference). Nesting keeps the load inside the
    // in-range branch. This guard is the outermost of the nest that bottoms out at the
    // phi-capacity gate (depth 6 == MAX_DEPTH); the match condition below is merged into
    // one `and` so that gate fits within the depth budget.
    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(num_nodes)),
        vec![Node::if_then(
            Expr::eq(Expr::load("ast_opcodes", t.clone()), Expr::u32(AST_ASSIGN)),
            ssa_dominance_scan(
                "ast_opcodes",
                "ast_rights",
                "ast_vals",
                "block_headers",
                Expr::u32(num_nodes),
                "out_phi_nodes",
                "out_phi_count",
                t,
            ),
        )],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("ast_opcodes", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("ast_rights", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("ast_vals", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("block_headers", 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_nodes),
            BufferDecl::storage("out_phi_nodes", 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(phi_words),
            BufferDecl::storage("out_phi_count", 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [num_nodes.max(1), 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

#[cfg(feature = "inventory-registry")]
fn fixture_u32(words: &[u32]) -> Vec<u8> {
    crate::wire::pack_u32_slice(words)
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || ssa_dominance_scan_program(4, 8),
        Some(|| vec![vec![
            fixture_u32(&[AST_ASSIGN, 0, AST_ASSIGN, 0]),
            fixture_u32(&[10, 0, 20, 0]),
            fixture_u32(&[7, 0, 7, 0]),
            fixture_u32(&[1, 0, 2, 0]),
            fixture_u32(&[0; 8]),
            fixture_u32(&[0]),
        ]]),
        Some(|| vec![vec![
            fixture_u32(&[7, 10, 20, 0, 0, 0, 0, 0]),
            fixture_u32(&[4]),
        ]]),
    )
}
