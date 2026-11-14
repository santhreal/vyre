//! The IR the exploded-supergraph CSR build emits.
//!
//! Count, prefix, and fill run on invocation zero because the row offsets must
//! be produced in a fixed order for the output bytes to be deterministic; the
//! grid in [`super::abi`] holds the launch to one block for that reason.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use super::abi::{IFDS_CSR_WORKGROUP_SIZE, OP_ID};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Build a GPU Program that emits the exploded-supergraph CSR.
///
/// Deterministic CSR construction: build a dense kill bitmap, count each
/// source row, prefix row counts, then fill `col_idx`. Count/prefix/fill
/// still run on invocation `0`, and the equality guard on that invocation is
/// what holds the launch to one block for the serial region.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_ifds_csr_program(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra_count: u32,
    inter_count: u32,
    gen_count: u32,
    kill_count: u32,
    max_col_count: u32,
) -> Program {
    let trap = |msg: String| trap_program(OP_ID, Some(("row_ptr", DataType::U32)), msg);
    if num_procs == 0 || blocks_per_proc == 0 || facts_per_proc == 0 {
        return trap(format!(
            "Fix: exploded IFDS dimensions must be nonzero, got procs={num_procs}, blocks={blocks_per_proc}, facts={facts_per_proc}."
        ));
    }
    let Some(slots_per_proc) = blocks_per_proc.checked_mul(facts_per_proc) else {
        return trap("Fix: exploded IFDS slots_per_proc overflowed u32.".into());
    };
    let Some(total_nodes) = num_procs.checked_mul(slots_per_proc) else {
        return trap("Fix: exploded IFDS total node count overflowed u32.".into());
    };
    let Some(row_ptr_count) = total_nodes.checked_add(1) else {
        return trap(format!(
            "Fix: exploded IFDS total_nodes={total_nodes} overflows row_ptr count. Shard the IFDS graph before GPU dispatch."
        ));
    };

    let idx_expr = |p: Expr, b: Expr, f: Expr| {
        Expr::add(
            Expr::add(
                Expr::mul(p, Expr::u32(slots_per_proc)),
                Expr::mul(b, Expr::u32(facts_per_proc)),
            ),
            f,
        )
    };
    let in_proc_block = |p: Expr, b: Expr| {
        Expr::and(
            Expr::lt(p, Expr::u32(num_procs)),
            Expr::lt(b, Expr::u32(blocks_per_proc)),
        )
    };
    let count_row = |src: Expr| {
        Node::store(
            "row_ptr",
            Expr::add(src.clone(), Expr::u32(1)),
            Expr::add(
                Expr::load("row_ptr", Expr::add(src, Expr::u32(1))),
                Expr::u32(1),
            ),
        )
    };
    let fill_pair = |src_expr: Expr, dst_expr: Expr| {
        vec![
            Node::let_bind("src_dense", src_expr),
            Node::let_bind("dst_dense", dst_expr),
            Node::let_bind(
                "emit_slot",
                Expr::load("row_cursor", Expr::var("src_dense")),
            ),
            Node::store("col_idx", Expr::var("emit_slot"), Expr::var("dst_dense")),
            Node::store(
                "row_cursor",
                Expr::var("src_dense"),
                Expr::add(Expr::var("emit_slot"), Expr::u32(1)),
            ),
        ]
    };
    let intra_loop = |fact_body: Vec<Node>, gen_body: Vec<Node>| {
        Node::loop_for(
            "intra_i",
            Expr::u32(0),
            Expr::u32(intra_count),
            vec![
                Node::let_bind("intra_p", Expr::load("intra_proc", Expr::var("intra_i"))),
                Node::let_bind(
                    "intra_src_b",
                    Expr::load("intra_src_block", Expr::var("intra_i")),
                ),
                Node::let_bind(
                    "intra_dst_b",
                    Expr::load("intra_dst_block", Expr::var("intra_i")),
                ),
                Node::if_then(
                    Expr::and(
                        in_proc_block(Expr::var("intra_p"), Expr::var("intra_src_b")),
                        Expr::lt(Expr::var("intra_dst_b"), Expr::u32(blocks_per_proc)),
                    ),
                    vec![
                        Node::loop_for("fact", Expr::u32(0), Expr::u32(facts_per_proc), fact_body),
                        Node::loop_for("gen_i", Expr::u32(0), Expr::u32(gen_count), gen_body),
                    ],
                ),
            ],
        )
    };
    let inter_loop = |fact_body: Vec<Node>| {
        Node::loop_for(
            "inter_i",
            Expr::u32(0),
            Expr::u32(inter_count),
            vec![
                Node::let_bind(
                    "inter_sp",
                    Expr::load("inter_src_proc", Expr::var("inter_i")),
                ),
                Node::let_bind(
                    "inter_sb",
                    Expr::load("inter_src_block", Expr::var("inter_i")),
                ),
                Node::let_bind(
                    "inter_dp",
                    Expr::load("inter_dst_proc", Expr::var("inter_i")),
                ),
                Node::let_bind(
                    "inter_db",
                    Expr::load("inter_dst_block", Expr::var("inter_i")),
                ),
                Node::if_then(
                    Expr::and(
                        in_proc_block(Expr::var("inter_sp"), Expr::var("inter_sb")),
                        in_proc_block(Expr::var("inter_dp"), Expr::var("inter_db")),
                    ),
                    vec![Node::loop_for(
                        "fact",
                        Expr::u32(0),
                        Expr::u32(facts_per_proc),
                        fact_body,
                    )],
                ),
            ],
        )
    };

    // Dense kill bitmap: O(total_nodes + kill_count) setup, O(1) lookup per fact.
    // (Replaces per-fact linear scans over all KILL rules.)
    let build_kill_bitmap = vec![
        Node::loop_for(
            "killed_i",
            Expr::u32(0),
            Expr::u32(total_nodes),
            vec![Node::store("killed", Expr::var("killed_i"), Expr::u32(0))],
        ),
        Node::loop_for(
            "kill_i",
            Expr::u32(0),
            Expr::u32(kill_count),
            vec![
                Node::let_bind("kill_p", Expr::load("kill_proc", Expr::var("kill_i"))),
                Node::let_bind("kill_b", Expr::load("kill_block", Expr::var("kill_i"))),
                Node::let_bind("kill_f", Expr::load("kill_fact", Expr::var("kill_i"))),
                Node::if_then(
                    in_proc_block(Expr::var("kill_p"), Expr::var("kill_b")),
                    vec![
                        Node::let_bind(
                            "kill_slot",
                            idx_expr(
                                Expr::var("kill_p"),
                                Expr::var("kill_b"),
                                Expr::var("kill_f"),
                            ),
                        ),
                        Node::store("killed", Expr::var("kill_slot"), Expr::u32(1)),
                    ],
                ),
            ],
        ),
    ];
    let kill_lookup = vec![Node::let_bind(
        "is_killed",
        Expr::load(
            "killed",
            idx_expr(
                Expr::var("intra_p"),
                Expr::var("intra_src_b"),
                Expr::var("fact"),
            ),
        ),
    )];

    let mut count_intra_fact = kill_lookup.clone();
    count_intra_fact.push(Node::if_then(
        Expr::eq(Expr::var("is_killed"), Expr::u32(0)),
        vec![
            Node::let_bind(
                "src_dense",
                idx_expr(
                    Expr::var("intra_p"),
                    Expr::var("intra_src_b"),
                    Expr::var("fact"),
                ),
            ),
            count_row(Expr::var("src_dense")),
        ],
    ));

    let gen_step = |on_valid: Vec<Node>| {
        vec![
            Node::let_bind("gen_p", Expr::load("gen_proc", Expr::var("gen_i"))),
            Node::let_bind("gen_b", Expr::load("gen_block", Expr::var("gen_i"))),
            Node::let_bind("gen_f", Expr::load("gen_fact", Expr::var("gen_i"))),
            Node::if_then(
                Expr::and(
                    Expr::and(
                        Expr::eq(Expr::var("gen_p"), Expr::var("intra_p")),
                        Expr::eq(Expr::var("gen_b"), Expr::var("intra_src_b")),
                    ),
                    Expr::lt(Expr::var("gen_f"), Expr::u32(facts_per_proc)),
                ),
                on_valid,
            ),
        ]
    };
    let count_gen = gen_step(vec![
        Node::let_bind(
            "src_dense",
            idx_expr(Expr::var("intra_p"), Expr::var("intra_src_b"), Expr::u32(0)),
        ),
        count_row(Expr::var("src_dense")),
    ]);

    let mut fill_intra_fact = kill_lookup;
    fill_intra_fact.push(Node::if_then(
        Expr::eq(Expr::var("is_killed"), Expr::u32(0)),
        fill_pair(
            idx_expr(
                Expr::var("intra_p"),
                Expr::var("intra_src_b"),
                Expr::var("fact"),
            ),
            idx_expr(
                Expr::var("intra_p"),
                Expr::var("intra_dst_b"),
                Expr::var("fact"),
            ),
        ),
    ));
    let fill_gen = gen_step(fill_pair(
        idx_expr(Expr::var("intra_p"), Expr::var("intra_src_b"), Expr::u32(0)),
        idx_expr(
            Expr::var("intra_p"),
            Expr::var("intra_dst_b"),
            Expr::var("gen_f"),
        ),
    ));

    let mut entry = build_kill_bitmap;
    entry.extend([
        Node::loop_for(
            "row_i",
            Expr::u32(0),
            Expr::u32(row_ptr_count),
            vec![Node::store("row_ptr", Expr::var("row_i"), Expr::u32(0))],
        ),
        Node::store("col_len", Expr::u32(0), Expr::u32(0)),
        intra_loop(count_intra_fact, count_gen),
        inter_loop(vec![
            Node::let_bind(
                "src_dense",
                idx_expr(
                    Expr::var("inter_sp"),
                    Expr::var("inter_sb"),
                    Expr::var("fact"),
                ),
            ),
            count_row(Expr::var("src_dense")),
        ]),
    ]);
    entry.extend([
        Node::let_bind("prefix_sum", Expr::u32(0)),
        Node::loop_for(
            "prefix_row",
            Expr::u32(0),
            Expr::u32(total_nodes),
            vec![
                Node::let_bind(
                    "row_count",
                    Expr::load("row_ptr", Expr::add(Expr::var("prefix_row"), Expr::u32(1))),
                ),
                Node::assign(
                    "prefix_sum",
                    Expr::add(Expr::var("prefix_sum"), Expr::var("row_count")),
                ),
                Node::store(
                    "row_ptr",
                    Expr::add(Expr::var("prefix_row"), Expr::u32(1)),
                    Expr::var("prefix_sum"),
                ),
            ],
        ),
        Node::store("col_len", Expr::u32(0), Expr::var("prefix_sum")),
        Node::loop_for(
            "cursor_row",
            Expr::u32(0),
            Expr::u32(total_nodes),
            vec![Node::store(
                "row_cursor",
                Expr::var("cursor_row"),
                Expr::load("row_ptr", Expr::var("cursor_row")),
            )],
        ),
    ]);
    entry.push(intra_loop(fill_intra_fact, fill_gen));
    entry.push(inter_loop(fill_pair(
        idx_expr(
            Expr::var("inter_sp"),
            Expr::var("inter_sb"),
            Expr::var("fact"),
        ),
        idx_expr(
            Expr::var("inter_dp"),
            Expr::var("inter_db"),
            Expr::var("fact"),
        ),
    )));

    let ro = |name: &str, slot: u32, count: u32| {
        BufferDecl::storage(name, slot, BufferAccess::ReadOnly, DataType::U32)
            .with_count(count.max(1))
    };
    let rw = |name: &str, slot: u32, count: u32| {
        BufferDecl::storage(name, slot, BufferAccess::ReadWrite, DataType::U32).with_count(count)
    };
    Program::wrapped(
        vec![
            ro("intra_proc", 0, intra_count),
            ro("intra_src_block", 1, intra_count),
            ro("intra_dst_block", 2, intra_count),
            ro("inter_src_proc", 3, inter_count),
            ro("inter_src_block", 4, inter_count),
            ro("inter_dst_proc", 5, inter_count),
            ro("inter_dst_block", 6, inter_count),
            ro("gen_proc", 7, gen_count),
            ro("gen_block", 8, gen_count),
            ro("gen_fact", 9, gen_count),
            ro("kill_proc", 10, kill_count),
            ro("kill_block", 11, kill_count),
            ro("kill_fact", 12, kill_count),
            rw("killed", 13, total_nodes.max(1)),
            rw("row_ptr", 14, row_ptr_count),
            rw("row_cursor", 15, total_nodes.max(1)),
            rw("col_idx", 16, max_col_count.max(1)),
            rw("col_len", 17, 1),
        ],
        IFDS_CSR_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::logical_index(0), Expr::u32(0)),
                entry,
            )],
        )],
    )
}
