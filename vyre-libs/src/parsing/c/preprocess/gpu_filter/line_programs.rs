use super::program_helpers::{
    packed_byte_load, packed_bytes_input_buffer, singleton_u32_read_buffer, u32_read_buffer,
    u32_rw_buffer, wrap_gpu_filter_program,
};
use vyre::ir::{Expr, Node, Program};

pub(super) fn simple_line_newline_flags_program(n: u32) -> Program {
    let i = Expr::var("i");
    let byte = packed_byte_load("bytes_in", i.clone());
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(n)),
            vec![
                Node::let_bind("n_real", Expr::load("line_n_real", Expr::u32(0))),
                Node::store(
                    "newline_flags",
                    i.clone(),
                    Expr::select(
                        Expr::and(
                            Expr::lt(i.clone(), Expr::var("n_real")),
                            Expr::eq(byte, Expr::u32(b'\n' as u32)),
                        ),
                        Expr::u32(1),
                        Expr::u32(0),
                    ),
                ),
            ],
        ),
    ];
    wrap_gpu_filter_program(
        "vyre-libs::parsing::c::preprocess::simple_line_newline_flags",
        vec![
            packed_bytes_input_buffer("bytes_in", 0, n),
            u32_rw_buffer("newline_flags", 1, n),
            singleton_u32_read_buffer("line_n_real", 2),
        ],
        body,
    )
}

pub(super) fn simple_line_comment_starts_program(n: u32) -> Program {
    let i = Expr::var("i");
    let b0 = packed_byte_load("bytes_in", i.clone());
    let b1_addr = Expr::add(i.clone(), Expr::u32(1));
    let b1 = Expr::select(
        Expr::lt(b1_addr.clone(), Expr::load("line_n_real", Expr::u32(0))),
        packed_byte_load("bytes_in", b1_addr),
        Expr::u32(0),
    );
    let row = Expr::saturating_sub(
        Expr::load("newline_scan", i.clone()),
        Expr::load("newline_flags", i.clone()),
    );
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::load("line_n_real", Expr::u32(0))),
            vec![Node::if_then(
                Expr::and(
                    Expr::eq(b0, Expr::u32(b'/' as u32)),
                    Expr::eq(b1, Expr::u32(b'/' as u32)),
                ),
                vec![Node::let_bind(
                    "line_comment_start_old",
                    Expr::atomic_min("row_comment_starts", row, i.clone()),
                )],
            )],
        ),
    ];
    wrap_gpu_filter_program(
        "vyre-libs::parsing::c::preprocess::simple_line_comment_starts",
        vec![
            packed_bytes_input_buffer("bytes_in", 0, n),
            u32_read_buffer("newline_flags", 1, n),
            u32_read_buffer("newline_scan", 2, n),
            u32_rw_buffer("row_comment_starts", 3, n),
            singleton_u32_read_buffer("line_n_real", 4),
        ],
        body,
    )
}

pub(super) fn simple_line_comment_masks_program(n: u32) -> Program {
    let i = Expr::var("i");
    let b = packed_byte_load("bytes_in", i.clone());
    let row = Expr::saturating_sub(
        Expr::load("newline_scan", i.clone()),
        Expr::load("newline_flags", i.clone()),
    );
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(n)),
            vec![
                Node::let_bind("n_real", Expr::load("line_n_real", Expr::u32(0))),
                Node::if_then_else(
                    Expr::lt(i.clone(), Expr::var("n_real")),
                    vec![
                        Node::let_bind("is_newline", Expr::eq(b, Expr::u32(b'\n' as u32))),
                        Node::let_bind("row", row),
                        Node::let_bind("start", Expr::load("row_comment_starts", Expr::var("row"))),
                        Node::let_bind(
                            "comment_mask",
                            Expr::select(
                                Expr::and(
                                    Expr::ne(Expr::var("start"), Expr::u32(u32::MAX)),
                                    Expr::and(
                                        Expr::ge(i.clone(), Expr::var("start")),
                                        Expr::not(Expr::var("is_newline")),
                                    ),
                                ),
                                Expr::select(
                                    Expr::eq(i.clone(), Expr::var("start")),
                                    Expr::u32(2),
                                    Expr::u32(1),
                                ),
                                Expr::u32(0),
                            ),
                        ),
                        Node::store("comment_mask_out", i.clone(), Expr::var("comment_mask")),
                        Node::store(
                            "final_keep",
                            i.clone(),
                            Expr::select(
                                Expr::ne(Expr::var("comment_mask"), Expr::u32(1)),
                                Expr::u32(1),
                                Expr::u32(0),
                            ),
                        ),
                    ],
                    vec![
                        Node::store("comment_mask_out", i.clone(), Expr::u32(0)),
                        Node::store("final_keep", i.clone(), Expr::u32(0)),
                    ],
                ),
            ],
        ),
    ];
    wrap_gpu_filter_program(
        "vyre-libs::parsing::c::preprocess::simple_line_comment_masks",
        vec![
            packed_bytes_input_buffer("bytes_in", 0, n),
            u32_read_buffer("newline_flags", 1, n),
            u32_read_buffer("newline_scan", 2, n),
            u32_read_buffer("row_comment_starts", 3, n),
            u32_rw_buffer("final_keep", 4, n),
            u32_rw_buffer("comment_mask_out", 5, n),
            singleton_u32_read_buffer("line_n_real", 6),
        ],
        body,
    )
}
