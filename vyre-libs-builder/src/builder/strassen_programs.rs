//! Strassen contraction program assembly, closed-form and one level deep.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::plumbing::operand::tensor_ref::TensorRefError;

/// Assemble 2x2 Strassen 7-multiplication closed-form Program.
pub(super) fn build_strassen_2x2(
    generator: &'static str,
    a: &str,
    b: &str,
    c: &str,
) -> Result<Program, TensorRefError> {
    let body = vec![
        Node::let_bind("a00", Expr::load(a, Expr::u32(0))),
        Node::let_bind("a01", Expr::load(a, Expr::u32(1))),
        Node::let_bind("a10", Expr::load(a, Expr::u32(2))),
        Node::let_bind("a11", Expr::load(a, Expr::u32(3))),
        Node::let_bind("b00", Expr::load(b, Expr::u32(0))),
        Node::let_bind("b01", Expr::load(b, Expr::u32(1))),
        Node::let_bind("b10", Expr::load(b, Expr::u32(2))),
        Node::let_bind("b11", Expr::load(b, Expr::u32(3))),
        Node::let_bind(
            "m1",
            Expr::mul(
                Expr::add(Expr::var("a00"), Expr::var("a11")),
                Expr::add(Expr::var("b00"), Expr::var("b11")),
            ),
        ),
        Node::let_bind(
            "m2",
            Expr::mul(
                Expr::add(Expr::var("a10"), Expr::var("a11")),
                Expr::var("b00"),
            ),
        ),
        Node::let_bind(
            "m3",
            Expr::mul(
                Expr::var("a00"),
                Expr::sub(Expr::var("b01"), Expr::var("b11")),
            ),
        ),
        Node::let_bind(
            "m4",
            Expr::mul(
                Expr::var("a11"),
                Expr::sub(Expr::var("b10"), Expr::var("b00")),
            ),
        ),
        Node::let_bind(
            "m5",
            Expr::mul(
                Expr::add(Expr::var("a00"), Expr::var("a01")),
                Expr::var("b11"),
            ),
        ),
        Node::let_bind(
            "m6",
            Expr::mul(
                Expr::sub(Expr::var("a10"), Expr::var("a00")),
                Expr::add(Expr::var("b00"), Expr::var("b01")),
            ),
        ),
        Node::let_bind(
            "m7",
            Expr::mul(
                Expr::sub(Expr::var("a01"), Expr::var("a11")),
                Expr::add(Expr::var("b10"), Expr::var("b11")),
            ),
        ),
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(0),
            value: Expr::add(
                Expr::sub(Expr::add(Expr::var("m1"), Expr::var("m4")), Expr::var("m5")),
                Expr::var("m7"),
            ),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(1),
            value: Expr::add(Expr::var("m3"), Expr::var("m5")),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(2),
            value: Expr::add(Expr::var("m2"), Expr::var("m4")),
        },
        Node::Store {
            buffer: c.into(),
            index: Expr::u32(3),
            value: Expr::add(
                Expr::add(Expr::sub(Expr::var("m1"), Expr::var("m2")), Expr::var("m3")),
                Expr::var("m6"),
            ),
        },
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::F32).with_count(4),
        BufferDecl::output(c, 2, DataType::F32).with_count(4),
    ];

    // Every store index is a constant, so the four output words are the same in
    // every invocation of the grid a backend derives from the output length, and
    // in every invocation a fusion widens this arm to. One invocation owns the
    // contraction, so the guard names it.
    let body = vec![Node::if_then(Expr::is_first_logical_point(), body)];
    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [1, 1, 1], vec![region]))
}

/// Assemble 1-level recursive Strassen 7-multiplication block Program.
pub(super) fn build_strassen_one_level(
    generator: &'static str,
    a: &str,
    b: &str,
    c: &str,
    n: u32,
) -> Result<Program, TensorRefError> {
    let half = n / 2;
    let total = n
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: c.to_string(),
            shape: vec![n, n],
        })?;

    let body = vec![
        Node::let_bind("flat", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("flat"), Expr::u32(total)),
            vec![
                Node::let_bind("row", Expr::div(Expr::var("flat"), Expr::u32(n))),
                Node::let_bind("col", Expr::rem(Expr::var("flat"), Expr::u32(n))),
                Node::let_bind("q_row", Expr::div(Expr::var("row"), Expr::u32(half))),
                Node::let_bind("q_col", Expr::div(Expr::var("col"), Expr::u32(half))),
                Node::let_bind("sr", Expr::rem(Expr::var("row"), Expr::u32(half))),
                Node::let_bind("sc", Expr::rem(Expr::var("col"), Expr::u32(half))),
                Node::let_bind("c_val", Expr::f32(0.0)),
                Node::let_bind("m1", Expr::f32(0.0)),
                Node::let_bind("m2", Expr::f32(0.0)),
                Node::let_bind("m3", Expr::f32(0.0)),
                Node::let_bind("m4", Expr::f32(0.0)),
                Node::let_bind("m5", Expr::f32(0.0)),
                Node::let_bind("m6", Expr::f32(0.0)),
                Node::let_bind("m7", Expr::f32(0.0)),
                Node::loop_for(
                    "k",
                    Expr::u32(0),
                    Expr::u32(half),
                    vec![
                        Node::let_bind(
                            "a11",
                            Expr::load(
                                a,
                                Expr::add(Expr::mul(Expr::var("sr"), Expr::u32(n)), Expr::var("k")),
                            ),
                        ),
                        Node::let_bind(
                            "a12",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(Expr::var("sr"), Expr::u32(n)),
                                    Expr::add(Expr::u32(half), Expr::var("k")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "a21",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("sr"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::var("k"),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "a22",
                            Expr::load(
                                a,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("sr"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::add(Expr::u32(half), Expr::var("k")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b11",
                            Expr::load(
                                b,
                                Expr::add(Expr::mul(Expr::var("k"), Expr::u32(n)), Expr::var("sc")),
                            ),
                        ),
                        Node::let_bind(
                            "b12",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(Expr::var("k"), Expr::u32(n)),
                                    Expr::add(Expr::u32(half), Expr::var("sc")),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b21",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("k"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::var("sc"),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "b22",
                            Expr::load(
                                b,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("k"), Expr::u32(half)),
                                        Expr::u32(n),
                                    ),
                                    Expr::add(Expr::u32(half), Expr::var("sc")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m1",
                            Expr::add(
                                Expr::var("m1"),
                                Expr::mul(
                                    Expr::add(Expr::var("a11"), Expr::var("a22")),
                                    Expr::add(Expr::var("b11"), Expr::var("b22")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m2",
                            Expr::add(
                                Expr::var("m2"),
                                Expr::mul(
                                    Expr::add(Expr::var("a21"), Expr::var("a22")),
                                    Expr::var("b11"),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m3",
                            Expr::add(
                                Expr::var("m3"),
                                Expr::mul(
                                    Expr::var("a11"),
                                    Expr::sub(Expr::var("b12"), Expr::var("b22")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m4",
                            Expr::add(
                                Expr::var("m4"),
                                Expr::mul(
                                    Expr::var("a22"),
                                    Expr::sub(Expr::var("b21"), Expr::var("b11")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m5",
                            Expr::add(
                                Expr::var("m5"),
                                Expr::mul(
                                    Expr::add(Expr::var("a11"), Expr::var("a12")),
                                    Expr::var("b22"),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m6",
                            Expr::add(
                                Expr::var("m6"),
                                Expr::mul(
                                    Expr::sub(Expr::var("a21"), Expr::var("a11")),
                                    Expr::add(Expr::var("b11"), Expr::var("b12")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "m7",
                            Expr::add(
                                Expr::var("m7"),
                                Expr::mul(
                                    Expr::sub(Expr::var("a12"), Expr::var("a22")),
                                    Expr::add(Expr::var("b21"), Expr::var("b22")),
                                ),
                            ),
                        ),
                    ],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(0)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(0)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(
                            Expr::sub(Expr::add(Expr::var("m1"), Expr::var("m4")), Expr::var("m5")),
                            Expr::var("m7"),
                        ),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(0)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(1)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(Expr::var("m3"), Expr::var("m5")),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(1)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(0)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(Expr::var("m2"), Expr::var("m4")),
                    )],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("q_row"), Expr::u32(1)),
                        Expr::eq(Expr::var("q_col"), Expr::u32(1)),
                    ),
                    vec![Node::assign(
                        "c_val",
                        Expr::add(
                            Expr::add(Expr::sub(Expr::var("m1"), Expr::var("m2")), Expr::var("m3")),
                            Expr::var("m6"),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: c.into(),
                    index: Expr::var("flat"),
                    value: Expr::var("c_val"),
                },
            ],
        ),
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::F32).with_count(total),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::F32).with_count(total),
        BufferDecl::output(c, 2, DataType::F32).with_count(total),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [64, 1, 1], vec![region]))
}
