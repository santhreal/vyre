//! Contraction program assembly routines for dense GEMM, batched matmul, projections, and Strassen.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::gemm_algebra::ContractionSemiring;
use super::ContractionEpilogue;
use crate::plumbing::operand::tensor_ref::TensorRefError;

/// Assemble 2D GEMM with 1D linear dispatch.
pub(crate) fn build_matmul_2d_linear(
    op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    bias: Option<&str>,
    out: &str,
    m: u32,
    k: u32,
    n: u32,
    dtype: &DataType,
    semiring: &ContractionSemiring,
    epilogue: &ContractionEpilogue,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let a_count = m
        .checked_mul(k)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![m, k],
        })?;
    let b_count = k
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![k, n],
        })?;
    let out_count = m
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![m, n],
        })?;

    let idx = Expr::LogicalIndex { axis: 0 };
    let row_expr = Expr::div(idx.clone(), Expr::u32(n));
    let col_expr = Expr::rem(idx.clone(), Expr::u32(n));

    let a_load = Expr::load(
        a,
        Expr::add(Expr::mul(Expr::var("row"), Expr::u32(k)), Expr::var("kk")),
    );
    let b_load = Expr::load(
        b,
        Expr::add(Expr::mul(Expr::var("kk"), Expr::u32(n)), Expr::var("col")),
    );
    let combined = semiring.combine_expr(a_load, b_load);
    let folded = semiring.accumulate_expr(Expr::var("acc"), combined);

    let (initial_acc, store_value) = match epilogue {
        ContractionEpilogue::None => (semiring.identity_expr(dtype), Expr::var("acc")),
        ContractionEpilogue::Bias { buffer, .. } => (
            semiring.identity_expr(dtype),
            Expr::add(Expr::var("acc"), Expr::load(buffer, Expr::var("col"))),
        ),
        ContractionEpilogue::Activation {
            bias: Some(bias_buf),
            activation,
        } => (
            Expr::load(bias_buf, Expr::var("col")),
            activation(Expr::var("acc")),
        ),
        ContractionEpilogue::Activation {
            bias: None,
            activation,
        } => (semiring.identity_expr(dtype), activation(Expr::var("acc"))),
        ContractionEpilogue::QuantizedScale {
            row_scales,
            batch_scales,
        } => (
            semiring.identity_expr(dtype),
            Expr::mul(
                Expr::mul(Expr::var("acc"), Expr::load(row_scales, Expr::var("row"))),
                Expr::load(batch_scales, Expr::var("batch")),
            ),
        ),
    };

    let body = vec![Node::if_then(
        Expr::lt(idx.clone(), Expr::buf_len(out)),
        vec![
            Node::let_bind("row", row_expr),
            Node::let_bind("col", col_expr),
            Node::let_bind("acc", initial_acc),
            Node::loop_for(
                "kk",
                Expr::u32(0),
                Expr::u32(k),
                vec![Node::assign("acc", folded)],
            ),
            Node::Store {
                buffer: out.into(),
                index: idx,
                value: store_value,
            },
        ],
    )];

    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
    ];
    let out_slot = if let Some(bias_name) = bias {
        buffers.push(
            BufferDecl::storage(bias_name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, out_slot, dtype.clone()).with_count(out_count));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 3D batched GEMM: `out[b, i, j] = sum_k a[b, i, k] * b[b, k, j]`.
pub(crate) fn build_batched_3d_contraction(
    _op_id: &'static str,
    generator: &'static str,
    a: &str,
    b: &str,
    out: &str,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    dtype: &DataType,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let a_batch_stride = m
        .checked_mul(k)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![batch, m, k],
        })?;
    let b_batch_stride = k
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![batch, k, n],
        })?;
    let out_batch_stride =
        m.checked_mul(n)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: out.to_string(),
                shape: vec![batch, m, n],
            })?;
    let a_count =
        batch
            .checked_mul(a_batch_stride)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: a.to_string(),
                shape: vec![batch, m, k],
            })?;
    let b_count =
        batch
            .checked_mul(b_batch_stride)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: b.to_string(),
                shape: vec![batch, k, n],
            })?;
    let out_count = batch.checked_mul(out_batch_stride).ok_or_else(|| {
        TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![batch, m, n],
        }
    })?;

    let idx = Expr::var("idx");
    let batch_idx = Expr::var("batch_idx");
    let row = Expr::var("row");
    let col = Expr::var("col");
    let local_idx = Expr::var("local_idx");

    let body = vec![
        Node::let_bind("idx", Expr::LogicalIndex { axis: 0 }),
        Node::let_bind(
            "batch_idx",
            Expr::div(idx.clone(), Expr::u32(out_batch_stride)),
        ),
        Node::let_bind(
            "local_idx",
            Expr::rem(idx.clone(), Expr::u32(out_batch_stride)),
        ),
        Node::let_bind("row", Expr::div(local_idx.clone(), Expr::u32(n))),
        Node::let_bind("col", Expr::rem(local_idx.clone(), Expr::u32(n))),
        Node::if_then(
            Expr::lt(idx.clone(), Expr::buf_len(out)),
            vec![
                Node::let_bind("acc", Expr::f32(0.0)),
                Node::loop_for(
                    "kk",
                    Expr::u32(0),
                    Expr::u32(k),
                    vec![Node::assign(
                        "acc",
                        Expr::add(
                            Expr::var("acc"),
                            Expr::mul(
                                Expr::load(
                                    a,
                                    Expr::add(
                                        Expr::mul(batch_idx.clone(), Expr::u32(a_batch_stride)),
                                        Expr::add(
                                            Expr::mul(row.clone(), Expr::u32(k)),
                                            Expr::var("kk"),
                                        ),
                                    ),
                                ),
                                Expr::load(
                                    b,
                                    Expr::add(
                                        Expr::mul(batch_idx.clone(), Expr::u32(b_batch_stride)),
                                        Expr::add(
                                            Expr::mul(Expr::var("kk"), Expr::u32(n)),
                                            col.clone(),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: idx,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    let buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
        BufferDecl::output(out, 2, dtype.clone()).with_count(out_count),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble row-batched affine projection with F32 accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_batched_rows_contraction(
    _op_id: &'static str,
    generator: &'static str,
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    dtype: &DataType,
    acc_dtype: &DataType,
    weight_out_in: bool,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let input_count =
        rows.checked_mul(in_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: x.to_string(),
                shape: vec![rows, in_dim],
            })?;
    let output_count =
        rows.checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: out.to_string(),
                shape: vec![rows, out_dim],
            })?;
    let weight_count =
        in_dim
            .checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: w.to_string(),
                shape: vec![in_dim, out_dim],
            })?;

    let index = Expr::var("index");
    let row = Expr::div(index.clone(), Expr::u32(out_dim));
    let column = Expr::rem(index.clone(), Expr::u32(out_dim));
    let accumulator = bias.map_or_else(
        || Expr::f32(0.0),
        |name| Expr::cast(acc_dtype.clone(), Expr::load(name, Expr::var("column"))),
    );
    let weight_index = if weight_out_in {
        Expr::add(
            Expr::mul(Expr::var("column"), Expr::u32(in_dim)),
            Expr::var("inner"),
        )
    } else {
        Expr::add(
            Expr::mul(Expr::var("inner"), Expr::u32(out_dim)),
            Expr::var("column"),
        )
    };

    let body = vec![
        Node::let_bind("index", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(output_count)),
            vec![
                Node::let_bind("row", row),
                Node::let_bind("column", column),
                Node::let_bind("accumulator", accumulator),
                Node::loop_for(
                    "inner",
                    Expr::u32(0),
                    Expr::u32(in_dim),
                    vec![Node::assign(
                        "accumulator",
                        Expr::add(
                            Expr::var("accumulator"),
                            Expr::mul(
                                Expr::cast(
                                    acc_dtype.clone(),
                                    Expr::load(
                                        x,
                                        Expr::add(
                                            Expr::mul(Expr::var("row"), Expr::u32(in_dim)),
                                            Expr::var("inner"),
                                        ),
                                    ),
                                ),
                                Expr::cast(acc_dtype.clone(), Expr::load(w, weight_index)),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index,
                    value: Expr::cast(dtype.clone(), Expr::var("accumulator")),
                },
            ],
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(input_count),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let output_slot = if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(output_count));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 1D block-tiled reference contraction loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_1d_contraction(
    _op_id: &'static str,
    generator: &'static str,
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    _m: u32,
    in_dim: u32,
    out_dim: u32,
    tile: u32,
    dtype: &DataType,
) -> Result<Program, TensorRefError> {
    let weight_count =
        in_dim
            .checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: w.to_string(),
                shape: vec![in_dim, out_dim],
            })?;
    let tile_count = in_dim.div_ceil(tile);
    let lane = Expr::var("lane");
    let kk = Expr::var("kk");

    let initial_acc = bias.map_or_else(|| Expr::u32(0), |b| Expr::load(b, lane.clone()));

    let body = vec![
        Node::let_bind("lane", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(lane.clone(), Expr::u32(out_dim)),
            vec![
                Node::let_bind("acc", initial_acc),
                Node::loop_for(
                    "tile_idx",
                    Expr::u32(0),
                    Expr::u32(tile_count),
                    vec![
                        Node::let_bind(
                            "tile_base",
                            Expr::mul(Expr::var("tile_idx"), Expr::u32(tile)),
                        ),
                        Node::loop_for(
                            "tile_k",
                            Expr::u32(0),
                            Expr::u32(tile),
                            vec![
                                Node::let_bind(
                                    "kk",
                                    Expr::add(Expr::var("tile_base"), Expr::var("tile_k")),
                                ),
                                Node::if_then(
                                    Expr::lt(kk.clone(), Expr::u32(in_dim)),
                                    vec![Node::assign(
                                        "acc",
                                        Expr::add(
                                            Expr::var("acc"),
                                            Expr::mul(
                                                Expr::load(x, kk.clone()),
                                                Expr::load(
                                                    w,
                                                    Expr::add(
                                                        Expr::mul(kk.clone(), Expr::u32(out_dim)),
                                                        lane.clone(),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )],
                                ),
                            ],
                        ),
                    ],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: lane,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(in_dim),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let output_slot = if let Some(b) = bias {
        buffers.push(
            BufferDecl::storage(b, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(out_dim));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [256, 1, 1], vec![region]))
}

/// Assemble Matrix-Vector contraction.
pub(crate) fn build_matvec_contraction(
    _op_id: &'static str,
    generator: &'static str,
    matrix: &str,
    vector: &str,
    out: &str,
    n: u32,
    matrix_cells: u32,
    dtype: &DataType,
    semiring: &ContractionSemiring,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let row = Expr::LogicalIndex { axis: 0 };
    let row_base = Expr::mul(row.clone(), Expr::u32(n));

    let combined = semiring.combine_expr(
        Expr::load(matrix, Expr::add(row_base, Expr::var("j"))),
        Expr::load(vector, Expr::var("j")),
    );
    let folded = semiring.accumulate_expr(Expr::var("acc"), combined);

    let body = vec![Node::if_then(
        Expr::lt(row.clone(), Expr::u32(n)),
        vec![
            Node::let_bind("acc", semiring.identity_expr(dtype)),
            Node::let_bind("row_base", Expr::mul(row.clone(), Expr::u32(n))),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::assign("acc", folded)],
            ),
            Node::store(out, row, Expr::var("acc")),
        ],
    )];

    let buffers = vec![
        BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, dtype.clone())
            .with_count(matrix_cells),
        BufferDecl::storage(vector, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(n),
        BufferDecl::storage(out, 2, BufferAccess::ReadWrite, dtype.clone()).with_count(n),
    ];

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 2x2 Strassen 7-multiplication closed-form Program.
pub(crate) fn build_strassen_2x2(
    _op_id: &'static str,
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
pub(crate) fn build_strassen_one_level(
    _op_id: &'static str,
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
