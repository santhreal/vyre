//! Backward for `partial_rope`: rotate by negated angle.
//!
//! For pair (2k, 2k+1), forward was:
//!
//! ```text
//! out[2k]   = x[2k]*cos - x[2k+1]*sin
//! out[2k+1] = x[2k]*sin + x[2k+1]*cos
//! ```
//!
//! Backward (transpose of rotation matrix = rotation by -θ):
//!
//! ```text
//! grad_x[2k]   = grad_out[2k]*cos  + grad_out[2k+1]*sin
//! grad_x[2k+1] = -grad_out[2k]*sin + grad_out[2k+1]*cos
//! ```
//!
//! Dims beyond rope_dims: grad_x = grad_out (identity).

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::nn::partial_rope_backward";

/// Backward for partial_rope (F32). Rotates by negated angle.
#[must_use]
pub fn partial_rope_backward(
    grad_out: &str,
    cos_table: &str,
    sin_table: &str,
    grad_in: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    rope_dims: u32,
) -> Program {
    let total = num_heads * seq_len * head_dim;
    let half_rope = rope_dims / 2;
    let per_head = seq_len * head_dim;

    let div_u32 = |lhs: Expr, rhs: u32| Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(lhs),
        right: Box::new(Expr::u32(rhs)),
    };
    let rotated_grad = Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(Expr::var("dy_even"), Expr::var("cos_val")),
                Expr::mul(Expr::var("dy_odd"), Expr::var("sin_val")),
            ),
            Expr::sub(Expr::f32(1.0), Expr::var("is_odd_f32")),
        ),
        Expr::mul(
            Expr::add(
                Expr::mul(
                    Expr::UnOp {
                        op: vyre_foundation::ir::UnOp::Negate,
                        operand: Box::new(Expr::var("dy_even")),
                    },
                    Expr::var("sin_val"),
                ),
                Expr::mul(Expr::var("dy_odd"), Expr::var("cos_val")),
            ),
            Expr::var("is_odd_f32"),
        ),
    );

    let body = vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(total)),
            vec![
                Node::let_bind("head", div_u32(Expr::var("i"), per_head)),
                Node::let_bind(
                    "pos_in_head",
                    Expr::sub(
                        Expr::var("i"),
                        Expr::mul(Expr::var("head"), Expr::u32(per_head)),
                    ),
                ),
                Node::let_bind("pos", div_u32(Expr::var("pos_in_head"), head_dim)),
                Node::let_bind(
                    "dim",
                    Expr::sub(
                        Expr::var("pos_in_head"),
                        Expr::mul(Expr::var("pos"), Expr::u32(head_dim)),
                    ),
                ),
                Node::if_then_else(
                    Expr::lt(Expr::var("dim"), Expr::u32(rope_dims)),
                    vec![
                        Node::let_bind("pair_idx", div_u32(Expr::var("dim"), 2)),
                        Node::let_bind(
                            "is_odd_f32",
                            Expr::cast(
                                DataType::F32,
                                Expr::sub(
                                    Expr::var("dim"),
                                    Expr::mul(Expr::var("pair_idx"), Expr::u32(2)),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "table_idx",
                            Expr::add(
                                Expr::mul(Expr::var("pos"), Expr::u32(half_rope)),
                                Expr::var("pair_idx"),
                            ),
                        ),
                        Node::let_bind(
                            "even_idx",
                            Expr::add(
                                Expr::add(
                                    Expr::mul(Expr::var("head"), Expr::u32(per_head)),
                                    Expr::mul(Expr::var("pos"), Expr::u32(head_dim)),
                                ),
                                Expr::mul(Expr::var("pair_idx"), Expr::u32(2)),
                            ),
                        ),
                        Node::let_bind("dy_even", Expr::load(grad_out, Expr::var("even_idx"))),
                        Node::let_bind(
                            "dy_odd",
                            Expr::load(grad_out, Expr::add(Expr::var("even_idx"), Expr::u32(1))),
                        ),
                        Node::let_bind("cos_val", Expr::load(cos_table, Expr::var("table_idx"))),
                        Node::let_bind("sin_val", Expr::load(sin_table, Expr::var("table_idx"))),
                        Node::store(grad_in, Expr::var("i"), rotated_grad),
                    ],
                    vec![Node::store(
                        grad_in,
                        Expr::var("i"),
                        Expr::load(grad_out, Expr::var("i")),
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(grad_out, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(total),
            BufferDecl::storage(cos_table, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(seq_len * half_rope),
            BufferDecl::storage(sin_table, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(seq_len * half_rope),
            BufferDecl::output(grad_in, 3, DataType::F32).with_count(total),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || partial_rope_backward("grad_out", "cos", "sin", "grad_in", 1, 1, 4, 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 0.0, 5.0, 6.0]), // grad_out
                to_f32(&[1.0]),                   // cos
                to_f32(&[0.0]),                   // sin
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA0, 0x40, 0x00, 0x00,
                0xC0, 0x40,
            ]]]
        }),
    )
    .with_category("nn")
    .with_uncharacterized()
}
