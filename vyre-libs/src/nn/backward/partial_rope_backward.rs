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

    let i = Expr::var("i");
    let head = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(i.clone()),
        right: Box::new(Expr::u32(per_head)),
    };
    let pos_in_head = Expr::sub(i.clone(), Expr::mul(head.clone(), Expr::u32(per_head)));
    let pos = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(pos_in_head.clone()),
        right: Box::new(Expr::u32(head_dim)),
    };
    let dim = Expr::sub(pos_in_head, Expr::mul(pos.clone(), Expr::u32(head_dim)));
    let pair_idx = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(dim.clone()),
        right: Box::new(Expr::u32(2)),
    };
    let is_odd = Expr::sub(dim.clone(), Expr::mul(pair_idx.clone(), Expr::u32(2)));
    let table_idx = Expr::add(
        Expr::mul(pos.clone(), Expr::u32(half_rope)),
        pair_idx.clone(),
    );
    let head_base = Expr::mul(head, Expr::u32(per_head));
    let row_base = Expr::add(head_base, Expr::mul(pos, Expr::u32(head_dim)));
    let even_idx = Expr::add(row_base.clone(), Expr::mul(pair_idx.clone(), Expr::u32(2)));
    let odd_idx = Expr::add(even_idx.clone(), Expr::u32(1));

    let dy_even = Expr::load(grad_out, even_idx);
    let dy_odd = Expr::load(grad_out, odd_idx);
    let cos_val = Expr::load(cos_table, table_idx.clone());
    let sin_val = Expr::load(sin_table, table_idx);

    // Backward rotation (transpose):
    // grad_even = dy_even*cos + dy_odd*sin
    let grad_even = Expr::add(
        Expr::mul(dy_even.clone(), cos_val.clone()),
        Expr::mul(dy_odd.clone(), sin_val.clone()),
    );
    // grad_odd = -dy_even*sin + dy_odd*cos
    let grad_odd = Expr::add(
        Expr::mul(
            Expr::UnOp {
                op: vyre_foundation::ir::UnOp::Negate,
                operand: Box::new(dy_even),
            },
            sin_val,
        ),
        Expr::mul(dy_odd, cos_val),
    );

    let is_odd_f32 = Expr::cast(DataType::F32, is_odd);
    let one_minus = Expr::sub(Expr::f32(1.0), is_odd_f32.clone());
    let rotated_grad = Expr::add(
        Expr::mul(grad_even, one_minus),
        Expr::mul(grad_odd, is_odd_f32),
    );

    let passthrough = Expr::load(grad_out, i.clone());

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(total)),
            vec![
                Node::if_then(
                    Expr::lt(dim.clone(), Expr::u32(rope_dims)),
                    vec![Node::Store {
                        buffer: grad_in.into(),
                        index: i.clone(),
                        value: rotated_grad,
                    }],
                ),
                Node::if_then(
                    Expr::ge(dim, Expr::u32(rope_dims)),
                    vec![Node::Store {
                        buffer: grad_in.into(),
                        index: i,
                        value: passthrough,
                    }],
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

const EXPECTED_PARTIAL_ROPE_BACKWARD_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xA0, 0x40, 0x00, 0x00, 0xC0, 0x40,
];

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
            vec![vec![EXPECTED_PARTIAL_ROPE_BACKWARD_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
